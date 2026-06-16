"""Minimal REPL — stateful code execution with top-level-await suspension.

One :class:`REPL` per agent. It runs an LLM code block in a persistent
namespace and captures stdout. If the block has a *top-level* ``await``
(i.e. ``await launch_subagents([...])``), the block is compiled as a
coroutine and driven with ``send()``; it suspends when it yields a
:class:`~rflow.graph.WaitRequest` and resumes when the engine sends the
child results back in.

stdout is captured through a thread-local buffer so REPLs running in
parallel threads don't clobber each other's output.
"""

from __future__ import annotations

import ast
import inspect
import io
import os
import re
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rflow.graph import WaitRequest

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_capture = threading.local()

# ``os.chdir`` mutates process-global state, so when a REPL runs with a fixed
# ``working_directory`` we serialize the chdir + execute window across threads.
# Without this, agent A could capture the cwd that agent B has already chdir'd
# into and "restore" the wrong directory, stranding the host. The body is one
# code block, so the contention is negligible; the lock is only taken when a
# working directory is actually set.
_CWD_LOCK = threading.Lock()


class DoneSignal(BaseException):
    """Raised by ``done()`` to stop the block. BaseException so a broad
    ``except Exception`` in agent code can't swallow a successful finish."""


def _has_top_level_await(tree: ast.AST) -> bool:
    """True iff ``tree`` has an ``await`` outside any nested scope."""

    boundary = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Lambda,
        ast.ClassDef,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )
    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Await):
            return True
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, boundary):
                stack.append(child)
    return False


class _StdoutProxy:
    """Thread-aware stdout: routes to a per-thread buffer when one is active."""

    def __init__(self, real):
        self.real = real

    def write(self, s):
        buf = getattr(_capture, "buf", None)
        return buf.write(s) if buf is not None else self.real.write(s)

    def flush(self):
        self.real.flush()

    def __getattr__(self, name):
        return getattr(self.real, name)


class REPL:
    """A stateful Python namespace that can suspend on top-level await.

    Outcomes from :meth:`start` / :meth:`resume` are ``(suspended, payload)``:

    * ``(False, stdout)`` — the block ran to completion (or errored; see
      :attr:`errored`);
    * ``(True, (WaitRequest, pre_output))`` — it suspended on a top-level
      ``await launch_subagents(...)``.
    """

    def __init__(self, working_directory: str | Path | None = None) -> None:
        self.namespace: dict[str, Any] = {"__builtins__": __builtins__}
        self.env: dict[str, Any] = {}
        self.coro = None
        self.errored = False
        self._buf = io.StringIO()
        # ``None`` → run in the process cwd as-is (the default; no chdir, no
        # lock). A path → each code block runs with the cwd switched into it
        # (created if missing), serialized via ``_CWD_LOCK`` across agents.
        self.working_directory: Path | None = None
        if working_directory is not None:
            self.working_directory = Path(working_directory).resolve()
            self.working_directory.mkdir(parents=True, exist_ok=True)
        if not isinstance(sys.stdout, _StdoutProxy):
            sys.stdout = _StdoutProxy(sys.stdout)

    # ── stdout capture ────────────────────────────────────────────────

    @contextmanager
    def _capture(self):
        """Run a fresh step with this thread's stdout routed into a buffer.

        Resets the buffer/error state on entry, and turns any agent error
        into captured output (``done()`` finishes cleanly).
        """
        self._buf = io.StringIO()
        self.errored = False
        _capture.buf = self._buf
        prev_cwd: str | None = None
        if self.working_directory is not None:
            _CWD_LOCK.acquire()
            prev_cwd = os.getcwd()
            os.chdir(self.working_directory)
        try:
            yield
        except DoneSignal:
            pass
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except BaseException as exc:  # noqa: BLE001 - agent errors become output
            self._buf.write(f"\n{type(exc).__name__}: {exc}")
            self.errored = True
        finally:
            _capture.buf = None
            if prev_cwd is not None:
                os.chdir(prev_cwd)
                _CWD_LOCK.release()

    @property
    def _output(self) -> str:
        return _ANSI_RE.sub("", self._buf.getvalue()).strip()

    # ── execution ─────────────────────────────────────────────────────

    def start(self, code: str) -> tuple[bool, object]:
        """Run a fresh code block in the persistent namespace."""
        self.coro = None
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            self.errored = True
            return False, f"SyntaxError: {exc}"
        with self._capture():
            self.coro = self._exec_block(tree)
        if self.coro is None:
            return False, self._output
        return self._drive(None)

    def resume(self, send_value: object) -> tuple[bool, object]:
        """Send ``send_value`` into the suspended coroutine and drive on."""
        return self._drive(send_value)

    def close(self) -> None:
        """No-op: in-process REPLs hold no external resources.

        Present so :class:`REPL` satisfies the ``ReplBackend`` protocol that
        remote backends (Docker/Modal) implement with real teardown.
        """

    def _exec_block(self, tree: ast.AST):
        """Execute the block. Return its coroutine if it has a top-level
        ``await``, else ``None`` (a plain block that already ran)."""
        if not _has_top_level_await(tree):
            exec(compile(tree, "<rlm>", "exec"), self.namespace)
            return None
        code = compile(tree, "<rlm>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        result = eval(code, self.namespace)  # creates the coroutine; runs nothing yet
        return result if inspect.iscoroutine(result) else None

    def _drive(self, send_value: object) -> tuple[bool, object]:
        """Advance the coroutine to its next ``await`` (suspend) or its end."""
        if self.coro is None:
            self.errored = True
            return False, "RuntimeError: no suspended coroutine to resume"
        with self._capture():
            try:
                request = self.coro.send(send_value)
            except StopIteration:
                self.coro = None
                return False, self._output
            if isinstance(request, WaitRequest):
                return True, (request, self._output)
            raise TypeError("only `await launch_subagents(...)` is supported")
        return False, self._output


__all__ = ["DoneSignal", "REPL"]
