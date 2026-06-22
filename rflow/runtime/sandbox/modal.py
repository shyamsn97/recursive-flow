"""Modal REPL backend — run an agent's code in a remote Modal Sandbox.

Requires ``modal`` (``pip install rlmflow[modal]``). The sandbox runs one
long-lived :mod:`rflow.runtime.repl_server` as its entrypoint; the host talks to
it over Modal's native Sandbox ``stdin`` / ``stdout`` streams.

Usage::

    import modal
    from rflow.runtime.sandbox.modal import ModalRuntime

    runtime = ModalRuntime(
        app_name="my-rlm-app",
        image=modal.Image.debian_slim().pip_install("rlmflow"),
    )
    flow = Flow(llm, runtime=runtime)
"""

from __future__ import annotations

import json
import shlex
import threading
from collections import deque
from queue import Empty, Queue
from typing import TYPE_CHECKING

from rflow.runtime.runtime import RemoteRepl, ReplBackend, Runtime

if TYPE_CHECKING:
    from rflow.graph import Graph

_REPL_ENTRYPOINT = "from rflow.runtime.repl_server import main; main()"


class ModalRepl(RemoteRepl):
    """A :class:`RemoteRepl` whose transport is a Modal Sandbox's stdio."""

    def __init__(
        self,
        app_name: str = "rlmflow",
        *,
        remote_workdir: str = "/workspace",
        image=None,
        timeout: int = 3600,
        repl_timeout: float = 30,
        **container_kwargs,
    ) -> None:
        super().__init__()
        self.app_name = app_name
        self.remote_workdir = remote_workdir
        self.image = image
        self.timeout = timeout
        self.repl_timeout = repl_timeout
        self.container_kwargs = container_kwargs
        self.container = None
        self._stdout_queue: Queue[str | None] | None = None
        self._stdout_iter = None
        self._stdout_pending = ""
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._closing = threading.Event()

    def _ensure_sandbox(self):
        if self.container is not None:
            return
        try:
            import modal
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "ModalRepl requires the optional `modal` dependency. "
                "Install it with `pip install rlmflow[modal]`."
            ) from exc
        app = modal.App.lookup(self.app_name, create_if_missing=True)
        image = self.image or modal.Image.debian_slim().pip_install("rlmflow")
        entrypoint = (
            f"mkdir -p {shlex.quote(self.remote_workdir)} && "
            f"exec python -u -c {shlex.quote(_REPL_ENTRYPOINT)} "
            f"--workdir {shlex.quote(self.remote_workdir)}"
        )
        self.container = modal.Sandbox.create(
            "sh",
            "-lc",
            entrypoint,
            app=app,
            image=image,
            timeout=self.timeout,
            **self.container_kwargs,
        )
        self._closing.clear()
        self._stdout_iter = iter(self.container.stdout)
        self._stdout_pending = ""

    def _start_reader(self, stream, output: Queue) -> None:
        def read() -> None:
            pending = ""
            try:
                for chunk in stream:
                    pending += _to_text(chunk)
                    while "\n" in pending:
                        line, pending = pending.split("\n", 1)
                        if line:
                            output.put(line)
                if pending:
                    output.put(pending)
            except Exception as exc:  # noqa: BLE001 - stream closes during teardown
                if not self._is_expected_stream_close(exc):
                    output.put(f'{{"error": "Modal stdout reader failed: {exc}"}}')
            finally:
                output.put(None)

        threading.Thread(target=read, daemon=True).start()

    def _start_stderr_reader(self, stream) -> None:
        def read() -> None:
            try:
                for line in stream:
                    self._stderr_tail.append(str(line))
            except Exception as exc:  # noqa: BLE001 - stream closes during teardown
                if not self._is_expected_stream_close(exc):
                    self._stderr_tail.append(f"Modal stderr reader failed: {exc}")

        threading.Thread(target=read, daemon=True).start()

    def send(self, msg: dict) -> None:
        self._ensure_sandbox()
        assert self.container is not None
        self.container.stdin.write(json.dumps(msg) + "\n")
        self.container.stdin.drain()

    def recv(self) -> dict:
        self._ensure_sandbox()
        if self._stdout_queue is not None:
            line = self._recv_from_queue()
        else:
            line = self._recv_from_stream()
        return json.loads(line)

    def _recv_from_queue(self) -> str:
        assert self._stdout_queue is not None
        try:
            line = self._stdout_queue.get(timeout=self.repl_timeout + 5)
        except Empty as exc:
            raise self._timeout_error() from exc
        if line is None:
            stderr = "".join(self._stderr_tail).strip()
            raise RuntimeError(
                f"Modal rlmflow REPL exited. stderr: {stderr or '<empty>'}"
            )
        return line

    def _recv_from_stream(self) -> str:
        if self._stdout_iter is None:
            raise RuntimeError("Modal stdout stream is not available")
        try:
            while True:
                if "\n" in self._stdout_pending:
                    line, self._stdout_pending = self._stdout_pending.split("\n", 1)
                    if line:
                        return line
                self._stdout_pending += _to_text(next(self._stdout_iter))
        except StopIteration as exc:
            stderr = "".join(self._stderr_tail).strip()
            raise RuntimeError(
                f"Modal rlmflow REPL exited. stderr: {stderr or '<empty>'}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            if self._is_expected_stream_close(exc):
                stderr = "".join(self._stderr_tail).strip()
                raise RuntimeError(
                    f"Modal rlmflow REPL exited. stderr: {stderr or '<empty>'}"
                ) from exc
            raise

    def _timeout_error(self) -> RuntimeError:
        stderr = "".join(self._stderr_tail).strip()
        return RuntimeError(
            "Modal rlmflow REPL did not respond within "
            f"{self.repl_timeout}s. stderr: {stderr or '<empty>'}"
        )

    def close(self) -> None:
        container, self.container = self.container, None
        self._closing.set()
        self._stdout_queue = None
        self._stdout_iter = None
        self._stdout_pending = ""
        self._stderr_tail.clear()
        if container is None:
            return
        try:
            container.terminate()
        except Exception:  # noqa: BLE001
            pass

    def _is_expected_stream_close(self, exc: Exception) -> bool:
        if self._closing.is_set():
            return True
        return exc.__class__.__name__ in {
            "ClientClosed",
            "StreamTerminatedError",
            "GRPCError",
        }


def _to_text(data: object) -> str:
    if isinstance(data, bytes):
        return data.decode(errors="replace")
    return str(data)


class ModalRuntime(Runtime):
    """Run each agent's code in a remote Modal Sandbox.

    The user-facing object you hand to ``Flow(runtime=...)``; :meth:`open` mints
    one :class:`ModalRepl` per agent. ``remote_workdir`` is the in-sandbox
    directory agent code runs in. All keyword arguments pass through to
    :class:`ModalRepl`.
    """

    def __init__(
        self,
        app_name: str = "rlmflow",
        *,
        remote_workdir: str = "/workspace",
        image=None,
        timeout: int = 3600,
        repl_timeout: float = 30,
        **container_kwargs,
    ) -> None:
        super().__init__(working_directory=remote_workdir)
        self.app_name = app_name
        self.remote_workdir = remote_workdir
        self.image = image
        self.timeout = timeout
        self.repl_timeout = repl_timeout
        self.container_kwargs = container_kwargs

    def open(self, agent: Graph) -> ReplBackend:
        return ModalRepl(
            self.app_name,
            remote_workdir=self.remote_workdir,
            image=self.image,
            timeout=self.timeout,
            repl_timeout=self.repl_timeout,
            **self.container_kwargs,
        )


__all__ = ["ModalRepl", "ModalRuntime"]
