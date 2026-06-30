"""Shared ``subprocess.Popen`` transport for JSON-line remote REPLs."""

from __future__ import annotations

import json
import selectors
import subprocess as sp
from pathlib import Path

from rflow.runtime.runtime import RemoteRepl


class PopenRepl(RemoteRepl):
    """A :class:`RemoteRepl` driven through a local subprocess's stdio."""

    def __init__(
        self,
        argv: list[str],
        *,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        label: str = "REPL subprocess",
        repl_timeout: float | None = None,
    ) -> None:
        super().__init__()
        self.argv = argv
        self.cwd = cwd
        self.env = env
        self.label = label
        self.repl_timeout = repl_timeout
        self.proc: sp.Popen | None = None

    def _ensure_proc(self) -> sp.Popen:
        if self.proc is None:
            self.proc = sp.Popen(
                self.argv,
                stdin=sp.PIPE,
                stdout=sp.PIPE,
                stderr=sp.PIPE,
                cwd=str(self.cwd) if self.cwd is not None else None,
                env=self.env,
                bufsize=0,
            )
        return self.proc

    def send(self, msg: dict) -> None:
        proc = self._ensure_proc()
        assert proc.stdin is not None
        proc.stdin.write((json.dumps(msg) + "\n").encode())
        proc.stdin.flush()

    def recv(self) -> dict:
        proc = self._ensure_proc()
        assert proc.stdout is not None
        if self.repl_timeout is not None:
            selector = selectors.DefaultSelector()
            try:
                selector.register(proc.stdout, selectors.EVENT_READ)
                events = selector.select(timeout=self.repl_timeout)
            finally:
                selector.close()
            if not events:
                self.close(force=True)
                raise TimeoutError(
                    f"{self.label} did not respond within {self.repl_timeout}s"
                )
        line = proc.stdout.readline()
        if not line:
            err = b""
            if proc.stderr is not None:
                try:
                    err = proc.stderr.read() or b""
                except Exception:  # noqa: BLE001
                    pass
            raise RuntimeError(
                f"{self.label} {self.argv!r} exited unexpectedly. "
                f"stderr: {err.decode(errors='replace')}"
            )
        return json.loads(line)

    def close(self, *, force: bool = False) -> None:
        """Tear down the subprocess and release pipe file descriptors."""

        proc, self.proc = self.proc, None
        if proc is None:
            return
        if force:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass
        try:
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.wait(timeout=0.5 if force else 2)
        except sp.TimeoutExpired:
            for action in (proc.terminate, proc.kill):
                try:
                    action()
                    proc.wait(timeout=0.5 if force else 2)
                    break
                except Exception:  # noqa: BLE001
                    continue
        except Exception:  # noqa: BLE001
            pass
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None and not stream.closed:
                    stream.close()
            except Exception:  # noqa: BLE001
                pass


__all__ = ["PopenRepl"]
