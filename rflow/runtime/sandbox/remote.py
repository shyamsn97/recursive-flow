"""File-bridge base for sandbox providers that only expose ``exec(command)``.

Some providers (E2B, Daytona) make it easy to run a shell command but don't hand
you a portable persistent stdin/stdout handle. :class:`RemoteFileRuntime` keeps
one long-lived :mod:`rflow.runtime.repl_server` process alive in the sandbox and
moves the JSON-line protocol through remote files:

- append outbound messages to an input file;
- ``tail -f`` feeds that file into the REPL process;
- poll the output file for the next response line.

Subclasses implement just :meth:`exec` (and the sandbox lifecycle). No workspace
sync: the sandbox starts empty unless your ``setup_commands`` populate it.
"""

from __future__ import annotations

import json
import shlex
import uuid

from rflow.runtime.runtime import RemoteRepl

_REPL_ENTRYPOINT = "from rflow.runtime.repl_server import main; main()"


class RemoteFileRuntime(RemoteRepl):
    """Base for provider SDKs with ``exec(command) -> stdout`` semantics."""

    #: First-boot commands for providers that don't ship rlmflow.
    DEFAULT_SETUP_COMMANDS = ("python -m pip install -q rlmflow",)

    def __init__(
        self,
        *,
        remote_workdir: str = "/workspace",
        repl_timeout: float = 30,
        setup_commands: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.remote_workdir = remote_workdir
        self.repl_timeout = repl_timeout
        self.setup_commands = self._resolve_setup_commands(setup_commands)
        self._run_id = uuid.uuid4().hex
        self._remote_dir = f"/tmp/rlmflow-{self._run_id}"
        self._input_path = f"{self._remote_dir}/in.jsonl"
        self._output_path = f"{self._remote_dir}/out.jsonl"
        self._stderr_path = f"{self._remote_dir}/stderr.log"
        self._pid_path = f"{self._remote_dir}/pid"
        self._output_offset = 0
        self._started = False
        self._setup_done = False

    @classmethod
    def _resolve_setup_commands(cls, setup_commands: list[str] | None) -> list[str]:
        """``None`` → default install command; an explicit (maybe empty) list verbatim."""
        if setup_commands is not None:
            return list(setup_commands)
        return list(cls.DEFAULT_SETUP_COMMANDS)

    # ── subclasses implement these ────────────────────────────────────

    def exec(self, command: str, *, timeout: float | None = None) -> str:
        """Run ``command`` in the sandbox and return its stdout."""
        raise NotImplementedError

    def _provider_prepare(self) -> None:
        """Create the sandbox / run setup before the REPL launches (override)."""

    def _close_sandbox(self) -> None:
        """Provider-specific resource cleanup (override)."""

    # ── REPL process lifecycle ────────────────────────────────────────

    def _run_setup(self) -> None:
        if self._setup_done:
            return
        for command in self.setup_commands:
            self.exec(command, timeout=self.repl_timeout)
        self._setup_done = True

    def _ensure_started(self) -> None:
        if self._started:
            return
        self._provider_prepare()
        remote_dir = shlex.quote(self._remote_dir)
        input_path = shlex.quote(self._input_path)
        output_path = shlex.quote(self._output_path)
        stderr_path = shlex.quote(self._stderr_path)
        pid_path = shlex.quote(self._pid_path)
        remote_workdir = shlex.quote(self.remote_workdir)
        command = " && ".join(
            [
                f"mkdir -p {remote_dir} {remote_workdir}",
                f": > {input_path}",
                f": > {output_path}",
                f": > {stderr_path}",
                (
                    "(nohup sh -lc "
                    + shlex.quote(
                        f"tail -n +1 -f {input_path} | "
                        "python -u -c "
                        f"{shlex.quote(_REPL_ENTRYPOINT)} "
                        f"--workdir {remote_workdir} "
                        f"> {output_path} 2> {stderr_path}"
                    )
                    + f" >/dev/null 2>&1 & echo $! > {pid_path})"
                ),
            ]
        )
        self.exec(command, timeout=self.repl_timeout)
        self._started = True

    # ── file-bridge transport ─────────────────────────────────────────

    def send(self, msg: dict) -> None:
        self._ensure_started()
        line = json.dumps(msg) + "\n"
        script = (
            "from pathlib import Path\n"
            f"Path({self._input_path!r}).open('a').write({line!r})\n"
        )
        self.exec(f"python -c {shlex.quote(script)}", timeout=self.repl_timeout)

    def recv(self) -> dict:
        self._ensure_started()
        script = (
            "from pathlib import Path\n"
            "import sys, time\n"
            f"out = Path({self._output_path!r})\n"
            f"err = Path({self._stderr_path!r})\n"
            f"offset = {self._output_offset}\n"
            f"deadline = time.time() + {self.repl_timeout!r}\n"
            "while time.time() < deadline:\n"
            "    text = out.read_text() if out.exists() else ''\n"
            "    idx = text.find('\\n', offset)\n"
            "    if idx >= 0:\n"
            "        print(text[offset:idx])\n"
            "        sys.exit(0)\n"
            "    time.sleep(0.05)\n"
            "stderr = err.read_text()[-4000:] if err.exists() else ''\n"
            "raise SystemExit('timed out waiting for rlmflow REPL output\\n' + stderr)\n"
        )
        line = self.exec(
            f"python -c {shlex.quote(script)}", timeout=self.repl_timeout + 5
        ).strip()
        self._output_offset += len(line) + 1
        return json.loads(line)

    def close(self) -> None:
        if self._started:
            try:
                self.exec(
                    "sh -lc "
                    + shlex.quote(
                        f"kill $(cat {shlex.quote(self._pid_path)}) 2>/dev/null || true; "
                        f"rm -rf {shlex.quote(self._remote_dir)}"
                    ),
                    timeout=5,
                )
            except Exception:  # noqa: BLE001
                pass
            self._started = False
        self._close_sandbox()


__all__ = ["RemoteFileRuntime"]
