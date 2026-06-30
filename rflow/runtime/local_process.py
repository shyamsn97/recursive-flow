"""Local subprocess REPL backend.

``SubprocessRuntime`` is the local runtime for true parallel code execution: it
starts one Python process per agent and drives the existing JSON-line
``repl_server`` protocol. Unlike in-process ``LocalRuntime``, cwd and
``RFLOW_*`` environment variables are process-local.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from rflow.runtime.popen import PopenRepl
from rflow.runtime.runtime import ReplBackend, Runtime

if TYPE_CHECKING:
    from rflow.graph import Graph


class SubprocessRepl(PopenRepl):
    """A :class:`RemoteRepl` backed by a local Python subprocess."""

    def __init__(
        self,
        *,
        working_directory: str | Path | None = None,
        python_executable: str | Path | None = None,
        env: dict[str, str] | None = None,
        launch_cwd: str | Path | None = None,
        repl_timeout: float | None = None,
    ) -> None:
        argv = [
            str(python_executable or sys.executable),
            "-u",
            "-m",
            "rflow.runtime.repl_server",
        ]
        if working_directory is not None:
            argv += ["--workdir", str(working_directory)]
        child_env = None
        if env is not None:
            child_env = {**os.environ, **{str(k): str(v) for k, v in env.items()}}
        super().__init__(
            argv,
            cwd=launch_cwd,
            env=child_env,
            label="local subprocess REPL",
            repl_timeout=repl_timeout,
        )


class SubprocessRuntime(Runtime):
    """Run each agent's code in its own local Python subprocess.

    This keeps the local development ergonomics of ``LocalRuntime`` while making
    cwd and ``RFLOW_*`` metadata process-local, so sibling child REPL blocks can
    execute concurrently.
    """

    def __init__(
        self,
        working_directory: str | Path | None = None,
        *,
        python_executable: str | Path | None = None,
        env: dict[str, str] | None = None,
        launch_cwd: str | Path | None = None,
        repl_timeout: float | None = None,
    ) -> None:
        super().__init__(working_directory=working_directory)
        self.python_executable = python_executable
        self.env = env
        self.launch_cwd = launch_cwd
        self.repl_timeout = repl_timeout

    def open(self, agent: Graph) -> ReplBackend:
        return SubprocessRepl(
            working_directory=self.working_directory,
            python_executable=self.python_executable,
            env=self.env,
            launch_cwd=self.launch_cwd,
            repl_timeout=self.repl_timeout,
        )


__all__ = ["SubprocessRepl", "SubprocessRuntime"]
