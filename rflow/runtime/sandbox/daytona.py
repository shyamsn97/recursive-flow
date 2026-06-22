"""Daytona REPL backend — run an agent's code inside a Daytona Sandbox.

Requires ``daytona`` (``pip install rlmflow[daytona]``) and Daytona
credentials configured for the SDK. Uses the :class:`RemoteFileRuntime` file
bridge: one persistent :mod:`rflow.runtime.repl_server` process, driven through
remote files via Daytona's ``process.exec``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rflow.runtime.runtime import ReplBackend, Runtime
from rflow.runtime.sandbox.common import command_output, optional_dependency_error
from rflow.runtime.sandbox.remote import RemoteFileRuntime

if TYPE_CHECKING:
    from rflow.graph import Graph


class DaytonaRepl(RemoteFileRuntime):
    """A :class:`RemoteFileRuntime` backed by a Daytona Sandbox."""

    def __init__(
        self,
        *,
        create_params: object = None,
        create_timeout: float = 60,
        env: dict[str, str] | None = None,
        remote_workdir: str = "/workspace",
        repl_timeout: float = 30,
        setup_commands: list[str] | None = None,
        daytona: object = None,
    ) -> None:
        super().__init__(
            remote_workdir=remote_workdir,
            repl_timeout=repl_timeout,
            setup_commands=setup_commands,
        )
        self.create_params = create_params
        self.create_timeout = create_timeout
        self.command_env = env
        self.daytona = daytona
        self.sandbox = None

    def _ensure_sandbox(self) -> None:
        if self.sandbox is not None:
            return
        if self.daytona is None:
            try:
                from daytona import Daytona
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
                raise ModuleNotFoundError(
                    optional_dependency_error("DaytonaRepl", "daytona")
                ) from exc
            self.daytona = Daytona()
        if self.create_params is None:
            self.sandbox = self.daytona.create(timeout=self.create_timeout)
        else:
            self.sandbox = self.daytona.create(
                self.create_params, timeout=self.create_timeout
            )

    def _provider_prepare(self) -> None:
        self._ensure_sandbox()
        self._run_setup()

    def exec(self, command: str, *, timeout: float | None = None) -> str:
        self._ensure_sandbox()
        assert self.sandbox is not None
        result = self.sandbox.process.exec(
            command,
            env=self.command_env,
            timeout=int(timeout or self.repl_timeout),
        )
        return command_output(result, "Daytona", stdout_getter=_stdout)

    def _close_sandbox(self) -> None:
        sandbox, self.sandbox = self.sandbox, None
        if sandbox is None:
            return
        for name in ("delete", "stop", "close"):
            method = getattr(sandbox, name, None)
            if callable(method):
                method()
                return


def _stdout(result: object) -> str:
    artifacts = getattr(result, "artifacts", None)
    if artifacts is not None and getattr(artifacts, "stdout", None) is not None:
        return artifacts.stdout
    for attr in ("stdout", "result", "output"):
        value = getattr(result, attr, None)
        if value is not None:
            return value
    return ""


class DaytonaRuntime(Runtime):
    """Run each agent's code in a remote Daytona Sandbox.

    The user-facing object you hand to ``Flow(runtime=...)``; :meth:`open` mints
    one :class:`DaytonaRepl` per agent. ``remote_workdir`` is the in-sandbox
    directory agent code runs in.
    """

    def __init__(
        self,
        *,
        create_params: object = None,
        create_timeout: float = 60,
        env: dict[str, str] | None = None,
        remote_workdir: str = "/workspace",
        repl_timeout: float = 30,
        setup_commands: list[str] | None = None,
        daytona: object = None,
    ) -> None:
        super().__init__(working_directory=remote_workdir)
        self.create_params = create_params
        self.create_timeout = create_timeout
        self.command_env = env
        self.remote_workdir = remote_workdir
        self.repl_timeout = repl_timeout
        self.setup_commands = setup_commands
        self.daytona = daytona

    def open(self, agent: Graph) -> ReplBackend:
        return DaytonaRepl(
            create_params=self.create_params,
            create_timeout=self.create_timeout,
            env=self.command_env,
            remote_workdir=self.remote_workdir,
            repl_timeout=self.repl_timeout,
            setup_commands=self.setup_commands,
            daytona=self.daytona,
        )


__all__ = ["DaytonaRepl", "DaytonaRuntime"]
