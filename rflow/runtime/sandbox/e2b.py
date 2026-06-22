"""E2B REPL backend — run an agent's code inside an E2B Sandbox.

Requires ``e2b`` (``pip install rlmflow[e2b]``) and an ``E2B_API_KEY``
(unless passed via ``sandbox_kwargs``). Uses the :class:`RemoteFileRuntime` file
bridge: one persistent :mod:`rflow.runtime.repl_server` process, driven through
remote files via E2B's ``commands.run``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rflow.runtime.runtime import ReplBackend, Runtime
from rflow.runtime.sandbox.common import command_output, optional_dependency_error
from rflow.runtime.sandbox.remote import RemoteFileRuntime

if TYPE_CHECKING:
    from rflow.graph import Graph


class E2BRepl(RemoteFileRuntime):
    """A :class:`RemoteFileRuntime` backed by an E2B Sandbox."""

    def __init__(
        self,
        *,
        template: str | None = None,
        timeout: int = 300,
        envs: dict[str, str] | None = None,
        remote_workdir: str = "/workspace",
        repl_timeout: float = 30,
        setup_commands: list[str] | None = None,
        sandbox_kwargs: dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            remote_workdir=remote_workdir,
            repl_timeout=repl_timeout,
            setup_commands=setup_commands,
        )
        self.template = template
        self.timeout = timeout
        self.envs = envs
        self.sandbox_kwargs = dict(sandbox_kwargs or {})
        self.sandbox = None

    def _ensure_sandbox(self) -> None:
        if self.sandbox is not None:
            return
        try:
            from e2b import Sandbox
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                optional_dependency_error("E2BRepl", "e2b")
            ) from exc
        self.sandbox = Sandbox.create(
            template=self.template,
            timeout=self.timeout,
            envs=self.envs,
            **self.sandbox_kwargs,
        )

    def _provider_prepare(self) -> None:
        self._ensure_sandbox()
        self._run_setup()

    def exec(self, command: str, *, timeout: float | None = None) -> str:
        self._ensure_sandbox()
        assert self.sandbox is not None
        result = self.sandbox.commands.run(
            command, timeout=timeout or self.repl_timeout
        )
        return command_output(result, "E2B")

    def _close_sandbox(self) -> None:
        sandbox, self.sandbox = self.sandbox, None
        if sandbox is None:
            return
        for name in ("kill", "close", "disconnect"):
            method = getattr(sandbox, name, None)
            if callable(method):
                method()
                return


class E2BRuntime(Runtime):
    """Run each agent's code in a remote E2B Sandbox.

    The user-facing object you hand to ``Flow(runtime=...)``; :meth:`open` mints
    one :class:`E2BRepl` per agent. ``remote_workdir`` is the in-sandbox
    directory agent code runs in.
    """

    def __init__(
        self,
        *,
        template: str | None = None,
        timeout: int = 300,
        envs: dict[str, str] | None = None,
        remote_workdir: str = "/workspace",
        repl_timeout: float = 30,
        setup_commands: list[str] | None = None,
        sandbox_kwargs: dict[str, object] | None = None,
    ) -> None:
        super().__init__(working_directory=remote_workdir)
        self.template = template
        self.timeout = timeout
        self.envs = envs
        self.remote_workdir = remote_workdir
        self.repl_timeout = repl_timeout
        self.setup_commands = setup_commands
        self.sandbox_kwargs = sandbox_kwargs

    def open(self, agent: Graph) -> ReplBackend:
        return E2BRepl(
            template=self.template,
            timeout=self.timeout,
            envs=self.envs,
            remote_workdir=self.remote_workdir,
            repl_timeout=self.repl_timeout,
            setup_commands=self.setup_commands,
            sandbox_kwargs=self.sandbox_kwargs,
        )


__all__ = ["E2BRepl", "E2BRuntime"]
