"""Docker REPL backend — run an agent's code inside a fresh container.

Each :class:`DockerRepl` owns one ``docker run -i --rm <image> python -m
rflow.runtime.repl_server`` subprocess and speaks the JSON-line protocol over its
stdin/stdout. The image must have ``rlmflow`` installed.

Example::

    runtime = DockerRuntime(
        image="rlmflow:local",
        working_directory="./myproject",  # bind-mounted to /workspace
        network="none",                   # air-gap the container
        cpus=1.0, memory="512m",
    )
    flow = Flow(llm, runtime=runtime)

Build a ready image once with ``docker build -t rlmflow:local .`` (any
image whose entrypoint can run ``python -m rflow.runtime.repl_server`` works).
:class:`DockerRepl` (the per-agent backend :class:`DockerRuntime` mints) stays
decoupled from any workspace abstraction: pass ``mounts`` / ``workdir`` / ``cwd``
directly if you want full control.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rflow.runtime.popen import PopenRepl
from rflow.runtime.runtime import ReplBackend, Runtime

if TYPE_CHECKING:
    from rflow.graph import Graph


class DockerRepl(PopenRepl):
    """A :class:`RemoteRepl` whose transport is a ``docker run`` subprocess."""

    def __init__(
        self,
        image: str,
        *,
        mounts: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
        network: str | None = None,
        cpus: float | None = None,
        memory: str | None = None,
        user: str | None = None,
        workdir: str | None = None,
        cwd: str | Path | None = None,
        extra_args: list[str] | None = None,
        docker_bin: str = "docker",
        entrypoint_argv: list[str] | None = None,
        repl_timeout: float | None = None,
    ) -> None:
        self.image = image
        super().__init__(
            build_argv(
                image,
                mounts=mounts,
                env=env,
                network=network,
                cpus=cpus,
                memory=memory,
                user=user,
                workdir=workdir,
                extra_args=extra_args,
                docker_bin=docker_bin,
                entrypoint_argv=entrypoint_argv,
            ),
            cwd=cwd,
            label="Docker REPL subprocess",
            repl_timeout=repl_timeout,
        )


class DockerRuntime(Runtime):
    """Run each agent's code inside a fresh Docker container.

    The user-facing object you hand to ``Flow(runtime=...)``. :meth:`open` mints
    one :class:`DockerRepl` per agent from the stored container options.

    ``working_directory`` is the **host** directory the agent's files live in. By
    default it is bind-mounted to ``/workspace`` in the container and used as the
    container ``--workdir`` (override with ``mounts`` / ``workdir``). All other
    keyword arguments are passed straight through to :class:`DockerRepl`.

    Example::

        runtime = DockerRuntime("rlmflow:local",
                                working_directory="./myproject", network="none")
        flow = Flow(llm, runtime=runtime)
    """

    def __init__(
        self,
        image: str,
        *,
        working_directory: str | Path | None = None,
        mounts: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
        network: str | None = None,
        cpus: float | None = None,
        memory: str | None = None,
        user: str | None = None,
        workdir: str | None = None,
        extra_args: list[str] | None = None,
        docker_bin: str = "docker",
        entrypoint_argv: list[str] | None = None,
        repl_timeout: float | None = None,
    ) -> None:
        super().__init__(working_directory=working_directory)
        self.image = image
        # When a working directory is given but no explicit mount/workdir, share
        # it with the container at /workspace and run there — the friendly default.
        if self.working_directory is not None:
            host = str(self.working_directory.resolve())
            if mounts is None:
                mounts = {host: "/workspace"}
            if workdir is None:
                workdir = "/workspace"
        self.options = dict(
            mounts=mounts,
            env=env,
            network=network,
            cpus=cpus,
            memory=memory,
            user=user,
            workdir=workdir,
            extra_args=extra_args,
            docker_bin=docker_bin,
            entrypoint_argv=entrypoint_argv,
            repl_timeout=repl_timeout,
        )

    def open(self, agent: Graph) -> ReplBackend:
        cwd = str(self.working_directory) if self.working_directory else None
        return DockerRepl(self.image, cwd=cwd, **self.options)


def build_argv(
    image: str,
    *,
    mounts: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
    network: str | None = None,
    cpus: float | None = None,
    memory: str | None = None,
    user: str | None = None,
    workdir: str | None = None,
    extra_args: list[str] | None = None,
    docker_bin: str = "docker",
    entrypoint_argv: list[str] | None = None,
) -> list[str]:
    """Build the ``docker run ...`` argv for :class:`DockerRepl`."""
    argv: list[str] = [docker_bin, "run", "-i", "--rm"]
    for host, container in (mounts or {}).items():
        argv += ["-v", f"{Path(host).resolve()}:{container}"]
    for key, value in (env or {}).items():
        argv += ["-e", f"{key}={value}"]
    if network is not None:
        argv += ["--network", network]
    if cpus is not None:
        argv += ["--cpus", str(cpus)]
    if memory is not None:
        argv += ["--memory", str(memory)]
    if user is not None:
        argv += ["--user", user]
    if workdir is not None:
        argv += ["--workdir", workdir]
    argv += list(extra_args or [])
    argv += [image]
    argv += list(entrypoint_argv or ["python", "-m", "rflow.runtime.repl_server"])
    return argv


__all__ = ["DockerRepl", "DockerRuntime", "build_argv"]
