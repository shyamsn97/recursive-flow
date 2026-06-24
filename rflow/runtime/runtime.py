"""Runtimes and REPL backends — where an agent's code runs.

This module gathers the whole execution seam in one place:

* :func:`serialize` / :func:`deserialize` — JSON-safe (de)serialization for the
  two control objects that cross the remote-REPL boundary.
* :class:`ReplBackend` — the protocol :meth:`rflow.flow.Flow.repl_for` depends
  on: run a code block, report ``(suspended, payload)``.
* :class:`RemoteRepl` — host-side base for sandbox backends that talk to a
  container-side :mod:`rflow.runtime.repl_server` over a JSON-line transport.
* :class:`Runtime` / :class:`LocalRuntime` — the **user-facing** object you
  construct and hand to :class:`~rflow.flow.Flow` (``Flow(runtime=...)``). A
  runtime owns the ``working_directory`` and the registered tools, and mints one
  :class:`ReplBackend` per agent via :meth:`Runtime.open`. ``DockerRuntime`` and
  the sandbox runtimes live next to their backends in :mod:`rflow.runtime.docker`
  and :mod:`rflow.runtime.sandbox`.

See ``docs/internal/runtime-working-directory.md`` for the design.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from rflow.graph import ChildHandle, WaitRequest
from rflow.repl import REPL, DoneSignal
from rflow.runtime.context import EngineContext

if TYPE_CHECKING:
    from rflow.graph import Graph

#: ``launch_subagents`` is composed *in the sandbox* from a private host-side
#: child-spawn proxy, so ``seed`` never ships the host closure itself.
REMOTE_LAUNCHER = ("launch_subagents",)


# ── serde ─────────────────────────────────────────────────────────────


def serialize(value: Any) -> Any:
    """Convert rlmflow objects to JSON-safe structures (recursively).

    Only the two control objects need special handling — everything else is
    JSON-native (the engine enforces ``str`` agent inputs).
    """
    if isinstance(value, (ChildHandle, WaitRequest)):
        return value.to_dict()
    if isinstance(value, (list, tuple)):
        return [serialize(v) for v in value]
    if isinstance(value, dict):
        return {k: serialize(v) for k, v in value.items()}
    return value


def deserialize(value: Any) -> Any:
    """Reconstruct rlmflow objects from JSON structures (recursively)."""
    if isinstance(value, list):
        return [deserialize(v) for v in value]
    if isinstance(value, dict):
        if "child_handle" in value:
            return ChildHandle.from_dict(value)
        if "wait_request" in value:
            return WaitRequest.from_dict(value)
        return {k: deserialize(v) for k, v in value.items()}
    return value


# ── backend protocol ──────────────────────────────────────────────────


@runtime_checkable
class ReplBackend(Protocol):
    """Where an agent's code runs (in-process REPL or a remote sandbox).

    Outcome contract (matches :class:`rflow.repl.REPL`):

    * ``start`` / ``resume`` return ``(False, stdout: str)`` on completion or
      error (``errored`` distinguishes them), or ``(True, (WaitRequest,
      pre_output: str))`` when the block suspends on
      ``await launch_subagents(...)``.
    * ``done(...)`` (a host tool) sets ``engine_context.done_result``; the engine
      reads it back after each turn.
    """

    #: Host-visible namespace; seedable for local, may be empty for remote.
    namespace: dict[str, Any]
    #: Trusted host-side control state for this agent.
    engine_context: EngineContext
    #: Public process environment variables exposed while agent code runs.
    process_env: dict[str, str]
    #: Set by the last ``start`` / ``resume`` (drives ExecOutput vs ErrorOutput).
    errored: bool

    def start(self, code: str) -> tuple[bool, object]:
        """Run a fresh code block."""
        ...

    def resume(self, send_value: object) -> tuple[bool, object]:
        """Resume a suspended block with ``send_value`` (``list[str]`` results)."""
        ...

    def close(self) -> None:
        """Release any external resources (no-op for in-process REPLs)."""
        ...


def parse_response(resp: dict) -> tuple[bool, object]:
    """Convert a REPL response dict into ``(suspended, payload)``.

    Callers set ``errored`` separately from ``resp.get("errored")``.
    """
    if resp.get("suspended"):
        return True, (
            WaitRequest(
                resp["agent_ids"],
                launch_id=resp.get("launch_id"),
                launch_specs=resp.get("launch_specs") or [],
                launch_names=resp.get("launch_names") or [],
            ),
            resp.get("pre_output", ""),
        )
    return False, resp.get("output", "")


# ── remote backend base ───────────────────────────────────────────────


class RemoteRepl(ABC):
    """A REPL running in a sandbox, driven over JSON-line stdio.

    Implements the host half of the protocol in :mod:`rflow.runtime.repl_server`.
    The ``namespace`` is empty host-side (code runs remotely);
    ``engine_context`` is the host-shared object the proxied ``done`` writes into.

    :meth:`seed` routes each tool by kind: host-bound tools proxy back (``done``
    writes the host context; private child spawn mutates the host graph;
    ``HISTORY`` reads the host graph), while ordinary tools (``FILE_TOOLS`` and
    friends) are shipped into the sandbox to run there. ``launch_subagents`` is
    built remotely from the private spawn proxy, so child recursion still spawns
    host-side while the parent's code runs in the sandbox.
    """

    def __init__(self) -> None:
        self.namespace: dict[str, Any] = {}
        self.engine_context = EngineContext()
        self.process_env: dict[str, str] = {}
        self.errored: bool = False
        self.proxied: dict[str, Callable[..., object]] = {}

    # ── transport (subclasses implement) ──────────────────────────────

    @abstractmethod
    def send(self, msg: dict) -> None:
        """Ship one JSON-serializable dict to the remote REPL."""

    @abstractmethod
    def recv(self) -> dict:
        """Block until the next dict arrives from the remote REPL."""

    # ── proxy loop ────────────────────────────────────────────────────

    def call(self, msg: dict) -> dict:
        """Send ``msg`` and service proxy calls until the REPL replies."""
        self.send(msg)
        while True:
            resp = self.recv()
            if "proxy" not in resp:
                return resp
            self.handle_proxy_call(resp)

    def handle_proxy_call(self, resp: dict) -> None:
        """Run one host call requested by the remote and ship the result back."""
        fn = self.proxied[resp["proxy"]]
        args = [deserialize(a) for a in resp.get("args", [])]
        kwargs = {k: deserialize(v) for k, v in resp.get("kwargs", {}).items()}
        try:
            result = fn(*args, **kwargs)
        except DoneSignal:
            self.send({"done": True})
            return
        except Exception as exc:  # noqa: BLE001 - surfaced to remote agent code
            self.send({"error": f"{type(exc).__name__}: {exc}"})
            return
        self.send({"value": serialize(result)})

    # ── seeding ───────────────────────────────────────────────────────

    def seed(
        self,
        tools: dict[str, Callable],
        inputs: dict[str, str],
        *,
        max_query_chars: int | None = None,
    ) -> None:
        """Bind this agent's inputs and tools into the remote REPL.

        Each tool is routed by kind (local-by-default, like smolagents):

        * ``launch_subagents`` is skipped — it is composed in the sandbox from a
          private host child-spawn proxy (``build_launcher``).
        * non-callables (``HISTORY``) are exposed as **object proxies** — public
          methods round-trip to the host, so a slice computed over the live host
          graph is all that crosses the wire.
        * ``@tool(proxy=True)`` callables (``done`` / private child spawn /
          ``llm_query_batched``) are **function proxies** — they touch host-only
          state, so calls run on the host.
        * everything else (``FILE_TOOLS``, ``runtime.register_tools(...)``) is
          **shipped into the sandbox** and runs there, against the sandbox's own
          working directory.

        ``process_env`` is copied into the sandbox's real ``os.environ``. Inputs
        are copied as a single ``INPUTS`` dict (read as ``INPUTS["key"]``) so a
        key can never shadow a real REPL variable in the sandbox.
        """
        from rflow.tools import get_tool_metadata

        self.inject_process_env(self.process_env)
        self.inject_literal("INPUTS", dict(inputs))
        for name, fn in tools.items():
            if name in REMOTE_LAUNCHER:
                continue
            if not callable(fn):
                self.inject_object_proxy(name, fn)
                continue
            meta = get_tool_metadata(fn)
            if meta is not None and meta.proxy:
                self.inject_function_proxy(name, fn)
            else:
                self.inject_local_tool(name, fn)
        # Build launch_subagents remotely from the private spawn proxy.
        self.call({"cmd": "build_launcher", "max_query_chars": max_query_chars})

    def inject_literal(self, name: str, value: object) -> None:
        """Copy a literal value (round-tripped through ``repr``) into the REPL."""
        self.call({"cmd": "inject", "name": name, "value": repr(value)})

    def inject_process_env(self, values: dict[str, str]) -> None:
        """Copy public agent metadata into the sandbox process environment."""
        self.call({"cmd": "set_env", "values": dict(values)})

    def inject_function_proxy(self, name: str, fn: Callable[..., object]) -> None:
        """Expose a host callable as a remote REPL function (calls round-trip)."""
        self.proxied[name] = fn
        self.call({"cmd": "inject_proxy", "name": name})

    def inject_object_proxy(self, name: str, obj: object) -> None:
        """Expose a host object's public methods through a remote proxy object.

        Each public (non-underscore) method is registered as ``"name.method"`` and
        the sandbox builds a stand-in whose methods forward to the host instance,
        so the object's state (and the work) stays host-side and only call results
        cross the wire.
        """
        methods = [
            m
            for m in dir(obj)
            if not m.startswith("_") and callable(getattr(obj, m, None))
        ]
        for method in methods:
            self.proxied[f"{name}.{method}"] = getattr(obj, method)
        self.call({"cmd": "inject_object", "name": name, "methods": methods})

    def inject_local_tool(self, name: str, fn: Callable[..., object]) -> None:
        """Materialize a tool *in* the sandbox so it runs there, not on the host.

        Importable tools (defined in a real module, e.g. all of ``FILE_TOOLS``)
        ship a reference the sandbox imports — which pulls the function and its
        module-level dependencies for free, since the image has rflow installed.
        Tools defined in ``__main__`` ship their source (``inspect.getsource``) to
        ``exec`` in the sandbox; closures/lambdas can't be shipped and raise an
        actionable error pointing at an importable module.
        """
        import inspect
        import textwrap

        module = getattr(fn, "__module__", None)
        qualname = getattr(fn, "__qualname__", "") or ""
        if module and module != "__main__" and "<locals>" not in qualname:
            self.call(
                {
                    "cmd": "inject_import",
                    "name": name,
                    "module": module,
                    "qualname": qualname,
                }
            )
            return
        try:
            source = textwrap.dedent(inspect.getsource(fn))
        except (OSError, TypeError) as exc:
            raise RuntimeError(
                f"cannot ship local tool {name!r} into the sandbox: define it in "
                f"an importable module (got module={module!r}, qualname="
                f"{qualname!r}); only top-level functions can be shipped by source."
            ) from exc
        self.call(
            {
                "cmd": "inject_source",
                "name": name,
                "func_name": getattr(fn, "__name__", name),
                "source": source,
            }
        )

    # ── ReplBackend outcome contract ──────────────────────────────────

    def start(self, code: str) -> tuple[bool, object]:
        return self._outcome(self.call({"cmd": "run", "code": code}))

    def resume(self, send_value: object) -> tuple[bool, object]:
        return self._outcome(self.call({"cmd": "resume", "value": send_value}))

    def _outcome(self, resp: dict) -> tuple[bool, object]:
        self.errored = bool(resp.get("errored"))
        return parse_response(resp)

    def close(self) -> None:
        """Release transport resources. Override in transports that hold them."""


# ── user-facing runtimes ──────────────────────────────────────────────


class Runtime(ABC):
    """Where and how an agent's code runs — the object you pass to ``Flow``.

    A runtime holds the configuration shared by a whole run: the
    ``working_directory`` agent code runs in and the tools registered with
    :meth:`register_tools`. :class:`~rflow.flow.Flow` calls :meth:`open` once per
    agent to mint a fresh :class:`ReplBackend` (the in-process
    :class:`~rflow.repl.REPL`, a Docker container, or a cloud sandbox).

    Subclass and implement :meth:`open` to add a backend; that is the supported
    extension point (there is no ``repl_factory`` — the runtime *is* the
    factory). See :class:`LocalRuntime` for the in-process default and
    :class:`~rflow.runtime.docker.DockerRuntime` for the sandboxed one.
    """

    def __init__(self, working_directory: str | Path | None = None) -> None:
        #: Directory agent code runs in. ``None`` → the process cwd as-is (no
        #: chdir, no serialization); a path → code runs with the cwd switched
        #: into it. Defaults to ``None`` ("the current directory").
        self.working_directory = (
            Path(working_directory) if working_directory is not None else None
        )
        #: name → callable, injected into each agent's REPL by ``Flow``.
        self.tools: dict[str, Callable[..., object]] = {}
        #: Runtime mirrors of graph-derived state, used to avoid redundant
        #: prompt renders and remote REPL sync calls. ``Graph`` remains the
        #: source of truth; these caches are disposable.
        self.prompt_fingerprints: dict[str, str] = {}
        self.repl_env_cache: dict[str, dict[str, str]] = {}
        self.repl_inputs_cache: dict[str, dict[str, str]] = {}

    # ── tool registration ─────────────────────────────────────────────

    def register_tool(
        self, fn: Callable[..., object], *, name: str | None = None
    ) -> None:
        """Expose one function to every agent's REPL (and its children)."""
        if name is None:
            from rflow.tools import get_tool_metadata

            meta = get_tool_metadata(fn)
            name = meta.name if meta is not None else fn.__name__
        self.tools[name] = fn

    def register_tools(self, tools: list[Callable[..., object]]) -> None:
        """Register a list of functions as tools (e.g. ``FILE_TOOLS``)."""
        for fn in tools:
            self.register_tool(fn)

    def clear_graph_sync_cache(self) -> None:
        """Forget graph-derived runtime mirror fingerprints."""
        self.prompt_fingerprints.clear()
        self.repl_env_cache.clear()
        self.repl_inputs_cache.clear()

    def drop_graph_sync_cache(self, agent_id: str) -> None:
        """Forget graph-derived runtime mirror fingerprints for one agent."""
        self.prompt_fingerprints.pop(agent_id, None)
        self.repl_env_cache.pop(agent_id, None)
        self.repl_inputs_cache.pop(agent_id, None)

    def discard_repl(self, repls: dict[str, ReplBackend], agent_id: str) -> None:
        """Close and forget an open backend plus its graph-derived caches."""
        repl = repls.pop(agent_id)
        self.drop_graph_sync_cache(agent_id)
        try:
            repl.close()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass

    def sync_repl(
        self,
        repl: ReplBackend,
        agent: Graph,
        *,
        env: dict[str, str],
        inputs: dict[str, str],
    ) -> None:
        """Bind a live backend to graph-derived runtime metadata.

        ``Graph`` remains the source of truth. This method updates only runtime
        mirrors: private tool context, public process env, and REPL ``INPUTS``.
        ``done_result`` is intentionally untouched; execution clears it per turn.
        """
        repl.engine_context.agent_id = agent.agent_id
        if repl.engine_context.output_schema != agent.output_schema:
            repl.engine_context.output_schema = agent.output_schema

        if self.repl_env_cache.get(agent.agent_id) != env:
            repl.process_env = dict(env)
            if isinstance(repl, RemoteRepl):
                repl.inject_process_env(env)
            self.repl_env_cache[agent.agent_id] = dict(env)

        if self.repl_inputs_cache.get(agent.agent_id) != inputs:
            if isinstance(repl, RemoteRepl):
                repl.inject_literal("INPUTS", inputs)
            else:
                repl.namespace["INPUTS"] = inputs
            self.repl_inputs_cache[agent.agent_id] = dict(inputs)

    # ── backend factory (subclasses implement) ────────────────────────

    @abstractmethod
    def open(self, agent: Graph) -> ReplBackend:
        """Mint the per-agent backend, bound to this runtime's config."""

    def close(self) -> None:
        """Release any runtime-owned resources (default: no-op)."""

    def __enter__(self) -> Runtime:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class LocalRuntime(Runtime):
    """Run agent code in the current Python process.

    :meth:`open` mints an in-process :class:`~rflow.repl.REPL` bound to
    ``working_directory`` (defaults to the current directory). When a working
    directory is set, each code block runs with the process cwd switched into it
    (serialized across agents), so the filesystem tools and any ``open(...)`` in
    agent code resolve relative to it.
    """

    def open(self, agent: Graph) -> ReplBackend:
        return REPL(working_directory=self.working_directory)


__all__ = [
    "REMOTE_LAUNCHER",
    "LocalRuntime",
    "RemoteRepl",
    "ReplBackend",
    "Runtime",
    "deserialize",
    "parse_response",
    "serialize",
]
