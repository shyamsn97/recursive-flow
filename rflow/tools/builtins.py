"""Engine-bound built-in tools and delegation launchers.

Each tool is a Python closure created per-runtime and bound to a specific
``runtime.env`` dict (and, for ``flow_delegate``, a ``spawn_child`` callable
that creates new sub-agents). They are registered through the normal
:meth:`Runtime.register_tool` path — ``LocalRuntime`` injects them
straight into the REPL namespace; remote runtimes expose proxy stubs that
round-trip back to the host closure.

The ``env`` dict captured here is the same object the engine reads back
via ``runtime.env`` after each execution to discover ``DONE_RESULT``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from rflow.graph import ChildHandle, WaitRequest
from rflow.runtime.env import (
    AGENT_ID,
    DONE_OUTPUT_SCHEMA,
    DONE_RESULT,
    PARENT_NODE_ID,
    replay_queue,
)
from rflow.tools import tool


class DoneSignal(BaseException):
    """Internal control-flow signal raised by ``done()`` to stop execution.

    This intentionally inherits from ``BaseException`` so agent code with a
    broad ``except Exception`` repair block cannot accidentally swallow a
    successful ``done(...)`` call and keep executing.
    """


def make_done(env: dict[str, Any], output_parser: Callable[[str, Any], Any]):
    """Closure that records the final answer and stops the current block."""

    @tool("Return this agent's final answer.")
    def done(answer: Any) -> str:
        if env.get(DONE_RESULT) is None:
            schema = env.get(DONE_OUTPUT_SCHEMA)
            if schema is None:
                env[DONE_RESULT] = str(answer).strip()
            else:
                content = json.dumps(answer, separators=(",", ":"), ensure_ascii=False)
                parsed = output_parser(content, schema)
                structured = (
                    parsed.model_dump(mode="json")
                    if hasattr(parsed, "model_dump")
                    else parsed
                )
                env[DONE_RESULT] = json.dumps(
                    structured,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            print(f"[done] {env[DONE_RESULT]}")
        raise DoneSignal(env[DONE_RESULT])

    return done


def make_wait():
    """Closure that packages :class:`ChildHandle`s into a :class:`WaitRequest`."""

    @tool("Wait for delegated children. Must be called with `await`.")
    def flow_wait(*handles: ChildHandle) -> WaitRequest:
        if not handles:
            raise ValueError("flow_wait() requires at least one child handle")
        bad = [(i, h) for i, h in enumerate(handles) if not isinstance(h, ChildHandle)]
        if bad:
            details = "; ".join(
                f"handles[{i}] is {type(h).__name__}: {h!r}" for i, h in bad
            )
            raise TypeError(
                f"flow_wait() got non-handle arguments — `flow_delegate()` likely "
                f"refused those calls and returned a refusal string instead of a "
                f"ChildHandle. Read the string(s), fix the cause (e.g. unknown "
                f"`model=` key, max depth reached), and retry. {details}"
            )
        return WaitRequest(agent_ids=[h.agent_id for h in handles])

    return flow_wait


@tool("Show current public REPL variable names and their type names.")
def SHOW_VARS() -> dict[str, str]:
    """Installed specially by Runtime so it can inspect the live REPL namespace."""

    raise RuntimeError("SHOW_VARS must be installed by the runtime")


def make_delegate(
    spawn_child: Callable[..., "ChildHandle | str"],
    env: dict[str, Any],
):
    """Closure that calls ``spawn_child(...)`` and tracks the new id.

    ``spawn_child`` is the engine's child-spawning seam (typically
    :meth:`RecursiveFlow.spawn_child` bound to a specific engine instance).
    Passing the callable instead of the whole engine keeps this
    module decoupled from :class:`RecursiveFlow`.

    In *replay mode* (``env[REPLAY_QUEUE]`` is a list), ``flow_delegate``
    does not spawn a new child — it pops the next expected agent id
    off the queue and returns a :class:`ChildHandle` to it. This lets
    the engine re-execute action code after a fork or cold start to
    re-create the suspended generator without duplicating children
    that already exist in the graph.
    """

    @tool(
        "Delegate one independent unit of work to a named child agent. "
        "Use for multi-file/component/chunk/trial fanout; the parent should "
        "pass shared requirements/data in context, then integrate and verify "
        "child results. Set output_schema to a JSON Schema dict when the child "
        "must return a validated JSON-compatible value."
    )
    def flow_delegate(
        *,
        name: str,
        query: str,
        context: str | list[str],
        model: str = "default",
        output_schema: Any | None = None,
    ) -> ChildHandle | str:
        queue = replay_queue(env)
        if queue is not None:
            if not queue:
                return (
                    f"[replay error: no expected child for flow_delegate({name!r}). "
                    "Recorded trajectory diverges from the action code.]"
                )
            return ChildHandle(queue.pop(0))
        context_text = "\n".join(context) if isinstance(context, list) else context
        return spawn_child(
            env[AGENT_ID],
            env[PARENT_NODE_ID],
            name,
            query,
            context_text,
            model=model,
            output_schema=output_schema,
        )

    return flow_delegate


def make_launch_subagents(
    flow_delegate: Callable[..., object],
    flow_wait: Callable[..., Any],
):
    """Build the public multi-child launcher from bound primitives."""

    @tool(
        "Launch sub-agents in parallel and wait for all. Must be awaited. "
        "Each spec may include output_schema as a JSON Schema dict for that "
        "child's done(value)."
    )
    async def launch_subagents(specs):
        """Launch sub-agents in parallel and wait for all. Must be awaited.

        ``specs`` is a list of dicts; each dict may set ``query`` (required),
        ``context``, ``name``, ``model``, and ``output_schema``.
        Returns child results in the same order as ``specs``. Children with
        ``output_schema`` return validated JSON-compatible values.
        """
        if not isinstance(specs, list):
            raise TypeError("launch_subagents(...) requires a list of dict specs")
        _results = [None] * len(specs)
        _handles = []
        _positions = []
        for _i, _spec in enumerate(specs):
            if not isinstance(_spec, dict):
                raise TypeError(
                    "launch_subagents(...) requires every spec to be a dict"
                )
            if "query" not in _spec:
                raise KeyError("launch_subagents(...) spec missing required 'query'")
            _handle = flow_delegate(
                name=_spec.get("name", "subagent"),
                query=_spec["query"],
                context=_spec.get("context", ""),
                model=_spec.get("model", "default"),
                output_schema=_spec.get("output_schema"),
            )
            if isinstance(_handle, str):
                _results[_i] = _handle
            else:
                _handles.append(_handle)
                _positions.append(_i)
        if _handles:
            _waited = await flow_wait(*_handles)
            for _pos, _result in zip(_positions, _waited):
                _results[_pos] = _result
        return _results

    return launch_subagents


def make_launcher(
    name: str,
    flow_delegate: Callable[..., object],
    flow_wait: Callable[..., Any],
):
    """Build a public launcher tool by registered name."""

    factories = {
        "launch_subagents": make_launch_subagents,
    }
    try:
        factory = factories[name]
    except KeyError as exc:
        raise KeyError(f"unknown launcher: {name}") from exc
    return factory(flow_delegate, flow_wait)


__all__ = [
    "DoneSignal",
    "SHOW_VARS",
    "make_delegate",
    "make_done",
    "make_launcher",
    "make_launch_subagents",
    "make_wait",
]
