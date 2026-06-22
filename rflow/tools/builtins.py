"""Core REPL tools, assembled once per agent by :meth:`rflow.base.BaseFlow.build_tools`.

Each ``make_*`` factory returns a closure bound to the live ``BaseFlow`` and the
agent's :class:`rflow.runtime.context.EngineContext` — *not* its ``Graph``. The
per-agent context is seeded by :meth:`rflow.base.BaseFlow.seed_agent_context` and
read at call time, so a single build serves the agent for its whole life. They
raise the existing :class:`rflow.repl.DoneSignal` and are
``@tool``-decorated so the prompt's tool list can describe them. ``History`` is
the read-only ``HISTORY`` object an agent uses to re-read its own past turns when
the model's context has been windowed (it does wrap the ``Graph``, since it is a
trajectory view rather than a control tool).
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from rflow.base import BaseFlow
from rflow.graph import ChildHandle, WaitRequest
from rflow.repl import DoneSignal
from rflow.runtime.context import EngineContext
from rflow.tools.registry import HIDDEN_REPL_TOOL_NAMES
from rflow.tools.tools import get_tool_metadata, tool

if TYPE_CHECKING:
    from collections.abc import Callable

    from rflow.graph import Graph

#: Default ceiling (in characters) for a child agent's ``query`` string. The
#: ``query`` is a short task instruction; large payloads belong in ``inputs``.
#: Configurable per run via ``Flow(max_query_chars=...)``.
DEFAULT_MAX_QUERY_CHARS = 2_000


def make_done(flow: BaseFlow, engine_context: EngineContext):
    """Build ``done(answer)``; enforces the schema in ``engine_context``.

    Reads context set per-agent by ``seed_agent_context`` so the same factory
    works for any agent — no Graph captured.
    """

    @tool("Finish and return this agent's final answer.", proxy=True)
    def done(answer: Any) -> None:
        schema = engine_context.output_schema
        if schema is not None:
            content = answer if isinstance(answer, str) else json.dumps(answer)
            # Validate against the schema; a mismatch raises StructuredOutputError
            # whose message is the recovery hint, recorded as a retryable error.
            flow.output_parser(content, schema)
            result = content
        else:
            result = str(answer).strip()
        engine_context.done_result = result
        print(f"[done] {result}")
        raise DoneSignal()

    return done


def make_wait():
    """Build ``flow_wait(*handles)`` with an actionable error on refusals."""

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
                "flow_wait() got non-handle arguments — flow_delegate() likely "
                "refused those calls and returned a refusal string instead of a "
                "ChildHandle. Read the string(s), fix the cause (unknown `model=` "
                f"key, max depth reached), and retry. {details}"
            )
        return WaitRequest([h.agent_id for h in handles])

    return flow_wait


def make_delegate(flow: BaseFlow, engine_context: EngineContext):
    """Build the hidden ``flow_delegate(...)`` primitive for this agent.

    Reads ``engine_context.agent_id`` at call time to know which agent is
    spawning, so the same factory works for any agent — no Graph captured.
    """

    @tool(
        "Delegate one unit of work to a named child agent. Pass shared "
        "requirements/data via inputs (a dict of str -> str bound as the "
        "child's own top-level variables); set output_schema to a JSON Schema "
        "dict when the child must return a validated value.",
        proxy=True,
    )
    def flow_delegate(
        *,
        name: str = "subagent",
        query: str,
        inputs: dict[str, str] | None = None,
        model: str = "default",
        output_schema: Any | None = None,
    ) -> ChildHandle | str:
        return flow.spawn_child(
            engine_context.agent_id, name, query, inputs, model, output_schema
        )

    return flow_delegate


def make_launch_subagents(
    flow_delegate, flow_wait, *, max_query_chars: int = DEFAULT_MAX_QUERY_CHARS
):
    """Build the public ``launch_subagents(specs)`` launcher from primitives.

    ``max_query_chars`` bounds each spec's ``query`` so a model cannot route a
    large payload through the child's task message; oversized queries raise
    before any child is spawned.
    """

    @tool(
        "Launch one or many sub-agents in parallel and wait for all. Must be "
        "awaited at the top level. specs is a list of dicts; each requires a "
        "short 'query' (the child's task; put bulk data in 'inputs', not here) "
        "and may set 'name', 'model', 'inputs', and 'output_schema'. "
        "Returns child results in spec order."
    )
    async def launch_subagents(specs):
        if not isinstance(specs, list):
            raise TypeError("launch_subagents(...) takes a list of dict specs")
        results: list = [None] * len(specs)
        handles, positions = [], []
        for i, spec in enumerate(specs):
            if not isinstance(spec, dict):
                raise TypeError(
                    "launch_subagents(...) requires every spec to be a dict"
                )
            if "query" not in spec:
                raise KeyError("launch_subagents(...) spec missing required 'query'")
            query = spec["query"]
            if not isinstance(query, str):
                raise TypeError(
                    "launch_subagents(...) spec 'query' must be a str; got "
                    f"{type(query).__name__}"
                )
            if len(query) > max_query_chars:
                raise ValueError(
                    f"launch_subagents(...) spec 'query' is too long "
                    f"({len(query)} chars > {max_query_chars} limit). 'query' "
                    "must be a short instruction. Move the large/helpful payload "
                    "(context, data, specs, file contents) into 'inputs' (a "
                    "str -> str dict) and have 'query' refer to it by key, e.g. "
                    "\"Answer INPUTS['question'] using INPUTS['corpus']\"."
                )
            inputs = spec.get("inputs")
            if isinstance(inputs, dict) and "query" in inputs:
                raise ValueError(
                    "launch_subagents(...) spec inputs must not contain reserved "
                    "key 'query'; put the child task in the top-level spec "
                    "'query' and use another input key for supporting text"
                )
            handle = flow_delegate(
                name=spec.get("name", "subagent"),
                query=spec["query"],
                inputs=inputs,
                model=spec.get("model", "default"),
                output_schema=spec.get("output_schema"),
            )
            if isinstance(handle, str):
                results[i] = handle
            else:
                handles.append(handle)
                positions.append(i)
        if handles:
            waited = await flow_wait(*handles)
            for pos, result in zip(positions, waited):
                results[pos] = result
        return results

    return launch_subagents


def make_show_vars(namespace: dict[str, Any]):
    """Build ``SHOW_VARS()`` over the live REPL ``namespace`` (off by default)."""

    @tool("Show current public REPL variable names and their type names.")
    def SHOW_VARS() -> dict[str, str]:
        out: dict[str, str] = {}
        for name, value in namespace.items():
            if name.startswith("_") or name in HIDDEN_REPL_TOOL_NAMES:
                continue
            if name == "SHOW_VARS" or get_tool_metadata(value) is not None:
                continue
            out[name] = type(value).__name__
        return out

    return SHOW_VARS


def make_history(flow: BaseFlow, engine_context: EngineContext) -> "History":
    """Build this agent's ``HISTORY`` view, mirroring ``make_done``.

    The view resolves the agent's *live* graph by id at call time (never captures
    a ``Graph``), so it stays correct across deep-copy-on-adopt, ``inject``, and
    ``truncate`` — and ships nothing until a method is actually called. ``HISTORY``
    is host-bound: a remote runtime object-proxies it so the slice computed over
    the full host trajectory is all that crosses the wire.
    """
    return History(lambda: flow.graph.agents.get(engine_context.agent_id))


class History:
    """Read-only view of THIS agent's own message trajectory (untruncated).

    ``build_messages`` may window or truncate what the model sees each turn;
    ``HISTORY`` always exposes the full user/assistant projection so code can
    recover earlier turns. It holds a *resolver* (not a ``Graph``) and reads the
    live trajectory on every call, so it reflects everything recorded so far —
    even after the run's graph has been copied or edited.
    """

    __slots__ = ("_resolve",)

    def __init__(self, resolve: "Callable[[], Graph | None]") -> None:
        self._resolve = resolve

    def messages(self) -> list[dict[str, str]]:
        """The full chat projection of this agent's turns (no system message)."""
        agent = self._resolve()
        return agent.messages("") if agent is not None else []

    def count(self) -> int:
        """Number of projected messages (so ``len`` works over the wire too)."""
        return len(self.messages())

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return f"HISTORY({self.count()} messages)"

    def read(self, i: int) -> dict[str, str]:
        """One message by index (negative indexes count from the end)."""
        return self.messages()[i]

    def last(self, n: int = 5) -> list[dict[str, str]]:
        """The most recent ``n`` messages."""
        if n <= 0:
            return []
        return self.messages()[-n:]

    def text(self, *, roles: tuple[str, ...] = ("user", "assistant")) -> str:
        """The selected turns flattened to ``[role] content`` blocks."""
        return "\n\n".join(
            f"[{m['role']}] {m['content']}"
            for m in self.messages()
            if m["role"] in roles
        )

    def grep(self, pattern: str, *, max_results: int = 50) -> list[str]:
        """Lines across all turns matching ``pattern`` (``idx[role]: line``)."""
        regex = re.compile(pattern)
        out: list[str] = []
        for idx, message in enumerate(self.messages()):
            for line in message["content"].splitlines():
                if regex.search(line):
                    out.append(f"{idx}[{message['role']}]: {line}")
                    if len(out) >= max_results:
                        return out
        return out


__all__ = [
    "DEFAULT_MAX_QUERY_CHARS",
    "History",
    "make_delegate",
    "make_done",
    "make_history",
    "make_launch_subagents",
    "make_show_vars",
    "make_wait",
]
