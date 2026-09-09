"""Child launch: resolve, attach, submit, and the ``launch_subagent`` factory."""

from __future__ import annotations

import asyncio
from typing import Any

from rlmflow.graph.nodes import DEFAULT_QUERY, AgentStart, ExecAction, Node, validate_agent_name
from rlmflow.runtime.repl_client import current_rpc_call_id
from rlmflow.structured import json_schema_for
from rlmflow.tools.agents import AgentHandle
from rlmflow.tools.tools import tool


def launch_tool(flow: Any, node: Node):
    mutation_lock = asyncio.Lock()

    @tool(
        "Spawn or resume one focused child agent with an explicitly chosen "
        "registered model; the caller remains responsible for final synthesis.",
        proxy=True,
    )
    async def launch_subagent(
        goal: str,
        *,
        model: str,
        name: str | None = None,
        inputs: dict[str, str] | None = None,
        output_schema: Any = None,
        prompt_profile: str | None = None,
        reuse_repl: bool = False,
    ) -> AgentHandle:
        if not isinstance(node, ExecAction):
            raise TypeError("launch_subagent requires an ExecAction")
        if not isinstance(goal, str):
            raise TypeError("launch_subagent goal must be a string")
        spec = {
            "query": goal,
            "name": name,
            "inputs": dict(inputs or {}),
            "model": model,
            "output_schema": output_schema,
            "prompt_profile": prompt_profile,
            "reuse_repl": reuse_repl,
        }
        async with mutation_lock:
            existing = {id(child) for child in node.children if isinstance(child, AgentStart)}
            resolved = flow.resolve_child(node, spec, current_rpc_call_id())
            if id(resolved) not in existing:
                flow.submit_child(resolved)
        return AgentHandle(
            agent_id=resolved.id,
            name=resolved.config.name,
            path=resolved.config.path,
        )

    return launch_subagent


def resolve_child(
    flow: Any,
    action: ExecAction,
    spec: dict[str, Any],
    call_id: int,
) -> AgentStart:
    """Resolve one launch call to a direct child or refusal."""
    explicit_name = spec.get("name")
    existing = next(
        (
            child
            for child in action.children
            if isinstance(child, AgentStart)
            and (
                child.config.launch_call_id == call_id
                or (explicit_name is not None and child.config.name == explicit_name)
            )
        ),
        None,
    )
    if existing is not None:
        return existing

    parent = action.parent_agent
    query = spec.get("query", "")
    if explicit_name is None:
        used = {child.config.name for child in parent.sub_agents}
        index = call_id
        while f"child{index}" in used:
            index += 1
        name = f"child{index}"
    else:
        name = explicit_name
    validate_agent_name(name)
    if any(child.config.name == name for child in parent.sub_agents):
        raise ValueError(f"duplicate child name {name!r}")

    if parent.config.depth >= parent.config.max_depth:
        raise ValueError(f"cannot launch beyond max depth {parent.config.max_depth}")
    if len(query) > parent.config.max_query_chars:
        raise ValueError(f"subagent goal exceeds {parent.config.max_query_chars} characters")
    model = spec["model"]
    if model not in flow._llm_clients:
        available = ", ".join(sorted(flow._llm_clients))
        raise ValueError(f"unknown model {model!r}; available models: {available}")
    return flow.new_child(action, name, spec, call_id=call_id)


def submit_child(flow: Any, child: AgentStart) -> None:
    """Submit a newly attached child and all unfinished restored leaves."""
    queue = flow.queue
    if queue is None:
        raise RuntimeError("launch_subagent requires an active stream")
    if child.terminal:
        return
    for leaf in child.leaves():
        owner = leaf.parent_agent
        if owner is not None and not owner.terminal:
            queue.submit(
                leaf,
                flow._drive,
                publish=isinstance(leaf, AgentStart),
            )


def new_child(
    flow: Any,
    node: Node,
    name: str,
    spec: dict[str, Any],
    *,
    call_id: int,
) -> AgentStart:
    """Open a child agent of ``node``'s agent, attached to ``node``."""
    schema = spec.get("output_schema")
    overrides = {
        key: value
        for key, value in {
            "inputs": dict(spec.get("inputs") or {}),
            "model": spec.get("model"),
            "prompt_profile": spec.get("prompt_profile"),
            "output_schema": (json_schema_for(schema) if schema is not None else None),
            "reuse_repl": spec.get("reuse_repl"),
            "launch_call_id": call_id,
        }.items()
        if value is not None
    }
    child = AgentStart(
        content=spec.get("query") or DEFAULT_QUERY,
        config=node.parent_agent.config.child(name, **overrides),
    )
    node.append(child)
    node.children.sort(
        key=lambda value: (
            not isinstance(value, AgentStart),
            (
                value.config.launch_call_id
                if isinstance(value, AgentStart) and value.config.launch_call_id is not None
                else 0
            ),
        )
    )
    parent = node.parent_agent
    parent.sub_agents.sort(
        key=lambda value: (
            value.parent.seq if value.parent is not None else 0,
            value.config.launch_call_id if value.config.launch_call_id is not None else 0,
        )
    )
    return child


__all__ = [
    "launch_tool",
    "new_child",
    "resolve_child",
    "submit_child",
]
