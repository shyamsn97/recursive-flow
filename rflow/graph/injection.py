"""Immutable graph injection helpers."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from rflow.graph.node import ActionNode, ExecOutput, Node
from rflow.graph.node_state import stamp_node_for_position


def inject(
    graph: Any,
    *,
    target: str | re.Pattern[str] | Callable[[Any], Iterable[str | Any]],
    node: Node,
    mode: str = "append",
    output_schema: dict[str, Any] | None = None,
    inherit_output_schema: bool = True,
) -> Any:
    """Return a new graph with ``node`` injected at ``target``."""

    if mode != "append":
        raise NotImplementedError("Graph.inject currently supports append mode only")
    out = graph.copy(deep=True)
    targets = resolve_injection_targets(out, target)
    if not targets:
        raise KeyError(f"no injection targets matched {target!r}")
    for sub in targets:
        cur = sub.current()
        fixed = node_for_injection(
            sub,
            node,
            output_schema=output_schema,
            inherit_output_schema=inherit_output_schema,
        )
        if cur is not None and cur.terminal:
            raise ValueError(f"cannot inject into finished agent {sub.agent_id!r}")
        if cur is not None and is_action_like(cur) and is_action_like(fixed):
            raise ValueError(
                f"cannot queue multiple pending actions for {sub.agent_id!r}"
            )
        sub.nodes.append(fixed)
    return out


def inject_output(
    graph: Any,
    *,
    target: str | re.Pattern[str] | Callable[[Any], Iterable[str | Any]],
    output: str,
    content: str | None = None,
) -> Any:
    return inject(
        graph,
        target=target,
        node=ExecOutput(output=output, content=content or output),
    )


def resolve_injection_targets(
    graph: Any,
    target: str | re.Pattern[str] | Callable[[Any], Iterable[str | Any]],
) -> list[Any]:
    if callable(target):
        from rflow.graph.graph import Graph

        raw = list(target(graph))
        out = []
        for item in raw:
            out.append(item if isinstance(item, Graph) else graph[item])
        return out
    if isinstance(target, str) and target in graph.agents:
        return [graph[target]]
    compiled = re.compile(target) if isinstance(target, str) else target
    return [g for g in graph.walk() if compiled.search(g.agent_id)]


def is_action_like(node: Node) -> bool:
    return isinstance(node, ActionNode)


def node_for_injection(
    sub: Any,
    node: Node,
    *,
    output_schema: dict[str, Any] | None = None,
    inherit_output_schema: bool = True,
) -> Node:
    return _node_for_injection(
        sub,
        node,
        output_schema=output_schema,
        inherit_output_schema=inherit_output_schema,
    )


def _node_for_injection(
    sub: Any,
    node: Node,
    *,
    output_schema: dict[str, Any] | None,
    inherit_output_schema: bool,
) -> Node:
    source = sub.current()
    next_seq = (sub.nodes[-1].seq + 1) if sub.nodes else 0
    return stamp_node_for_position(
        source=source,
        replacement=node,
        agent_id=sub.agent_id,
        seq=next_seq,
        graph_output_schema=getattr(sub, "output_schema", None),
        output_schema=output_schema,
        inherit_output_schema=inherit_output_schema,
    )


__all__ = [
    "inject",
    "inject_output",
    "is_action_like",
    "node_for_injection",
    "resolve_injection_targets",
]
