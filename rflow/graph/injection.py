"""Immutable graph injection (copy → append → return).

Append a node onto one or more agents' trajectories out of band — e.g. feed an
:class:`ExecOutput` to steer a paused run, or fan an edit across every agent
matching a regex / predicate. Append mode only; the engine still owns live
appends via ``Flow._append``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from rflow.graph.graph import ActionNode, ExecOutput, Node
from rflow.graph.node_state import stamp_node_for_position

if TYPE_CHECKING:
    from rflow.graph.graph import Graph


def inject(
    graph: "Graph",
    *,
    target: "str | re.Pattern[str] | Callable[[Graph], Iterable[str | Graph]]",
    node: Node,
    mode: str = "append",
) -> "Graph":
    """Return a new graph with ``node`` appended at every matched ``target``."""
    if mode != "append":
        raise NotImplementedError("inject currently supports append mode only")
    out = graph.copy(deep=True)
    targets = resolve_injection_targets(out, target)
    if not targets:
        raise KeyError(f"no injection targets matched {target!r}")
    global_step = out.next_global_step()
    for sub in targets:
        cur = sub.current()
        fixed = node_for_injection(sub, node, global_step=global_step)
        if cur is not None and cur.terminal:
            raise ValueError(f"cannot inject into finished agent {sub.agent_id!r}")
        if cur is not None and is_action_like(cur) and is_action_like(fixed):
            raise ValueError(
                f"cannot queue multiple pending actions for {sub.agent_id!r}"
            )
        sub.nodes.append(fixed)
    return out


def inject_output(
    graph: "Graph",
    *,
    target: "str | re.Pattern[str] | Callable[[Graph], Iterable[str | Graph]]",
    output: str,
    content: str | None = None,
) -> "Graph":
    """Inject an :class:`ExecOutput` (convenience wrapper over :func:`inject`)."""
    return inject(
        graph, target=target, node=ExecOutput(output=output, content=content or output)
    )


def resolve_injection_targets(
    graph: "Graph",
    target: "str | re.Pattern[str] | Callable[[Graph], Iterable[str | Graph]]",
) -> "list[Graph]":
    if callable(target):
        from rflow.graph.graph import Graph as _Graph

        out: list[Any] = []
        for item in list(target(graph)):
            out.append(item if isinstance(item, _Graph) else graph[item])
        return out
    if isinstance(target, str) and target in graph.agents:
        return [graph[target]]
    compiled = re.compile(target) if isinstance(target, str) else target
    return [g for g in graph.walk() if compiled.search(g.agent_id)]


def is_action_like(node: Node) -> bool:
    return isinstance(node, ActionNode)


def node_for_injection(
    sub: "Graph", node: Node, *, global_step: int | None = None
) -> Node:
    source = sub.current()
    next_seq = (sub.nodes[-1].seq + 1) if sub.nodes else 0
    return stamp_node_for_position(
        source=source,
        replacement=node,
        agent_id=sub.agent_id,
        seq=next_seq,
        global_step=global_step if global_step is not None else sub.next_global_step(),
    )


__all__ = [
    "inject",
    "inject_output",
    "is_action_like",
    "node_for_injection",
    "resolve_injection_targets",
]
