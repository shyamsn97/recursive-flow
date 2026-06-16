"""Immutable node replacement (copy → swap → return).

Replace any node by id (or the latest action/observation of an agent) and
choose how much of its local future to drop via ``truncate``. Replacing an
observation that followed an action keeps the pair's ``global_step`` so the
timeline stays consistent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rflow.graph.node_state import (
    next_global_step_for_position,
    stamp_node_for_position,
)
from rflow.graph.truncation import (
    apply_descendant_truncation,
    invalidate_ancestors_after_child_edit,
    validate_truncate,
)

if TYPE_CHECKING:
    from rflow.graph.graph import ActionNode, Graph, Node, ObservationNode


def replace_node(
    graph: "Graph",
    target: "str | Node",
    node: "Node",
    *,
    truncate: str = "descendants",
) -> "Graph":
    """Return a copy with ``target`` replaced by ``node``.

    ``target`` may be a node id or a :class:`Node` (its ``id`` is used).
    ``truncate``: ``"none"`` keeps the local future, ``"after"`` drops local
    nodes past the target, ``"descendants"`` also prunes spawned children.
    """
    from rflow.graph.graph import Node as _Node

    node_id = target.id if isinstance(target, _Node) else target
    validate_truncate(truncate)
    out = graph.copy(deep=True)
    owner = out.node_owner(node_id)
    index = owner._index_of(node_id)
    old = owner.nodes[index]
    global_step = next_global_step_for_position(
        source=old, next_global_step=owner.next_global_step(), replacement=node
    )
    fixed = stamp_node_for_position(
        source=old,
        replacement=node,
        agent_id=old.agent_id,
        seq=old.seq,
        global_step=global_step,
    )
    if truncate == "none":
        owner.nodes[index] = fixed
    else:
        owner.nodes = [*owner.nodes[:index], fixed]
    invalidate_ancestors_after_child_edit(out, owner, truncate=truncate)
    if truncate == "descendants":
        apply_descendant_truncation(out, owner, old)
    return out


def replace_last_action(
    graph: "Graph",
    agent_id: str,
    node: "ActionNode",
    *,
    truncate: str = "descendants",
) -> "Graph":
    """Return a copy replacing ``agent_id``'s latest action node."""
    last = graph.last_action(agent_id)
    if last is None:
        raise KeyError(f"agent {agent_id!r} has no action node")
    return replace_node(graph, last.id, node, truncate=truncate)


def replace_last_observation(
    graph: "Graph",
    agent_id: str,
    node: "ObservationNode",
    *,
    truncate: str = "descendants",
) -> "Graph":
    """Return a copy replacing ``agent_id``'s latest observation node."""
    last = graph.last_observation(agent_id)
    if last is None:
        raise KeyError(f"agent {agent_id!r} has no observation node")
    return replace_node(graph, last.id, node, truncate=truncate)


__all__ = ["replace_last_action", "replace_last_observation", "replace_node"]
