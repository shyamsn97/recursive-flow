"""Graph truncation and pruning helpers (pure: copy → mutate → return).

These operate on an already-copied graph; the public entry points
(:func:`truncate_after`, :func:`truncate_agent`, ...) copy first. Editing a
child's trajectory can strand a parent's resume path, so the helpers also drop
stale ancestor states and unreachable children to keep the tree consistent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rflow.graph.graph import Graph, Node


def validate_truncate(truncate: str) -> None:
    if truncate not in {"none", "after", "descendants"}:
        raise ValueError("truncate must be 'none', 'after', or 'descendants'")


def truncate_after(
    graph: "Graph", node_id: str, *, descendants: bool = True
) -> "Graph":
    """Return a copy with local nodes after ``node_id`` removed."""
    out = graph.copy(deep=True)
    owner = out.node_owner(node_id)
    index = owner._index_of(node_id)
    owner.nodes = owner.nodes[: index + 1]
    if descendants:
        prune_unreachable_children(out)
    return out


def truncate_agent(graph: "Graph", agent_id: str, *, after_seq: int) -> "Graph":
    """Return a copy with ``agent_id`` nodes after ``after_seq`` removed."""
    out = graph.copy(deep=True)
    out[agent_id].nodes = [
        node for node in out[agent_id].nodes if node.seq <= after_seq
    ]
    prune_unreachable_children(out)
    return out


def prune_descendants_spawned_after(graph: "Graph", agent_id: str, seq: int) -> "Graph":
    """Return a copy pruning children spawned after ``agent_id`` ``seq``."""
    out = graph.copy(deep=True)
    owner = out[agent_id]
    kept_ids = {node.id for node in owner.nodes if node.seq <= seq}
    owner.children = {
        aid: child
        for aid, child in owner.children.items()
        if child.parent_node_id in kept_ids
    }
    for child in owner.children.values():
        prune_unreachable_children(child)
    return out


def invalidate_ancestors_after_child_edit(
    graph: "Graph", owner: "Graph", *, truncate: str
) -> None:
    """Drop stale ancestor resume paths after a child timeline edit."""
    if truncate == "none" or owner.parent_agent_id is None:
        return
    truncate_ancestors_waiting_on(graph, owner.agent_id)


def apply_descendant_truncation(graph: "Graph", owner: "Graph", old: "Node") -> None:
    """Apply child cleanup for route-changing edits."""
    prune_waiting_on_children(owner, old)
    prune_unreachable_children(graph)


def prune_waiting_on_children(owner: "Graph", old: "Node") -> None:
    """Drop children that were only part of a replaced supervisor route."""
    for child_id in getattr(old, "waiting_on", ()):
        owner.children.pop(child_id, None)


def prune_unreachable_children(graph: "Graph") -> None:
    """Drop children whose ``parent_node_id`` no longer exists."""
    valid_ids = {node.id for node in graph.nodes}
    graph.children = {
        aid: child
        for aid, child in graph.children.items()
        if child.parent_node_id in valid_ids
    }
    for child in graph.children.values():
        prune_unreachable_children(child)


def truncate_ancestors_waiting_on(graph: "Graph", agent_id: str) -> None:
    """Drop stale parent resume states that depended on ``agent_id``."""
    child = graph[agent_id]
    if child.parent_agent_id is None:
        return
    parent = graph[child.parent_agent_id]
    wait_index = None
    for index, node in enumerate(parent.nodes):
        if agent_id in getattr(node, "waiting_on", ()):
            wait_index = index
    if wait_index is None:
        return
    parent.nodes = parent.nodes[: wait_index + 1]
    if parent.parent_agent_id is not None:
        truncate_ancestors_waiting_on(graph, parent.agent_id)


__all__ = [
    "apply_descendant_truncation",
    "invalidate_ancestors_after_child_edit",
    "prune_descendants_spawned_after",
    "prune_unreachable_children",
    "prune_waiting_on_children",
    "truncate_after",
    "truncate_agent",
    "truncate_ancestors_waiting_on",
    "validate_truncate",
]
