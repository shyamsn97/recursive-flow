"""Graph node replacement helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rflow.graph.node import ActionNode, Node, ObservationNode
from rflow.graph.node_state import stamp_node_for_position
from rflow.graph.truncation import (
    apply_descendant_truncation,
    invalidate_ancestors_after_child_edit,
    validate_truncate,
)

if TYPE_CHECKING:
    from rflow.graph.graph import Graph


def replace_node(
    graph: Graph,
    target: str | Node,
    node: Node,
    *,
    truncate: str = "descendants",
    branch_id: str | None = None,
    output_schema: dict[str, Any] | None = None,
    inherit_output_schema: bool = True,
) -> Graph:
    """Return a copy with ``target`` replaced by ``node``.

    ``target`` may be a node id or a :class:`Node` (its ``id`` is used).
    """

    node_id = target.id if isinstance(target, Node) else target
    validate_truncate(truncate)
    out = graph.copy(deep=True)
    owner = out.node_owner(node_id)
    index = owner._index_of(node_id)
    global_step = out.next_global_step()
    old, _fixed = _replace_node_at_index(
        owner,
        index,
        node,
        global_step=global_step,
        branch_id=branch_id,
        truncate=truncate,
        output_schema=output_schema,
        inherit_output_schema=inherit_output_schema,
    )
    invalidate_ancestors_after_child_edit(out, owner, truncate=truncate)
    if truncate == "descendants":
        apply_descendant_truncation(out, owner, old)
    return out


def replace_last_action(
    graph: Graph,
    agent_id: str,
    node: ActionNode,
    *,
    truncate: str = "descendants",
    branch_id: str | None = None,
    output_schema: dict[str, Any] | None = None,
    inherit_output_schema: bool = True,
) -> Graph:
    """Return a copy replacing ``agent_id``'s latest action node."""

    last = graph.last_action(agent_id)
    if last is None:
        raise KeyError(f"agent {agent_id!r} has no action node")
    return replace_node(
        graph,
        last.id,
        node,
        truncate=truncate,
        branch_id=branch_id,
        output_schema=output_schema,
        inherit_output_schema=inherit_output_schema,
    )


def replace_last_observation(
    graph: Graph,
    agent_id: str,
    node: ObservationNode,
    *,
    truncate: str = "descendants",
    branch_id: str | None = None,
    output_schema: dict[str, Any] | None = None,
    inherit_output_schema: bool = True,
) -> Graph:
    """Return a copy replacing ``agent_id``'s latest observation node."""

    last = graph.last_observation(agent_id)
    if last is None:
        raise KeyError(f"agent {agent_id!r} has no observation node")
    return replace_node(
        graph,
        last.id,
        node,
        truncate=truncate,
        branch_id=branch_id,
        output_schema=output_schema,
        inherit_output_schema=inherit_output_schema,
    )


def _replace_node_at_index(
    owner: Graph,
    index: int,
    node: Node,
    *,
    global_step: int,
    branch_id: str | None,
    truncate: str,
    output_schema: dict[str, Any] | None,
    inherit_output_schema: bool,
) -> tuple[Node, Node]:
    """Replace one local state and optionally drop its local future."""

    old = owner.nodes[index]
    fixed = _node_for_replacement(
        owner,
        old,
        node,
        global_step=global_step,
        branch_id=branch_id,
        output_schema=output_schema,
        inherit_output_schema=inherit_output_schema,
    )
    if truncate == "none":
        owner.nodes[index] = fixed
    else:
        owner.nodes = [*owner.nodes[:index], fixed]
    return old, fixed


def _node_for_replacement(
    owner: Graph,
    old: Node,
    new: Node,
    *,
    global_step: int | None = None,
    branch_id: str | None = None,
    output_schema: dict[str, Any] | None = None,
    inherit_output_schema: bool = True,
) -> Node:
    """Stamp ``new`` into ``old``'s graph position with a fresh id."""

    return stamp_node_for_position(
        source=old,
        replacement=new,
        agent_id=old.agent_id,
        seq=old.seq,
        global_step=(
            global_step if global_step is not None else owner.next_global_step()
        ),
        branch_id=branch_id,
        graph_output_schema=owner.output_schema,
        output_schema=output_schema,
        inherit_output_schema=inherit_output_schema,
    )
