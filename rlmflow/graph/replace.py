"""Graph node replacement helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rlmflow.graph.node import ActionNode, Node, ObservationNode
from rlmflow.graph.node_state import inherit_node_state
from rlmflow.graph.truncation import (
    apply_descendant_truncation,
    invalidate_ancestors_after_child_edit,
    validate_truncate,
)

if TYPE_CHECKING:
    from rlmflow.graph.graph import Graph


def replace_node(
    graph: Graph,
    node_id: str,
    node: Node,
    *,
    truncate: str = "descendants",
    output_schema: dict[str, Any] | None = None,
    inherit_output_schema: bool = True,
) -> Graph:
    """Return a copy with ``node_id`` replaced by ``node``."""

    validate_truncate(truncate)
    out = graph.copy(deep=True)
    owner = out.node_owner(node_id)
    index = owner._index_of(node_id)
    old, _fixed = _replace_node_at_index(
        owner,
        index,
        node,
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
        output_schema=output_schema,
        inherit_output_schema=inherit_output_schema,
    )


def replace_last_observation(
    graph: Graph,
    agent_id: str,
    node: ObservationNode,
    *,
    truncate: str = "descendants",
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
        output_schema=output_schema,
        inherit_output_schema=inherit_output_schema,
    )


def _replace_node_at_index(
    owner: Graph,
    index: int,
    node: Node,
    *,
    truncate: str,
    output_schema: dict[str, Any] | None,
    inherit_output_schema: bool,
) -> tuple[Node, Node]:
    """Replace one local state and optionally drop its local future."""

    old = owner.nodes[index]
    fixed = _node_for_replacement(
        old,
        node,
        output_schema=output_schema,
        inherit_output_schema=inherit_output_schema,
    )
    if (
        output_schema is None
        and inherit_output_schema
        and fixed.output_schema is None
        and owner.output_schema is not None
    ):
        fixed = fixed.update(output_schema=owner.output_schema)
    if truncate == "none":
        owner.nodes[index] = fixed
    else:
        owner.nodes = [*owner.nodes[:index], fixed]
    return old, fixed


def _node_for_replacement(
    old: Node,
    new: Node,
    *,
    output_schema: dict[str, Any] | None = None,
    inherit_output_schema: bool = True,
) -> Node:
    """Stamp ``new`` into ``old``'s graph position with a fresh id."""

    fields = new.model_dump(exclude={"id", "agent_id", "seq"}, mode="python")
    fixed = new.__class__(agent_id=old.agent_id, seq=old.seq, **fields)
    return inherit_node_state(
        source=old,
        replacement=fixed,
        output_schema=output_schema,
        inherit_output_schema=inherit_output_schema,
    )
