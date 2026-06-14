"""Helpers for carrying active execution state across graph edits."""

from __future__ import annotations

from typing import Any

from rflow.graph.node import ActionNode, Node, ObservationNode


def inherit_node_state(
    *,
    source: Node | None,
    replacement: Node,
    output_schema: dict[str, Any] | None = None,
    inherit_output_schema: bool = True,
) -> Node:
    """Return ``replacement`` with active path state inherited or overridden.

    By default the replacement keeps its own schema if present, otherwise it
    inherits from ``source``. Pass ``output_schema`` to override. Pass
    ``inherit_output_schema=False`` with no schema to explicitly clear it.
    """

    next_schema = output_schema
    if next_schema is None:
        next_schema = replacement.output_schema
    if next_schema is None and inherit_output_schema and source is not None:
        next_schema = source.output_schema

    if replacement.output_schema == next_schema:
        return replacement
    return replacement.update(output_schema=next_schema)


def next_global_step_for_position(
    *,
    source: Node | None,
    next_global_step: int,
    replacement: Node | None = None,
) -> int:
    """Return the logical step for a node placed after/replacing ``source``."""

    if (
        source is not None
        and source.global_step is not None
        and isinstance(source, ActionNode)
        and (replacement is None or isinstance(replacement, ObservationNode))
    ):
        return source.global_step
    return next_global_step


def stamp_node_for_position(
    *,
    source: Node | None,
    replacement: Node,
    agent_id: str,
    seq: int,
    global_step: int | None = None,
    branch_id: str | None = None,
    graph_output_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    inherit_output_schema: bool = True,
) -> Node:
    """Stamp ``replacement`` into a graph position and carry active state."""

    fields = replacement.model_dump(
        exclude={"id", "agent_id", "seq", "global_step", "branch_id"},
        mode="python",
    )
    stamped = replacement.__class__(
        agent_id=agent_id,
        seq=seq,
        global_step=global_step,
        branch_id=branch_id if branch_id is not None else replacement.branch_id,
        **fields,
    )
    stamped = inherit_node_state(
        source=source,
        replacement=stamped,
        output_schema=output_schema,
        inherit_output_schema=inherit_output_schema,
    )
    if (
        output_schema is None
        and inherit_output_schema
        and stamped.output_schema is None
        and graph_output_schema is not None
    ):
        stamped = stamped.update(output_schema=graph_output_schema)
    return stamped


__all__ = [
    "inherit_node_state",
    "next_global_step_for_position",
    "stamp_node_for_position",
]
