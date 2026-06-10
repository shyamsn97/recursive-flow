"""Helpers for carrying active execution state across graph edits."""

from __future__ import annotations

from typing import Any

from rlmflow.graph.node import Node


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


__all__ = ["inherit_node_state"]
