"""Position stamping for out-of-band graph edits.

When an edit (replace/inject) drops a node into a graph position, the new node
must take that slot's identity — ``agent_id``/``seq`` and the right
``global_step`` — while getting a fresh node id. An observation that replaces or
follows an action in the same engine tick keeps that action's ``global_step``
(they're one tick); anything else takes the next step.
"""

from __future__ import annotations

from rflow.graph.graph import ActionNode, Node, ObservationNode


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
) -> Node:
    """Stamp ``replacement`` into a graph position with a fresh id."""
    fields = replacement.model_dump(
        exclude={"id", "agent_id", "seq", "global_step"}, mode="python"
    )
    return replacement.__class__(
        agent_id=agent_id, seq=seq, global_step=global_step, **fields
    )


__all__ = ["next_global_step_for_position", "stamp_node_for_position"]
