"""Small shared helpers used across the engine.

These are pure (or close to it) utilities that don't fit on a specific
transition module. Keeping them in one place avoids cross-imports between
``transitions``, ``actions``, ``scheduling``, and ``rlm``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from rflow.graph import Graph, Node, is_llm_action
from rflow.graph.node_state import inherit_node_state, next_global_step_for_position
from rflow.prompts.messages import EXECUTION_OUTPUT
from rflow.utils.pool import CallablePool, Pool, SequentialPool, ThreadPool
from rflow.workspace import Session

if TYPE_CHECKING:
    from rflow.engine.config import FlowConfig

ROOT_RUNTIME_ID = "root"


def prepare_node_for_append(
    graph: Graph,
    node: Node,
    *,
    global_step: int | None = None,
    inherit_output_schema: bool = True,
) -> Node:
    """Stamp ``node`` for the next position in ``graph`` without persisting it.

    Callers pass a node with only payload fields populated. This helper assigns
    the owning ``agent_id``, the next ``seq``, and any active node-carried state
    such as ``output_schema``.
    """

    previous = graph.nodes[-1] if graph.nodes else None
    next_seq = (previous.seq + 1) if previous else 0
    next_step = (
        global_step
        if global_step is not None
        else next_global_step_for_position(
            source=previous,
            replacement=node,
            next_global_step=graph.next_global_step(),
        )
    )
    payload = node.model_dump(
        exclude={"id", "agent_id", "seq", "global_step"},
        mode="python",
    )
    prepared = node.__class__(
        agent_id=graph.agent_id,
        seq=next_seq,
        global_step=next_step,
        **payload,
    )
    prepared = inherit_node_state(
        source=previous,
        replacement=prepared,
        inherit_output_schema=inherit_output_schema,
    )
    if prepared.output_schema is None and graph.output_schema is not None:
        prepared = prepared.update(output_schema=graph.output_schema)
    return prepared


def append_node(
    session: Session,
    graph: Graph,
    node: Node,
    *,
    global_step: int | None = None,
) -> Node:
    """Prepare, persist, and mirror ``node`` onto ``graph.nodes``.

    This is the normal transition helper after an agent has already been written
    to the session. Use it for append-only execution events such as
    ``LLMAction -> LLMOutput`` or ``ExecAction -> DoneOutput``.

    Do not use this for graph bootstrap or graph surgery. Those paths first
    construct/edit an in-memory graph snapshot, then sync the whole graph with
    ``rewrite_graph`` / workspace sync.

    This also mirrors the new node into the local ``graph.nodes`` list so
    consecutive appends within one transition compute ``seq`` correctly without
    needing to reload from the session between calls.
    """

    previous = graph.nodes[-1] if graph.nodes else None
    planned_global_step = (
        global_step
        if global_step is not None
        else next_global_step_for_position(
            source=previous,
            replacement=node,
            next_global_step=session.load_graph().next_global_step(),
        )
    )
    paired_global_step = next_global_step_for_position(
        source=previous,
        replacement=node,
        next_global_step=planned_global_step,
    )
    prepared = prepare_node_for_append(graph, node, global_step=paired_global_step)
    session.write_state(prepared)
    graph.nodes.append(prepared)
    return prepared


def unique_child_id(parent_aid: str, name: str, existing: set[str]) -> str:
    base = f"{parent_aid}.{name}"
    if base not in existing:
        return base
    i = 1
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


def create_pool(config: "FlowConfig", pool: Pool | Callable | None = None) -> Pool:
    if pool is not None:
        return pool if hasattr(pool, "execute") else CallablePool(pool)
    if config.max_concurrency is None or config.max_concurrency <= 1:
        return SequentialPool()
    return ThreadPool(config.max_concurrency)


def iteration_count(graph: Graph) -> int:
    """How many :class:`LLMAction` nodes the agent has emitted so far."""
    return sum(is_llm_action(s) for s in graph.nodes)


def budget_exceeded(graph: Graph, max_budget: int | None) -> int | None:
    """Return total tokens if the run is over budget, else ``None``."""
    if max_budget is None:
        return None
    total = graph.total_tokens()
    return total if total >= max_budget else None


def truncate_output(raw: object, max_length: int) -> object:
    """Cap REPL output at ``max_length`` chars; passthrough non-strings."""
    if isinstance(raw, str) and len(raw) > max_length:
        return raw[:max_length] + "\n...<truncated>"
    return raw


def format_exec_output(output: str) -> str:
    return EXECUTION_OUTPUT.format(output=output or "none")


__all__ = [
    "ROOT_RUNTIME_ID",
    "append_node",
    "budget_exceeded",
    "create_pool",
    "format_exec_output",
    "iteration_count",
    "prepare_node_for_append",
    "truncate_output",
    "unique_child_id",
]
