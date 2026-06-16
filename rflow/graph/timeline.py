"""Retrace a final :class:`Graph` into per-tick snapshots.

The engine appends one node at a time but typically only the final graph is
persisted. :func:`retrace_steps` reconstructs the intermediate snapshots — one
per parallel tick under unbounded concurrency — so tools (viewer slider,
exporters) can step through the run without re-executing any code.

When every node carries a ``global_step`` (live runs stamp them), the fast path
slices on those steps. Otherwise a slow path simulates readiness: a child can't
advance before its parent's spawning :class:`SupervisingOutput`, and a parent's
resume waits for every awaited child to finish.
"""

from __future__ import annotations

from rflow.graph.graph import Graph

# Observation types where an agent rests between engine steps; each tick
# advances forward until it lands on (and includes) one of these.
_STABLE_TYPES: frozenset[str] = frozenset(
    {
        "user_query",
        "llm_output",
        "exec_output",
        "supervising_output",
        "error_output",
        "done_output",
    }
)


def retrace_steps(graph: Graph) -> list[Graph]:
    """Return one :class:`Graph` snapshot per parallel tick, in order.

    The first snapshot is the root's :class:`UserQuery`; each subsequent
    snapshot adds one ``(action, observation)`` pair per ready agent. The final
    snapshot equals ``graph``.
    """
    ticks = (
        _global_step_ticks(graph)
        if _all_nodes_have_global_steps(graph)
        else _execution_ticks(graph)
    )
    return _snapshots_from_ticks(graph, ticks)


def _snapshots_from_ticks(
    graph: Graph, ticks: list[list[tuple[str, int]]]
) -> list[Graph]:
    if not ticks:
        return [graph]
    snapshots: list[Graph] = []
    counts: dict[str, int] = dict.fromkeys(graph.agents, 0)
    for tick in ticks:
        for aid, index in tick:
            counts[aid] = max(counts.get(aid, 0), index + 1)
        snap = graph.copy(deep=True)
        for sub in snap.walk():
            keep = counts.get(sub.agent_id, 0)
            del sub.nodes[keep:]
        snapshots.append(snap)
    return snapshots


def _all_nodes_have_global_steps(graph: Graph) -> bool:
    nodes = list(graph.all_nodes)
    return bool(nodes) and all(node.global_step is not None for node in nodes)


def _global_step_ticks(graph: Graph) -> list[list[tuple[str, int]]]:
    steps = sorted(
        {node.global_step for node in graph.all_nodes if node.global_step is not None}
    )
    counts: dict[str, int] = dict.fromkeys(graph.agents, 0)
    ticks: list[list[tuple[str, int]]] = []
    for cutoff in steps:
        tick: list[tuple[str, int]] = []
        for sub in graph.walk():
            keep = counts.get(sub.agent_id, 0)
            while keep < len(sub.nodes):
                step = sub.nodes[keep].global_step
                if step is None or step > cutoff:
                    break
                tick.append((sub.agent_id, keep))
                keep += 1
            counts[sub.agent_id] = keep
        if tick:
            ticks.append(tick)
    return ticks


def _execution_ticks(graph: Graph) -> list[list[tuple[str, int]]]:
    """Return execution events grouped into parallel ticks (slow path)."""
    states = {aid: list(sub.nodes) for aid, sub in graph.agents.items()}

    spawn_dep: dict[str, tuple[str, int]] = {}
    for aid, agent_states in states.items():
        for i, s in enumerate(agent_states):
            if s.type != "supervising_output":
                continue
            for child in getattr(s, "waiting_on", []):
                spawn_dep.setdefault(child, (aid, i))

    pos: dict[str, int] = dict.fromkeys(states, 0)
    ticks: list[list[tuple[str, int]]] = []

    def is_ready(aid: str) -> bool:
        i = pos[aid]
        if i >= len(states[aid]):
            return False
        if i == 0:
            dep = spawn_dep.get(aid)
            if dep is None:
                return True
            parent_aid, parent_idx = dep
            return pos.get(parent_aid, 0) > parent_idx
        prev = states[aid][i - 1]
        if prev.type == "supervising_output":
            for child in getattr(prev, "waiting_on", []):
                if pos.get(child, 0) < len(states.get(child, [])):
                    return False
        return True

    def step_count(aid: str) -> int:
        i = pos[aid]
        agent_states = states[aid]
        n = len(agent_states)
        j = i
        while j < n:
            if agent_states[j].type in _STABLE_TYPES:
                return j - i + 1
            j += 1
        return max(1, n - i)

    while True:
        ready = sorted(aid for aid in pos if is_ready(aid))
        if not ready:
            break
        tick: list[tuple[str, int]] = []
        for aid in ready:
            count = step_count(aid)
            for _ in range(count):
                tick.append((aid, pos[aid]))
                pos[aid] += 1
        ticks.append(tick)

    for aid in sorted(pos):
        while pos[aid] < len(states[aid]):
            ticks.append([(aid, pos[aid])])
            pos[aid] += 1

    return ticks


__all__ = ["retrace_steps"]
