"""Scheduling helpers for :class:`rflow.flow.RecursiveFlow`.

This module owns the outer ``step`` loop and async-child refill policy. It is
intentionally not a standalone scheduler object: ``RecursiveFlow`` still owns engine
state, public override methods, sessions, runtimes, and pools.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from rflow.engine.actions import act
from rflow.engine.replay import can_resume
from rflow.graph import (
    Graph,
    SupervisingOutput,
)


def _sync_and_plan_step(
    engine,
    graph: Graph,
    *,
    task_prefix: str = "",
    allow_eager_children: bool = True,
) -> tuple[Any, Graph, list[tuple[str, Callable[[], None]]]]:
    if engine.workspace is not None:
        graph = engine.workspace.sync_graph_if_changed(graph)
    else:
        persisted = engine.session.load_graph()
        persisted_agents = set(persisted.agents)
        changed = persisted_agents != set(graph.agents)
        if not changed:
            for aid in persisted_agents:
                current = persisted.agents[aid].nodes
                proposed = graph.agents[aid].nodes
                if len(proposed) != len(current):
                    changed = True
                    break
                for old, new in zip(current, proposed, strict=True):
                    if old.id != new.id:
                        changed = True
                        break
                    if old.model_dump(mode="python") != new.model_dump(mode="python"):
                        changed = True
                        break
                if changed:
                    break
        if changed:
            graph = engine.commit_graph(graph)

    if engine.config.eager_children and not allow_eager_children:
        raise ValueError(
            "parallel_step(...) does not support eager_children=True. "
            "Set eager_children=False so the coordinator owns global scheduling."
        )

    runnable = engine.node_scheduler.runnable_agents(graph)
    if not runnable:
        return engine, graph, []
    plan = act(
        graph,
        config=engine.config,
        runnable=runnable,
        terminate_requested=engine.terminate_requested,
    )
    if not plan:
        return engine, graph, []
    prefix = f"{task_prefix}:" if task_prefix else ""
    tasks = [
        (f"{prefix}{aid}", lambda action=action: engine.apply_one(action))
        for aid, action in plan.items()
    ]
    return engine, graph, tasks


def step(engine, graph: Graph, *, pool: Any = None) -> Graph:
    """Advance the run by one synchronized or async-child batch.

    If ``graph`` has been edited outside normal execution (injection,
    replacement, truncation, fork repair), sync that whole graph snapshot before
    planning. Once planning starts, transitions use append-only ``write_state``
    via ``append_node`` rather than rewriting the graph for every new node.
    """

    engine, graph, tasks = _sync_and_plan_step(engine, graph)
    if not tasks:
        return graph

    runner = pool or engine.pool
    if engine.config.eager_children:
        runner.run_until_idle(tasks, engine._refill_eager_children)
    else:
        runner.execute(tasks)

    graph = engine.session.load_graph()
    if engine.workspace is not None:
        engine.workspace.mark_graph_synced(graph)
    return graph


def parallel_step(
    pairs: Sequence[tuple[Any, Graph]],
    *,
    pool: Any = None,
) -> list[Graph]:
    """Advance several independent graphs with one shared action batch.

    This is the cross-graph equivalent of :func:`step`: all graphs are synced
    and planned first, then every runnable action is executed through one pool.
    ``eager_children=True`` is intentionally rejected because its refill loop is
    graph-local; a global eager scheduler needs a separate design.
    """

    synced: list[tuple[Any, Graph]] = []
    tasks: list[tuple[str, Callable[[], None]]] = []
    prefix_tasks = len(pairs) > 1
    for index, (agent, graph) in enumerate(pairs):
        task_prefix = str(index) if prefix_tasks else ""
        engine, graph, planned = _sync_and_plan_step(
            agent,
            graph,
            task_prefix=task_prefix,
            allow_eager_children=False,
        )
        synced.append((engine, graph))
        tasks.extend(planned)

    if tasks:
        runner = pool or synced[0][0].pool
        runner.execute(tasks)

    out: list[Graph] = []
    for engine, graph in synced:
        if tasks:
            graph = engine.session.load_graph()
            if engine.workspace is not None:
                engine.workspace.mark_graph_synced(graph)
        out.append(graph)
    return out


def refill_eager_children(
    engine,
    _done_id: str,
    _result: object,
    active_ids: set[str],
) -> list[tuple[str, Callable[[], None]]]:
    """Return newly runnable eager-child tasks after one task completes."""

    graph = engine.session.load_graph()
    tasks: list[tuple[str, Callable[[], None]]] = []
    scheduled: set[str] = set(active_ids)

    for supervisor in graph.walk():
        cur = supervisor.current()
        if not isinstance(cur, SupervisingOutput):
            continue
        if not supervisor.config.get("eager_children", engine.config.eager_children):
            continue

        runnable = (
            [supervisor.agent_id]
            if can_resume(supervisor, cur)
            else engine.node_scheduler.runnable_descendants(supervisor)
        )
        runnable = [aid for aid in runnable if aid not in scheduled]
        if not runnable:
            continue

        plan = act(
            graph,
            config=engine.config,
            runnable=runnable,
            terminate_requested=engine.terminate_requested,
        )
        for aid, action in plan.items():
            if aid in scheduled:
                continue
            scheduled.add(aid)
            tasks.append((aid, lambda action=action: engine.apply_one(action)))

    return tasks


__all__ = ["parallel_step", "refill_eager_children", "step"]
