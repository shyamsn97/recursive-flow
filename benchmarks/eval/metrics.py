"""Aggregation helpers for benchmark result rows."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

from rflow import Graph

from benchmarks.eval.core import EvalResult


class MetricsAggregator:
    """Facade over the pure metric helpers used by the orchestrator."""

    def graph_metrics(self, graph: Graph | None) -> dict[str, Any]:
        return graph_metrics(graph)

    def summarize(self, results: list[EvalResult]) -> dict[str, Any]:
        return summarize(results)


def graph_metrics(graph: Graph | None) -> dict[str, Any]:
    if graph is None:
        return {}
    agents = list(graph.walk())
    node_counts = [len(agent.nodes) for agent in agents]
    child_counts = [len(agent.children) for agent in agents]
    llm_turns = sum(
        1 for agent in agents for node in agent.nodes if node.type == "llm_output"
    )
    return {
        "agents": len(agents),
        "nodes": sum(node_counts),
        "llm_turns": llm_turns,
        "max_depth": max((agent.depth for agent in agents), default=0),
        "max_branching": max(child_counts, default=0),
    }


def summarize(results: list[EvalResult]) -> dict[str, Any]:
    """Summarize rows overall and by runner/task pair."""

    if not results:
        return {"count": 0, "accuracy_by_task": {}}

    def summarize_group(rows: list[EvalResult]) -> dict[str, Any]:
        return {
            "count": len(rows),
            "accuracy": mean(1.0 if row.correct else 0.0 for row in rows),
            "score": mean(row.score for row in rows),
            "time_seconds": mean(row.time_seconds for row in rows),
            "input_tokens": mean(row.input_tokens for row in rows),
            "output_tokens": mean(row.output_tokens for row in rows),
            "total_tokens": mean(row.total_tokens for row in rows),
            "errors": sum(1 for row in rows if row.error),
        }

    by_runner: dict[str, list[EvalResult]] = defaultdict(list)
    by_task: dict[str, list[EvalResult]] = defaultdict(list)
    by_pair: dict[str, list[EvalResult]] = defaultdict(list)
    runner_order = list(dict.fromkeys(row.runner for row in results))
    for row in results:
        by_runner[row.runner].append(row)
        by_task[row.task_name].append(row)
        by_pair[f"{row.runner}/{row.task_name}"].append(row)
    pair_summary = {
        key: summarize_group(rows) for key, rows in sorted(by_pair.items())
    }
    accuracy_by_task = _accuracy_by_task(by_task, pair_summary, runner_order)
    return {
        "count": len(results),
        "overall": summarize_group(results),
        "by_runner": {
            key: summarize_group(rows) for key, rows in sorted(by_runner.items())
        },
        "by_task": {
            key: summarize_group(rows) for key, rows in sorted(by_task.items())
        },
        "by_runner_task": pair_summary,
        "accuracy_by_task": accuracy_by_task,
        "tasks_won": _task_wins(by_task, pair_summary, runner_order),
    }


def _accuracy_by_task(
    by_task: dict[str, list[EvalResult]],
    pair_summary: dict[str, dict[str, Any]],
    runner_order: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for task_name in sorted(by_task):
        matrix[task_name] = {}
        for runner in runner_order:
            values = pair_summary.get(f"{runner}/{task_name}")
            if values is None:
                continue
            matrix[task_name][runner] = {
                "accuracy": values["accuracy"],
                "score": values["score"],
                "count": values["count"],
                "errors": values["errors"],
                "total_tokens": values["total_tokens"],
                "time_seconds": values["time_seconds"],
            }
    return matrix


def _task_wins(
    by_task: dict[str, list[EvalResult]],
    pair_summary: dict[str, dict[str, Any]],
    runner_order: list[str],
) -> dict[str, Any]:
    wins = {runner: 0 for runner in runner_order}
    winners: dict[str, str] = {}
    for task_name in sorted(by_task):
        candidates = [
            runner
            for runner in runner_order
            if f"{runner}/{task_name}" in pair_summary
        ]
        if not candidates:
            continue
        winner = max(
            candidates,
            key=lambda runner: (
                pair_summary[f"{runner}/{task_name}"]["accuracy"],
                -runner_order.index(runner),
            ),
        )
        wins[winner] += 1
        winners[task_name] = winner
    return {"counts": wins, "by_task": winners}


__all__ = ["MetricsAggregator", "graph_metrics", "summarize"]
