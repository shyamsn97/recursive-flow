"""Optional Weights & Biases integration for eval runs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from benchmarks.eval.core import EvalResult


class WandbLogger:
    """Tiny wrapper that keeps the CLI usable when wandb is not installed."""

    def __init__(
        self,
        *,
        enabled: bool,
        project: str,
        entity: str | None,
        run_id: str,
        config: dict[str, Any],
    ) -> None:
        self.enabled = enabled
        self._run = None
        self._results: list[EvalResult] = []
        self._step = 0
        if not enabled:
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "wandb logging requested but wandb is not installed. "
                "Install with `pip install wandb` or `pip install -e .[eval]`."
            ) from exc
        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=run_id,
            config=config,
            reinit="finish_previous",
        )
        wandb.define_metric("eval/row")
        wandb.define_metric("eval/*", step_metric="eval/row")
        wandb.define_metric("overall/*", step_metric="eval/row")
        wandb.define_metric("by_runner/*", step_metric="eval/row")
        wandb.define_metric("by_task/*", step_metric="eval/row")
        wandb.define_metric("by_runner_task/*", step_metric="eval/row")
        wandb.define_metric("task_accuracy/*", step_metric="eval/row")
        wandb.define_metric("task_count/*", step_metric="eval/row")

    def log_result(self, result: EvalResult) -> None:
        if not self.enabled:
            return
        self._results.append(result)
        self._step += 1
        payload: dict[str, Any] = {
            "eval/row": self._step,
            "eval/correct": int(result.correct),
            "eval/score": result.score,
            "eval/time_seconds": result.time_seconds,
            "eval/input_tokens": result.input_tokens,
            "eval/output_tokens": result.output_tokens,
            "eval/total_tokens": result.total_tokens,
            "eval/iterations": result.iterations,
            "eval/error": int(bool(result.error)),
            "runner": result.runner,
            "task": result.task_name,
            "task_id": result.task_id,
            "seed": result.seed,
        }
        payload.update(self._running_metrics())
        for key, value in result.graph.items():
            if isinstance(value, int | float):
                payload[f"graph/{key}"] = value
        self._wandb.log(payload)

    def log_summary(self, summary: dict[str, Any]) -> None:
        if not self.enabled:
            return
        flat = _flatten_summary(summary)
        self._wandb.summary.update(flat)
        self._wandb.log(
            {
                **flat,
                "tables/summary": self._summary_table(summary),
                "tables/task_accuracy": self._task_accuracy_table(summary),
                "tables/results": self._results_table(),
            }
        )

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            self._wandb.finish()

    def _running_metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        groups: dict[str, list[EvalResult]] = defaultdict(list)
        groups["overall"].extend(self._results)
        for result in self._results:
            groups[f"by_runner/{result.runner}"].append(result)
            groups[f"by_task/{result.task_name}"].append(result)
            groups[f"by_runner_task/{result.runner}/{result.task_name}"].append(result)
        for prefix, rows in groups.items():
            metrics[f"{prefix}/count"] = len(rows)
            metrics[f"{prefix}/accuracy"] = sum(1 for row in rows if row.correct) / len(rows)
            metrics[f"{prefix}/score"] = sum(row.score for row in rows) / len(rows)
            metrics[f"{prefix}/errors"] = sum(1 for row in rows if row.error)
            if prefix.startswith("by_runner_task/"):
                _, runner, task = prefix.split("/", 2)
                metrics[f"task_accuracy/{task}/{_runner_metric_name(runner)}"] = metrics[
                    f"{prefix}/accuracy"
                ]
                metrics[f"task_count/{task}/{_runner_metric_name(runner)}"] = len(rows)
        return metrics

    def _results_table(self):
        table = self._wandb.Table(
            columns=[
                "task",
                "task_id",
                "runner",
                "seed",
                "model",
                "correct",
                "score",
                "error",
                "answer",
                "expected",
                "time_seconds",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "iterations",
                "graph_agents",
                "graph_nodes",
                "graph_llm_turns",
                "graph_max_depth",
                "graph_max_branching",
            ]
        )
        for result in self._results:
            table.add_data(
                result.task_name,
                result.task_id,
                result.runner,
                result.seed,
                result.model,
                result.correct,
                result.score,
                result.error,
                _truncate(result.answer),
                _truncate(result.expected),
                result.time_seconds,
                result.input_tokens,
                result.output_tokens,
                result.total_tokens,
                result.iterations,
                result.graph.get("agents"),
                result.graph.get("nodes"),
                result.graph.get("llm_turns"),
                result.graph.get("max_depth"),
                result.graph.get("max_branching"),
            )
        return table

    def _summary_table(self, summary: dict[str, Any]):
        table = self._wandb.Table(
            columns=[
                "group",
                "count",
                "accuracy",
                "score",
                "errors",
                "time_seconds",
                "input_tokens",
                "output_tokens",
                "total_tokens",
            ]
        )
        overall = summary.get("overall")
        if isinstance(overall, dict):
            _add_summary_row(table, "overall", overall)
        by_pair = summary.get("by_runner_task", {})
        if isinstance(by_pair, dict):
            for name, values in sorted(by_pair.items()):
                if isinstance(values, dict):
                    _add_summary_row(table, f"by_runner_task/{name}", values)
        for group_name in ("by_runner", "by_task"):
            values_by_name = summary.get(group_name, {})
            if isinstance(values_by_name, dict):
                for name, values in sorted(values_by_name.items()):
                    if isinstance(values, dict):
                        _add_summary_row(table, f"{group_name}/{name}", values)
        return table

    def _task_accuracy_table(self, summary: dict[str, Any]):
        table = self._wandb.Table(
            columns=[
                "task",
                "runner",
                "accuracy",
                "score",
                "n",
                "errors",
                "avg_tokens",
                "avg_latency_seconds",
            ]
        )
        matrix = summary.get("accuracy_by_task", {})
        if not isinstance(matrix, dict):
            return table
        for task_name, by_runner in sorted(matrix.items()):
            if not isinstance(by_runner, dict):
                continue
            for runner, values in sorted(by_runner.items()):
                if isinstance(values, dict):
                    table.add_data(
                        task_name,
                        _runner_label(runner),
                        values.get("accuracy"),
                        values.get("score"),
                        values.get("count"),
                        values.get("errors"),
                        values.get("total_tokens"),
                        values.get("time_seconds"),
                    )
        return table


def _add_summary_row(table, name: str, values: dict[str, Any]) -> None:
    table.add_data(
        name,
        values.get("count"),
        values.get("accuracy"),
        values.get("score"),
        values.get("errors"),
        values.get("time_seconds"),
        values.get("input_tokens"),
        values.get("output_tokens"),
        values.get("total_tokens"),
    )


def _flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    overall = summary.get("overall")
    if isinstance(overall, dict):
        for key, value in overall.items():
            if isinstance(value, int | float):
                flat[f"overall/{key}"] = value
    by_pair = summary.get("by_runner_task", {})
    if isinstance(by_pair, dict):
        for pair, values in by_pair.items():
            if not isinstance(values, dict):
                continue
            runner, _, task = pair.partition("/")
            prefix = f"by_runner_task/{runner}/{task}"
            for key, value in values.items():
                if isinstance(value, int | float):
                    flat[f"{prefix}/{key}"] = value
    for group_name in ("by_runner", "by_task"):
        values_by_name = summary.get(group_name, {})
        if not isinstance(values_by_name, dict):
            continue
        for name, values in values_by_name.items():
            if not isinstance(values, dict):
                continue
            for key, value in values.items():
                if isinstance(value, int | float):
                    flat[f"{group_name}/{name}/{key}"] = value
    tasks_won = summary.get("tasks_won", {})
    if isinstance(tasks_won, dict) and isinstance(tasks_won.get("counts"), dict):
        for runner, count in tasks_won["counts"].items():
            if isinstance(count, int | float):
                flat[f"tasks_won/{runner}"] = count
    if isinstance(summary.get("count"), int):
        flat["overall/result_rows"] = summary["count"]
    matrix = summary.get("accuracy_by_task", {})
    if isinstance(matrix, dict):
        for task_name, by_runner in matrix.items():
            if not isinstance(by_runner, dict):
                continue
            for runner, values in by_runner.items():
                if not isinstance(values, dict):
                    continue
                accuracy = values.get("accuracy")
                count = values.get("count")
                label = _runner_metric_name(runner)
                if isinstance(accuracy, int | float):
                    flat[f"task_accuracy/{task_name}/{label}"] = accuracy
                if isinstance(count, int | float):
                    flat[f"task_count/{task_name}/{label}"] = count
    return flat


def _truncate(value: Any, limit: int = 1000) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _runner_label(runner: str) -> str:
    return {
        "rflow": "RLMFlow",
        "official": "Official RLM",
        "vanilla": "Vanilla",
        "minrlm-reasoning": "minRLM",
    }.get(runner, runner)


def _runner_metric_name(runner: str) -> str:
    return {
        "rflow": "rlmflow",
        "official": "official_rlm",
        "vanilla": "vanilla",
        "minrlm-reasoning": "minrlm",
    }.get(runner, runner.replace(" ", "_"))


__all__ = ["WandbLogger"]
