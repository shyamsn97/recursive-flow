"""Optional Weights & Biases logger."""

from __future__ import annotations

from pathlib import Path

from benchmarks.eval import logger
from benchmarks.eval.metrics import summarize
from benchmarks.eval.types import Logger, Row


@logger("wandb")
class WandbLogger(Logger):
    def __init__(
        self,
        project: str = "rflow-eval",
        entity: str | None = None,
        root: Path | str | None = None,
    ) -> None:
        self.project = project
        self.entity = entity
        self.root = Path(root) if root is not None else None
        self._wandb = None

    def start(self, config: dict) -> None:
        import wandb

        self._wandb = wandb
        wandb.init(project=self.project, entity=self.entity, config=config)

    def row(self, row: Row) -> None:
        if self._wandb is None:
            return
        self._wandb.log(
            {
                "dataset": row.dataset,
                "runner": row.runner,
                "score": row.score.value,
                "correct": row.score.correct,
                "error": 1 if row.prediction.error else 0,
                "input_tokens": row.prediction.usage.get("input_tokens", 0),
                "output_tokens": row.prediction.usage.get("output_tokens", 0),
                "time_seconds": row.prediction.metrics.get("time_seconds", 0.0),
            }
        )

    def summary(self, rows: list[Row]) -> None:
        if self._wandb is None:
            return
        summary = summarize(rows)
        overall = summary.get("overall", {})
        payload = _flatten_summary("overall", overall)
        for name, values in summary.get("by_dataset", {}).items():
            payload.update(_flatten_summary(f"benchmark/{name}", values))
        for name, values in summary.get("by_runner", {}).items():
            payload.update(_flatten_summary(f"runner/{name}", values))
        for name, values in summary.get("by_runner_dataset", {}).items():
            payload.update(_flatten_summary(f"runner_benchmark/{name}", values))
        self._wandb.summary.update(payload)
        self._wandb.log(
            {
                "summary/by_benchmark": _summary_table(self._wandb, summary.get("by_dataset", {})),
                "summary/by_runner": _summary_table(self._wandb, summary.get("by_runner", {})),
                "summary/by_runner_benchmark": _summary_table(
                    self._wandb, summary.get("by_runner_dataset", {})
                ),
            }
        )
        if self.root is not None:
            report_path = self.root / "report.md"
            if report_path.exists():
                self._wandb.save(str(report_path), policy="now")

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()


def _flatten_summary(prefix: str, values: dict) -> dict[str, object]:
    return {
        f"{prefix}/count": values.get("count"),
        f"{prefix}/graded_count": values.get("graded_count"),
        f"{prefix}/correct": values.get("correct"),
        f"{prefix}/incorrect": values.get("incorrect"),
        f"{prefix}/accuracy": values.get("accuracy"),
        f"{prefix}/accuracy_pct": values.get("accuracy_pct"),
        f"{prefix}/score": values.get("score"),
        f"{prefix}/errors": values.get("errors"),
        f"{prefix}/input_tokens": values.get("input_tokens"),
        f"{prefix}/output_tokens": values.get("output_tokens"),
        f"{prefix}/time_seconds": values.get("time_seconds"),
    }


def _summary_table(wandb, values: dict[str, dict]):
    table = wandb.Table(
        columns=[
            "name",
            "rows",
            "graded",
            "correct",
            "pct_correct",
            "score",
            "errors",
            "avg_input_tokens",
            "avg_output_tokens",
            "avg_time_seconds",
        ]
    )
    for name, item in values.items():
        table.add_data(
            name,
            item.get("count"),
            item.get("graded_count"),
            item.get("correct"),
            item.get("accuracy_pct"),
            item.get("score"),
            item.get("errors"),
            item.get("input_tokens"),
            item.get("output_tokens"),
            item.get("time_seconds"),
        )
    return table


__all__ = ["WandbLogger"]
