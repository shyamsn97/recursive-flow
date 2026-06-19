"""Optional Weights & Biases logger."""

from __future__ import annotations

from benchmarks.eval import logger
from benchmarks.eval.metrics import summarize
from benchmarks.eval.types import Logger, Row


@logger("wandb")
class WandbLogger(Logger):
    def __init__(self, project: str = "rflow-eval", entity: str | None = None) -> None:
        self.project = project
        self.entity = entity
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
        self._wandb.summary.update(
            {
                "overall/score": overall.get("score"),
                "overall/accuracy": overall.get("accuracy"),
                "overall/errors": overall.get("errors"),
            }
        )

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()


__all__ = ["WandbLogger"]
