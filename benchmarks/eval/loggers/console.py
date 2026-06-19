"""Console progress logger."""

from __future__ import annotations

from benchmarks.eval import logger
from benchmarks.eval.metrics import summarize
from benchmarks.eval.types import Example, Logger, Row


@logger("console")
class ConsoleLogger(Logger):
    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet

    def example_start(self, example: Example, *, runner: str, model: str) -> None:
        if not self.quiet:
            print(f"[bench] {example.id} runner={runner} model={model}")

    def row(self, row: Row) -> None:
        if self.quiet:
            return
        status = "ERR" if row.prediction.error else f"{row.score.value:.3g}"
        print(f"[bench] row {row.dataset}/{row.example_id}/{row.runner}: {status}")

    def summary(self, rows: list[Row]) -> None:
        if self.quiet:
            return
        overall = summarize(rows).get("overall", {})
        print(f"[bench] done count={len(rows)} score={overall.get('score')}")


__all__ = ["ConsoleLogger"]
