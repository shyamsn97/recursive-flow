"""Markdown report logger."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from benchmarks.eval import logger
from benchmarks.eval.metrics import summarize
from benchmarks.eval.types import Logger, Row

MAX_ANSWER_CHARS = 300


@logger("report")
class ReportLogger(Logger):
    def __init__(self, root: Path | str = "benchmarks/runs/latest") -> None:
        self.root = Path(root)
        self._rows: list[Row] = []

    def row(self, row: Row) -> None:
        self._rows.append(row)
        self._write(self._rows)

    def summary(self, rows: list[Row]) -> None:
        self._write(rows)

    def _write(self, rows: list[Row]) -> None:
        summary = summarize(rows)
        lines = ["# Benchmark Report", ""]
        overall = summary.get("overall", {})
        if overall:
            lines.extend(
                [
                    "## Overall",
                    "",
                    f"- Count: {summary.get('count', 0)}",
                    f"- Accuracy: {overall.get('accuracy')}",
                    f"- Score: {overall.get('score')}",
                    f"- Errors: {overall.get('errors')}",
                    "",
                ]
            )
        lines.extend(["## By Runner", ""])
        for runner, values in summary.get("by_runner", {}).items():
            lines.append(f"- `{runner}`: score={values.get('score')} errors={values.get('errors')}")
        lines.extend(["", "## By Dataset", ""])
        for dataset, values in summary.get("by_dataset", {}).items():
            lines.append(f"- `{dataset}`: score={values.get('score')} errors={values.get('errors')}")
        lines.extend(["", *_examples_section(rows)])
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _examples_section(rows: list[Row]) -> list[str]:
    """Per-example answer-vs-expected comparison across every runner."""
    if not rows:
        return []
    grouped: "OrderedDict[tuple, list[Row]]" = OrderedDict()
    for row in rows:
        grouped.setdefault((row.dataset, row.example_id, row.seed), []).append(row)

    lines = ["## Examples", ""]
    for (dataset, example_id, seed), group in grouped.items():
        seed_label = "" if seed is None else f" (seed={seed})"
        lines.append(f"### `{dataset}` / `{example_id}`{seed_label}")
        lines.append("")
        lines.append(f"- Expected: {_format_cell(_expected(group))}")
        lines.append("")
        lines.append("| runner | correct | answer |")
        lines.append("| --- | --- | --- |")
        for row in group:
            answer = row.prediction.error or row.prediction.answer
            mark = "x" if row.prediction.error else _mark(row.score.correct, row.score.value)
            lines.append(f"| `{row.runner}` | {mark} | {_format_cell(answer)} |")
        lines.append("")
    return lines


def _expected(group: list[Row]):
    for row in group:
        expected = row.score.details.get("expected")
        if expected not in (None, [], ""):
            return expected
    return None


def _mark(correct: bool | None, value: float) -> str:
    if correct is True:
        return "PASS"
    if correct is False:
        return "FAIL"
    return f"{value:.2f}"


def _format_cell(value: object) -> str:
    if isinstance(value, (list, tuple)):
        text = " || ".join(str(item) for item in value)
    else:
        text = str(value)
    text = text.replace("\n", " ").replace("|", "\\|").strip()
    if len(text) > MAX_ANSWER_CHARS:
        text = text[:MAX_ANSWER_CHARS].rstrip() + "..."
    return text or "_(empty)_"


__all__ = ["ReportLogger"]
