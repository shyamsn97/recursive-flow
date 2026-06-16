"""Filesystem reports for model-oriented benchmark review."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol

from benchmarks.eval.config import RunConfig
from benchmarks.eval.core import EvalResult, TaskInstance


class ReportWriter(Protocol):
    def write(
        self,
        *,
        config: RunConfig,
        summary: dict[str, Any],
        results: list[EvalResult],
        instances: dict[str, TaskInstance],
    ) -> Path: ...


class ModelReportWriter:
    """Default report writer for the eval-runs/<model>/<benchmark> layout."""

    def write(
        self,
        *,
        config: RunConfig,
        summary: dict[str, Any],
        results: list[EvalResult],
        instances: dict[str, TaskInstance],
    ) -> Path:
        return write_model_report(
            report_dir=config.report_dir,
            model=config.model,
            benchmark=config.report_name,
            run_id=config.run_id,
            config=config.to_dict(),
            summary=summary,
            results=results,
            instances=instances,
        )


def write_model_report(
    *,
    report_dir: Path,
    model: str,
    benchmark: str,
    run_id: str,
    config: dict[str, Any],
    summary: dict[str, Any],
    results: list[EvalResult],
    instances: dict[str, TaskInstance],
) -> Path:
    """Write per-model summary, Markdown, and problem/solution JSON files."""

    model_dir = report_dir / _slug(model)
    benchmark_dir = model_dir / _slug(benchmark)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    write_json(benchmark_dir / "config.json", config)
    write_json(benchmark_dir / "summary.json", summary)
    grouped = _group_results(results)
    for task_id, rows in sorted(grouped.items()):
        instance = instances.get(task_id)
        write_json(
            benchmark_dir / f"{_slug(task_id)}.json",
            _problem_solution_record(
                model=model,
                benchmark=benchmark,
                run_id=run_id,
                instance=instance,
                rows=rows,
            ),
        )

    report_path = benchmark_dir / "report.md"
    report_path.write_text(
        _render_markdown(
            model=model,
            benchmark=benchmark,
            run_id=run_id,
            summary=summary,
            grouped=grouped,
        ),
        encoding="utf-8",
    )
    _write_model_index(model_dir)
    return benchmark_dir


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _group_results(results: list[EvalResult]) -> dict[str, list[EvalResult]]:
    grouped: dict[str, list[EvalResult]] = defaultdict(list)
    for row in results:
        grouped[row.task_id].append(row)
    return grouped


def _problem_solution_record(
    *,
    model: str,
    benchmark: str,
    run_id: str,
    instance: TaskInstance | None,
    rows: list[EvalResult],
) -> dict[str, Any]:
    first = rows[0]
    return {
        "model": model,
        "benchmark": benchmark,
        "run_id": run_id,
        "task_name": first.task_name,
        "task_id": first.task_id,
        "seed": first.seed,
        "prompt": instance.prompt if instance else None,
        "inputs": instance.inputs if instance else {},
        "expected": instance.expected if instance else first.expected,
        "metadata": instance.metadata if instance else {},
        "solutions": [
            {
                "runner": row.runner,
                "answer": row.answer,
                "correct": row.correct,
                "score": row.score,
                "error": row.error,
                "input_tokens": row.input_tokens,
                "output_tokens": row.output_tokens,
                "total_tokens": row.total_tokens,
                "time_seconds": row.time_seconds,
                "iterations": row.iterations,
                "graph": row.graph,
                "artifacts": row.artifacts,
                "metadata": row.metadata,
            }
            for row in sorted(rows, key=lambda row: row.runner)
        ],
    }


def _render_markdown(
    *,
    model: str,
    benchmark: str,
    run_id: str,
    summary: dict[str, Any],
    grouped: dict[str, list[EvalResult]],
) -> str:
    lines = [
        f"# {model} - {benchmark}",
        "",
        f"- Run ID: `{run_id}`",
        f"- Result rows: `{summary.get('count', 0)}`",
        f"- Questions: `{len(grouped)}`",
        "",
        "## Per-Task Accuracy",
        "",
    ]
    runners = list(summary.get("by_runner", {}).keys())
    if runners:
        lines.append("| Task | " + " | ".join(_runner_label(runner) for runner in runners) + " |")
        lines.append("|---|" + "---:|" * len(runners))
        matrix = summary.get("accuracy_by_task", {})
        for task_name, by_runner in sorted(matrix.items()):
            cells = []
            for runner in runners:
                values = by_runner.get(runner, {})
                if values:
                    cells.append(
                        "{accuracy:.1%} ({count})".format(
                            accuracy=values.get("accuracy", 0.0),
                            count=values.get("count", 0),
                        )
                    )
                else:
                    cells.append("—")
            lines.append(f"| {task_name} | " + " | ".join(cells) + " |")
    else:
        lines.append("_No completed rows yet._")
    lines.extend(
        [
            "",
        "## By Runner",
        "",
        "| Runner | Accuracy | Score | Avg Tokens | Avg Latency | Errors | Tasks Won |",
        "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    wins = summary.get("tasks_won", {}).get("counts", {})
    for runner, values in sorted(summary.get("by_runner", {}).items()):
        lines.append(
            "| {runner} | {accuracy:.1%} | {score:.3f} | {tokens:.0f} | {latency:.2f}s | {errors} | {wins} |".format(
                runner=_runner_label(runner),
                accuracy=values.get("accuracy", 0.0),
                score=values.get("score", 0.0),
                tokens=values.get("total_tokens", 0.0),
                latency=values.get("time_seconds", 0.0),
                errors=values.get("errors", 0),
                wins=wins.get(runner, 0),
            )
        )
    lines.extend(["", "## By Task And Runner", "", "| Task | Runner | Accuracy | Score | N | Errors |", "|---|---|---:|---:|---:|---:|"])
    for key, values in sorted(summary.get("by_runner_task", {}).items()):
        runner, _, task = key.partition("/")
        lines.append(
            "| {task} | {runner} | {accuracy:.1%} | {score:.3f} | {count} | {errors} |".format(
                task=task,
                runner=_runner_label(runner),
                accuracy=values.get("accuracy", 0.0),
                score=values.get("score", 0.0),
                count=values.get("count", 0),
                errors=values.get("errors", 0),
            )
        )
    lines.extend(["", "## Problem/Solution Files", ""])
    for task_id, rows in sorted(grouped.items()):
        status = ", ".join(
            f"{_runner_label(row.runner)}:{'ok' if row.correct else 'fail'}" for row in sorted(rows, key=lambda row: row.runner)
        )
        lines.append(f"- [`{_slug(task_id)}.json`]({_slug(task_id)}.json) - {status}")
    lines.append("")
    return "\n".join(lines)


def _runner_label(runner: str) -> str:
    return {
        "rflow": "RLMFlow",
        "official": "Official RLM",
        "vanilla": "Vanilla",
        "minrlm-reasoning": "minRLM",
    }.get(runner, runner)


def _write_model_index(model_dir: Path) -> None:
    reports = sorted(model_dir.glob("*/report.md"))
    lines = [f"# {model_dir.name} Eval Reports", ""]
    for report in reports:
        lines.append(f"- [{report.parent.name}]({report.parent.name}/report.md)")
    lines.append("")
    (model_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def _slug(value: str, *, max_length: int = 120) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.replace("/", "_").replace(":", "_"))
    slug = slug.strip("-") or "unnamed"
    if len(slug) <= max_length:
        return slug
    digest = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:8]
    return f"{slug[: max_length - 9].rstrip('-')}-{digest}"


__all__ = ["ModelReportWriter", "ReportWriter", "write_model_report"]
