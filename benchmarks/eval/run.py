"""CLI for running the shared benchmark suite.

The task/runner sweep shape is adapted from avilum/minrlm's eval suite:
https://github.com/avilum/minrlm/tree/master/eval

Example:
    python -m benchmarks.eval.run --tasks sniah --runners fake rflow --provider fake
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.eval.clients import ClientFactory
from benchmarks.eval.config import (
    RunConfig,
    expand_runners,
    expand_tasks,
    make_run_id,
    parse_seed_spec,
    parse_task_params,
)
from benchmarks.eval.core import EvalResult
from benchmarks.eval.logging import NullLogger, WandbLogger
from benchmarks.eval.metrics import MetricsAggregator
from benchmarks.eval.orchestrator import EvalOrchestrator
from benchmarks.eval.reporting import ModelReportWriter
from benchmarks.eval.runners import RUNNER_REGISTRY, list_runners
from benchmarks.eval.store import RunStore
from benchmarks.eval.tasks import TASK_REGISTRY, list_tasks


def run_eval(args: argparse.Namespace) -> tuple[list[EvalResult], dict[str, Any], Path]:
    config = RunConfig.from_namespace(args)
    wandb_config = config.wandb
    logger = (
        WandbLogger(
            enabled=True,
            project=wandb_config.project,
            entity=wandb_config.entity,
            run_id=config.run_id,
            config=config.to_dict(),
        )
        if wandb_config
        else NullLogger()
    )
    eval_run = EvalOrchestrator(
        config=config,
        task_registry=TASK_REGISTRY,
        runner_registry=RUNNER_REGISTRY,
        client_factory=ClientFactory(),
        store=RunStore(config.root, config=config),
        metrics=MetricsAggregator(),
        reporter=ModelReportWriter(),
        logger=logger,
    ).run()
    return eval_run.results, eval_run.summary, eval_run.root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run recursive-flow eval benchmarks.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["sniah"],
        help=(
            "Tasks to run. Accepts spaces or commas. Use `official` for the "
            f"RLM-Bench official suite, `all` for everything. Available: {', '.join(list_tasks())}"
        ),
    )
    parser.add_argument(
        "--runners",
        nargs="+",
        default=["rflow"],
        help=(
            "Runners to compare. Accepts spaces or commas. Use `all` for every "
            f"runner. Available: {', '.join(list_runners())}"
        ),
    )
    parser.add_argument("--seeds", default="0:5")
    parser.add_argument(
        "--task-param",
        action="append",
        default=[],
        help="Task generation parameter as key=value; value may be JSON.",
    )
    parser.add_argument("--provider", default="openai", choices=["fake", "openai", "anthropic"])
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-iters", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/eval/runs"))
    parser.add_argument("--report-dir", type=Path, default=Path("eval-runs"))
    parser.add_argument("--report-name")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip task/runner/seed rows already present in results.jsonl.",
    )
    parser.add_argument("--live-save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--official-data-dir", default="evals/data")
    parser.add_argument("--official-split")
    parser.add_argument("--official-max-samples", type=int)
    parser.add_argument("--official-max-context-chars", type=int)
    parser.add_argument("--official-max-context-tokens", type=int)
    parser.add_argument("--browsecomp-max-docs", type=int)
    parser.add_argument("--wandb", action="store_true", help="Log per-row metrics to W&B.")
    parser.add_argument("--wandb-project", default="rflow-eval")
    parser.add_argument("--wandb-entity")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _results, summary, run_root = run_eval(args)
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))
    print(f"Saved results to {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_parser",
    "expand_runners",
    "expand_tasks",
    "main",
    "make_run_id",
    "parse_seed_spec",
    "parse_task_params",
    "run_eval",
]
