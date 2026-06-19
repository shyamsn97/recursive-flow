"""CLI and orchestration for the clean benchmark harness."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.eval import DATASETS, LOGGERS, MODELS, RUNNERS
from benchmarks.eval.loggers import MultiLogger
from benchmarks.eval.loggers.jsonl import load_rows
from benchmarks.eval.metrics import summarize
from benchmarks.eval.types import (
    ComponentSpec,
    ModelSpec,
    Row,
    RunContext,
    SuiteConfig,
)

# Import built-ins so decorators register.
from benchmarks.eval import loggers as _loggers  # noqa: F401,E402
from benchmarks.eval import models as _models  # noqa: F401,E402
from benchmarks.eval import runners as _runners  # noqa: F401,E402
from benchmarks.eval import tasks as _tasks  # noqa: F401,E402


def run_suite(config: SuiteConfig) -> list[Row]:
    config.root.mkdir(parents=True, exist_ok=True)
    datasets = [DATASETS.make(spec.name, **spec.params) for spec in config.datasets]
    runners = [RUNNERS.make(spec.name, **spec.params) for spec in config.runners]
    model = MODELS.make(config.model.provider, name=config.model.name, **config.model.params)
    logger = build_logger(config)
    rows = load_rows(config.root / "rows.jsonl") if config.resume else []
    seen = {
        (row.dataset, row.example_id, row.runner, row.model, row.seed)
        for row in rows
    }

    logger.start(config.to_dict())
    try:
        for dataset in datasets:
            for seed in config.seeds:
                examples = dataset.examples(
                    split=config.split,
                    limit=config.limit,
                    seed=seed,
                )
                for example in examples:
                    for runner in runners:
                        key = (dataset.name, example.id, runner.name, model.name, seed)
                        if key in seen:
                            continue
                        artifact_dir = (
                            config.root
                            / "artifacts"
                            / dataset.name
                            / example.id
                            / runner.name
                        )
                        ctx = RunContext(
                            run_id=config.run_id,
                            root=config.root,
                            artifact_dir=artifact_dir,
                        )
                        logger.example_start(example, runner=runner.name, model=model.name)
                        prediction = runner.run(example, model, ctx)
                        score = dataset.score(example, prediction)
                        row = Row(
                            run_id=config.run_id,
                            dataset=dataset.name,
                            example_id=example.id,
                            runner=runner.name,
                            model=model.name,
                            seed=seed,
                            prediction=prediction,
                            score=score,
                            metadata=example.metadata,
                        )
                        rows.append(row)
                        seen.add(key)
                        logger.row(row)
        logger.summary(rows)
        return rows
    finally:
        logger.finish()


def build_logger(config: SuiteConfig) -> MultiLogger:
    instances = []
    for spec in config.loggers:
        params = dict(spec.params)
        if spec.name in {"jsonl", "report"}:
            params.setdefault("root", config.root)
        instances.append(LOGGERS.make(spec.name, **params))
    return MultiLogger(instances)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run recursive-flow benchmarks.")
    parser.add_argument("--dataset", "--datasets", nargs="+", default=["oolong"])
    parser.add_argument("--runner", "--runners", nargs="+", default=["rflow-local"])
    parser.add_argument("--model", default="openai:gpt-5-mini")
    parser.add_argument("--logger", "--loggers", nargs="+", default=["jsonl", "console", "report"])
    parser.add_argument("--seeds", default="0:5")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/eval/runs"))
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset-param", action="append", default=[])
    parser.add_argument("--runner-param", action="append", default=[])
    parser.add_argument("--model-param", action="append", default=[])
    parser.add_argument("--logger-param", action="append", default=[])
    parser.add_argument("--wandb", action="store_true", help="Add the W&B logger.")
    parser.add_argument("--wandb-project", default="rflow-eval")
    parser.add_argument("--wandb-entity")
    return parser


def config_from_args(args: argparse.Namespace) -> SuiteConfig:
    dataset_names = DATASETS.expand(_flatten(args.dataset))
    runner_names = RUNNERS.expand(_flatten(args.runner))
    logger_names = LOGGERS.expand(_flatten(args.logger))
    if args.wandb and "wandb" not in logger_names:
        logger_names.append("wandb")
    model_spec = parse_model_spec(args.model, parse_params(args.model_param).get("_", {}))
    dataset_params = parse_params(args.dataset_param)
    runner_params = parse_params(args.runner_param)
    logger_params = parse_params(args.logger_param)
    if args.wandb:
        logger_params.setdefault("wandb", {})
        logger_params["wandb"].setdefault("project", args.wandb_project)
        if args.wandb_entity:
            logger_params["wandb"]["entity"] = args.wandb_entity
    config = SuiteConfig(
        run_id=args.run_id
        or make_run_id(
            datasets=dataset_names,
            runners=runner_names,
            model=model_spec.label,
        ),
        datasets=[
            ComponentSpec(name=name, params=dataset_params.get(name, {}))
            for name in dataset_names
        ],
        runners=[
            ComponentSpec(name=name, params=runner_params.get(name, {}))
            for name in runner_names
        ],
        model=model_spec,
        loggers=[
            ComponentSpec(name=name, params=logger_params.get(name, {}))
            for name in logger_names
        ],
        seeds=parse_seed_spec(args.seeds),
        split=args.split,
        limit=args.limit,
        output_root=args.out_dir,
        resume=args.resume,
    )
    return config


def parse_model_spec(value: str, params: dict[str, Any] | None = None) -> ModelSpec:
    params = params or {}
    if ":" not in value:
        provider = value
        name = value
    else:
        provider, name = value.split(":", 1)
    return ModelSpec(provider=provider, name=name, params=params)


def parse_params(values: list[str]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"parameter must be key=value: {value}")
        raw_key, raw_value = value.split("=", 1)
        if "." in raw_key:
            scope, key = raw_key.split(".", 1)
        else:
            scope, key = "_", raw_key
        grouped.setdefault(scope, {})[key] = _parse_value(raw_value)
    return grouped


def parse_seed_spec(spec: str) -> list[int]:
    if ":" in spec:
        parts = [int(part) for part in spec.split(":")]
        if len(parts) == 2:
            start, stop = parts
            step = 1
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise ValueError(f"invalid seed range: {spec}")
        return list(range(start, stop, step))
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def make_run_id(*, datasets: list[str], runners: list[str], model: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text = f"{stamp}_{_slug(model)}_{_compact(datasets)}_{_compact(runners)}"
    return text[:180].rstrip("-")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    rows = run_suite(config)
    summary = summarize(rows)
    print(json.dumps(summary.get("overall", {}), indent=2, sort_keys=True))
    print(f"Saved results to {config.root}")
    return 0


def _flatten(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out


def _parse_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value).strip("-")


def _compact(values: list[str], *, max_length: int = 80) -> str:
    text = "-".join(_slug(value) for value in values)
    if len(text) <= max_length:
        return text
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{text[: max_length - 9].rstrip('-')}-{digest}"


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_logger",
    "build_parser",
    "config_from_args",
    "main",
    "make_run_id",
    "parse_model_spec",
    "parse_params",
    "parse_seed_spec",
    "run_suite",
]
