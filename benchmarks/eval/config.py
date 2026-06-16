"""Run configuration helpers for the shared benchmark harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.eval.runners import list_runners
from benchmarks.eval.tasks import expand_tasks as expand_task_ids


@dataclass(frozen=True)
class WandbConfig:
    """Configuration for optional Weights & Biases logging."""

    project: str
    entity: str | None


@dataclass(frozen=True)
class RunConfig:
    """Single source of truth for one benchmark sweep."""

    run_id: str
    tasks: tuple[str, ...]
    runners: tuple[str, ...]
    seeds: tuple[int, ...]
    provider: str
    model: str
    max_iters: int
    max_depth: int
    out_dir: Path
    report_dir: Path
    report_name: str
    live_save: bool
    task_params: dict[str, Any]
    official_params: dict[str, Any]
    wandb: WandbConfig | None = None
    resume: bool = False

    @property
    def root(self) -> Path:
        return self.out_dir / self.run_id

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "RunConfig":
        requested_tasks = _flatten_names(args.tasks)
        task_params = parse_task_params(args.task_param)
        official_params = {
            key: value
            for key, value in (
                ("data_dir", args.official_data_dir),
                ("split", args.official_split),
                ("max_samples", args.official_max_samples),
                ("max_context_chars", args.official_max_context_chars),
                ("max_context_tokens", args.official_max_context_tokens),
                ("max_docs", args.browsecomp_max_docs),
            )
            if value is not None
        }
        return cls(
            run_id=args.run_id or make_run_id(args),
            tasks=tuple(expand_tasks(args.tasks)),
            runners=tuple(expand_runners(args.runners)),
            seeds=tuple(parse_seed_spec(args.seeds)),
            provider=args.provider,
            model=args.model,
            max_iters=args.max_iters,
            max_depth=args.max_depth,
            out_dir=args.out_dir,
            report_dir=args.report_dir,
            report_name=args.report_name or _compact_slug(requested_tasks, max_length=80),
            live_save=args.live_save,
            task_params=task_params,
            official_params=official_params,
            wandb=(
                WandbConfig(project=args.wandb_project, entity=args.wandb_entity)
                if args.wandb
                else None
            ),
            resume=args.resume,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "tasks": list(self.tasks),
            "runners": list(self.runners),
            "seeds": list(self.seeds),
            "provider": self.provider,
            "model": self.model,
            "max_iters": self.max_iters,
            "max_depth": self.max_depth,
            "task_param": self.task_params,
            "official_task_param": self.official_params,
            "live_save": self.live_save,
            "report_dir": str(self.report_dir),
            "report_name": self.report_name,
            "resume": self.resume,
        }


def parse_seed_spec(spec: str) -> list[int]:
    """Parse `0:10`, `0:10:2`, or `1,5,9`."""

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


def parse_task_params(values: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"task parameter must be key=value: {value}")
        key, raw = value.split("=", 1)
        try:
            params[key] = json.loads(raw)
        except json.JSONDecodeError:
            params[key] = raw
    return params


def expand_tasks(values: list[str]) -> list[str]:
    return expand_task_ids(values)


def expand_runners(values: list[str]) -> list[str]:
    names = _flatten_names(values)
    expanded = list_runners() if "all" in names else names
    available = set(list_runners())
    unknown = [name for name in expanded if name not in available]
    if unknown:
        raise ValueError(
            f"unknown runner(s): {', '.join(unknown)}. "
            f"available: {', '.join(sorted(available))}"
        )
    return list(dict.fromkeys(expanded))


def make_run_id(args: argparse.Namespace) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runners = _compact_slug(_flatten_names(args.runners))
    tasks = _compact_slug(_flatten_names(args.tasks))
    model = _slug(args.model)
    return _compact_run_id(f"{stamp}_{model}_{tasks}_{runners}")


def _flatten_names(values: list[str]) -> list[str]:
    names: list[str] = []
    for value in values:
        names.extend(part.strip() for part in value.split(",") if part.strip())
    return names


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.replace("/", "_").replace(":", "_"))
    return slug.strip("-") or "unnamed"


def _compact_slug(values: list[str], *, max_length: int = 80) -> str:
    text = "-".join(_slug(value) for value in values)
    if len(text) <= max_length:
        return text
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{text[: max_length - 9].rstrip('-')}-{digest}"


def _compact_run_id(run_id: str, *, max_length: int = 180) -> str:
    if len(run_id) <= max_length:
        return run_id
    digest = hashlib.sha1(run_id.encode("utf-8")).hexdigest()[:8]
    return f"{run_id[: max_length - 9].rstrip('-')}-{digest}"


__all__ = [
    "RunConfig",
    "WandbConfig",
    "expand_runners",
    "expand_tasks",
    "make_run_id",
    "parse_seed_spec",
    "parse_task_params",
]
