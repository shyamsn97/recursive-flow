"""Runner registry for benchmark execution backends.

Inspired by avilum/minrlm's eval runner registry:
https://github.com/avilum/minrlm/tree/master/eval
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from rflow.clients import LLMClient

from benchmarks.eval.core import RunResult, TaskInstance


class Runner(Protocol):
    name: str

    def run(
        self,
        instance: TaskInstance,
        *,
        client: LLMClient,
        model: str,
        out_dir: Path,
        max_iters: int,
        max_depth: int,
        live_save: bool,
    ) -> RunResult: ...


RUNNER_REGISTRY: dict[str, type[Runner]] = {}


def register_runner(name: str):
    """Register a runner class under a CLI name."""

    def decorator(cls: type[Runner]) -> type[Runner]:
        RUNNER_REGISTRY[name] = cls
        cls.name = name
        return cls

    return decorator


def get_runner(name: str) -> Runner:
    if name not in RUNNER_REGISTRY:
        available = ", ".join(sorted(RUNNER_REGISTRY))
        raise ValueError(f"unknown runner {name!r}. available: {available}")
    return RUNNER_REGISTRY[name]()


def list_runners() -> list[str]:
    return sorted(RUNNER_REGISTRY)


# Import built-ins so decorators run.
from benchmarks.eval.runners import fake, official, rflow_runner, vanilla  # noqa: E402,F401

__all__ = ["RUNNER_REGISTRY", "Runner", "get_runner", "list_runners", "register_runner"]
