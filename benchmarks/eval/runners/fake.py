"""Deterministic no-LLM runner used to smoke-test the harness."""

from __future__ import annotations

import time
from pathlib import Path

from rflow.clients import LLMClient

from benchmarks.eval.core import RunResult, TaskInstance
from benchmarks.eval.runners import register_runner


@register_runner("fake")
class FakeRunner:
    """Return the expected answer directly; useful for CI and CLI validation."""

    name = "fake"

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
    ) -> RunResult:
        del client, model, out_dir, max_iters, max_depth, live_save
        start = time.perf_counter()
        return RunResult(
            answer=str(instance.expected),
            time_seconds=time.perf_counter() - start,
            iterations=0,
        )
