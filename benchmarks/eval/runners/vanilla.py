"""Single-call LLM baseline runner."""

from __future__ import annotations

import time
from pathlib import Path

from rflow.clients import LLMClient

from benchmarks.eval.core import RunResult, TaskInstance
from benchmarks.eval.runners import register_runner


@register_runner("vanilla")
class VanillaRunner:
    """No REPL, no delegation: one chat completion."""

    name = "vanilla"

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
        del max_iters, max_depth, live_save
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the benchmark task directly. Return only the final "
                    "answer value, with no explanation."
                ),
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        instance.prompt,
                        "INPUTS:",
                        *(
                            f"INPUT {key}:\n{value}"
                            for key, value in sorted(instance.inputs.items())
                        ),
                    ]
                ),
            },
        ]
        start = time.perf_counter()
        try:
            answer, usage = client.completion(messages)
            error = None
        except Exception as exc:  # benchmark rows should record failures
            answer = ""
            usage = client.last_usage
            error = f"{type(exc).__name__}: {exc}"
        return RunResult(
            answer=answer,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            time_seconds=time.perf_counter() - start,
            iterations=1,
            error=error,
            metadata={"model": model},
        )
