"""recursive-flow benchmark runner."""

from __future__ import annotations

import time
from pathlib import Path

from rflow import Flow, LocalRuntime
from rflow.clients import LLMClient

from benchmarks.eval.core import RunResult, TaskInstance
from benchmarks.eval.metrics import graph_metrics
from benchmarks.eval.runners import register_runner


@register_runner("rflow")
class RFlowRunner:
    """Drive `Flow` with the functional step API and save each checkpoint."""

    name = "rflow"

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
        graph_dir = out_dir / "graph"
        work_dir = out_dir / "workdir"
        work_dir.mkdir(parents=True, exist_ok=True)
        flow = Flow(
            client,
            max_iters=max_iters,
            max_depth=max_depth,
            runtime=LocalRuntime(working_directory=work_dir),
        )
        start = time.perf_counter()
        graph = None
        error = None
        steps = 0
        try:
            graph = flow.start(instance.prompt, instance.inputs)
            if live_save:
                graph.save(graph_dir)
            max_steps = max(200, max_iters * max(1, max_depth + 1) * 25)
            while not graph.finished:
                graph = flow.step(graph)
                steps += 1
                if live_save:
                    graph.save(graph_dir)
                if steps >= max_steps:
                    raise RuntimeError(f"run exceeded step cap ({max_steps})")
            graph.save(graph_dir)
            answer = graph.result()
            input_tokens, output_tokens = graph.tokens()
        except Exception as exc:  # benchmark rows should record failures
            answer = graph.result() if graph is not None else ""
            input_tokens, output_tokens = graph.tokens() if graph is not None else (0, 0)
            error = f"{type(exc).__name__}: {exc}"
            if graph is not None:
                graph.save(graph_dir)
        finally:
            flow.close()
        return RunResult(
            answer=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            time_seconds=time.perf_counter() - start,
            iterations=steps,
            error=error,
            graph_path=str(graph_dir),
            metadata={
                "graph": graph_metrics(graph),
                "model": model,
                "live_save": live_save,
            },
        )
