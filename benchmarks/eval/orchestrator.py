"""Run orchestration for the shared benchmark harness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.eval.clients import ClientFactory
from benchmarks.eval.config import RunConfig
from benchmarks.eval.core import EvalResult, RunResult, Score, TaskInstance
from benchmarks.eval.logging import MetricsLogger
from benchmarks.eval.metrics import MetricsAggregator
from benchmarks.eval.progress import ProgressReporter, TqdmProgress
from benchmarks.eval.reporting import ReportWriter
from benchmarks.eval.runners import Runner
from benchmarks.eval.tasks import TaskRegistry
from benchmarks.eval.store import RunStore


@dataclass(frozen=True)
class EvalRun:
    results: list[EvalResult]
    summary: dict[str, Any]
    root: Path
    report_path: Path


class EvalOrchestrator:
    """Coordinates task generation, runner execution, scoring, and side effects."""

    def __init__(
        self,
        *,
        config: RunConfig,
        task_registry: TaskRegistry,
        runner_registry: dict[str, type[Runner]],
        client_factory: ClientFactory,
        store: RunStore,
        metrics: MetricsAggregator,
        reporter: ReportWriter,
        logger: MetricsLogger,
        progress: ProgressReporter | None = None,
    ) -> None:
        self.config = config
        self.task_registry = task_registry
        self.runner_registry = runner_registry
        self.client_factory = client_factory
        self.store = store
        self.metrics = metrics
        self.reporter = reporter
        self.logger = logger
        self.progress = progress or TqdmProgress()

    def run(self) -> EvalRun:
        self.store.initialize()
        client = self.client_factory.create(self.config.provider, self.config.model)
        results = self.store.load_results() if self.config.resume else []
        instances: dict[str, TaskInstance] = {}

        try:
            task_cache = {}
            for task_name in self.progress.task_loader(
                self.config.tasks,
                total=len(self.config.tasks),
            ):
                task_cache[task_name] = self.task_registry.make(
                    task_name,
                    **self._task_params_for(task_name),
                )
            runner_cache = {
                runner_name: self.runner_registry[runner_name]()
                for runner_name in self.config.runners
            }
            jobs = [
                (task_name, runner_name, seed)
                for task_name in self.config.tasks
                for seed in self.config.seeds
                for runner_name in self.config.runners
            ]
            for task_name, runner_name, seed in self.progress.eval_jobs(
                jobs,
                total=len(jobs),
            ):
                task = task_cache[task_name]
                if self.config.resume and self.store.has_result(task_name, runner_name, seed):
                    instance = task.generate(seed)
                    instances.setdefault(instance.task_id, instance)
                    continue
                instance = task.generate(seed)
                instances.setdefault(instance.task_id, instance)
                runner = runner_cache[runner_name]
                artifact_dir = self.store.artifact_dir(
                    runner_name,
                    task_name,
                    instance.task_id,
                )
                run_result = runner.run(
                    instance,
                    client=client,
                    model=self.config.model,
                    out_dir=artifact_dir,
                    max_iters=self.config.max_iters,
                    max_depth=self.config.max_depth,
                    live_save=self.config.live_save,
                )
                score = task.score(run_result.answer, instance.expected, instance.metadata)
                row = build_result(
                    run_id=self.config.run_id,
                    task_name=task_name,
                    runner_name=runner_name,
                    model=self.config.model,
                    instance=instance,
                    run_result=run_result,
                    score=score,
                )
                result_path = self.store.job_result_path(
                    runner_name,
                    task_name,
                    instance.task_id,
                )
                row.artifacts["result_path"] = str(result_path)
                self.store.write_job_result(row)
                self.store.append_result(row)
                results.append(row)
                self.logger.log_result(row)
                summary = self.metrics.summarize(results)
                self.store.write_json("summary.json", summary)
                self.store.write_json(
                    "task_accuracy.json",
                    summary.get("accuracy_by_task", {}),
                )

            summary = self.metrics.summarize(results)
            self.store.write_json("summary.json", summary)
            self.store.write_json("task_accuracy.json", summary.get("accuracy_by_task", {}))
            report_path = self.reporter.write(
                config=self.config,
                summary=summary,
                results=results,
                instances=instances,
            )
            self.logger.log_summary(summary)
            return EvalRun(
                results=results,
                summary=summary,
                root=self.store.root,
                report_path=report_path,
            )
        finally:
            self.logger.finish()

    def _task_params_for(self, task_name: str) -> dict[str, Any]:
        if task_name.startswith("official_"):
            return {**self.config.task_params, **self.config.official_params}
        return dict(self.config.task_params)


def build_result(
    *,
    run_id: str,
    task_name: str,
    runner_name: str,
    model: str,
    instance: TaskInstance,
    run_result: RunResult,
    score: Score,
) -> EvalResult:
    metadata = {
        **instance.metadata,
        "score_details": score.details,
        "runner": run_result.metadata,
    }
    graph = run_result.metadata.get("graph", {})
    return EvalResult(
        run_id=run_id,
        task_name=task_name,
        task_id=instance.task_id,
        seed=instance.seed,
        runner=runner_name,
        model=model,
        correct=score.correct,
        score=score.value,
        answer=run_result.answer,
        expected=instance.expected,
        input_tokens=run_result.input_tokens,
        output_tokens=run_result.output_tokens,
        total_tokens=run_result.total_tokens,
        time_seconds=run_result.time_seconds,
        iterations=run_result.iterations,
        error=run_result.error,
        graph=graph,
        artifacts={
            "graph_path": run_result.graph_path,
            "trace_path": run_result.trace_path,
        },
        metadata=metadata,
    )


__all__ = ["EvalOrchestrator", "EvalRun", "build_result"]
