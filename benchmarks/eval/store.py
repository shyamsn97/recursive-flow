"""Filesystem storage for benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.eval.config import RunConfig
from benchmarks.eval.core import EvalResult


class RunStore:
    """Owns the run directory, config, result rows, and artifact paths."""

    def __init__(self, root: Path, *, config: RunConfig) -> None:
        self.root = root
        self.config = config
        self.config_path = root / "config.json"
        self.results_path = root / "results.jsonl"
        self._result_keys: set[tuple[str, str, int]] | None = None

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.config_path, self.config.to_dict())
        if not self.config.resume:
            self.results_path.write_text("", encoding="utf-8")
        self._result_keys = None

    def append_result(self, row: EvalResult) -> None:
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with self.results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
            handle.flush()
        if self._result_keys is not None:
            self._result_keys.add(self._key(row.task_name, row.runner, row.seed))

    def write_job_result(self, row: EvalResult) -> Path:
        path = self.job_result_path(row.runner, row.task_name, row.task_id)
        write_json(path, row.to_dict())
        return path

    def job_result_path(self, runner: str, task_name: str, task_id: str) -> Path:
        return self.artifact_dir(runner, task_name, task_id) / "result.json"

    def load_results(self) -> list[EvalResult]:
        if not self.results_path.exists():
            return []
        rows = []
        for line in self.results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(EvalResult.from_dict(json.loads(line)))
        return rows

    def has_result(self, task_name: str, runner: str, seed: int) -> bool:
        if self._result_keys is None:
            self._result_keys = {
                self._key(row.task_name, row.runner, row.seed)
                for row in self.load_results()
            }
        return self._key(task_name, runner, seed) in self._result_keys

    def artifact_dir(self, runner: str, task_name: str, task_id: str) -> Path:
        path = self.root / "artifacts" / runner / task_name / task_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, name: str, data: Any) -> Path:
        path = self.root / name
        write_json(path, data)
        return path

    @staticmethod
    def _key(task_name: str, runner: str, seed: int) -> tuple[str, str, int]:
        return (task_name, runner, seed)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["RunStore", "write_json"]
