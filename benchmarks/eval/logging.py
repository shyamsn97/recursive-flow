"""Metrics logging interfaces for benchmark runs."""

from __future__ import annotations

from typing import Any, Protocol

from benchmarks.eval.core import EvalResult
from benchmarks.eval.wandb_logging import WandbLogger


class MetricsLogger(Protocol):
    def log_result(self, row: EvalResult) -> None: ...

    def log_summary(self, summary: dict[str, Any]) -> None: ...

    def finish(self) -> None: ...


class NullLogger:
    """No-op logger used when integrations are disabled."""

    def log_result(self, row: EvalResult) -> None:
        return None

    def log_summary(self, summary: dict[str, Any]) -> None:
        return None

    def finish(self) -> None:
        return None


__all__ = ["MetricsLogger", "NullLogger", "WandbLogger"]
