"""Dataclasses shared by the benchmark tasks, runners, and metrics.

The normalized task/result schema follows the shape of avilum/minrlm's eval
suite, adapted for rflow graph artifacts:
https://github.com/avilum/minrlm/tree/master/eval
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class TaskInstance:
    """One deterministic task sample."""

    task_name: str
    task_id: str
    seed: int
    prompt: str
    inputs: dict[str, str] = field(default_factory=dict)
    expected: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Score:
    """Task-level score for one answer."""

    correct: bool
    value: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Raw result from a runner before task scoring."""

    answer: str
    input_tokens: int = 0
    output_tokens: int = 0
    time_seconds: float = 0.0
    iterations: int = 0
    error: str | None = None
    graph_path: str | None = None
    trace_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class EvalResult:
    """Normalized result row saved to results.jsonl."""

    run_id: str
    task_name: str
    task_id: str
    seed: int
    runner: str
    model: str
    correct: bool
    score: float
    answer: str
    expected: Any
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    time_seconds: float = 0.0
    iterations: int = 0
    error: str | None = None
    graph: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str | None] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalResult":
        return cls(**data)
