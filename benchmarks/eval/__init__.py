"""Shared evaluation harness for recursive-flow benchmarks.

Architecture adapted from avilum/minrlm's eval suite:
https://github.com/avilum/minrlm/tree/master/eval
"""

from benchmarks.eval.core import (
    EvalResult,
    RunResult,
    Score,
    TaskInstance,
)
from benchmarks.eval.runners import get_runner, list_runners, register_runner
from benchmarks.eval.tasks import get_task, list_tasks, register_task

__all__ = [
    "EvalResult",
    "RunResult",
    "Score",
    "TaskInstance",
    "get_runner",
    "get_task",
    "list_runners",
    "list_tasks",
    "register_runner",
    "register_task",
]
