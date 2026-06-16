"""Task registry for the shared benchmark harness.

Inspired by avilum/minrlm's eval task registry:
https://github.com/avilum/minrlm/tree/master/eval
"""

from __future__ import annotations

from benchmarks.eval.tasks.registry import TASK_REGISTRY, Task, TaskRegistry, TaskSpec

register_task = TASK_REGISTRY.register
get_task = TASK_REGISTRY.make
list_tasks = TASK_REGISTRY.names
expand_tasks = TASK_REGISTRY.expand

OFFICIAL_TASKS = [
    "official_sniah",
    "official_oolong",
    "official_longbench_v2",
    "official_codeqa",
    "official_repoqa",
    "official_browsecomp",
    "official_gdpval",
    "official_aime_2025",
    "official_gpqa_diamond",
    "official_mmlu_pro",
    "official_ifeval",
    "official_livecodebench",
    "official_sudoku_extreme",
]


# Import built-ins so decorators run.
from benchmarks.eval.tasks import code as _code  # noqa: E402,F401
from benchmarks.eval.tasks import long_context as _long_context  # noqa: E402,F401
from benchmarks.eval.tasks import reasoning as _reasoning  # noqa: E402,F401
from benchmarks.eval.tasks import synthetic as _synthetic  # noqa: E402,F401
from benchmarks.eval.tasks import work as _work  # noqa: E402,F401

TASK_REGISTRY.alias("official", lambda _registry: OFFICIAL_TASKS)
TASK_REGISTRY.alias("all", lambda registry: registry.names())

__all__ = [
    "OFFICIAL_TASKS",
    "TASK_REGISTRY",
    "Task",
    "TaskRegistry",
    "TaskSpec",
    "expand_tasks",
    "get_task",
    "list_tasks",
    "register_task",
]
