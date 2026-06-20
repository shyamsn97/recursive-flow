"""Execution pools for running step tasks in parallel.

A pool has a barrier method, ``execute(tasks) -> results``, and a dynamic
method, ``run_until_idle(tasks, refill) -> results``. In both cases *tasks* is a
list of ``(id, callable)`` pairs and *results* is a ``dict[str, Any]`` mapping
ids to return values.

Pass a pool to ``Flow(pool=...)``. A plain callable is wrapped in
``CallablePool`` automatically. The work-conserving ``run_until_idle`` is what
``eager_children`` scheduling rides on: as each task finishes it can enqueue
newly-runnable descendants without waiting for the whole wave.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

Task = tuple[str, Callable[[], Any]]
Refill = Callable[[str, Any, set[str]], list[Task]]


class Pool(ABC):
    """Base class for execution pools."""

    @abstractmethod
    def execute(self, tasks: list[Task]) -> dict[str, Any]:
        """Run callables in parallel, keyed by id."""

    def run_until_idle(self, tasks: list[Task], refill: Refill) -> dict[str, Any]:
        """Run tasks and let completions enqueue follow-up tasks.

        The base implementation runs in barrier batches so execute-only custom
        pools still work. Pools with native as-completed support override this
        for true work-conserving refill.
        """
        results: dict[str, Any] = {}
        pending = list(tasks)
        while pending:
            batch = self.execute(pending)
            pending = []
            for task_id, result in batch.items():
                results[task_id] = result
                pending.extend(refill(task_id, result, set()))
        return results


class SequentialPool(Pool):
    """Run everything one at a time — useful for testing and debugging."""

    def execute(self, tasks: list[Task]) -> dict[str, Any]:
        return {task_id: fn() for task_id, fn in tasks}


class ThreadPool(Pool):
    """Run steps concurrently in a long-lived ``ThreadPoolExecutor``."""

    def __init__(self, max_concurrency: int = 8) -> None:
        self.max_concurrency = max_concurrency
        self.executor = ThreadPoolExecutor(max_workers=max_concurrency)

    def execute(self, tasks: list[Task]) -> dict[str, Any]:
        if self.max_concurrency <= 1 or len(tasks) <= 1:
            return {task_id: fn() for task_id, fn in tasks}
        futures = {self.executor.submit(fn): task_id for task_id, fn in tasks}
        return {futures[f]: f.result() for f in as_completed(futures)}

    def run_until_idle(self, tasks: list[Task], refill: Refill) -> dict[str, Any]:
        if self.max_concurrency <= 1:
            return super().run_until_idle(tasks, refill)
        if len(tasks) <= 1:
            results: dict[str, Any] = {}
            pending = list(tasks)
            while len(pending) == 1:
                task_id, fn = pending.pop()
                result = fn()
                results[task_id] = result
                pending.extend(refill(task_id, result, set()))
            if pending:
                results.update(self.run_until_idle(pending, refill))
            return results
        futures = {self.executor.submit(fn): task_id for task_id, fn in tasks}
        results: dict[str, Any] = {}
        while futures:
            for future in as_completed(list(futures)):
                task_id = futures.pop(future)
                result = future.result()
                results[task_id] = result
                active_ids = set(futures.values())
                for new_id, fn in refill(task_id, result, active_ids):
                    futures[self.executor.submit(fn)] = new_id
                break
        return results

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)


class CallablePool(Pool):
    """Wrap a plain ``execute``-style function as a pool."""

    def __init__(self, fn: Callable[[list[Task]], dict[str, Any]]) -> None:
        self.fn = fn

    def execute(self, tasks: list[Task]) -> dict[str, Any]:
        return self.fn(tasks)


def create_pool(pool: Any, *, max_concurrency: int) -> Pool:
    """Resolve the ``Flow(pool=...)`` argument into a :class:`Pool`."""
    if pool is None:
        return ThreadPool(max_concurrency) if max_concurrency > 1 else SequentialPool()
    if isinstance(pool, Pool):
        return pool
    if callable(pool):
        return CallablePool(pool)
    raise TypeError("pool must be a Pool, a callable, or None")


__all__ = [
    "CallablePool",
    "Pool",
    "Refill",
    "SequentialPool",
    "Task",
    "ThreadPool",
    "create_pool",
]
