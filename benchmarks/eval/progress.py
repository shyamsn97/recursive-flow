"""Injectable progress reporting for eval sweeps."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, TypeVar

T = TypeVar("T")


class ProgressReporter(Protocol):
    def task_loader(self, iterable: Iterable[T], *, total: int) -> Iterable[T]: ...

    def eval_jobs(self, iterable: Iterable[T], *, total: int) -> Iterable[T]: ...


class TqdmProgress:
    """Progress reporter that gracefully degrades when tqdm is not installed."""

    def task_loader(self, iterable: Iterable[T], *, total: int) -> Iterable[T]:
        return _tqdm(iterable, total=total, desc="load tasks")

    def eval_jobs(self, iterable: Iterable[T], *, total: int) -> Iterable[T]:
        return _tqdm(iterable, total=total, desc="eval")


class NullProgress:
    def task_loader(self, iterable: Iterable[T], *, total: int) -> Iterable[T]:
        return iterable

    def eval_jobs(self, iterable: Iterable[T], *, total: int) -> Iterable[T]:
        return iterable


def _tqdm(iterable: Iterable[T], **kwargs) -> Iterable[T]:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


__all__ = ["NullProgress", "ProgressReporter", "TqdmProgress"]
