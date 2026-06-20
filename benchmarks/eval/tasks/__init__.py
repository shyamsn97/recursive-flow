"""Built-in benchmark datasets."""

from __future__ import annotations

from benchmarks.eval import DATASETS, dataset

# Explicit built-ins so decorator registration remains grep-able.
from benchmarks.eval.tasks import livecodebench  # noqa: E402,F401
from benchmarks.eval.tasks import longbench  # noqa: E402,F401
from benchmarks.eval.tasks import oolong  # noqa: E402,F401
from benchmarks.eval.tasks import sudoku  # noqa: E402,F401
from benchmarks.eval.tasks import synthetic_needle  # noqa: E402,F401

DATASETS.alias("smoke", ["synthetic_needle"])
DATASETS.alias("needle", ["synthetic_needle"])
DATASETS.alias("all", DATASETS.names())

__all__ = ["DATASETS", "dataset"]
