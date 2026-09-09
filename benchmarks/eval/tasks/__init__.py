"""Built-in benchmark datasets."""

from __future__ import annotations

from benchmarks.eval import DATASETS, dataset
from benchmarks.eval.sets import register_sets

# Explicit built-ins so decorator registration remains grep-able.
from benchmarks.eval.tasks import (
    aime,  # noqa: F401
    arc_agi,  # noqa: F401
    browsecomp,  # noqa: F401
    dabstep,  # noqa: F401
    delegation_codeqa,  # noqa: F401
    delegation_iteration,  # noqa: F401
    delegation_sudoku,  # noqa: F401
    entailmentbank,  # noqa: F401
    grsqa,  # noqa: F401
    livecodebench,  # noqa: F401
    longbench,  # noqa: F401
    musique,  # noqa: F401
    natural_plan,  # noqa: F401
    oolong,  # noqa: F401
    oolong_pairs,  # noqa: F401
    parallelqa,  # noqa: F401
    planbench,  # noqa: F401
    sniah,  # noqa: F401
    sudoku,  # noqa: F401
    synthetic_needle,  # noqa: F401
    twowiki,  # noqa: F401
)

register_sets(DATASETS)
DATASETS.alias("all", DATASETS.names())

__all__ = ["DATASETS", "dataset"]
