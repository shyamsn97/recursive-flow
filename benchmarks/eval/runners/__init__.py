"""Built-in benchmark runners."""

from __future__ import annotations

from benchmarks.eval import RUNNERS, runner

from benchmarks.eval.runners import fake as _fake  # noqa: E402,F401
from benchmarks.eval.runners import official_rlm as _official_rlm  # noqa: E402,F401
from benchmarks.eval.runners import rflow as _rflow  # noqa: E402,F401
from benchmarks.eval.runners import vanilla as _vanilla  # noqa: E402,F401

__all__ = ["RUNNERS", "runner"]
