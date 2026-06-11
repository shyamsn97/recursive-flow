"""Backwards-compatible re-export of :class:`FlowConfig`.

The canonical home is :mod:`rflow.engine.config` — engine helpers
import it from there so they never need to reach back into
:mod:`rflow.flow` for typing. This shim keeps existing
``from rflow.config import FlowConfig`` imports working.
"""

from __future__ import annotations

from rflow.engine.config import FlowConfig

__all__ = ["FlowConfig"]
