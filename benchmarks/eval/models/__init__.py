"""Built-in benchmark models."""

from __future__ import annotations

from benchmarks.eval import MODELS, model

from benchmarks.eval.models import anthropic as _anthropic  # noqa: E402,F401
from benchmarks.eval.models import fake as _fake  # noqa: E402,F401
from benchmarks.eval.models import openai as _openai  # noqa: E402,F401

__all__ = ["MODELS", "model"]
