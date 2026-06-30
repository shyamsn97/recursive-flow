"""Optional integrations for the minimal stack.

``structured`` (JSON-schema / pydantic output parsing) and ``tui`` have only core
import-time deps and are re-exported here. The DSPy adapter lives in
:mod:`rflow.integrations.dspy` and is **not** imported here so
``import rflow.integrations`` never requires the ``[dspy]`` extra.
"""

from rflow.integrations.structured import (
    Schema,
    StructuredOutputError,
    StructuredOutputParser,
    json_schema_for,
)
from rflow.integrations.tui import run_tui, tui

__all__ = [
    "Schema",
    "StructuredOutputError",
    "StructuredOutputParser",
    "json_schema_for",
    "run_tui",
    "tui",
]
