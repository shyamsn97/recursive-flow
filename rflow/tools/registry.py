"""REPL namespace partitioning for prompt generation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SHOW_VARS_NAME = "SHOW_VARS"
HIDDEN_REPL_TOOL_NAMES = frozenset()


def partition_repl_namespace(
    namespace: Mapping[str, Any],
    *,
    hidden_names: frozenset[str] | set[str] = HIDDEN_REPL_TOOL_NAMES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a REPL namespace into ``(visible, hidden)`` callables.

    Skips private names, ``SHOW_VARS``, and non-callables. Any name in
    ``hidden_names`` goes in the hidden bucket.
    """
    visible: dict[str, Any] = {}
    hidden: dict[str, Any] = {}
    for name, value in namespace.items():
        if name.startswith("_") or name == SHOW_VARS_NAME or not callable(value):
            continue
        if name in hidden_names:
            hidden[name] = value
        else:
            visible[name] = value
    return visible, hidden


__all__ = [
    "HIDDEN_REPL_TOOL_NAMES",
    "SHOW_VARS_NAME",
    "partition_repl_namespace",
]
