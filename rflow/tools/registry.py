"""REPL namespace partitioning for prompt generation.

The REPL namespace holds both model-facing tools and a couple of hidden control
primitives (``flow_delegate``/``flow_wait``) that ``launch_subagents`` composes
but the model should not call directly. :func:`partition_repl_namespace` splits a
namespace into ``(visible, hidden)`` so the prompt's tool list only advertises
the visible ones.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SHOW_VARS_NAME = "SHOW_VARS"
HIDDEN_REPL_TOOL_NAMES = frozenset({"flow_delegate", "flow_wait"})


def partition_repl_namespace(
    namespace: Mapping[str, Any],
    *,
    hidden_names: frozenset[str] | set[str] = HIDDEN_REPL_TOOL_NAMES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a REPL namespace into ``(visible, hidden)`` callables.

    Skips private names, ``SHOW_VARS``, and non-callables. ``flow_delegate`` /
    ``flow_wait`` (or any name in ``hidden_names``) go in the hidden bucket.
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
