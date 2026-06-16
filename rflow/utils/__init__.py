"""Utility re-exports.

Code-parsing helpers (:mod:`rflow.code`) are cheap and imported eagerly — the
engine hot path pulls ``find_code_blocks`` / ``check_wait_syntax`` every turn.

The viewer/export figure helpers live in :mod:`rflow.utils.viewer`, which
transitively imports plotly (and, on demand, gradio). To keep that weight off
the engine path, those names are re-exported **lazily** via ``__getattr__`` —
they only import ``viewer`` when first accessed. ``trace`` / ``tracing`` /
``export`` / ``viz`` are plain submodules (``from rflow.utils import export``).
"""

from __future__ import annotations

from typing import Any

from rflow.code import check_wait_syntax, find_code_blocks, replace_code_block

_LAZY_VIEWER = {
    "open_viewer",
    "render_html",
    "resolve_graphs",
    "graph_tree",
    "agent_transcript",
    "save_gif",
    "save_html",
    "save_image",
    "save_steps",
}

__all__ = [
    "check_wait_syntax",
    "find_code_blocks",
    "replace_code_block",
    "agent_transcript",
    "graph_tree",
    "open_viewer",
    "render_html",
    "resolve_graphs",
    "save_gif",
    "save_html",
    "save_image",
    "save_steps",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_VIEWER:
        from rflow.utils import viewer

        return getattr(viewer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
