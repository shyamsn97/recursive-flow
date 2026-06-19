"""Helpers for saving example run graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def example_run_dir(source_file: str | Path, name: str) -> Path:
    """Return `examples/_runs/<name>` for an example source file."""

    source = Path(source_file).resolve()
    for parent in (source.parent, *source.parents):
        if parent.name == "examples":
            return parent / "_runs" / name
    return source.parent / "_runs" / name


def save_example_graph(
    graph: Any,
    source_file: str | Path,
    name: str,
    *,
    out_dir: str | Path | None = None,
    label: str = "Graph saved to",
) -> Path:
    """Save a final example graph and print the destination."""

    path = graph.save(
        Path(out_dir) if out_dir is not None else example_run_dir(source_file, name)
    )
    print(f"{label} {path}")
    return path


__all__ = ["example_run_dir", "save_example_graph"]
