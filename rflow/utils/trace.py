"""The :class:`Trace`: the canonical input for every visualization.

A trace is a list of :class:`~rflow.graph.Graph` snapshots — typically one per
:meth:`Flow.step` call — plus optional metadata. Every viewer/exporter consumes
a ``Trace`` (or anything coercible to one via :meth:`Trace.of`).

Build one in whichever way matches what you have:

- :meth:`Trace.from_graphs` — an explicit list of snapshots you captured.
- :meth:`Trace.from_graph` — a single *final* graph, expanded into per-tick
  snapshots with :func:`~rflow.graph.timeline.retrace_steps` (also ``graph.trace()``).
- :meth:`Trace.load` — a ``trace.json`` / ``graph.json`` / directory on disk.

The file format is JSON: a top-level dict with ``"steps"`` (a list of
``Graph.to_dict()`` payloads) and optional ``"metadata"``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Union

from rflow.graph import Graph

if TYPE_CHECKING:
    #: Anything the viz layer accepts; coerced to a :class:`Trace` by ``Trace.of``.
    TraceSource = Union[str, Path, Graph, Iterable[Graph], "Trace"]


class Trace(NamedTuple):
    """An ordered list of graph snapshots plus optional metadata."""

    graphs: list[Graph]
    metadata: dict[str, Any]

    # ── constructors ──────────────────────────────────────────────────

    @classmethod
    def from_graphs(
        cls, graphs: Iterable[Graph], *, metadata: dict | None = None
    ) -> "Trace":
        """Wrap an explicit list of snapshots (no reconstruction)."""
        return cls(graphs=list(graphs), metadata=dict(metadata or {}))

    @classmethod
    def from_graph(cls, graph: Graph, *, metadata: dict | None = None) -> "Trace":
        """Expand a single final graph into per-tick snapshots.

        Uses :func:`rflow.graph.timeline.retrace_steps`, so a graph you only
        kept the final state of (e.g. loaded from ``graph.json``) still produces
        a real stepped timeline — no code is re-executed.
        """
        from rflow.graph.timeline import retrace_steps

        return cls(graphs=retrace_steps(graph), metadata=dict(metadata or {}))

    @classmethod
    def of(cls, source: "TraceSource") -> "Trace":
        """Coerce any supported source into a :class:`Trace`.

        A :class:`Trace` passes through; a path is loaded; an iterable of graphs
        is wrapped as-is; a single :class:`Graph` becomes a one-frame trace (use
        :meth:`from_graph` / ``graph.trace()`` to expand it into steps instead).
        """
        if isinstance(source, Trace):
            return source
        if isinstance(source, Graph):
            return cls(graphs=[source], metadata={})
        if isinstance(source, (str, Path)):
            return cls.load(source)
        graphs = list(source)
        if not all(isinstance(graph, Graph) for graph in graphs):
            raise TypeError("expected a Trace, Graph, path, or iterable of Graphs")
        return cls.from_graphs(graphs)

    @classmethod
    def load(cls, path: str | Path) -> "Trace":
        """Load a trace from disk.

        Accepts a ``trace.json`` (``{"steps": [...], "metadata": {...}}``), a
        single ``Graph.to_dict()`` dump, a JSON list of dumps, or a directory
        holding ``trace.json`` (preferred) or ``graph.json``.
        """
        p = Path(path)
        if p.is_dir():
            trace = p / "trace.json"
            if trace.exists():
                p = trace
            else:
                graph_json = p / "graph.json"
                if not graph_json.exists():
                    raise ValueError(f"{p} has neither trace.json nor graph.json")
                from rflow.graph.run_layout import is_run_manifest

                manifest = json.loads(graph_json.read_text(encoding="utf-8"))
                if is_run_manifest(manifest):
                    graph = Graph.load(p)
                    return cls.from_graph(
                        graph, metadata=manifest.get("metadata") or {}
                    )
                p = graph_json
        if not p.is_file():
            raise ValueError(f"no such file or directory: {p}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p} is not valid JSON: {exc}") from exc

        if _looks_like_trace(data):
            graphs = [Graph.from_dict(step) for step in data.get("steps", [])]
            return cls(graphs=graphs, metadata=data.get("metadata") or {})
        if _looks_like_graph_dump(data):
            return cls(graphs=[Graph.from_dict(data)], metadata={})
        if isinstance(data, list) and all(_looks_like_graph_dump(d) for d in data):
            return cls(graphs=[Graph.from_dict(d) for d in data], metadata={})
        raise ValueError(f"{p} does not look like a graph dump or trace")

    # ── conveniences ──────────────────────────────────────────────────

    @property
    def latest(self) -> Graph:
        """The final snapshot (raises if the trace is empty)."""
        if not self.graphs:
            raise ValueError("trace has no snapshots")
        return self.graphs[-1]

    def save(self, path: str | Path = "trace.json") -> Path:
        """Persist this trace's snapshots (and metadata) to ``path``."""
        return save_trace(self.graphs, path, self.metadata or None)


def _looks_like_graph_dump(data: Any) -> bool:
    return (
        isinstance(data, dict)
        and "agent_id" in data
        and ("nodes" in data or "states" in data)
    )


def _looks_like_trace(data: Any) -> bool:
    return isinstance(data, dict) and "steps" in data


def _resolve(path: str | Path, *, writing: bool) -> Path:
    p = Path(path)
    if p.is_dir() or not p.suffix:
        if writing:
            p.mkdir(parents=True, exist_ok=True)
        p = p / "trace.json"
    elif writing:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_trace(
    graphs: list[Graph],
    path: str | Path = "trace.json",
    metadata: dict | None = None,
) -> Path:
    """Persist a list of :class:`Graph` snapshots."""
    p = _resolve(path, writing=True)
    data: dict[str, Any] = {"steps": [graph.to_dict() for graph in graphs]}
    if metadata:
        data["metadata"] = metadata
    p.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
    return p


def load_trace(path: str | Path) -> Trace:
    """Load a ``trace.json`` (or a directory containing one) into a :class:`Trace`."""
    p = _resolve(path, writing=False)
    data = json.loads(p.read_text(encoding="utf-8"))
    graphs = [Graph.from_dict(step) for step in data.get("steps", [])]
    return Trace(graphs=graphs, metadata=data.get("metadata") or {})


__all__ = ["Trace", "load_trace", "save_trace"]
