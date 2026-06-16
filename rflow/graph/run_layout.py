"""Run-directory persistence: manifest + per-agent JSONL logs.

See ``docs/internal/run-layout.md`` for the on-disk layout. The in-memory source of
truth remains :class:`~rflow.graph.Graph`; this module is the projection to
and from disk.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rflow.graph.graph import Graph, Node, parse_node_obj

_AGENT_DIR_RE = re.compile(r"[^A-Za-z0-9_.-]+")
AGENTS_DIRNAME = "agents"


def safe_dirname(name: str) -> str:
    """Filesystem-safe directory name (dots and dashes are kept)."""
    return _AGENT_DIR_RE.sub("_", name).strip("_") or "agent"


def local_dirname(agent_id: str, parent_agent_id: str | None) -> str:
    """The directory name for an agent under its parent's directory.

    Agents nest by the graph's parent→child relationship, so each folder is
    named by the child's id with the parent prefix stripped (e.g. child
    ``root.cols.cols.4_1`` under parent ``root.cols`` becomes ``cols.4_1``).
    The root keeps its full id. Dots inside a name are preserved — the name is
    never split into path segments.
    """
    local = agent_id
    if parent_agent_id and agent_id.startswith(parent_agent_id + "."):
        local = agent_id[len(parent_agent_id) + 1 :]
    return safe_dirname(local)


def _agents_search_root(run_root: Path) -> tuple[Path, bool]:
    """Where agent dirs live: ``agents/`` if present, else the run root (legacy)."""
    agents_root = run_root / AGENTS_DIRNAME
    if agents_root.is_dir():
        return agents_root, True
    return run_root, False


def resolve_agent_dir(run_root: Path, agent_id: str) -> Path | None:
    """Locate an agent's directory by reading ids from ``agent.json`` files.

    Works for the nested layout (and legacy flat layouts) because the lookup
    matches the stored ``agent_id`` rather than reconstructing a path.
    """
    search_root, nested = _agents_search_root(run_root)
    candidates = (
        search_root.rglob("agent.json") if nested else search_root.glob("*/agent.json")
    )
    for meta_path in candidates:
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("agent_id") == agent_id:
            return meta_path.parent
    return None


def is_run_manifest(data: Any) -> bool:
    """True when ``data`` is a run manifest (not a nested graph snapshot)."""
    return (
        isinstance(data, dict)
        and "root_agent_id" in data
        and "agents" in data
        and "nodes" not in data
        and "states" not in data
    )


def is_graph_snapshot(data: Any) -> bool:
    """True when ``data`` is a monolithic :meth:`Graph.to_dict` payload."""
    return (
        isinstance(data, dict)
        and "agent_id" in data
        and ("nodes" in data or "states" in data)
    )


def agent_meta_dict(graph: Graph) -> dict[str, Any]:
    """Metadata fields persisted in ``{agent_id}/agent.json``."""
    return {
        "agent_id": graph.agent_id,
        "depth": graph.depth,
        "query": graph.query,
        "system_prompt": graph.system_prompt,
        "inputs": dict(graph.inputs),
        "model": graph.model,
        "max_iters": graph.max_iters,
        "output_schema": graph.output_schema,
        "parent_agent_id": graph.parent_agent_id,
        "parent_node_id": graph.parent_node_id,
    }


def latest_dict(node: Node) -> dict[str, Any]:
    """Tip summary written to ``{agent_id}/latest.json``."""
    from rflow.graph.graph import ErrorOutput, SupervisingOutput

    payload: dict[str, Any] = {
        "agent_id": node.agent_id,
        "latest_node_id": node.id,
        "seq": node.seq,
        "type": node.type,
        "terminal": node.terminal,
    }
    result = getattr(node, "result", None)
    if result:
        payload["result"] = result
    if isinstance(node, SupervisingOutput):
        payload["waiting_on"] = list(node.waiting_on)
    if isinstance(node, ErrorOutput):
        payload["error"] = node.error
    return payload


def build_graph(
    *,
    root_agent_id: str,
    agent_dicts: dict[str, dict[str, Any]],
    agent_nodes: dict[str, tuple[Node, ...]],
) -> Graph:
    """Recover the recursive :class:`Graph` from flat per-agent dicts."""
    children_by_parent: dict[str, list[str]] = {}
    for aid, data in agent_dicts.items():
        if aid == root_agent_id:
            continue
        parent = data.get("parent_agent_id") or root_agent_id
        children_by_parent.setdefault(parent, []).append(aid)

    def build(aid: str) -> Graph:
        data = agent_dicts.get(aid, {"agent_id": aid})
        nodes = agent_nodes.get(aid, ())
        children = {
            child_aid: build(child_aid) for child_aid in children_by_parent.get(aid, [])
        }
        return Graph.from_meta_dict(data, nodes=list(nodes), children=children)

    if root_agent_id not in agent_dicts:
        return Graph(agent_id=root_agent_id)
    return build(root_agent_id)


def _write_agent_dir(graph: Graph, dir_path: Path) -> None:
    """Write one agent's files, then recurse into child subdirectories."""
    dir_path.mkdir(parents=True, exist_ok=True)

    (dir_path / "agent.json").write_text(
        json.dumps(agent_meta_dict(graph), default=str, indent=2),
        encoding="utf-8",
    )

    lines = [
        json.dumps(node.to_dict(), default=str, ensure_ascii=False)
        for node in graph.nodes
    ]
    session_text = "\n".join(lines)
    if session_text:
        session_text += "\n"
    (dir_path / "session.jsonl").write_text(session_text, encoding="utf-8")

    latest_path = dir_path / "latest.json"
    if graph.nodes:
        latest_path.write_text(
            json.dumps(latest_dict(graph.nodes[-1]), default=str, indent=2),
            encoding="utf-8",
        )
    elif latest_path.exists():
        latest_path.unlink()

    kept: set[str] = set()
    for child in graph.children.values():
        name = local_dirname(child.agent_id, graph.agent_id)
        kept.add(name)
        _write_agent_dir(child, dir_path / name)

    # Drop child agent dirs that are no longer part of this subtree.
    for path in dir_path.iterdir():
        if path.is_dir() and path.name not in kept and (path / "agent.json").is_file():
            shutil.rmtree(path)


def save_run(
    graph: Graph,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a run directory (manifest + nested per-agent logs).

    Agents are stored under ``agents/`` nested by their parent→child
    relationship (see ``docs/internal/run-layout.md``). Returns the run root.
    """
    run_root = Path(path)
    run_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "root_agent_id": graph.agent_id,
        "agents": [agent.agent_id for agent in graph.walk()],
        "metadata": {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        },
    }
    result = graph.result()
    if result:
        manifest["metadata"]["result"] = result

    (run_root / "graph.json").write_text(
        json.dumps(manifest, default=str, indent=2),
        encoding="utf-8",
    )

    agents_root = run_root / AGENTS_DIRNAME
    agents_root.mkdir(parents=True, exist_ok=True)
    root_local = safe_dirname(graph.agent_id)
    _write_agent_dir(graph, agents_root / root_local)

    # Prune stale top-level agent dirs (e.g. a previous flat layout) and any
    # legacy flat agent dirs written directly at the run root.
    for entry in agents_root.iterdir():
        if (
            entry.is_dir()
            and entry.name != root_local
            and (entry / "agent.json").is_file()
        ):
            shutil.rmtree(entry)
    for entry in run_root.iterdir():
        if (
            entry.is_dir()
            and entry.name != AGENTS_DIRNAME
            and (entry / "agent.json").is_file()
        ):
            shutil.rmtree(entry)

    return run_root


def load_run(path: str | Path) -> Graph:
    """Rebuild a :class:`Graph` from a run directory.

    Discovers agents by reading the ``agent_id`` recorded in each
    ``agent.json``, so the same loader handles the nested layout and older
    flat layouts. The recursive shape is rebuilt from ``parent_agent_id``.
    """
    run_root = Path(path)
    if not run_root.is_dir():
        raise ValueError(f"run directory does not exist: {run_root}")

    manifest_path = run_root / "graph.json"
    if not manifest_path.is_file():
        raise ValueError(f"{run_root} has no graph.json manifest")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not is_run_manifest(manifest):
        raise ValueError(f"{manifest_path} is not a run manifest")

    root_agent_id = manifest["root_agent_id"]
    agent_dicts: dict[str, dict[str, Any]] = {}
    agent_nodes: dict[str, tuple[Node, ...]] = {}

    search_root, nested = _agents_search_root(run_root)
    meta_paths = (
        search_root.rglob("agent.json") if nested else search_root.glob("*/agent.json")
    )
    for meta_path in meta_paths:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        aid = data.get("agent_id")
        if not aid:
            continue
        agent_dicts[aid] = data

        session_path = meta_path.parent / "session.jsonl"
        nodes: list[Node] = []
        if session_path.is_file():
            for line in session_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    nodes.append(parse_node_obj(json.loads(line)))
        agent_nodes[aid] = tuple(nodes)

    return build_graph(
        root_agent_id=root_agent_id,
        agent_dicts=agent_dicts,
        agent_nodes=agent_nodes,
    )


def save_snapshot(graph: Graph, path: str | Path) -> Path:
    """Write a monolithic nested ``graph.json`` snapshot."""
    p = Path(path)
    if p.is_dir() or not p.suffix:
        p.mkdir(parents=True, exist_ok=True)
        p = p / "graph.json"
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(graph.to_dict(), default=str, indent=2), encoding="utf-8")
    return p


def load_snapshot(path: str | Path) -> Graph:
    """Load a monolithic nested ``graph.json`` snapshot."""
    p = Path(path)
    if p.is_dir():
        p = p / "graph.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    if is_run_manifest(data):
        raise ValueError(f"{p} is a run manifest, not a graph snapshot")
    return Graph.from_dict(data)


__all__ = [
    "AGENTS_DIRNAME",
    "agent_meta_dict",
    "build_graph",
    "is_graph_snapshot",
    "is_run_manifest",
    "latest_dict",
    "load_run",
    "load_snapshot",
    "local_dirname",
    "resolve_agent_dir",
    "safe_dirname",
    "save_run",
    "save_snapshot",
]
