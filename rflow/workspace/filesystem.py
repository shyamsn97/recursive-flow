"""Filesystem-backed workspace implementations."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from rflow.graph import ContextPayload, Graph, parse_node_obj
from rflow.workspace.artifacts import ArtifactStore
from rflow.workspace.base import BaseWorkspace, Context, Session, build_graph
from rflow.workspace.context_helpers import (
    context_keys_for_agents,
    first_context_value,
)
from rflow.workspace.store import Store, copy_workspace_paths, resolve_backend


def _safe_name(value: str, *, default: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or default


class FileContext(Context):
    """Store-backed context persistence."""

    def __init__(self, root: Store | str | Path) -> None:
        self.store, self.root = resolve_backend(root)

    def _context_paths(self, key: str, *, agent_id: str) -> tuple[Path, Path]:
        safe = _safe_name(key, default="context")
        base = Path("context") / _safe_name(agent_id, default="root")
        if safe == "context":
            return base / "context.txt", base / "context_metadata.json"
        return base / f"{safe}.txt", base / f"{safe}_metadata.json"

    def write(
        self,
        key: str,
        value: str,
        *,
        agent_id: str = "root",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        path, meta_path = self._context_paths(key, agent_id=agent_id)
        self.store.write_text(str(path), value)
        self.store.write_json(
            str(meta_path),
            {
                "key": key,
                "agent_id": agent_id,
                "chars": len(value),
                "metadata": metadata or {},
            },
        )

    def read(self, key: str = "context", *, agent_id: str = "root") -> str:
        return first_context_value(
            key,
            agent_id=agent_id,
            exists=lambda aid, k: self.store.exists(
                str(self._context_paths(k, agent_id=aid)[0])
            ),
            read=lambda aid, k: self.store.read_text(
                str(self._context_paths(k, agent_id=aid)[0])
            ),
        )

    def list_contexts(self, *, agent_id: str | None = None) -> list[str]:
        def keys_for_agent(aid: str) -> list[str]:
            base = Path("context") / _safe_name(aid, default="root")
            return [
                Path(path).stem
                for path in self.store.list(str(base))
                if path.endswith(".txt")
            ]

        return context_keys_for_agents(agent_id, keys_for_agent)

    def fork(self, new_location: object) -> Context:
        return FileContext(copy_workspace_paths(self.store, new_location, ("context",)))


class FileSession(Session):
    """Filesystem-backed :class:`Session`."""

    def __init__(self, root: Store | str | Path) -> None:
        self.store, self.root = resolve_backend(root)

    def write_agent(self, graph) -> None:
        self.store.write_json(
            f"session/{_safe_name(graph.agent_id, default='root')}/agent.json",
            graph.meta_dict(),
        )
        self._touch_graph_agent(graph.agent_id)

    def write_state(self, state) -> None:
        path = f"session/{_safe_name(state.agent_id, default='root')}/session.jsonl"
        self.store.append_jsonl(path, state)
        self._write_latest(state)

    def rewrite_graph(self, graph) -> None:
        agent_ids = [agent.agent_id for agent in graph.walk()]
        self.store.write_json(
            "graph.json",
            {
                "root_agent_id": graph.agent_id,
                "agents": agent_ids,
            },
        )
        for agent in graph.walk():
            safe = _safe_name(agent.agent_id, default="root")
            self.store.write_json(f"session/{safe}/agent.json", agent.meta_dict())
            lines = [
                json.dumps(node.model_dump(mode="json"), ensure_ascii=False)
                for node in agent.nodes
            ]
            text = "\n".join(lines)
            if text:
                text += "\n"
            self.store.write_text(f"session/{safe}/session.jsonl", text)
            if agent.nodes:
                self._write_latest(agent.nodes[-1])
            else:
                self.store.remove(f"session/{safe}/latest.json")

    def read_transcript(self, agent_id: str) -> dict[str, Any] | None:
        path = f"session/{_safe_name(agent_id, default='root')}/transcript.json"
        if not self.store.exists(path):
            return None
        return self.store.read_json(path)

    def write_transcript(self, agent_id: str, transcript: dict[str, Any]) -> None:
        path = f"session/{_safe_name(agent_id, default='root')}/transcript.json"
        self.store.write_json(path, transcript)

    def load_graph(self):
        manifest = self._load_manifest()
        agent_dicts: dict[str, dict[str, Any]] = {}
        agent_states = {}
        for aid in manifest["agents"]:
            safe = _safe_name(aid, default="root")
            meta_path = f"session/{safe}/agent.json"
            if not self.store.exists(meta_path):
                continue
            agent_dicts[aid] = self.store.read_json(meta_path)
            contexts = self._read_contexts(aid)
            if contexts:
                agent_dicts[aid]["contexts"] = contexts
            session_path = f"session/{safe}/session.jsonl"
            agent_states[aid] = tuple(
                parse_node_obj(line) for line in self.store.read_jsonl(session_path)
            )
        return build_graph(
            root_agent_id=manifest["root_agent_id"],
            agent_dicts=agent_dicts,
            agent_nodes=agent_states,
        )

    def _read_contexts(self, agent_id: str) -> dict[str, dict[str, Any]]:
        safe = _safe_name(agent_id, default="root")
        base = f"context/{safe}"
        contexts: dict[str, dict[str, Any]] = {}
        for path in self.store.list(base):
            if not path.endswith(".txt"):
                continue
            stem = Path(path).stem
            meta_path = (
                f"{base}/context_metadata.json"
                if stem == "context"
                else f"{base}/{stem}_metadata.json"
            )
            metadata: dict[str, Any] = {}
            key = stem
            if self.store.exists(meta_path):
                raw_meta = self.store.read_json(meta_path)
                key = raw_meta.get("key") or key
                metadata = dict(raw_meta.get("metadata") or {})
            contexts[key] = ContextPayload(
                text=self.store.read_text(path),
                metadata=metadata,
            ).model_dump(mode="json")
        return contexts

    def _load_manifest(self) -> dict[str, Any]:
        if self.store.exists("graph.json"):
            return self.store.read_json("graph.json")
        return {"root_agent_id": "root", "agents": []}

    def _touch_graph_agent(self, agent_id: str) -> None:
        manifest = self._load_manifest()
        dirty = False
        if not manifest["agents"]:
            manifest["root_agent_id"] = agent_id
            dirty = True
        if agent_id not in manifest["agents"]:
            manifest["agents"].append(agent_id)
            dirty = True
        if dirty:
            self.store.write_json("graph.json", manifest)

    def _write_latest(self, state) -> None:
        self.store.write_json(
            f"session/{_safe_name(state.agent_id, default='root')}/latest.json",
            {
                "agent_id": state.agent_id,
                "latest_node_id": state.id,
                "seq": state.seq,
                "type": state.type,
                "terminal": state.terminal,
                "result": getattr(state, "result", None),
            },
        )

    def fork(self, new_location: object) -> Session:
        return FileSession(
            copy_workspace_paths(
                self.store,
                new_location,
                ("graph.json", "session"),
            )
        )


class Workspace(BaseWorkspace):
    """Local filesystem workspace."""

    def __init__(
        self,
        root: str | Path,
        *,
        session: Session | None = None,
        context: Context | None = None,
        uri: str | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.session = session or FileSession(self.root)
        self.context = context or FileContext(self.root)
        self.artifacts = ArtifactStore(self.root)
        self.uri = uri or str(self.root)

    @classmethod
    def create(
        cls,
        dir: str | Path,
        *,
        session: Session | None = None,
        context: Context | None = None,
    ) -> Workspace:
        return cls(dir, session=session, context=context)

    @classmethod
    def from_graph(
        cls,
        graph: Graph,
        dir: str | Path | None = None,
    ) -> Workspace:
        """Create a workspace from ``graph``."""

        if dir is None:
            raise TypeError("Workspace.from_graph(...) requires a workspace path")
        root = Path(dir).resolve()
        if root.exists():
            shutil.rmtree(root)
        workspace = cls.create(root)
        workspace.sync_graph(graph)
        return workspace

    @classmethod
    def open_path(
        cls,
        dir: str | Path,
    ) -> Workspace:
        return cls.create(dir)

    @staticmethod
    def check_path(path: str | Path) -> bool:
        root = Path(path)
        return (
            root.is_dir()
            and (root / "graph.json").is_file()
            and (root / "session").is_dir()
        )

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def materialize(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def commit(self) -> None:
        """Local filesystem workspaces are already durable."""

    def prune_graph_payloads(self, graph) -> None:
        """Remove per-agent payload dirs for agents absent from ``graph``."""

        agent_ids = [agent.agent_id for agent in graph.walk()]
        _prune_agent_dirs(self.root / "session", agent_ids)
        _prune_agent_dirs(self.root / "context", agent_ids)

    def fork(
        self,
        new_location: str | Path | None = None,
        *,
        new_dir: str | Path | None = None,
    ) -> Workspace:
        location = new_location if new_location is not None else new_dir
        if location is None:
            raise TypeError("fork() requires a new workspace location")

        new_root = Path(location).resolve()
        if new_root.exists():
            shutil.rmtree(new_root)
        new_root.mkdir(parents=True, exist_ok=True)

        reserved = {
            "session",
            "context",
            "graph.json",
            "trace",
            "checkpoint.json",
        }
        for item in self.root.iterdir():
            if item.name in reserved:
                continue
            dst = new_root / item.name
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)

        forked = type(self)(
            new_root,
            session=self.session.fork(new_root),
            context=self.context.fork(new_root),
        )
        graph = forked.session.load_graph()
        if graph.nodes or graph.children:
            forked.sync_graph(graph)
        return forked


def _prune_agent_dirs(base: Path, agent_ids: list[str]) -> None:
    if not base.exists():
        return
    keep = {_safe_name(agent_id, default="root") for agent_id in agent_ids}
    for child in base.iterdir():
        if child.is_dir() and child.name not in keep:
            shutil.rmtree(child)


__all__ = ["FileContext", "FileSession", "Workspace"]
