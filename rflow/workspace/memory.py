"""In-memory workspace data implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rflow.graph import Graph
from rflow.workspace.artifacts import ArtifactStore
from rflow.workspace.base import (
    BaseWorkspace,
    Context,
    Session,
    build_graph,
    graph_fingerprint,
)
from rflow.workspace.context_helpers import (
    context_keys_for_agents,
    first_context_value,
)
from rflow.workspace.store import MemoryStore


class InMemoryContext(Context):
    """Process-local payload store for runs without a filesystem workspace."""

    def __init__(self) -> None:
        self.blobs: dict[tuple[str, str], str] = {}

    def write(
        self,
        key: str,
        value: str,
        *,
        agent_id: str = "root",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.blobs[(agent_id, key)] = value

    def read(self, key: str = "context", *, agent_id: str = "root") -> str:
        return first_context_value(
            key,
            agent_id=agent_id,
            exists=lambda aid, k: (aid, k) in self.blobs,
            read=lambda aid, k: self.blobs[(aid, k)],
        )

    def list_contexts(self, *, agent_id: str | None = None) -> list[str]:
        return context_keys_for_agents(
            agent_id,
            lambda target: (key for aid, key in self.blobs if aid == target),
        )

    def fork(self, new_location: object) -> Context:
        out = InMemoryContext()
        out.blobs = dict(self.blobs)
        return out


class InMemorySession(Session):
    """Process-local session for runs without a filesystem workspace."""

    def __init__(self) -> None:
        self.agent_dicts: dict[str, dict[str, Any]] = {}
        self.agent_states = {}
        self.agent_transcripts: dict[str, dict[str, Any]] = {}
        self.root_agent_id: str = "root"

    def write_agent(self, graph) -> None:
        if not self.agent_dicts:
            self.root_agent_id = graph.agent_id
        self.agent_dicts[graph.agent_id] = graph.meta_dict()
        self.agent_states.setdefault(graph.agent_id, [])
        self.agent_transcripts.setdefault(graph.agent_id, {})

    def write_state(self, state) -> None:
        self.agent_states.setdefault(state.agent_id, []).append(state)

    def rewrite_graph(self, graph) -> None:
        self.root_agent_id = graph.agent_id
        self.agent_dicts = {agent.agent_id: agent.meta_dict() for agent in graph.walk()}
        self.agent_states = {
            agent.agent_id: list(agent.nodes) for agent in graph.walk()
        }
        self.agent_transcripts = {
            aid: self.agent_transcripts.get(aid, {}) for aid in self.agent_dicts
        }

    def read_transcript(self, agent_id: str) -> dict[str, Any] | None:
        existing = self.agent_transcripts.get(agent_id)
        if not existing:
            return None
        return {k: list(v) if isinstance(v, list) else v for k, v in existing.items()}

    def write_transcript(self, agent_id: str, transcript: dict[str, Any]) -> None:
        self.agent_transcripts[agent_id] = {
            k: list(v) if isinstance(v, list) else v for k, v in transcript.items()
        }

    def load_graph(self):
        return build_graph(
            root_agent_id=self.root_agent_id,
            agent_dicts=self.agent_dicts,
            agent_nodes={aid: tuple(s) for aid, s in self.agent_states.items()},
        )

    def fork(self, new_location: object) -> Session:
        out = InMemorySession()
        out.agent_dicts = {aid: dict(d) for aid, d in self.agent_dicts.items()}
        out.agent_states = {aid: list(s) for aid, s in self.agent_states.items()}
        out.agent_transcripts = {
            aid: {k: list(v) if isinstance(v, list) else v for k, v in t.items()}
            for aid, t in self.agent_transcripts.items()
        }
        out.root_agent_id = self.root_agent_id
        return out


class InMemoryWorkspace(BaseWorkspace):
    """Process-local workspace for graph/session tests and in-memory runs."""

    def __init__(
        self,
        *,
        session: Session | None = None,
        context: Context | None = None,
        store: MemoryStore | None = None,
    ) -> None:
        self.root = Path("<memory>")
        self.store = store or MemoryStore()
        self.session = session or InMemorySession()
        self.context = context or InMemoryContext()
        self.artifacts = ArtifactStore(self.store)
        self.uri = "memory://workspace"

    @classmethod
    def create(cls) -> InMemoryWorkspace:
        return cls()

    @classmethod
    def from_graph(
        cls,
        graph: Graph,
        dir: str | Path | None = None,
    ) -> InMemoryWorkspace:
        workspace = cls.create()
        workspace.sync_graph(graph)
        return workspace

    def materialize(self) -> Path:
        raise TypeError("InMemoryWorkspace cannot be materialized as a filesystem path")

    def commit(self) -> None:
        """In-memory workspaces are already current."""

    def fork(
        self,
        new_location: str | Path | None = None,
    ) -> InMemoryWorkspace:
        return type(self)(
            session=self.session.fork(new_location),
            context=self.context.fork(new_location),
            store=self.store.fork(new_location),
        )

    def sync_graph(
        self,
        graph: Graph,
        *,
        prune: bool = True,
        restamp: bool = True,
    ) -> Graph:
        synced = graph.copy(deep=True)
        self._ensure_graph_context_payloads(synced)
        self.session.rewrite_graph(synced)
        self._sync_graph_contexts(synced)
        self._remember_graph_fingerprint(graph_fingerprint(synced))
        return self.session.load_graph()

    def sync_graph_if_changed(self, graph: Graph) -> Graph:
        candidate = graph.copy(deep=True)
        self._ensure_graph_context_payloads(candidate)
        candidate_hash = graph_fingerprint(candidate)
        if getattr(self, "_graph_fingerprint", None) == candidate_hash:
            persisted = self.session.load_graph()
            self._sync_graph_contexts(persisted)
            return persisted

        persisted = self.session.load_graph()
        persisted_hash = graph_fingerprint(persisted)
        if persisted_hash == candidate_hash:
            self._sync_graph_contexts(persisted)
            self._remember_graph_fingerprint(persisted_hash)
            return persisted

        return self.sync_graph(candidate, restamp=False)

    def mark_graph_synced(self, graph: Graph) -> Graph:
        synced = graph.copy(deep=True)
        self._remember_graph_fingerprint(graph_fingerprint(synced))
        return graph

    def prune_graph_payloads(self, graph: Graph) -> None:
        """In-memory payloads are cheap and scoped to this workspace."""

    def _remember_graph_fingerprint(self, fingerprint: str) -> None:
        self._graph_fingerprint = fingerprint

    def load_graph(self) -> Graph:
        return self.session.load_graph()


__all__ = ["InMemoryContext", "InMemorySession", "InMemoryWorkspace"]
