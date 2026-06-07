"""The :class:`Graph` data model.

A :class:`Graph` is **one agent**, mutable. It holds:

* the agent's per-run invariants (``agent_id``, ``depth``, ``query``,
  ``system_prompt``, ``config``, ``workspace``, ``runtime``, ``model``,
  ``branch_id``, ``parent_agent_id``, ``parent_node_id``) as flat fields;
* ``nodes`` — this agent's trajectory of :class:`Node` instances;
* ``children`` — a ``dict[str, Graph]`` of sub-agents spawned from this one.

Recursion lives in ``children``. Indexing by id (``graph[aid]``) walks the
tree; ``graph.agents`` is a flat :class:`Mapping` view over every agent in
the subtree; ``graph.all_nodes`` / ``graph.edges`` are flat views over every
node / derived edge in the subtree.

Per-state payload lives on :class:`Node`. Per-agent invariants live on
``Graph``. There is no ``AgentMeta`` class — its fields are inlined here.
There is no stored ``Edge`` list — flow / spawn edges are derived from
the recursive structure on demand.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from rlmflow.graph.node import ActionNode, Node, ObservationNode, parse_node_obj
from rlmflow.graph.views import AgentsView, Edge, EdgesView, NodesView

# ── refs (small Pydantic models for serializable external pointers) ──


class WorkspaceRef(BaseModel):
    """Serializable reference to branch-local workspace storage."""

    root: str
    branch_id: str = "main"


class RuntimeRef(BaseModel):
    """Serializable reference to a durable runtime / REPL session."""

    id: str


# ── Graph ────────────────────────────────────────────────────────────


@dataclass
class Graph:
    """One agent's view of a run, recursive through ``children``.

    Every field is per-agent invariant (set at spawn) or this agent's
    trajectory. Sub-agents live in ``children``; ``graph[other_aid]``
    or ``graph.agents[other_aid]`` walks the subtree to find them.

    Graphs are mutable — use the editing helpers (``add_node``,
    ``update_node``, ``remove_node``, ``add_child``, ``remove_child``,
    ``update``) or just assign fields directly. ``graph.all_nodes`` /
    ``graph.children`` are live subtree views, so mutations show up immediately.
    """

    agent_id: str
    depth: int = 0
    query: str = ""
    system_prompt: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    workspace: WorkspaceRef | None = None
    runtime: RuntimeRef | None = None
    model: str | None = None
    branch_id: str = "main"
    parent_agent_id: str | None = None
    parent_node_id: str | None = None

    nodes: list[Node] = field(default_factory=list)
    children: dict[str, Graph] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Be forgiving: callers may pass tuples / iterables.
        if not isinstance(self.nodes, list):
            self.nodes = list(self.nodes)
        if not isinstance(self.children, dict):
            self.children = dict(self.children)

    # ── identity / aliases ───────────────────────────────────────────

    @property
    def root_agent_id(self) -> str:
        """Alias for :attr:`agent_id`, reads better at the top level."""
        return self.agent_id

    @property
    def parent_id(self) -> str | None:
        """Alias for :attr:`parent_agent_id`."""
        return self.parent_agent_id

    @property
    def model_key(self) -> str:
        return str(self.config.get("model") or "default")

    @property
    def model_label(self) -> str:
        actual = self.model
        if actual is None:
            actual = next(
                (
                    str(model)
                    for state in reversed(self.nodes)
                    if (model := getattr(state, "model", None))
                ),
                None,
            )
        if actual and actual != self.model_key:
            return f"{self.model_key}:{actual}"
        return self.model_key

    # ── current-node accessors ───────────────────────────────────────

    def current(self) -> Node | None:
        """Latest node of this agent (last by insertion)."""
        return self.nodes[-1] if self.nodes else None

    @property
    def finished(self) -> bool:
        cur = self.current()
        if not (cur and cur.terminal):
            return False
        return all(child.finished for child in self.children.values())

    def result(self) -> str:
        """Terminal result string from the deepest terminal leaf, or ``""``."""
        g: Graph = self
        while True:
            cur = g.current()
            if cur is None:
                return ""
            if cur.terminal:
                return getattr(cur, "result", "") or ""
            kids = list(g.children.values())
            if not kids:
                return ""
            g = kids[-1]

    @property
    def root(self) -> Node | None:
        """First node of this agent (the :class:`UserQuery` at ``seq=0``)."""
        return self.nodes[0] if self.nodes else None

    # ── subtree iteration ────────────────────────────────────────────

    def walk(self) -> Iterator[Graph]:
        """Yield self plus every descendant sub-:class:`Graph`, depth-first."""
        yield self
        for child in self.children.values():
            yield from child.walk()

    def subtree(self) -> list[Graph]:
        """List form of :meth:`walk` (self + all descendants)."""
        return list(self.walk())

    def leaves(self) -> list[Graph]:
        """Agents with no child agents."""
        return [g for g in self.walk() if not g.children]

    def unfinished_agents(self) -> list[Graph]:
        """Agents whose current state is not terminal."""
        return [g for g in self.walk() if not g.finished]

    def finished_agents(self) -> list[Graph]:
        """Agents whose current state is terminal."""
        return [g for g in self.walk() if g.finished]

    def children_of(self, agent_id: str) -> list[Graph]:
        """Direct children of ``agent_id``."""
        return list(self[agent_id].children.values())

    def descendants_of(self, agent_id: str) -> list[Graph]:
        """All descendants of ``agent_id``, excluding that agent."""
        root = self[agent_id]
        return [g for g in root.walk() if g.agent_id != root.agent_id]

    def where(self, predicate: Callable[[Graph], bool]) -> list[Graph]:
        """Agents for which ``predicate(agent)`` is true."""
        return [g for g in self.walk() if predicate(g)]

    def match(self, pattern: str | re.Pattern[str]) -> list[Graph]:
        """Agents whose id matches ``pattern``."""
        compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
        return [g for g in self.walk() if compiled.search(g.agent_id)]

    # ── flat views over the subtree ──────────────────────────────────

    @property
    def agents(self) -> AgentsView:
        return AgentsView(self)

    @property
    def all_nodes(self) -> NodesView:
        return NodesView(self)

    @property
    def edges(self) -> EdgesView:
        return EdgesView(self)

    def find(self, node_id: str) -> Node | None:
        """Bare :class:`Node` lookup by id across the whole subtree."""
        for g in self.walk():
            for n in g.nodes:
                if n.id == node_id:
                    return n
        return None

    # ── sub-rooting ──────────────────────────────────────────────────

    def __getitem__(self, ident: str) -> Graph:
        """Sub-:class:`Graph` for an agent id (bare or dotted)."""
        if ident == self.agent_id:
            return self
        if ident in self.children:
            return self.children[ident]
        # Dotted-path walk: "root.scanner.deep" descends children stepwise.
        if ident.startswith(self.agent_id + "."):
            cur: Graph = self
            for prefix in _path_prefixes(self.agent_id, ident):
                if prefix in cur.children:
                    cur = cur.children[prefix]
                    if cur.agent_id == ident:
                        return cur
                else:
                    break
        # Fallback: search the subtree for any descendant with this id.
        for g in self.walk():
            if g.agent_id == ident:
                return g
        raise KeyError(ident)

    def __contains__(self, ident: object) -> bool:
        if not isinstance(ident, str):
            return False
        try:
            self[ident]
        except KeyError:
            return False
        return True

    # ── token rollups ────────────────────────────────────────────────

    def tokens(self, *, recursive: bool = True) -> tuple[int, int]:
        inp = sum(getattr(s, "input_tokens", 0) for s in self.nodes)
        out = sum(getattr(s, "output_tokens", 0) for s in self.nodes)
        if recursive:
            for child in self.children.values():
                ci, co = child.tokens()
                inp += ci
                out += co
        return inp, out

    def total_tokens(self) -> int:
        i, o = self.tokens()
        return i + o

    # ── editing helpers (mutate in place) ────────────────────────────

    def _index_of(self, node_id: str) -> int:
        for i, s in enumerate(self.nodes):
            if s.id == node_id:
                return i
        raise KeyError(node_id)

    def add_node(self, node: Node) -> Node:
        """Append a node to this agent's trajectory."""
        self.nodes.append(node)
        return node

    def set_node(self, node_id: str, new_node: Node) -> Node:
        """Swap a node on **this** agent by id. For subtree-wide replacement,
        use ``graph.all_nodes.replace(id, node)``."""
        self.nodes[self._index_of(node_id)] = new_node
        return new_node

    def update_node(self, node_id: str, **changes: Any) -> Node:
        """Replace a node on this agent with a copy carrying ``changes``."""
        i = self._index_of(node_id)
        self.nodes[i] = self.nodes[i].update(**changes)
        return self.nodes[i]

    def remove_node(self, node_id: str) -> Node:
        """Drop a node by id and return it."""
        return self.nodes.pop(self._index_of(node_id))

    def pop_node(self) -> Node:
        """Drop and return the most recent node of this agent."""
        return self.nodes.pop()

    def clear_nodes(self) -> Graph:
        self.nodes.clear()
        return self

    def add_state(self, node: Node) -> Node:
        """Deprecated alias for :meth:`add_node`."""

        return self.add_node(node)

    def replace_state(self, node_id: str, new_node: Node) -> Node:
        """Deprecated alias for :meth:`set_node`."""

        return self.set_node(node_id, new_node)

    def update_state(self, node_id: str, **changes: Any) -> Node:
        """Deprecated alias for :meth:`update_node`."""

        return self.update_node(node_id, **changes)

    def remove_state(self, node_id: str) -> Node:
        """Deprecated alias for :meth:`remove_node`."""

        return self.remove_node(node_id)

    def pop_state(self) -> Node:
        """Deprecated alias for :meth:`pop_node`."""

        return self.pop_node()

    def clear_states(self) -> Graph:
        """Deprecated alias for :meth:`clear_nodes`."""

        return self.clear_nodes()

    def last_node(self, agent_id: str) -> Node | None:
        """Latest node for ``agent_id``."""

        return self[agent_id].current()

    def last_action(self, agent_id: str) -> ActionNode | None:
        """Latest action node for ``agent_id``."""

        for node in reversed(self[agent_id].nodes):
            if isinstance(node, ActionNode):
                return node
        return None

    def last_observation(self, agent_id: str) -> ObservationNode | None:
        """Latest observation node for ``agent_id``."""

        for node in reversed(self[agent_id].nodes):
            if isinstance(node, ObservationNode):
                return node
        return None

    def node_owner(self, node_id: str) -> Graph:
        """Agent graph that owns ``node_id``."""

        for graph in self.walk():
            if any(node.id == node_id for node in graph.nodes):
                return graph
        raise KeyError(node_id)

    def replace_node(
        self,
        node_id: str,
        node: Node,
        *,
        truncate: str = "after",
    ) -> Graph:
        """Return a copy with ``node_id`` replaced by ``node``.

        ``truncate`` may be:
        - ``"none"``: replace only the node;
        - ``"after"``: drop later states in the same agent;
        - ``"descendants"``: also prune children whose spawn node disappeared,
          and children that were only reachable through a replaced supervisor's
          ``waiting_on`` list.
        """

        if truncate not in {"none", "after", "descendants"}:
            raise ValueError("truncate must be 'none', 'after', or 'descendants'")
        out = self.copy(deep=True)
        owner = out.node_owner(node_id)
        index = owner._index_of(node_id)
        old, _fixed = out._replace_node_at_index(
            owner,
            index,
            node,
            truncate=truncate,
        )
        out._invalidate_ancestors_after_child_edit(owner, truncate=truncate)
        if truncate == "descendants":
            out._apply_descendant_truncation(owner, old)
        return out

    def replace_last_action(
        self,
        agent_id: str,
        node: ActionNode,
        *,
        truncate: str = "after",
    ) -> Graph:
        """Return a copy replacing ``agent_id``'s latest action node."""

        last = self.last_action(agent_id)
        if last is None:
            raise KeyError(f"agent {agent_id!r} has no action node")
        return self.replace_node(last.id, node, truncate=truncate)

    def replace_last_observation(
        self,
        agent_id: str,
        node: ObservationNode,
        *,
        truncate: str = "after",
    ) -> Graph:
        """Return a copy replacing ``agent_id``'s latest observation node."""

        last = self.last_observation(agent_id)
        if last is None:
            raise KeyError(f"agent {agent_id!r} has no observation node")
        return self.replace_node(last.id, node, truncate=truncate)

    def truncate_after(self, node_id: str, *, descendants: bool = True) -> Graph:
        """Return a copy with states after ``node_id`` removed."""

        out = self.copy(deep=True)
        owner = out.node_owner(node_id)
        index = owner._index_of(node_id)
        owner.nodes = owner.nodes[: index + 1]
        if descendants:
            out._prune_unreachable_children()
        return out

    def truncate_agent(self, agent_id: str, *, after_seq: int) -> Graph:
        """Return a copy with ``agent_id`` states after ``after_seq`` removed."""

        out = self.copy(deep=True)
        out[agent_id].nodes = [
            node for node in out[agent_id].nodes if node.seq <= after_seq
        ]
        out._prune_unreachable_children()
        return out

    def prune_descendants_spawned_after(self, agent_id: str, seq: int) -> Graph:
        """Return a copy pruning children spawned after ``agent_id`` ``seq``."""

        out = self.copy(deep=True)
        owner = out[agent_id]
        kept_ids = {node.id for node in owner.nodes if node.seq <= seq}
        owner.children = {
            aid: child
            for aid, child in owner.children.items()
            if child.parent_node_id in kept_ids
        }
        for child in owner.children.values():
            child._prune_unreachable_children()
        return out

    def _replace_node_at_index(
        self,
        owner: Graph,
        index: int,
        node: Node,
        *,
        truncate: str,
    ) -> tuple[Node, Node]:
        """Replace one local state and optionally drop its local future."""

        old = owner.nodes[index]
        fixed = _node_for_replacement(old, node)
        if truncate == "none":
            owner.nodes[index] = fixed
        else:
            owner.nodes = [*owner.nodes[:index], fixed]
        return old, fixed

    def _invalidate_ancestors_after_child_edit(
        self,
        owner: Graph,
        *,
        truncate: str,
    ) -> None:
        """Drop stale ancestor resume paths after a child timeline edit."""

        if truncate == "none" or owner.parent_agent_id is None:
            return
        self._truncate_ancestors_waiting_on(owner.agent_id)

    def _apply_descendant_truncation(self, owner: Graph, old: Node) -> None:
        """Apply child cleanup for route-changing edits."""

        self._prune_waiting_on_children(owner, old)
        self._prune_unreachable_children()

    def _prune_waiting_on_children(self, owner: Graph, old: Node) -> None:
        """Drop children that were only part of a replaced supervisor route."""

        for child_id in getattr(old, "waiting_on", ()):
            owner.children.pop(child_id, None)

    def _prune_unreachable_children(self) -> None:
        """Drop children whose ``parent_node_id`` no longer exists."""

        valid_ids = {node.id for node in self.nodes}
        self.children = {
            aid: child
            for aid, child in self.children.items()
            if child.parent_node_id in valid_ids
        }
        for child in self.children.values():
            child._prune_unreachable_children()

    def _truncate_ancestors_waiting_on(self, agent_id: str) -> None:
        """Drop stale parent resume states that depended on ``agent_id``."""

        child = self[agent_id]
        if child.parent_agent_id is None:
            return

        parent = self[child.parent_agent_id]
        wait_index = None
        for index, node in enumerate(parent.nodes):
            if agent_id in getattr(node, "waiting_on", ()):
                wait_index = index

        if wait_index is None:
            return

        parent.nodes = parent.nodes[: wait_index + 1]
        if parent.parent_agent_id is not None:
            self._truncate_ancestors_waiting_on(parent.agent_id)

    def add_child(self, child: Graph) -> Graph:
        """Attach (or replace) a sub-agent under this graph."""
        self.children[child.agent_id] = child
        return child

    def remove_child(self, agent_id: str) -> Graph:
        """Drop a sub-agent and return it."""
        return self.children.pop(agent_id)

    def update(self, **fields: Any) -> Graph:
        """Bulk-assign top-level fields (``query``, ``config``, ``model``, …)."""
        for key, value in fields.items():
            if not hasattr(self, key):
                raise AttributeError(f"Graph has no field {key!r}")
            setattr(self, key, value)
        return self

    def copy(self, *, deep: bool = True) -> Graph:
        """Return a copy of this graph. ``deep`` copies nodes + subtree."""
        from copy import deepcopy

        return deepcopy(self) if deep else replace(self)

    def inject(
        self,
        *,
        target: str | re.Pattern[str] | Callable[[Graph], Iterable[str | Graph]],
        node: Node,
        mode: str = "append",
    ) -> Graph:
        """Return a new graph with ``node`` injected at ``target``.

        ``target`` may be an exact agent id, a regex/pattern over agent ids,
        or a callable returning agent ids / subgraphs. Only append mode is
        supported for now.
        """
        from rlmflow.graph.injection import inject

        return inject(self, target=target, node=node, mode=mode)

    def inject_output(
        self,
        *,
        target: str | re.Pattern[str] | Callable[[Graph], Iterable[str | Graph]],
        output: str,
        content: str | None = None,
    ) -> Graph:
        from rlmflow.graph.injection import inject_output

        return inject_output(
            self,
            target=target,
            output=output,
            content=content,
        )

    def _resolve_injection_targets(
        self, target: str | re.Pattern[str] | Callable[[Graph], Iterable[str | Graph]]
    ) -> list[Graph]:
        from rlmflow.graph.injection import resolve_injection_targets

        return resolve_injection_targets(self, target)

    # ── rendering ────────────────────────────────────────────────────

    def tree(self) -> str:
        from rlmflow.utils.viewer import graph_tree

        return graph_tree(self)

    def session(self, *, include_system: bool = False) -> str:
        from rlmflow.utils.viewer import graph_session

        return graph_session(self, include_system=include_system)

    def transcript(
        self, agent_id: str | None = None, *, include_system: bool = True
    ) -> str:
        from rlmflow.utils.viewer import agent_transcript

        target = self[agent_id] if agent_id else self
        return agent_transcript(target, include_system=include_system)

    def plot(self, kind: str = "graph", **kwargs: Any) -> Any:
        from rlmflow.utils.viewer import graph_plot

        return graph_plot(self, kind, **kwargs)

    def save_image(self, path: str | Path, **kwargs: Any) -> Path:
        from rlmflow.utils.viewer import save_image

        return save_image(self, path, **kwargs)

    def save_html(self, path: str | Path, **kwargs: Any) -> Path:
        from rlmflow.utils.viewer import save_html

        return save_html([self], path, **kwargs)

    # ── persistence ──────────────────────────────────────────────────

    def meta_dict(self) -> dict[str, Any]:
        """Flat per-agent invariants — what gets persisted to ``agent.json``."""
        return {
            "agent_id": self.agent_id,
            "depth": self.depth,
            "query": self.query,
            "system_prompt": self.system_prompt,
            "config": dict(self.config),
            "workspace": (
                self.workspace.model_dump(mode="json") if self.workspace else None
            ),
            "runtime": (self.runtime.model_dump(mode="json") if self.runtime else None),
            "model": self.model,
            "branch_id": self.branch_id,
            "parent_agent_id": self.parent_agent_id,
            "parent_node_id": self.parent_node_id,
        }

    @classmethod
    def from_meta_dict(
        cls,
        data: dict[str, Any],
        *,
        nodes: Iterable[Node] = (),
        children: dict[str, Graph] | None = None,
    ) -> Graph:
        """Build a :class:`Graph` from a flat agent dict + nodes + children."""
        return cls(
            agent_id=data["agent_id"],
            depth=data.get("depth", 0),
            query=data.get("query", ""),
            system_prompt=data.get("system_prompt", ""),
            config=dict(data.get("config") or {}),
            workspace=(
                WorkspaceRef.model_validate(data["workspace"])
                if data.get("workspace")
                else None
            ),
            runtime=(
                RuntimeRef.model_validate(data["runtime"])
                if data.get("runtime")
                else None
            ),
            model=data.get("model"),
            branch_id=data.get("branch_id", "main"),
            parent_agent_id=data.get("parent_agent_id"),
            parent_node_id=data.get("parent_node_id"),
            nodes=list(nodes),
            children=dict(children or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Recursive JSON dump of the whole subtree."""
        return {
            **self.meta_dict(),
            "nodes": [s.to_dict() for s in self.nodes],
            "children": {aid: c.to_dict() for aid, c in self.children.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Graph:
        raw_nodes = data.get("nodes", data.get("states", []))
        return cls.from_meta_dict(
            data,
            nodes=[parse_node_obj(s) for s in raw_nodes],
            children={
                aid: cls.from_dict(child)
                for aid, child in (data.get("children") or {}).items()
            },
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> Graph:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ── dunder ───────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[str]:
        """Iterate agent ids in the subtree (insertion order)."""
        for g in self.walk():
            yield g.agent_id

    def __len__(self) -> int:
        return sum(1 for _ in self.walk())

    def __repr__(self) -> str:
        return (
            f"Graph(agent_id={self.agent_id!r}, depth={self.depth}, "
            f"nodes={len(self.nodes)}, children={len(self.children)})"
        )


def _path_prefixes(start: str, full: str) -> Iterator[str]:
    """Yield intermediate dotted prefixes between ``start`` and ``full``.

    For ``start="root"`` and ``full="root.a.b.c"`` yields
    ``"root.a"``, ``"root.a.b"``, ``"root.a.b.c"``.
    """
    if not full.startswith(start + "."):
        return
    rest = full[len(start) + 1 :].split(".")
    cur = start
    for piece in rest:
        cur = f"{cur}.{piece}"
        yield cur


def _node_for_replacement(old: Node, new: Node) -> Node:
    """Stamp ``new`` into ``old``'s graph position with a fresh id."""

    fields = new.model_dump(exclude={"id", "agent_id", "seq"}, mode="python")
    return new.__class__(agent_id=old.agent_id, seq=old.seq, **fields)


__all__ = [
    "AgentsView",
    "Edge",
    "EdgesView",
    "Graph",
    "NodesView",
    "RuntimeRef",
    "WorkspaceRef",
]
