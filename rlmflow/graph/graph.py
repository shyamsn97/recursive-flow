"""The :class:`Graph` data model.

A :class:`Graph` is **one agent**, mutable. It holds:

* the agent's per-run invariants (``agent_id``, ``depth``, ``query``,
  ``system_prompt``, ``config``, ``workspace``, ``runtime``, ``model``,
  ``branch_id``, ``parent_agent_id``, ``parent_node_id``) as flat fields;
* ``states`` — this agent's trajectory of :class:`Node` instances;
* ``children`` — a ``dict[str, Graph]`` of sub-agents spawned from this one.

Recursion lives in ``children``. Indexing by id (``graph[aid]``) walks the
tree; ``graph.agents`` is a flat :class:`Mapping` view over every agent in
the subtree; ``graph.nodes`` / ``graph.edges`` are flat views over every
node / derived edge in the subtree.

Per-state payload lives on :class:`Node`. Per-agent invariants live on
``Graph``. There is no ``AgentMeta`` class — its fields are inlined here.
There is no stored ``Edge`` list — flow / spawn edges are derived from
the recursive structure on demand.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, NamedTuple

from pydantic import BaseModel

from rlmflow.graph.node import (
    ActionNode,
    ErrorNode,
    Node,
    ObservationNode,
    QueryNode,
    ResultNode,
    ResumeNode,
    SupervisingNode,
    parse_node_obj,
)

_MISSING = object()


# ── refs (small Pydantic models for serializable external pointers) ──


class WorkspaceRef(BaseModel):
    """Serializable reference to branch-local workspace storage."""

    root: str
    branch_id: str = "main"

    @property
    def context_dir(self) -> Path:
        return Path(self.root) / "context"


class RuntimeRef(BaseModel):
    """Serializable reference to a durable runtime / REPL session."""

    id: str


# ── derived edges (for viz) ──────────────────────────────────────────


class Edge(NamedTuple):
    """A derived flow- or spawn-edge between two nodes.

    Edges are *not* stored on a :class:`Graph` — they're computed on
    demand from the recursive structure (``states`` ordering yields
    ``flows_to``; ``children`` + ``parent_node_id`` yield ``spawns``).
    """

    from_: str
    to: str
    kind: str  # "flows_to" | "spawns"


# ── Graph ────────────────────────────────────────────────────────────


@dataclass
class Graph:
    """One agent's view of a run, recursive through ``children``.

    Every field is per-agent invariant (set at spawn) or this agent's
    trajectory. Sub-agents live in ``children``; ``graph[other_aid]``
    or ``graph.agents[other_aid]`` walks the subtree to find them.

    Graphs are mutable — use the editing helpers (``add_state``,
    ``replace_state``, ``remove_state``, ``add_child``, ``remove_child``,
    ``update``) or just assign fields directly. ``graph.nodes`` /
    ``graph.children`` are live views, so mutations show up immediately.
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

    states: list[Node] = field(default_factory=list)
    children: dict[str, Graph] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Be forgiving: callers may pass tuples / iterables.
        if not isinstance(self.states, list):
            self.states = list(self.states)
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
        if actual and actual != self.model_key:
            return f"{self.model_key}:{actual}"
        return self.model_key

    # ── current-state accessors ──────────────────────────────────────

    def current(self) -> Node | None:
        """Latest state of this agent (last by insertion)."""
        return self.states[-1] if self.states else None

    @property
    def finished(self) -> bool:
        cur = self.current()
        return bool(cur and cur.terminal)

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
        """First state of this agent (the :class:`QueryNode` at ``seq=0``)."""
        return self.states[0] if self.states else None

    # ── subtree iteration ────────────────────────────────────────────

    def walk(self) -> Iterator[Graph]:
        """Yield self plus every descendant sub-:class:`Graph`, depth-first."""
        yield self
        for child in self.children.values():
            yield from child.walk()

    def subtree(self) -> list[Graph]:
        """List form of :meth:`walk` (self + all descendants)."""
        return list(self.walk())

    # ── flat views over the subtree (back-compat) ────────────────────

    @property
    def agents(self) -> AgentsView:
        return AgentsView(self)

    @property
    def nodes(self) -> NodesView:
        return NodesView(self)

    @property
    def edges(self) -> EdgesView:
        return EdgesView(self)

    def find(self, node_id: str) -> Node | None:
        """Bare :class:`Node` lookup by id across the whole subtree."""
        for g in self.walk():
            for n in g.states:
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
        inp = sum(getattr(s, "input_tokens", 0) for s in self.states)
        out = sum(getattr(s, "output_tokens", 0) for s in self.states)
        if recursive:
            for child in self.children.values():
                ci, co = child.tokens()
                inp += ci
                out += co
        return inp, out

    def total_tokens(self) -> int:
        i, o = self.tokens()
        return i + o

    # ── editing helpers (mutate in place; return self for chaining) ──

    def add_state(self, node: Node) -> Node:
        """Append a state to this agent's trajectory."""
        self.states.append(node)
        return node

    def replace_state(self, node_id: str, new_node: Node) -> Node:
        """Swap a state on **this** agent by id. Raises ``KeyError`` if absent.

        For subtree-wide replacement, use ``graph.nodes.replace(id, node)``.
        """
        for i, s in enumerate(self.states):
            if s.id == node_id:
                self.states[i] = new_node
                return new_node
        raise KeyError(node_id)

    def update_state(self, node_id: str, **changes: Any) -> Node:
        """Replace a state on this agent with a copy carrying ``changes``."""
        for i, s in enumerate(self.states):
            if s.id == node_id:
                self.states[i] = s.update(**changes)
                return self.states[i]
        raise KeyError(node_id)

    def remove_state(self, node_id: str) -> Node:
        """Drop a state by id and return it. Raises ``KeyError`` if absent."""
        for i, s in enumerate(self.states):
            if s.id == node_id:
                return self.states.pop(i)
        raise KeyError(node_id)

    def pop_state(self) -> Node:
        """Drop and return the most recent state of this agent."""
        return self.states.pop()

    def clear_states(self) -> Graph:
        self.states.clear()
        return self

    def add_child(self, child: Graph) -> Graph:
        """Attach (or replace) a sub-agent under this graph."""
        self.children[child.agent_id] = child
        return child

    def remove_child(self, agent_id: str) -> Graph:
        """Drop a sub-agent and return it. Raises ``KeyError`` if absent."""
        if agent_id not in self.children:
            raise KeyError(agent_id)
        return self.children.pop(agent_id)

    def update(self, **fields: Any) -> Graph:
        """Bulk-assign top-level fields (``query``, ``config``, ``model``, …)."""
        for key, value in fields.items():
            if not hasattr(self, key):
                raise AttributeError(f"Graph has no field {key!r}")
            setattr(self, key, value)
        return self

    def copy(self, *, deep: bool = True) -> Graph:
        """Return a copy of this graph. ``deep`` copies states + subtree."""
        from copy import deepcopy

        return deepcopy(self) if deep else replace(self)

    # ── rendering ────────────────────────────────────────────────────

    def tree(self) -> str:
        return _render_tree(self)

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
        states: Iterable[Node] = (),
        children: dict[str, Graph] | None = None,
    ) -> Graph:
        """Build a :class:`Graph` from a flat agent dict + states + children."""
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
            states=list(states),
            children=dict(children or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Recursive JSON dump of the whole subtree."""
        return {
            **self.meta_dict(),
            "states": [s.to_dict() for s in self.states],
            "children": {aid: c.to_dict() for aid, c in self.children.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Graph:
        return cls.from_meta_dict(
            data,
            states=[parse_node_obj(s) for s in data.get("states", [])],
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
            f"states={len(self.states)}, children={len(self.children)})"
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


# ── flat views over the subtree ──────────────────────────────────────


class NodesView:
    """``graph.nodes`` — flat view over every node in the subtree."""

    __slots__ = ("_g",)

    def __init__(self, g: Graph) -> None:
        self._g = g

    def _iter(self) -> Iterator[Node]:
        for g in self._g.walk():
            yield from g.states

    def __iter__(self) -> Iterator[Node]:
        return self._iter()

    def __len__(self) -> int:
        return sum(1 for _ in self._iter())

    def __contains__(self, node_id: object) -> bool:
        return any(n.id == node_id for n in self._iter())

    def __repr__(self) -> str:
        return f"NodesView({len(self)} nodes)"

    def find(self, node_id: str) -> Node | None:
        return self._g.find(node_id)

    # ── mutations across the subtree ─────────────────────────────────

    def replace(self, node_id: str, new_node: Node) -> Node:
        """Find a node anywhere in the subtree and swap it in place."""
        for g in self._g.walk():
            for i, s in enumerate(g.states):
                if s.id == node_id:
                    g.states[i] = new_node
                    return new_node
        raise KeyError(node_id)

    def update(self, node_id: str, **changes: Any) -> Node:
        """Apply ``changes`` to the node with ``node_id`` (anywhere in subtree)."""
        for g in self._g.walk():
            for i, s in enumerate(g.states):
                if s.id == node_id:
                    g.states[i] = s.update(**changes)
                    return g.states[i]
        raise KeyError(node_id)

    def remove(self, node_id: str) -> Node:
        """Drop a node from the subtree by id and return it."""
        for g in self._g.walk():
            for i, s in enumerate(g.states):
                if s.id == node_id:
                    return g.states.pop(i)
        raise KeyError(node_id)

    # ── filters ──────────────────────────────────────────────────────

    def where(
        self,
        predicate: Callable[[Node], bool] | None = None,
        /,
        **filters: Any,
    ) -> list[Node]:
        return _filter(self, predicate, filters)

    def queries(self) -> list[Node]:
        return [n for n in self if isinstance(n, QueryNode)]

    def actions(self) -> list[Node]:
        return [n for n in self if isinstance(n, ActionNode)]

    def observations(self) -> list[Node]:
        return [n for n in self if isinstance(n, ObservationNode)]

    def supervising(self) -> list[Node]:
        return [n for n in self if isinstance(n, SupervisingNode)]

    def resumes(self) -> list[Node]:
        return [n for n in self if isinstance(n, ResumeNode)]

    def results(self) -> list[Node]:
        return [n for n in self if isinstance(n, ResultNode)]

    def errors(self) -> list[Node]:
        return [n for n in self if isinstance(n, ErrorNode)]


class AgentsView(Mapping[str, Graph]):
    """``graph.agents`` — Mapping[agent_id, sub-Graph] across the subtree."""

    __slots__ = ("_g",)

    def __init__(self, g: Graph) -> None:
        self._g = g

    def __iter__(self) -> Iterator[str]:
        for g in self._g.walk():
            yield g.agent_id

    def __len__(self) -> int:
        return sum(1 for _ in self._g.walk())

    def __getitem__(self, aid: str) -> Graph:
        return self._g[aid]

    def __repr__(self) -> str:
        return f"AgentsView({list(self)})"


class EdgesView:
    """``graph.edges`` — derived flow + spawn edges across the subtree.

    Recomputed on every call so it stays consistent with a mutable graph.
    """

    __slots__ = ("_g",)

    def __init__(self, g: Graph) -> None:
        self._g = g

    def _build(self) -> list[Edge]:
        out: list[Edge] = []
        for g in self._g.walk():
            for prev, curr in zip(g.states, g.states[1:]):
                out.append(Edge(from_=prev.id, to=curr.id, kind="flows_to"))
            for child in g.children.values():
                if child.parent_node_id and child.states:
                    out.append(
                        Edge(
                            from_=child.parent_node_id,
                            to=child.states[0].id,
                            kind="spawns",
                        )
                    )
        return out

    def __iter__(self) -> Iterator[Edge]:
        return iter(self._build())

    def __len__(self) -> int:
        return len(self._build())

    def __repr__(self) -> str:
        return f"EdgesView({len(self)} edges)"

    def where(
        self,
        predicate: Callable[[Edge], bool] | None = None,
        /,
        **filters: Any,
    ) -> list[Edge]:
        return _filter(self._build(), predicate, filters)

    def spawns(self) -> list[Edge]:
        return [e for e in self._build() if e.kind == "spawns"]

    def flows_to(self) -> list[Edge]:
        return [e for e in self._build() if e.kind == "flows_to"]


def _filter(items, predicate, filters):
    def matches(x):
        if predicate is not None and not predicate(x):
            return False
        return all(getattr(x, k, _MISSING) == v for k, v in filters.items())

    return [x for x in items if matches(x)]


# ── tree rendering ──────────────────────────────────────────────────


def _render_tree(graph: Graph) -> str:
    lines: list[str] = []

    def walk(g: Graph, indent: str) -> None:
        head = f"{indent}● {g.agent_id} ({g.model_label})"
        if g.query:
            head += f" — {_short(g.query, 60)}"
        lines.append(head)

        # Decide where each child attaches visually. Prefer the supervising
        # state whose ``waiting_on`` lists the child — that's the lifecycle
        # state during which the child actually runs, so the children block
        # belongs *under* it. Fall back to ``parent_node_id`` (the action
        # that physically spawned it) when no supervising state is waiting
        # on the child (e.g. mid-run, before the wait was committed).
        state_ids = {s.id for s in g.states}
        sup_for_agent: dict[str, str] = {}
        for s in g.states:
            if isinstance(s, SupervisingNode):
                for aid in s.waiting_on:
                    sup_for_agent[aid] = s.id

        attach_at: dict[str, list[Graph]] = {}
        unplaced: list[Graph] = []
        for child in g.children.values():
            key = sup_for_agent.get(child.agent_id) or (
                child.parent_node_id if child.parent_node_id in state_ids else None
            )
            if key:
                attach_at.setdefault(key, []).append(child)
            else:
                unplaced.append(child)

        for s in g.states:
            lines.append(f"{indent}  - [{s.seq:>2}] {_label(s)}")
            for child in attach_at.get(s.id, []):
                walk(child, indent + "    ")
        for child in unplaced:
            walk(child, indent + "    ")

    walk(graph, "")
    return "\n".join(lines)


def _label(s: Node) -> str:
    t = s.type
    if isinstance(s, SupervisingNode) and s.waiting_on:
        return f"{t} waiting_on={s.waiting_on}"
    if isinstance(s, ResultNode) and s.result:
        return f"{t} -> {_short(s.result, 60)}"
    if isinstance(s, ErrorNode):
        return f"{t} ({s.error or 'error'})"
    if isinstance(s, ActionNode) and s.code:
        return f"{t} code={_short(s.code, 40)}"
    if isinstance(s, ObservationNode) and s.content:
        return f"{t} {_short(s.content, 60)}"
    return t


def _short(text: str, n: int) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[: n - 1] + "…"


__all__ = [
    "AgentsView",
    "Edge",
    "EdgesView",
    "Graph",
    "NodesView",
    "RuntimeRef",
    "WorkspaceRef",
]
