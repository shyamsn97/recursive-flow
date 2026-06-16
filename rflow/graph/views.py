"""Flat query and edge views over a recursive :class:`~rflow.graph.graph.Graph`.

These are pure, read-mostly helpers — no engine or runtime state. ``graph.all_nodes``
flattens every node in the subtree; ``graph.edges`` derives the flow- and spawn-edges
between them. Both walk the live graph on each access, so they always reflect the
current trajectory.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from rflow.graph.graph import Graph, Node

_MISSING = object()


class Edge(NamedTuple):
    """A derived flow- or spawn-edge between two nodes."""

    from_: str
    to: str
    kind: str  # "flows_to" | "spawns"


class NodesView:
    """``graph.all_nodes`` — flat view over every node in the subtree."""

    __slots__ = ("_g",)

    def __init__(self, g: Graph) -> None:
        self._g = g

    def _iter(self) -> Iterator[Node]:
        for g in self._g.walk():
            yield from g.nodes

    def __iter__(self) -> Iterator[Node]:
        return self._iter()

    def __len__(self) -> int:
        return sum(1 for _ in self._iter())

    def __contains__(self, node_id: object) -> bool:
        return any(n.id == node_id for n in self._iter())

    def __repr__(self) -> str:
        return f"NodesView({len(self)} nodes)"

    def find(self, node_id: str) -> Node | None:
        """Return the node with ``node_id`` anywhere in the subtree, or ``None``."""
        for n in self._iter():
            if n.id == node_id:
                return n
        return None

    def _locate(self, node_id: str) -> tuple[Graph, int]:
        for g in self._g.walk():
            for i, n in enumerate(g.nodes):
                if n.id == node_id:
                    return g, i
        raise KeyError(node_id)

    def replace(self, node_id: str, new_node: Node) -> Node:
        """Find a node anywhere in the subtree and swap it in place."""
        g, i = self._locate(node_id)
        g.nodes[i] = new_node
        return new_node

    def update(self, node_id: str, **changes: Any) -> Node:
        """Apply ``changes`` to the node with ``node_id`` anywhere in subtree."""
        g, i = self._locate(node_id)
        g.nodes[i] = g.nodes[i].update(**changes)
        return g.nodes[i]

    def remove(self, node_id: str) -> Node:
        """Drop a node from the subtree by id and return it."""
        g, i = self._locate(node_id)
        return g.nodes.pop(i)

    def where(
        self,
        predicate: Callable[[Node], bool] | None = None,
        /,
        **filters: Any,
    ) -> list[Node]:
        return _filter(self, predicate, filters)

    def queries(self) -> list[Node]:
        """Bootstrap user queries (``type == "user_query"``)."""
        return self.where(type="user_query")

    def llm_actions(self) -> list[Node]:
        """LLM action records (``type == "llm_action"``)."""
        return self.where(type="llm_action")

    def llm_outputs(self) -> list[Node]:
        """LLM replies (``type == "llm_output"``)."""
        return self.where(type="llm_output")

    def exec_actions(self) -> list[Node]:
        """Code-execution actions (``type == "exec_action"``)."""
        return self.where(type="exec_action")

    def resume_actions(self) -> list[Node]:
        """Resume actions (``type == "resume_action"``)."""
        return self.where(type="resume_action")

    def observations(self) -> list[Node]:
        """:class:`ExecOutput` nodes that were not produced by a resume."""
        return [
            n
            for n in self._iter()
            if n.type == "exec_output" and not getattr(n, "resumed_from", None)
        ]

    def supervising(self) -> list[Node]:
        """Yielded code observations (``type == "supervising_output"``)."""
        return self.where(type="supervising_output")

    def resumes(self) -> list[Node]:
        """Code observations produced by a :class:`ResumeAction`."""
        return [
            n
            for n in self._iter()
            if n.type
            in ("exec_output", "supervising_output", "error_output", "done_output")
            and bool(getattr(n, "resumed_from", None))
        ]

    def results(self) -> list[Node]:
        """Terminal results (``type == "done_output"``)."""
        return self.where(type="done_output")

    def errors(self) -> list[Node]:
        """Error observations (``type == "error_output"``)."""
        return self.where(type="error_output")


class EdgesView:
    """``graph.edges`` — derived flow + spawn edges across the subtree.

    ``flows_to`` edges connect consecutive nodes *within* one agent's trajectory;
    ``spawns`` edges connect a parent's spawning node to a child's bootstrap query.
    """

    __slots__ = ("_g",)

    def __init__(self, g: Graph) -> None:
        self._g = g

    def _build(self) -> list[Edge]:
        out: list[Edge] = []
        for g in self._g.walk():
            for prev, curr in zip(g.nodes, g.nodes[1:]):
                out.append(Edge(from_=prev.id, to=curr.id, kind="flows_to"))
            for child in g.children.values():
                if child.parent_node_id and child.nodes:
                    out.append(
                        Edge(
                            from_=child.parent_node_id,
                            to=child.nodes[0].id,
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


__all__ = ["Edge", "EdgesView", "NodesView"]
