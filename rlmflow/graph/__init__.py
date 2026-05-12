"""RLMFlow's data model — one recursive class.

* :class:`Graph` — one agent, mutable, with per-agent invariants as flat
  fields, ``states`` as its trajectory, and ``children`` for sub-agents.
  Recursion lives in ``children``; cross-agent navigation goes through
  ``graph[other_aid]`` or ``graph.agents``.
* :class:`AgentsView`, :class:`NodesView`, :class:`EdgesView` — flat
  query / mutation views over the subtree (``graph.agents``,
  ``graph.nodes``, ``graph.edges``).
* :class:`Node` (and subclasses) — one immutable per-state payload.
* :class:`WorkspaceRef`, :class:`RuntimeRef` — serializable handles to
  external systems (branch storage, durable REPL).
* :class:`ChildHandle`, :class:`WaitRequest` — REPL protocol handles
  the engine inspects for delegation / suspension.
"""

from rlmflow.graph.graph import (
    AgentsView,
    Edge,
    EdgesView,
    Graph,
    NodesView,
    RuntimeRef,
    WorkspaceRef,
)
from rlmflow.graph.handles import ChildHandle, WaitRequest
from rlmflow.graph.node import (
    ActionNode,
    ErrorNode,
    Node,
    ObservationNode,
    QueryNode,
    ResultNode,
    ResumeNode,
    SupervisingNode,
    new_id,
    parse_node_obj,
)

__all__ = [
    "ActionNode",
    "AgentsView",
    "ChildHandle",
    "Edge",
    "EdgesView",
    "ErrorNode",
    "Graph",
    "Node",
    "NodesView",
    "ObservationNode",
    "QueryNode",
    "ResultNode",
    "ResumeNode",
    "RuntimeRef",
    "SupervisingNode",
    "WaitRequest",
    "WorkspaceRef",
    "new_id",
    "parse_node_obj",
]
