"""Typed agent states.

A :class:`Node` is one immutable state in an agent's trajectory. Subclasses
encode the state kind via the ``type`` discriminator: ``query``, ``action``,
``observation``, ``supervising``, ``resume``, ``result``, ``error``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


def new_id() -> str:
    return f"n_{uuid4().hex[:12]}"


class Node(BaseModel):
    """One immutable state in an agent's trajectory.

    Subclasses carry the state payload (content / code / output / reply /
    result / error / token deltas). Agent-invariant data lives directly
    on :class:`~rlmflow.graph.Graph`; cross-agent topology is recovered
    from the recursive ``Graph.children`` structure (no separate edge
    objects are stored).
    """

    model_config = ConfigDict(frozen=True)

    type: str
    id: str = Field(default_factory=new_id)
    agent_id: str = "root"
    seq: int = 0

    @property
    def terminal(self) -> bool:
        return False

    def update(self, **changes: Any) -> Node:
        return self.model_copy(update=changes)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


class ObservationNode(Node):
    type: Literal["observation"] = "observation"
    content: str = ""
    code: str | None = None
    output: str | None = None


class QueryNode(ObservationNode):
    type: Literal["query"] = "query"


class ErrorNode(ObservationNode):
    type: Literal["error"] = "error"
    error: str = ""


class ResumeNode(ObservationNode):
    type: Literal["resume"] = "resume"
    resumed_from: list[str] = Field(default_factory=list)


class ResultNode(ObservationNode):
    type: Literal["result"] = "result"
    result: str = ""

    @property
    def terminal(self) -> bool:
        return True


class ActionNode(Node):
    type: Literal["action"] = "action"
    reply: str = ""
    code: str = ""
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class SupervisingNode(ActionNode):
    type: Literal["supervising"] = "supervising"
    output: str = ""
    waiting_on: list[str] = Field(default_factory=list)


# ── parser ───────────────────────────────────────────────────────────


NodeUnion = Annotated[
    Union[
        QueryNode,
        ErrorNode,
        ResumeNode,
        ResultNode,
        SupervisingNode,
        ActionNode,
        ObservationNode,
    ],
    Field(discriminator="type"),
]


_NODE_ADAPTER: TypeAdapter[Node] = TypeAdapter(NodeUnion)


def parse_node_obj(data: dict) -> Node:
    return _NODE_ADAPTER.validate_python(data)


__all__ = [
    "ActionNode",
    "ErrorNode",
    "Node",
    "ObservationNode",
    "QueryNode",
    "ResultNode",
    "ResumeNode",
    "SupervisingNode",
    "new_id",
    "parse_node_obj",
]
