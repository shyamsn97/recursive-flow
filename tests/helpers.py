"""Shared scaffolding for the minimal-stack (``rflow.Flow`` / ``rflow.Graph``) tests."""

from __future__ import annotations

from collections.abc import Callable

from rflow import Flow, Graph, LLMClient


class ScriptedLLM(LLMClient):
    """Pick a reply from the live message list via a callback.

    ``reply_for`` receives the chat messages the engine built for this turn and
    returns the assistant string. Stateless by design so it can be reused across
    parallel agents in one run.
    """

    def __init__(self, reply_for: Callable[[list[dict[str, str]]], str]) -> None:
        self._reply_for = reply_for
        self.calls = 0

    def chat(self, messages: list[dict[str, str]], *args, **kwargs) -> str:
        self.calls += 1
        return self._reply_for(messages)


class StubLLM(ScriptedLLM):
    """Always returns the same reply (good for one-shot agents)."""

    def __init__(self, reply: str = '```repl\ndone("ok")\n```') -> None:
        super().__init__(lambda _messages: reply)


def make_flow(reply: str = '```repl\ndone("ok")\n```', **kwargs) -> Flow:
    kwargs.setdefault("max_iters", 5)
    kwargs.setdefault("max_depth", 2)
    return Flow(StubLLM(reply), **kwargs)


def run_to_completion(
    flow: Flow, query: str, inputs: dict[str, str] | None = None, **start_kwargs
) -> Graph:
    """Drive a fresh run to completion and return the final root graph.

    Extra keyword args (e.g. ``output_schema=...``) are forwarded to ``start``.
    Capped so a misbehaving stub can't hang the suite (``pytest-timeout`` is the
    backstop, but failing the assertion gives a clearer message).
    """
    flow.start(query, inputs, **start_kwargs)
    steps = 0
    while not flow.graph.finished:
        flow.step()
        steps += 1
        assert steps < 200, "run did not finish within 200 steps"
    return flow.graph


def types(graph: Graph) -> list[str]:
    """Node types of one agent's trajectory, in order."""
    return [n.type for n in graph.nodes]


def first_user_text(messages: list[dict[str, str]]) -> str:
    """The first ``user`` message content (an agent's bootstrap task line)."""
    return next((m["content"] for m in messages if m["role"] == "user"), "")


__all__ = [
    "ScriptedLLM",
    "StubLLM",
    "make_flow",
    "run_to_completion",
    "types",
    "first_user_text",
]
