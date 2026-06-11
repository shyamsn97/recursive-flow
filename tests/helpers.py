"""Shared test scaffolding for engine-style tests."""

from __future__ import annotations

from rflow import Graph, LLMClient, FlowConfig, RecursiveFlow
from rflow.runtime.local import LocalRuntime


class StaticLLM(LLMClient):
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def chat(self, messages, *args, **kwargs) -> str:
        return self.reply


def run_to_completion(agent: RecursiveFlow, graph: Graph) -> Graph:
    while not graph.finished:
        graph = agent.step(graph)
    return graph


def make_agent(reply: str = '```repl\ndone("ok")\n```', **config_kwargs) -> RecursiveFlow:
    config_kwargs.setdefault("max_iterations", 3)
    return RecursiveFlow(
        StaticLLM(reply),
        runtime=LocalRuntime(),
        config=FlowConfig(**config_kwargs),
    )


__all__ = ["StaticLLM", "make_agent", "run_to_completion"]
