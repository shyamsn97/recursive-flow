"""Nested delegation behavior for the typed RLMFlow graph."""

from __future__ import annotations

from rlmflow import (
    Graph,
    LLMClient,
    RLMConfig,
    RLMFlow,
    ResultNode,
    SupervisingNode,
)
from rlmflow.runtime.local import LocalRuntime


class RecursiveLLM(LLMClient):
    def __init__(self, *, max_child_depth: int) -> None:
        self.max_child_depth = max_child_depth
        self.calls: list[str] = []

    def chat(self, messages, *args, **kwargs):
        depth, max_depth = self._depth(messages)
        self.calls.append(f"depth:{depth}")
        if depth < max_depth and depth < self.max_child_depth:
            return (
                "```repl\n"
                'h = delegate("child", "go deeper", "")\n'
                "results = yield wait(h)\n"
                'done(AGENT_ID + "->" + results[0])\n'
                "```"
            )
        return '```repl\ndone("leaf:" + AGENT_ID)\n```'

    @staticmethod
    def _depth(messages: list[dict]) -> tuple[int, int]:
        system = (
            messages[0]["content"]
            if messages and messages[0].get("role") == "system"
            else ""
        )
        marker = "You are at recursion depth **"
        if marker not in system:
            return 0, 0
        rest = system.split(marker, 1)[1]
        depth_text, rest = rest.split("**", 1)
        max_text = rest.split("max **", 1)[1].split("**", 1)[0]
        return int(depth_text), int(max_text)


def _run(agent: RLMFlow, graph: Graph) -> Graph:
    while not graph.finished:
        graph = agent.step(graph)
    return graph


def test_root_can_delegate_to_child_at_depth_one():
    agent = RLMFlow(
        llm_client=RecursiveLLM(max_child_depth=1),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=1),
    )

    final = _run(agent, agent.start("test"))

    assert isinstance(final.current(), ResultNode)
    assert final.result() == "root->leaf:root.child"
    assert final["root.child"].depth == 1


def test_nested_delegation_reaches_grandchild_at_depth_two():
    agent = RLMFlow(
        llm_client=RecursiveLLM(max_child_depth=2),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )

    final = _run(agent, agent.start("test"))

    assert isinstance(final.current(), ResultNode)
    assert final["root.child"].agent_id == "root.child"
    assert final["root.child.child"].depth == 2
    assert final.result() == "root->root.child->leaf:root.child.child"


def test_deep_supervising_chain_completes_at_depth_four():
    """Regression: result propagates through a depth-4 supervising chain."""
    agent = RLMFlow(
        llm_client=RecursiveLLM(max_child_depth=4),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=4),
    )

    final = _run(agent, agent.start("test"))

    assert final.finished
    expected = (
        "root->root.child->root.child.child->root.child.child.child->"
        "leaf:root.child.child.child.child"
    )
    assert final.result() == expected
    assert final["root.child.child.child.child"].depth == 4


def test_max_depth_zero_turns_delegate_into_direct_llm_work():
    agent = RLMFlow(
        llm_client=RecursiveLLM(max_child_depth=3),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=0, max_iterations=3),
    )

    final = _run(agent, agent.start("test"))

    assert isinstance(final.current(), ResultNode)
    assert len(final) == 1  # no spawned agents
    assert final.result() == "leaf:root"


def test_each_step_advances_runnable_agents_once():
    agent = RLMFlow(
        llm_client=RecursiveLLM(max_child_depth=2),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )

    graph = agent.step(agent.start("test"))
    assert isinstance(graph.current(), SupervisingNode)
    assert list(graph.children) == ["root.child"]

    graph = agent.step(graph)
    # Child should now have spawned a grandchild and be supervising.
    assert "root.child" in graph
    assert isinstance(graph["root.child"].current(), SupervisingNode)
    assert list(graph["root.child"].children) == ["root.child.child"]


def test_model_routing_is_stored_on_child_agent_meta():
    class ModelAwareLLM(LLMClient):
        model = "strong-model"

        def chat(self, messages, *args, **kwargs):
            return (
                "```repl\n"
                'h = delegate("worker", "use fast", "", model="fast")\n'
                "results = yield wait(h)\n"
                "done(results[0])\n"
                "```"
            )

    class FastLLM(LLMClient):
        model = "fast-model"

        def chat(self, messages, *args, **kwargs):
            return '```repl\ndone("fast-result")\n```'

    agent = RLMFlow(
        llm_client=ModelAwareLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=1),
        llm_clients={"fast": {"model": FastLLM(), "description": "quick worker"}},
    )

    final = _run(agent, agent.start("test"))

    assert final.result() == "fast-result"
    worker = final["root.worker"]
    assert worker.config["model"] == "fast"
