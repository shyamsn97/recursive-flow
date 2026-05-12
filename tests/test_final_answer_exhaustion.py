"""Iteration-budget exhaustion and explicit ``terminate()`` semantics."""

from __future__ import annotations

from rlmflow import (
    Graph,
    LLMClient,
    LLMUsage,
    RLMConfig,
    RLMFlow,
    ResultNode,
    SupervisingNode,
)
from rlmflow.prompts.messages import FINAL_ANSWER_ACTION
from rlmflow.runtime.local import LocalRuntime


class StallingThenFinalLLM(LLMClient):
    """Stalls with ``x = 1`` until the engine forces a final-answer turn."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_messages: list[dict] = []
        self.last_usage = LLMUsage(input_tokens=1, output_tokens=1)

    def chat(self, messages, *args, **kwargs):
        self.calls += 1
        self.last_messages = list(messages)
        if any("full iteration budget" in m.get("content", "") for m in messages):
            return '```repl\ndone("final answer")\n```'
        return "```repl\nx = 1\n```"


def _run(agent: RLMFlow, graph: Graph) -> Graph:
    while not graph.finished:
        graph = agent.step(graph)
    return graph


def test_exhaustion_runs_one_more_repl_turn_with_final_answer_message():
    llm = StallingThenFinalLLM()
    agent = RLMFlow(
        llm_client=llm,
        runtime=LocalRuntime(),
        config=RLMConfig(max_iterations=1, max_depth=0),
    )

    final = _run(agent, agent.start("answer the question"))

    assert isinstance(final.current(), ResultNode)
    assert final.result() == "final answer"
    assert llm.calls == 2
    user_messages = [
        m["content"] for m in llm.last_messages if m.get("role") == "user"
    ]
    assert user_messages[-1] == FINAL_ANSWER_ACTION


def test_explicit_terminate_marks_agent_and_drives_one_final_turn():
    llm = StallingThenFinalLLM()
    agent = RLMFlow(
        llm_client=llm,
        runtime=LocalRuntime(),
        config=RLMConfig(max_iterations=10, max_depth=0),
    )

    graph = agent.terminate(agent.start("answer the question"))
    assert "root" in agent._terminate_requested
    assert llm.calls == 0

    final = _run(agent, graph)
    assert final.result() == "final answer"
    assert llm.calls == 1


def test_terminate_marks_every_running_agent():
    """``terminate(graph)`` flips every still-running agent into final-answer mode."""

    class DelegatingThenStallingLLM(LLMClient):
        def __init__(self) -> None:
            self.last_usage = LLMUsage(input_tokens=1, output_tokens=1)

        def chat(self, messages, *args, **kwargs):
            prompt = messages[-1]["content"].lower()
            if "full iteration budget" in prompt:
                return '```repl\ndone("forced final")\n```'
            if "child" in prompt:
                return "```repl\ny = 2\n```"
            return (
                "```repl\n"
                'h = delegate("child", "child task", "")\n'
                "r = yield wait(h)\n"
                "done(r[0])\n"
                "```"
            )

    agent = RLMFlow(
        llm_client=DelegatingThenStallingLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_iterations=10, max_depth=2),
    )
    graph = agent.step(agent.start("kickoff"))
    assert isinstance(graph.current(), SupervisingNode)
    assert "root.child" in graph

    graph = agent.terminate(graph)
    assert {"root", "root.child"} <= agent._terminate_requested

    final = _run(agent, graph)
    assert final.finished
    assert final["root.child"].result() == "forced final"
