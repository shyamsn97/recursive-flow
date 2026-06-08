from __future__ import annotations

import pytest
from pydantic import BaseModel

from rlmflow import Graph, LLMClient, RLMConfig, RLMFlow, is_done, is_errored
from rlmflow.runtime.local import LocalRuntime
from tests.helpers import StaticLLM


class WeatherAdvice(BaseModel):
    city: str
    temp_f: float


class ScriptedLLM(LLMClient):
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)

    def chat(self, messages, *args, **kwargs) -> str:
        if not self.replies:
            raise AssertionError("ScriptedLLM ran out of replies")
        return self.replies.pop(0)


def _run_to_completion(agent: RLMFlow, graph: Graph) -> Graph:
    while not graph.finished:
        graph = agent.step(graph)
    return graph


def test_run_with_pydantic_output_schema_returns_model_and_persists_json_result():
    agent = RLMFlow(
        StaticLLM('```repl\ndone({"city": "Austin", "temp_f": 95.5})\n```'),
        runtime=LocalRuntime(),
        config=RLMConfig(max_iterations=3),
    )

    result = agent.run("give weather advice", output_schema=WeatherAdvice)
    graph = agent.session.load_graph()
    terminal = graph.current()

    assert isinstance(result, WeatherAdvice)
    assert result.city == "Austin"
    assert result.temp_f == 95.5
    assert is_done(terminal)
    assert terminal.result == '{"city":"Austin","temp_f":95.5}'
    assert terminal.structured_result == {"city": "Austin", "temp_f": 95.5}
    assert graph.output_schema is None
    assert graph.root.output_schema == terminal.output_schema
    assert all(node.output_schema == terminal.output_schema for node in graph.nodes)
    assert graph.result() == {"city": "Austin", "temp_f": 95.5}


def test_reusing_flow_for_plain_run_clears_root_output_schema():
    agent = RLMFlow(
        ScriptedLLM(
            [
                '```repl\ndone({"city": "Austin", "temp_f": 95.5})\n```',
                '```repl\ndone("plain ok")\n```',
            ]
        ),
        runtime=LocalRuntime(),
        config=RLMConfig(max_iterations=3),
    )

    assert isinstance(agent.run("structured", output_schema=WeatherAdvice), WeatherAdvice)

    result = agent.run("plain")
    graph = agent.session.load_graph()

    assert result == "plain ok"
    assert graph.output_schema is None
    assert all(node.output_schema is None for node in graph.nodes)
    assert graph.current().result == "plain ok"


def test_run_can_continue_finished_graph_with_plain_phase():
    agent = RLMFlow(
        ScriptedLLM(
            [
                '```repl\ndone({"city": "Austin", "temp_f": 95.5})\n```',
                '```repl\ndone("plain followup")\n```',
            ]
        ),
        runtime=LocalRuntime(),
        config=RLMConfig(max_iterations=3),
    )

    assert isinstance(agent.run("structured", output_schema=WeatherAdvice), WeatherAdvice)
    first_graph = agent.session.load_graph()

    result = agent.run("plain", graph=first_graph)
    graph = agent.session.load_graph()

    assert result == "plain followup"
    assert first_graph.current().output_schema is not None
    assert graph.current().output_schema is None
    assert graph.current().result == "plain followup"
    assert any(node.output_schema is not None for node in graph.nodes)


def test_run_can_continue_finished_graph_with_new_structured_phase():
    agent = RLMFlow(
        ScriptedLLM(
            [
                '```repl\ndone("plain first")\n```',
                '```repl\ndone({"city": "Austin", "temp_f": 95.5})\n```',
            ]
        ),
        runtime=LocalRuntime(),
        config=RLMConfig(max_iterations=3),
    )

    assert agent.run("plain") == "plain first"
    first_graph = agent.session.load_graph()

    result = agent.run("structured", graph=first_graph, output_schema=WeatherAdvice)
    graph = agent.session.load_graph()

    assert isinstance(result, WeatherAdvice)
    assert result.city == "Austin"
    assert graph.current().structured_result == {"city": "Austin", "temp_f": 95.5}
    assert graph.current().output_schema is not None


def test_start_from_graph_rejects_unfinished_graph():
    agent = RLMFlow(
        StaticLLM('```repl\ndone("ok")\n```'),
        runtime=LocalRuntime(),
        config=RLMConfig(max_iterations=3),
    )
    graph = agent.start("unfinished")

    with pytest.raises(ValueError, match="requires a finished graph"):
        agent.start("new phase", graph=graph)


def test_invalid_structured_done_records_error_then_repairs():
    schema = {
        "type": "object",
        "properties": {"score": {"type": "integer"}},
        "required": ["score"],
        "additionalProperties": False,
    }
    agent = RLMFlow(
        ScriptedLLM(
            [
                '```repl\ndone({"score": "bad"})\n```',
                '```repl\ndone({"score": 3})\n```',
            ]
        ),
        runtime=LocalRuntime(),
        config=RLMConfig(max_iterations=5),
    )
    graph = agent.start("return a score", output_schema=schema)

    graph = _run_to_completion(agent, graph)

    errors = [node for node in graph.nodes if is_errored(node)]
    assert errors
    assert "StructuredOutputError" in errors[0].output
    assert "Expected JSON Schema" in errors[0].output
    assert '{"score":"bad"}' in errors[0].output
    assert graph.result() == {"score": 3}
    assert graph.current().result == '{"score":3}'


def test_plain_child_under_structured_parent_does_not_inherit_parent_schema():
    agent = RLMFlow(
        StaticLLM('```repl\ndone({"city": "Austin", "temp_f": 95.5})\n```'),
        runtime=LocalRuntime(),
        config=RLMConfig(max_iterations=3),
    )
    graph = agent.start("parent wants weather advice", output_schema=WeatherAdvice)
    parent_node_id = graph.current().id

    handle = agent.spawn_child(
        "root",
        parent_node_id,
        "plain_child",
        "return prose",
        "child context",
    )
    assert not isinstance(handle, str)
    graph = agent.session.load_graph()
    child = graph.agents[handle.agent_id]

    assert graph.root.output_schema is not None
    assert child.root.output_schema is None
    assert child.output_schema is None


def test_structured_child_gets_its_own_schema():
    child_schema = {
        "type": "object",
        "properties": {"score": {"type": "integer"}},
        "required": ["score"],
    }
    agent = RLMFlow(
        StaticLLM('```repl\ndone({"city": "Austin", "temp_f": 95.5})\n```'),
        runtime=LocalRuntime(),
        config=RLMConfig(max_iterations=3),
    )
    graph = agent.start("parent wants weather advice", output_schema=WeatherAdvice)
    parent_node_id = graph.current().id

    handle = agent.spawn_child(
        "root",
        parent_node_id,
        "structured_child",
        "return a score",
        "child context",
        output_schema=child_schema,
    )
    assert not isinstance(handle, str)
    graph = agent.session.load_graph()
    child = graph.agents[handle.agent_id]

    assert child.root.output_schema == child_schema
    assert child.output_schema is None
