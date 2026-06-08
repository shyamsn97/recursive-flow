from __future__ import annotations

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
    assert graph.current().result == "plain ok"


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

    assert any(is_errored(node) for node in graph.nodes)
    assert graph.result() == {"score": 3}
    assert graph.current().result == '{"score":3}'
