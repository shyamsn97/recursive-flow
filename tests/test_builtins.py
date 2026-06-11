"""Tests for REPL builtins exposed to agent code."""

from __future__ import annotations

from inspect import Parameter, signature

import pytest

from rflow import LLMClient, FlowConfig, RecursiveFlow
from rflow.graph.handles import ChildHandle
from rflow.runtime.local import LocalRuntime
from rflow.tools import get_repl_tools, tool
from rflow.tools.builtins import make_delegate


def test_flow_delegate_is_keyword_only():
    spawned: list[tuple[str, str, str]] = []

    def spawn_child(parent_agent_id, parent_node_id, name, query, context, **kwargs):
        spawned.append((name, query, context))
        return ChildHandle(f"{parent_agent_id}.{name}")

    delegate = make_delegate(
        spawn_child,
        {"AGENT_ID": "root", "PARENT_NODE_ID": "node-1"},
    )

    params = signature(delegate).parameters
    assert params["name"].kind is Parameter.KEYWORD_ONLY
    assert params["query"].kind is Parameter.KEYWORD_ONLY
    assert params["context"].kind is Parameter.KEYWORD_ONLY
    assert "max_iterations" not in params

    with pytest.raises(TypeError):
        delegate("child", "task", "")
    with pytest.raises(TypeError):
        delegate(name="child", query="task", context="payload", max_iterations=1)

    handle = delegate(name="child", query="task", context="payload")
    assert handle.agent_id == "root.child"
    assert spawned == [("child", "task", "payload")]


class _EchoLLM(LLMClient):
    def chat(self, messages, *args, **kwargs) -> str:
        return messages[-1]["content"].upper()


def test_llm_query_batched_validates_list_shape(tmp_path):
    agent = RecursiveFlow(
        _EchoLLM(),
        runtime=LocalRuntime(workspace=tmp_path / "workspace"),
        config=FlowConfig(max_concurrency=1),
    )

    params = signature(agent.llm_query_batched).parameters
    assert params["model"].kind is Parameter.KEYWORD_ONLY

    assert agent.llm_query_batched(["a", "b"]) == ["A", "B"]

    assert agent.llm_query_batched([]) == []

    with pytest.raises(TypeError):
        agent.llm_query_batched("not a list")
    with pytest.raises(TypeError):
        agent.llm_query_batched(["ok", 3])


def test_llm_query_batched_validates_structured_outputs(tmp_path):
    class _InventoryLLM(LLMClient):
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def chat(self, messages, *args, **kwargs) -> str:
            prompt = messages[-1]["content"]
            self.prompts.append(prompt)
            if "apple" in prompt:
                return '{"name":"apple","count":2}'
            return '{"name":"orange","count":3}'

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": ["name", "count"],
        "additionalProperties": False,
    }
    llm = _InventoryLLM()
    agent = RecursiveFlow(
        llm,
        runtime=LocalRuntime(workspace=tmp_path / "workspace"),
        config=FlowConfig(max_concurrency=2),
    )

    results = agent.llm_query_batched(
        ["extract apple count", "extract orange count"],
        output_schema=schema,
    )

    assert results == [
        {"name": "apple", "count": 2},
        {"name": "orange", "count": 3},
    ]
    assert all("Return only a JSON value matching this JSON Schema" in p for p in llm.prompts)


def test_get_repl_tools_lets_local_tool_call_visible_tool(tmp_path):
    @tool("Return a greeting.")
    def greet(name: str) -> str:
        return f"hello {name}"

    @tool("Call another visible tool.")
    def call_greet(name: str) -> str:
        return get_repl_tools()["greet"](name)

    runtime = LocalRuntime(workspace=tmp_path / "workspace")
    runtime.register_tool(greet)
    runtime.register_tool(call_greet)

    assert runtime.execute("print(call_greet('rlm'))") == "hello rlm"


def test_get_repl_tools_hides_internal_primitives_by_default(tmp_path):
    runtime = LocalRuntime(workspace=tmp_path / "workspace")
    RecursiveFlow(_EchoLLM(), runtime=runtime, config=FlowConfig(max_depth=1))

    visible = runtime.execute(
        "from rflow.tools import get_repl_tools\n"
        "tools = get_repl_tools()\n"
        "print('flow_delegate' in tools, 'flow_wait' in tools, 'done' in tools)"
    )
    assert visible == "False False True"

    hidden = runtime.execute(
        "from rflow.tools import get_repl_tools\n"
        "tools = get_repl_tools(include_hidden=True)\n"
        "print('flow_delegate' in tools, 'flow_wait' in tools)"
    )
    assert hidden == "True True"


def test_get_repl_tools_requires_active_context():
    with pytest.raises(RuntimeError, match="No active RecursiveFlow tool context"):
        get_repl_tools()
