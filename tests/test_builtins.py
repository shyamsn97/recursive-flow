"""Tests for REPL builtins exposed to agent code."""

from __future__ import annotations

from inspect import Parameter, signature

import pytest

from rlmflow.graph.handles import ChildHandle
from rlmflow.tools.builtins import make_delegate, make_llm_query_batched


def test_rlm_delegate_is_keyword_only():
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

    with pytest.raises(TypeError):
        delegate("child", "task", "")

    handle = delegate(name="child", query="task", context="payload")
    assert handle.agent_id == "root.child"
    assert spawned == [("child", "task", "payload")]


def test_llm_query_batched_validates_list_shape():
    calls: list[tuple[list[str], str]] = []

    def query_batch(prompts: list[str], *, model: str = "default") -> list[str]:
        calls.append((prompts, model))
        return [prompt.upper() for prompt in prompts]

    llm_query_batched = make_llm_query_batched(query_batch)

    params = signature(llm_query_batched).parameters
    assert params["model"].kind is Parameter.KEYWORD_ONLY

    assert llm_query_batched(["a", "b"], model="fast") == ["A", "B"]
    assert calls == [(["a", "b"], "fast")]

    assert llm_query_batched([]) == []

    with pytest.raises(TypeError):
        llm_query_batched("not a list")
    with pytest.raises(TypeError):
        llm_query_batched(["ok", 3])
