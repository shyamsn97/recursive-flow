"""PromptBuilder structural tests."""

from __future__ import annotations

from rflow import FlowConfig, RecursiveFlow
from rflow.prompts.builder import PromptBuilder
from rflow.prompts.default import DEFAULT_BUILDER
from rflow.runtime.local import LocalRuntime
from tests.helpers import StaticLLM, make_agent


def test_default_builder_has_expected_section_shape():
    # role/strategy/format/examples/final render headless and back-to-back
    # (byte-identical to the single-block narrative); `tools` and `status` are
    # callable sections filled from the current engine and graph.
    assert DEFAULT_BUILDER.names == [
        "role",
        "strategy",
        "format",
        "examples",
        "final",
        "structured-output",
        "tools",
        "status",
    ]


def test_prompt_builder_arranges_sections_by_insertion_and_anchors():
    base = (
        PromptBuilder()
        .section("role", "role body", title="Role")
        .section("tools", "tools body", title="Tools")
        .section("status", "status body", title="Status")
    )
    derived = (
        base.section("strategy", "strategy body", title="Strategy", after="role")
        .section("preamble", "preamble body", title="Preamble", before="role")
        .update("tools", "updated tools")
    )

    assert base.names == ["role", "tools", "status"]
    assert derived.names == ["preamble", "role", "strategy", "tools", "status"]


def test_prompt_builder_renders_callable_sections_with_engine_and_graph():
    class Engine:
        label = "engine"

    class Graph:
        agent_id = "root.child"

    prompt = PromptBuilder().section(
        "memory",
        lambda engine, graph: f"{engine.label}:{graph.agent_id}",
        title="Memory",
    )

    rendered = prompt.build(Engine(), Graph())

    assert rendered == "## Memory\n\nengine:root.child\n"


def test_prompt_builder_overrides_win_over_callable_sections():
    prompt = PromptBuilder().section(
        "memory",
        lambda engine, graph: "dynamic",
        title="Memory",
    )

    rendered = prompt.build(memory="forced")

    assert rendered == "## Memory\n\nforced\n"


def test_default_prompt_skips_structured_output_section_without_schema():
    graph = make_agent().start("say ok")

    assert "## Structured Output" not in graph.system_prompt


def test_default_prompt_documents_child_output_schema_specs():
    graph = make_agent().start("say ok")

    assert "structured one-shot batch" in graph.system_prompt
    assert "llm_query_batched(" in graph.system_prompt
    assert "output_schema=fact_schema" in graph.system_prompt
    assert "structured child results" in graph.system_prompt
    assert "`output_schema`" in graph.system_prompt
    assert '"output_schema": item_schema' in graph.system_prompt
    assert "JSON Schema dict" in graph.system_prompt
    assert "validated JSON-compatible values" in graph.system_prompt


def test_structured_prompt_hints_can_be_disabled():
    agent = RecursiveFlow(
        StaticLLM('```repl\ndone("ok")\n```'),
        runtime=LocalRuntime(),
        config=FlowConfig(enable_structured_output=False),
    )

    graph = agent.start("say ok")

    assert "structured one-shot batch" not in graph.system_prompt
    assert "structured child results" not in graph.system_prompt
    assert "output_schema" not in graph.system_prompt
    assert "validated JSON-compatible values" not in graph.system_prompt
    assert "**Example 4 — multi-file app fanout.**" in graph.system_prompt


def test_default_prompt_includes_structured_output_section_with_schema():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    graph = make_agent().start("say ok", output_schema=schema)

    assert "## Structured Output" in graph.system_prompt
    assert '"answer"' in graph.system_prompt


def test_structured_output_section_can_be_disabled_even_with_schema():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    agent = RecursiveFlow(
        StaticLLM('```repl\ndone("ok")\n```'),
        runtime=LocalRuntime(),
        config=FlowConfig(enable_structured_output=False),
    )

    graph = agent.start("say ok", output_schema=schema)

    assert "## Structured Output" not in graph.system_prompt

