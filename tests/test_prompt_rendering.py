"""PromptBuilder structural tests."""

from __future__ import annotations

from rlmflow.prompts.builder import PromptBuilder
from rlmflow.prompts.default import DEFAULT_BUILDER


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

