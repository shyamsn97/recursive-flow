"""Rendered prompt invariants for the default recursive prompt."""

from __future__ import annotations

from rlmflow.prompts.default import DEFAULT_BUILDER
from rlmflow.prompts.messages import (
    STATUS_DEPTH_MID,
    STATUS_DEPTH_NEAR_MAX,
    STATUS_DEPTH_ROOT,
)


def _render_default_prompt() -> str:
    return DEFAULT_BUILDER.build(
        tools="",
        status="You are at recursion depth **0** of **3**. You have the full recursion budget available.",
    )


def test_default_prompt_has_alex_style_inventory_order():
    prompt = _render_default_prompt()

    markers = [
        "1. `CONTEXT`",
        "2. `llm_query_batched(prompts)`",
        "3. `rlm_delegate(...)`",
        "4. `yield rlm_wait(*handles)`",
        "5. `SHOW_VARS()`",
        "6. `print(...)`",
        "7. `SESSION`",
        "8. `done(answer)`",
    ]
    positions = [prompt.index(marker) for marker in markers]
    assert positions == sorted(positions)


def test_default_prompt_teaches_both_batch_lanes_and_final_answer():
    prompt = _render_default_prompt()

    assert "prompts = [" in prompt
    assert "notes = llm_query_batched(prompts)" in prompt
    assert "specs = [" in prompt
    assert "rlm_delegate(" in prompt
    assert "yield rlm_wait(*handles)" in prompt
    assert "done(answer)" in prompt
    assert "query` is the task/output contract" in prompt
    assert "context` is the data/scope" in prompt
    assert "Signature: `llm_query_batched(prompts: list[str]" in prompt
    assert "Signature: `rlm_delegate(*, name: str, query: str, context: str" in prompt
    assert "`CONTEXT.grep(pattern, max_results=50)` - regex search inside the `CONTEXT` payload only." in prompt
    assert "Do not pass file paths or `path=` to `CONTEXT` methods." in prompt
    assert "use available tools/functions to inspect the referenced items" in prompt
    assert "parent fans out by default" in prompt
    assert "many independent units" in prompt
    for unit in ("chunks", "documents", "files", "paths", "records", "trials", "checks", "components", "artifacts", "subproblems"):
        assert unit in prompt
    assert "Do not serially loop many independent units in root" in prompt
    assert "context` must be a string" in prompt
    assert "never a Python list/dict object" in prompt
    assert '"\\n".join(items)' in prompt
    assert "actual result returned to parent/user" in prompt


def test_default_prompt_examples_cover_key_behaviors_without_bad_taxonomy():
    prompt = _render_default_prompt()

    for heading in (
        "**Inspect, then choose a lane:**",
        "**Recursive fanout over independent units:**",
        "**One-shot semantic batch:**",
    ):
        assert heading in prompt

    bad_phrases = [
        "assigned worker task",
        "semantic checks/extractions",
        "Use delegation aggressively",
        "finish it directly unless asked",
    ]
    for phrase in bad_phrases:
        assert phrase not in prompt


def test_status_depth_text_is_budget_only():
    status = STATUS_DEPTH_ROOT + STATUS_DEPTH_MID + STATUS_DEPTH_NEAR_MAX

    for phrase in (
        "multiple files",
        "review/test",
        "naturally separable",
        "independent subtask",
        "Delegate based on task structure",
    ):
        assert phrase not in status
