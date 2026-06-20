"""Prompt and message projection helpers."""

from __future__ import annotations

from rflow.graph import Graph, append_message
from rflow.prompts import messages


def build_messages(
    graph: Graph,
    *,
    system_prompt: str,
    max_messages: int | None,
    continue_nudge: str,
    final_nudge: str,
    force_final: bool,
) -> list[dict[str, str]]:
    """Project one graph trajectory into chat messages for an LLM call."""
    msgs = graph.messages(system_prompt)
    if max_messages is not None and len(msgs) > max_messages:
        head: list[dict[str, str]] = []
        body = msgs
        if msgs and msgs[0]["role"] == "system":
            head, body = msgs[:1], msgs[1:]
        keep = max(1, max_messages - len(head) - 1)
        summary = {"role": "user", "content": messages.TRUNCATION_SUMMARY}
        msgs = [*head, summary, *body[-keep:]]
    if force_final:
        append_message(msgs, "user", final_nudge)
    elif not msgs or msgs[-1]["role"] != "user":
        append_message(msgs, "user", continue_nudge)
    return msgs


def build_system_prompt(
    system_prompt: str, *, schema_instruction: str | None = None
) -> str:
    """Render an explicit system prompt plus optional schema instruction."""
    if schema_instruction is None:
        return system_prompt
    return f"{system_prompt}\n\n{schema_instruction}"


def schema_instruction(schema_hint: str) -> str:
    """Render the structured-output instruction block for a schema."""
    return (
        "When you finish, call done(value) with a JSON-compatible Python "
        "value (not a JSON string) that matches this JSON Schema:\n"
        f"{schema_hint}"
    )


def format_exec_output(output: str) -> str:
    """Wrap captured REPL stdout for the model's next user turn."""
    return f"REPL output:\n{output or '(no output)'}"


def first_prompt(
    query: str, inputs: dict[str, str], *, depth: int, max_depth: int
) -> str:
    """Build an agent's bootstrap user message."""
    return messages.first_prompt(
        query,
        inputs,
        depth=depth,
        max_depth=max_depth,
    )


__all__ = [
    "build_messages",
    "build_system_prompt",
    "first_prompt",
    "format_exec_output",
    "schema_instruction",
]
