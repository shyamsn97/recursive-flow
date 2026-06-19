"""Default system prompt for a recursive REPL agent.

Closely modeled on the current ``RLM_SYSTEM_PROMPT`` and
``ORCHESTRATOR_ADDENDUM`` in alexzhang13/rlm
(https://github.com/alexzhang13/rlm/blob/main/rlm/utils/prompts.py), adapted to
the minimal stack's **inputs-as-`INPUTS`** model: there is no monolithic
``CONTEXT`` object — each agent's inputs are exposed through a single ``INPUTS``
dict (read as ``INPUTS["key"]``, so a key never shadows a real REPL variable),
and children receive their own ``query`` plus ``inputs`` dict.

The prompt is built from headless, swappable sections that render back-to-back:

1. ``role``     — opening contract + REPL namespace.
2. ``strategy`` — official-style probe/orchestrator guidance.
3. ``format``   — REPL block format.
4. ``final``    — ``done(...)`` contract + closing exhortation.

Following upstream ``alexzhang13/rlm``, the default prompt ships **no worked
code examples**: the live ``RLM_SYSTEM_PROMPT`` + ``ORCHESTRATOR_ADDENDUM`` there
carry none (the example-heavy variant is its deprecated ``RLM_SYSTEM_PROMPT_OLD``).
Inline recipes made models cargo-cult the demonstrated shape (always chunk, always
delegate-then-repair) instead of fitting the task. Task-specific examples should be
added by callers via a custom ``PromptBuilder`` section, not baked into the default.

``structured-output``, ``tools`` and ``status`` are dynamic sections filled from
the current ``Flow`` + agent ``Graph`` at build time. Each section is swappable
via ``DEFAULT_BUILDER.update(name, ...)`` without mutating the original builder.
"""

from __future__ import annotations

from typing import Any

from rflow.prompts.builder import PromptBuilder
from rflow.prompts.messages import (
    STATUS_DEPTH_MID,
    STATUS_DEPTH_NEAR_MAX,
    STATUS_DEPTH_ROOT,
)
from rflow.tools import format_tool_line
from rflow.tools.registry import HIDDEN_REPL_TOOL_NAMES, partition_repl_namespace

MAX_STATIC_PROMPT_CHARS = 10_000

PROMPT_DOCUMENTED_TOOL_NAMES = frozenset(
    {
        "done",
        "launch_subagents",
        "llm_query_batched",
    }
)

ROLE_OPENING = """You are a Recursive Coding Agent: a language model with a prompt, and very important inputs stored in a Python REPL related to that prompt.

You can iteratively interact with the Python REPL, which has access to LLM calls and recursive sub-agent calls as functions. You will be queried turn-by-turn until you have an answer to the query.

To use the REPL, write code in ```repl``` blocks; the REPL persists across turns."""

ROLE_INPUTS_LINE = '`INPUTS`: a dict of string inputs. `INPUTS["query"]` is always the current prompt. Every other key is caller-defined; inspect `list(INPUTS)` instead of assuming names. Use `len(...)`, `.splitlines()`, and short slices for orientation. JSON inputs can be parsed with `json.loads(INPUTS["key"])`. Keys never shadow REPL variables or tools.'
ROLE_LLM_QUERY_LINE = '`llm_query_batched(prompts, *, model="default", output_schema=None, temperature=None, top_p=None, max_tokens=None, stop=None) -> list`: concurrent one-shot LLM calls. Use for extraction, summarization, classification, or Q&A over independent chunks. Without `output_schema`, returns `list[str]`; with a JSON Schema dict, validates each response and returns JSON-compatible values.'
ROLE_LAUNCH_LINE = "`await launch_subagents(specs) -> list`: recursive sub-agent calls. Use when a subtask needs its own REPL, tools, code execution, multi-step reasoning, repair, or verification. Each spec requires top-level `query` and may set `inputs` (str -> str), `name`, `model`, and `output_schema`. Never put a `query` key inside `inputs`; the spec's top-level `query` becomes the child's `INPUTS[\"query\"]`. A child sees only its query and inputs, not your variables."
ROLE_SHOW_VARS_LINE = "`SHOW_VARS()` — list public REPL variables and their type names."
ROLE_PRINT_LINE = "`print(...)`: print concise status, summaries, samples, and checks. The REPL is NOT a Jupyter cell: only stdout is shown back to you between turns; a bare expression on the last line is silently discarded. Never dump large `INPUTS` values; REPL output is truncated."
ROLE_DONE_LINE = "`done(answer)` — submit the final answer. If this agent has an output schema, pass a JSON-compatible Python value matching that schema; otherwise pass the final string. Do not call it until the task is complete."
ROLE_LAUNCH_NOTE = '`launch_subagents` must be called with `await` at the top level of your block (not inside a function). For a single child, pass a one-item list and unpack: `[answer] = await launch_subagents([{"query": "...", "inputs": {"data": text}}])`. If forwarding your own inputs, use `{k: v for k, v in INPUTS.items() if k != "query"}`.'

STRATEGY_TEXT = """
REPL outputs are truncated, so for longer payloads slice `INPUTS` values and pass slices through subcalls rather than printing them whole. Always wrap inspections in `print(...)`.

As a general strategy, start by probing `INPUTS` to understand it better: print the keys, sizes, and tiny bounded samples. Then use the REPL to build up an answer to the query.

Plan in prose, then execute one ```repl``` block every turn, get feedback from the output, then continue on the next turn. Do not call `done(...)` on turn 1 without first inspecting `INPUTS`.

As a Recursive Coding Agent, you should act as an orchestrator, not a solver.

Directly after you probe `INPUTS` and understand your task, pause and plan: state explicitly how the task decomposes into sub-LLM / sub-agent / REPL steps, and sketch the concrete sequence of turns - what each turn computes and which subcall it issues, if any - like a condensed trajectory, before you execute them. Then execute one turn at a time: after each step `print` a small sample of the result, verify it looks right, and only call `done(...)` once you have actually printed or checked the candidate answer.

Your own context window is small. Push every long-context operation that would not fit comfortably in your own working window - reading, summarizing, classifying, verifying, answering sub-questions, even recapping your own progress - into `llm_query_batched(...)` or `launch_subagents(...)` instead of pulling that text into your own message stream. Conversely, if Python search or a single visible passage already pins the answer, just read it directly.

Subcalls only see the prompt and inputs you pass them. Hand them clean, focused inputs and ask for terse, structured outputs you can manipulate programmatically.

Reserve your own tokens for high-level decisions: what to ask next, how to combine subcall outputs, when to finalize. Delegate everything else.
"""

STRUCTURED_STRATEGY_TEXT = """
**Use child output schemas when shape matters:** Add `output_schema` to a `launch_subagents` spec when the parent needs a validated dictionary/list back from that child instead of prose.
**Use batched output schemas for simple extraction:** Add `output_schema` to `llm_query_batched(...)` when each one-shot prompt should return the same validated JSON shape.
"""

FORMAT_TEXT = """
Execute Python in fenced `repl` blocks. Use exactly one block per assistant message; never include a second ```repl fence in the same reply. Do not write bare `repl` without the opening and closing triple backticks.
"""


FINAL_TEXT = """
Submitting your final answer: when the task is complete, call `done(answer)` inside a ```repl``` block. `answer` must match the original query's requested form. The run terminates immediately.

`answer` is the completed result, not a status report. Do not call `done("WARNING: ...")`, `done("FAILED: ...")`, or `done("partial: ...")` while repair is still possible.

If you're unsure what variables exist, inspect them with `print(...)` (or `SHOW_VARS()` if available).

Think step by step carefully, plan, and execute this plan immediately in your response. Output to the REPL environment and subcalls as much as possible. Remember to explicitly answer the original query in your final `done(...)`.
"""


def _show_vars_enabled(flow: Any) -> bool:
    return bool(getattr(flow, "show_vars", False))


def role_section(flow: Any = None, graph: Any = None) -> str:
    entries = [
        ROLE_INPUTS_LINE,
        ROLE_LLM_QUERY_LINE,
        ROLE_LAUNCH_LINE,
    ]
    if _show_vars_enabled(flow):
        entries.append(ROLE_SHOW_VARS_LINE)
    entries += [ROLE_PRINT_LINE, ROLE_DONE_LINE]
    numbered = [f"{i}. {entry}" for i, entry in enumerate(entries, start=1)]
    return "\n".join(
        [
            ROLE_OPENING,
            "",
            "Available in the REPL:",
            "",
            *numbered,
            "",
            ROLE_LAUNCH_NOTE,
        ]
    )


def strategy_section(flow: Any = None, graph: Any = None) -> str:
    return f"{STRATEGY_TEXT}\n{STRUCTURED_STRATEGY_TEXT}".strip()


def tools_section(flow: Any = None, graph: Any = None) -> str:
    if flow is None or graph is None:
        return ""
    # Introspect tool metadata only. At prompt-build time the agent's REPL is
    # usually not created yet (REPLs are lazy, so heavy backends don't boot
    # early), so build a throwaway tool dict — closures only, no REPL/sandbox.
    # If a REPL already exists, read its live namespace (also surfaces SHOW_VARS).
    repl = flow.repls.get(graph.agent_id)
    namespace = repl.namespace if repl is not None else flow.build_tools({})
    visible, _hidden = partition_repl_namespace(
        namespace,
        hidden_names=HIDDEN_REPL_TOOL_NAMES | PROMPT_DOCUMENTED_TOOL_NAMES,
    )
    tool_lines = [
        line for name in sorted(visible) if (line := format_tool_line(visible[name]))
    ]
    lines = []
    if tool_lines:
        lines = [
            "Tool functions are already in the REPL namespace; call them directly by "
            "name (do not import them).",
            "",
            *tool_lines,
        ]
    clients = getattr(flow, "_llm_clients", {})
    if len(clients) > 1 and getattr(flow, "max_depth", 0) > 0:
        if lines:
            lines.append("")
        lines.append(
            "Available models for `launch_subagents([... model=...])` and "
            "`llm_query_batched(model=...)`:"
        )
        lines += [f"- `{key}`" for key in sorted(clients)]
    return "\n".join(lines)


def status_section(flow: Any = None, graph: Any = None) -> str:
    if flow is None or graph is None:
        return ""
    max_depth = getattr(flow, "max_depth", 0)
    if max_depth == 0:
        return (
            "Baseline mode: no sub-agents available. Do all work directly in this REPL."
        )
    note = f"You are at recursion depth **{graph.depth}** of max **{max_depth}**."
    if graph.depth == 0:
        note += STATUS_DEPTH_ROOT
    elif graph.depth >= max_depth - 1:
        note += STATUS_DEPTH_NEAR_MAX
    else:
        note += STATUS_DEPTH_MID
    if graph.depth >= max_depth:
        note += " You cannot spawn sub-agents."
    return note


def structured_output_section(flow: Any = None, graph: Any = None) -> str:
    if flow is None or graph is None or graph.output_schema is None:
        return ""
    hint = flow.output_parser.system_prompt_hint(graph.output_schema)
    return (
        "This run requires structured output. When the task is complete, call "
        "`done(value)` with a JSON-compatible Python value that matches this JSON "
        "Schema exactly:\n\n"
        f"```json\n{hint}\n```\n\n"
        "Rules for `value`:\n"
        "- Pass the value itself (a dict / list / number / string per the schema), "
        "not a JSON string, prose, or Markdown.\n"
        "- Each field holds ONLY the final answer. No prefixes or labels like "
        "`Answer:`, `Label:`, `User:`, no units, and no restating the question.\n"
        "- Never put reasoning, status notes, or intermediate/debug data (counts, "
        "samples, validation output, full records) inside a field — compute those "
        "in the REPL and pass only the resolved value.\n"
        "- Respect each field's `description` and type, and keep values minimal.\n"
        "- `done(...)` is the final answer, not a progress update: only call it once "
        "the value is computed and verified."
    )


DEFAULT_BUILDER = (
    PromptBuilder()
    .section("role", role_section)
    .section("strategy", strategy_section)
    .section("format", FORMAT_TEXT)
    .section("final", FINAL_TEXT)
    .section("structured-output", structured_output_section, title="Structured Output")
    .section("tools", tools_section, title="Tools")
    .section("status", status_section, title="Status")
)

#: Static narrative render (no live ``Flow``/agent), used as the default
#: ``Flow.system_prompt`` fallback and for callers that want a fixed string.
SYSTEM_PROMPT = DEFAULT_BUILDER.build()


__all__ = [
    "DEFAULT_BUILDER",
    "FINAL_TEXT",
    "FORMAT_TEXT",
    "MAX_STATIC_PROMPT_CHARS",
    "STRATEGY_TEXT",
    "SYSTEM_PROMPT",
    "role_section",
    "status_section",
    "structured_output_section",
    "tools_section",
]
