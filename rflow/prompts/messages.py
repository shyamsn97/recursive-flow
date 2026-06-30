"""User-facing message templates for the :class:`~rflow.flow.Flow` engine.

All bootstrap / nudge / error text lives here so ``flow.py`` stays logic-only.
Adapted from alexzhang13/rlm (``rlm/utils/prompts.py``) to the minimal stack's
**inputs-as-`INPUTS`** model: there is no monolithic ``CONTEXT`` variable — each
agent's inputs are exposed through a single ``INPUTS`` dict (read as
``INPUTS["key"]``) so a key never shadows a real REPL variable.
"""

from __future__ import annotations

REPL_BLOCK_RULE = """Use exactly one fenced REPL code block per assistant message. Your entire reply must have this shape:
```repl
# Python code here
```
Do not write bare `repl` without the opening and closing triple backticks."""

USER_PROMPT_WITH_ROOT = """Think step-by-step on what to do using the REPL environment. Your task is the prompt above; any supporting material is in the REPL `INPUTS` dict (inspect `list(INPUTS)`).

Continue using the REPL environment and querying sub-LLMs / sub-agents by writing ```repl``` blocks, and determine your answer. Your next action:"""

CONTINUE_ACTION = "Continue your next action:"

FIRST_TURN_INSPECTION_NUDGE = """\
If `INPUTS` is non-empty, make your first REPL block an inspection-only observation turn: print keys, sizes, line counts, likely formats, and the constraints or line windows needed to understand the task. Wait for that REPL output before planning, calling `done(...)`, launching subagents, or running effectful tools. For small instruction-like inputs, read or outline them fully enough that the next turn can reason from the actual constraints. For large inputs, keep full values in variables and print targeted windows instead of dumping the payload.
"""


def build_decomposition_nudge(*, depth: int = 0, max_depth: int = 0) -> str:
    """Depth-aware planning and delegation hint for the bootstrap turn.

    The root gets coordinator framing. Children keep ownership of their assigned
    task and delegate only when a separable subtask benefits from its own agent.
    """
    if depth <= 0:
        return (
            "You are this task's root coordinator. After the relevant context has "
            "been observed, either act directly for simple work or write a short "
            "plan for multi-step work. When there are separable pieces, launch "
            "them as sub-agents in parallel with `await launch_subagents([...])`; "
            "make the launch block after the observation turn. Use the root for "
            "preparing focused inputs, integrating child results, verifying the "
            "combined work, and calling done(...)."
        )
    if max_depth > 0 and depth >= max_depth - 1:
        return (
            "Own your assigned task. You are near the recursion limit, so prefer "
            "solving locally after observing relevant context. You may still "
            "delegate a clearly bounded leaf subtask with "
            "`await launch_subagents([...])` when that is the best way to finish, "
            "but keep responsibility for integrating the result and calling "
            "done(...)."
        )
    return (
        "Own your assigned task. After the relevant context has been observed, "
        "either solve simple work directly or write a short plan for multi-step "
        "work. You may delegate clearly separable subtasks with "
        "`await launch_subagents([...])` when another agent would help, but keep "
        "responsibility for integrating the result, verifying it, and calling "
        "done(...)."
    )


FINAL_ANSWER_ACTION = """You have used the full iteration budget without calling done().

Based on the work above, provide the final answer now. The block must call done(answer). The done() argument must be only the final answer in the exact form the query requested. Do not do more investigation."""

NO_CODE_BLOCK = f"ERROR: Your previous reply did not contain a ```repl``` code block. {REPL_BLOCK_RULE} Try again."

EXECUTION_OUTPUT = "REPL output for previous block:\n{output}"

CONTINUE_NUDGE = "Continue. Reply with one ```repl``` block, or call done(...)."

TRUNCATION_SUMMARY = (
    "[earlier turns omitted to fit the context window; the most recent turns " "follow]"
)

FIRST_TURN_SAFEGUARD = (
    "You have not interacted with the REPL environment or seen your inputs yet. "
    "Your next action should be to inspect them and figure out how to answer the "
    "prompt, so don't just give a final answer yet."
)

STATUS_DEPTH_ROOT = " You have the full recursion budget available."
STATUS_DEPTH_MID = " Some recursion budget remains available."
STATUS_DEPTH_NEAR_MAX = " You are near the recursion limit."

BASELINE_NOTE = "Baseline mode: no sub-agents available. Do all work in this REPL."


def create_nudge_message() -> str:
    """Nudge appended when the model needs to produce another REPL action."""
    return CONTINUE_NUDGE


def create_final_action_message() -> str:
    """Nudge appended when iteration budget forces a final ``done(...)`` action."""
    return FINAL_ANSWER_ACTION


def build_inputs_manifest(
    inputs: dict[str, str],
) -> str:
    """List inputs by name + size (no value dumps), with a big-input chunk hint.

    Mirrors the size-signal idea from ``build_rlm_system_prompt`` so the model
    knows how large each REPL-visible input is before choosing a chunking /
    fanout strategy. The query is delivered as the first user message, not as an
    input, so it is not listed here. Returns ``""`` when the agent has no inputs.
    """
    if not inputs:
        return ""
    lines = [f"- {name}: str, {len(value)} chars" for name, value in inputs.items()]
    total = sum(len(value) for value in inputs.values())
    manifest = (
        "Your REPL INPUTS contain:\n"
        + "\n".join(lines)
        + f"\nTotal input chars: {total}.\n"
        + "Choose an inspection strategy for each input: small instruction-like "
        "inputs can be read or outlined fully; large inputs should be "
        "summarized by structure plus targeted line windows around relevant "
        "constraints."
    )

    if total > 50_000:
        manifest += (
            f"\n\nThese inputs total ~{total} characters (~{total // 4} tokens). "
            "Keep large values in variables, print only the structure and "
            "task-relevant windows, then process focused pieces in later turns "
            "or subagents."
        )
    return manifest


def depth_note(depth: int, max_depth: int) -> str:
    """One-line recursion-budget note for the bootstrap turn."""
    if max_depth == 0:
        return BASELINE_NOTE
    note = f"You are at recursion depth {depth} of max {max_depth}."
    if depth == 0:
        note += STATUS_DEPTH_ROOT
    elif depth >= max_depth - 1:
        note += STATUS_DEPTH_NEAR_MAX
    else:
        note += STATUS_DEPTH_MID
    if depth >= max_depth:
        note += " You cannot spawn sub-agents."
    return note


def first_prompt(
    query: str,
    inputs: dict[str, str],
    *,
    depth: int = 0,
    max_depth: int = 0,
) -> str:
    """Build an agent's bootstrap user message.

    The query is delivered here as plain text (the model's task), followed by the
    first-interaction safeguard, the inputs manifest (name + size, never the
    values), the step-by-step framing, the first-turn nudges, and the
    recursion-depth note.
    """
    parts = [FIRST_TURN_SAFEGUARD, f"Your task:\n{query}"]
    parts.append(build_inputs_manifest(inputs))
    parts.append(USER_PROMPT_WITH_ROOT)
    parts.append(FIRST_TURN_INSPECTION_NUDGE.strip())
    if max_depth > 0 and depth < max_depth:
        parts.append(build_decomposition_nudge(depth=depth, max_depth=max_depth))
    parts.append(depth_note(depth, max_depth))
    return "\n\n".join(p for p in parts if p)


def followup_prompt(
    query: str,
    *,
    depth: int = 0,
    max_depth: int = 0,
) -> str:
    """Build a lightweight user message for a follow-up task.

    Follow-ups happen inside an existing REPL trajectory, so they do not repeat
    first-turn inspection language. They still restate the task boundary,
    orchestration behavior, and recursion budget.
    """
    parts = [
        f"New user task:\n{query}",
        (
            "Continue using the REPL environment. After you understand the new "
            "task, write a short plan. If the remaining work has independent "
            "branches, delegate them with `await launch_subagents([...])`; use "
            "the root for coordination, integration, verification, and "
            "`done(...)`."
        ),
        depth_note(depth, max_depth),
    ]
    return "\n\n".join(parts)
