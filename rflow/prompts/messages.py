"""User-facing message templates for the :class:`~rflow.flow.Flow` engine.

All bootstrap / nudge / error text lives here so ``flow.py`` stays logic-only.
Adapted from alexzhang13/rlm (``rlm/utils/prompts.py``) to the minimal stack's
**inputs-as-`INPUTS`** model: there is no monolithic ``CONTEXT`` variable — each
agent's inputs are exposed through a single ``INPUTS`` dict (read as
``INPUTS["key"]``) so a key never shadows a real REPL variable, and an agent
re-reads its own past turns through ``HISTORY``.
"""

from __future__ import annotations

REPL_BLOCK_RULE = """Use exactly one fenced REPL code block per assistant message. Your entire reply must have this shape:
```repl
# Python code here
```
Do not write bare `repl` without the opening and closing triple backticks."""

USER_PROMPT_WITH_ROOT = """Think step-by-step on what to do using the REPL environment (your inputs are available in the `INPUTS` dict, e.g. `INPUTS["key"]`) to answer the original prompt: "{query}".

Continue using the REPL environment and querying sub-LLMs / sub-agents by writing ```repl``` blocks, and determine your answer. Your next action:"""

CONTINUE_ACTION = "Continue your next action:"

FIRST_TURN_INSPECTION_NUDGE = """\
Your first REPL block should usually inspect your inputs (e.g. `print(list(INPUTS))`, `print(len(INPUTS["key"]))`, `print(INPUTS["key"][:2000])`) and, if needed, your own `HISTORY`, then stop so you can use that output in the next turn.
"""

FIRST_TURN_DECOMPOSITION_NUDGE = """\
After the initial inspection, delegate independent units with
`await launch_subagents([...])` or `llm_query_batched(...)` instead of doing all
work yourself. The parent should combine child results, make final decisions,
and verify the outcome.

Put each child's data and requirements in its `inputs` (a dict of str -> str),
since a child sees only the query and the `INPUTS` you pass it — never your own
`INPUTS` or variables.
"""

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


def build_inputs_manifest(inputs: dict[str, str]) -> str:
    """List inputs by name + size (no value dumps), with a big-input chunk hint.

    Returns ``""`` when there are no inputs. Mirrors the size-signal idea from
    ``build_rlm_system_prompt`` so the model knows how large each input is
    before choosing a chunking / fanout strategy.
    """
    if not inputs:
        return ""
    lines = [
        f'  INPUTS["{name}"]  (str, {len(value)} chars)'
        for name, value in inputs.items()
    ]
    manifest = 'Your inputs (read with `INPUTS["name"]`):\n' + "\n".join(lines)

    total = sum(len(value) for value in inputs.values())
    if total > 50_000:
        manifest += (
            f"\n\nThese inputs total ~{total} characters (~{total // 4} tokens). "
            "Chunk large inputs and process the pieces in parallel with "
            "`llm_query_batched(...)` or `await launch_subagents([...])`."
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
    query: str, inputs: dict[str, str], *, depth: int = 0, max_depth: int = 0
) -> str:
    """Build an agent's bootstrap user message.

    Folds in the first-interaction safeguard, the inputs manifest (name + size,
    never the values), the step-by-step framing, the first-turn nudges, and the
    recursion-depth note.
    """
    parts = [FIRST_TURN_SAFEGUARD]
    manifest = build_inputs_manifest(inputs)
    if manifest:
        parts.append(manifest)
    parts.append(USER_PROMPT_WITH_ROOT.format(query=query))
    parts.append(FIRST_TURN_INSPECTION_NUDGE.strip())
    if max_depth > 0 and depth < max_depth:
        parts.append(FIRST_TURN_DECOMPOSITION_NUDGE.strip())
    parts.append(depth_note(depth, max_depth))
    return "\n\n".join(p for p in parts if p)
