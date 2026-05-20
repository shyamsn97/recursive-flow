"""Message templates used by the RLMFlow engine.

All user-facing text lives here so rlm.py stays logic-only.
"""

from typing import Any

DEFAULT_QUERY = (
    "Please read through the context and answer any queries "
    "or respond to any instructions contained within it."
)

REPL_BLOCK_RULE = (
    "Use exactly one fenced REPL code block per assistant message. "
    "Your entire reply must have this shape:\n"
    "```repl\n"
    "# Python code here\n"
    "```\n"
    "Do not write bare `repl` without the opening and closing triple backticks."
)

FIRST_ACTION = (
    "Query: {query}\n\n"
    "First inspect/decompose. If the query has many independent units/scopes, "
    "spawn the child batch now with `rlm_delegate(...)` and "
    "`yield rlm_wait(*handles)`. Do not draft/write all unit outputs in root "
    "first. Do not final-answer before doing the needed REPL work. Your next "
    "action:\n\n"
    "{context_hint}"
    f"{REPL_BLOCK_RULE}"
)

CONTINUE_ACTION = (
    "The history before is your previous interactions with the REPL environment. "
    'Continue using the REPL environment to answer the original query: "{query}". '
    "Use `CONTEXT` and batched or recursive LLM calls as useful, build on saved "
    "variables, and determine your answer. Your next action:\n\n"
    "{context_hint}"
)

RESUME_VERIFY_ACTION = (
    "Children just finished: {child_ids}. Before any new `rlm_delegate`, "
    "inspect their outputs/state with saved REPL variables, `SESSION.read(...)`, "
    "or `SESSION.grep(...)`. Verify their requested output contracts, then "
    "either call `done(answer)`, repair only failed pieces, or print a concrete "
    "reason for another delegation batch.\n\n"
    "{context_hint}"
)

CONTEXT_HINT_PRESENT = (
    "Relevant data is available as the `CONTEXT` REPL variable - "
    "inspect it with `CONTEXT.info/read/lines/grep`. If `CONTEXT` contains "
    "references or assigned scope rather than the target data itself, use "
    "available tools/functions to inspect those referenced items; do not treat "
    "the reference list itself as the evidence.\n\n"
)
CONTEXT_HINT_ABSENT = ""


def format_context_hint(
    info: dict[str, Any] | None = None,
    *,
    context_keys: list[str] | None = None,
) -> str:
    """Render compact context metadata for first/continue action messages."""

    keys = sorted(context_keys or [])
    if not info and not keys:
        return CONTEXT_HINT_ABSENT

    lines = [CONTEXT_HINT_PRESENT.rstrip()]
    if info:
        parts = []
        for key in ("chars", "approx_tokens", "lines"):
            if key in info:
                parts.append(f"{key}={info[key]}")
        if parts:
            lines.append("Context metadata: " + ", ".join(parts) + ".")
    if keys:
        shown = ", ".join(keys[:8])
        suffix = f", ... +{len(keys) - 8} more" if len(keys) > 8 else ""
        lines.append(f"Available context keys: {shown}{suffix}.")
    return "\n".join(lines) + "\n\n"


FINAL_ANSWER_ACTION = (
    "You have used the full iteration budget without calling done().\n\n"
    "Based on the work above, provide the final answer now. "
    "The block must call done(answer). "
    "The done() argument must be only the final answer string in the exact form "
    "the query requested. Do not do more investigation."
)

NO_CODE_BLOCK = (
    "ERROR: Your previous reply did not contain a ```repl``` code block. "
    f"{REPL_BLOCK_RULE} "
    "Try again."
)

EXECUTION_OUTPUT = "REPL output:\n{output}"

ORPHANED_DELEGATES = (
    "You delegated [{names}] but never called `yield rlm_wait(...)`. "
    "You must use `yield rlm_wait(*handles)` to collect results."
)

STATUS_DEPTH_ROOT = " You have the full recursion budget available."

STATUS_DEPTH_MID = " Some recursion budget remains available."

STATUS_DEPTH_NEAR_MAX = " You are near the recursion limit."

TRUNCATION_SUMMARY = (
    "## Query\n{query}\n\n"
    "## History\n{total} messages so far, showing the last {cap}.{session_hint}"
)

TRUNCATION_SESSION_HINT = ""
