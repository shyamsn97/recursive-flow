"""Message templates used by the RLMFlow engine.

All user-facing text lives here so rlm.py stays logic-only.
"""

from typing import Any

DEFAULT_QUERY = "Please read through the context and answer any queries or respond to any instructions contained within it."

REPL_BLOCK_RULE = """Use exactly one fenced REPL code block per assistant message. Your entire reply must have this shape:
```repl
# Python code here
```
Do not write bare `repl` without the opening and closing triple backticks."""

FIRST_ACTION = f"""Query: {{query}}

Use the REPL to inspect/decompose and act now. Choose the lane in code: `llm_query_batched(prompts)` for independent one-shot LLM prompts that do not need tools/files/iteration; `rlm_delegate(...)` plus `yield rlm_wait(*handles)` for independent units that need tools, files, execution, repair, verification, or multi-turn work. For multi-file or multi-component artifacts, spawn the child batch before writing unit outputs in root unless there is a hard sequential dependency. When delegating artifacts/components, keep each child `query` short and put the shared brief, owned scope, dependencies, interfaces, and acceptance checks in `context`; do not pass only the filename/unit name as context. Work directly only for one small local scope or truly sequential work. Do not call `done(...)` until the needed REPL work is complete.

{{context_hint}}{REPL_BLOCK_RULE}"""

CONTINUE_ACTION = """Continue working on the original query: "{query}". Use saved REPL variables, `CONTEXT`, `llm_query_batched(...)`, and `rlm_delegate(...)` as appropriate. If independent units remain, batch them before waiting; if a prior step failed, repair the specific failure. Your next action:

{context_hint}"""

RESUME_VERIFY_ACTION = """Your previous `yield rlm_wait(...)` has resumed; children finished: {child_ids}. Use the wait-result variables already saved by your code and any other saved REPL variables to verify the child outputs. Then call `done(answer)`, repair only failed pieces, or explain why another delegation batch is needed.

{context_hint}"""

CONTEXT_HINT_PRESENT = """Relevant data is available as the `CONTEXT` REPL variable - inspect it with `CONTEXT.info/read/lines/grep`. If `CONTEXT` contains references or assigned scope rather than the target data itself, use available tools/functions to inspect those referenced items; do not treat the reference list itself as the evidence.

"""
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


FINAL_ANSWER_ACTION = """You have used the full iteration budget without calling done().

Based on the work above, provide the final answer now. The block must call done(answer). The done() argument must be only the final answer string in the exact form the query requested. Do not do more investigation."""

NO_CODE_BLOCK = f"ERROR: Your previous reply did not contain a ```repl``` code block. {REPL_BLOCK_RULE} Try again."

EXECUTION_OUTPUT = "REPL output:\n{output}"

ORPHANED_DELEGATES = "You delegated [{names}] but never called `yield rlm_wait(...)`. You must use `yield rlm_wait(*handles)` to collect results."

STATUS_DEPTH_ROOT = " You have the full recursion budget available."

STATUS_DEPTH_MID = " Some recursion budget remains available."

STATUS_DEPTH_NEAR_MAX = " You are near the recursion limit."

TRUNCATION_SUMMARY = """## Query
{query}

## History
{total} messages so far, showing the last {cap}.{session_hint}"""

TRUNCATION_SESSION_HINT = ""
