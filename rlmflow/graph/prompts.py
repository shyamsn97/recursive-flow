"""Turn-facing prompt text attached by graph node types.

Distinct from ``rlmflow.prompts``, which builds the inherited system protocol.
"""

from __future__ import annotations

from typing import Any

FINAL_ANSWER_ACTION = """This is your last turn: the run is out of budget to keep working.
Based on the work above, call finish(answer) now with only the final answer, in the
exact form the query requested. Do not investigate further, and do not hold the answer
back for verification — no turn follows this one to read a print, so submit your best
inference rather than ending the run with nothing."""

TRUNCATION_SUMMARY = """[earlier turns omitted to fit the context window; the most recent
turns follow. The REPL kept running, so variables, imports, and helpers defined in those
turns are still bound — reuse them instead of redefining them.]"""

COLD_REPL_NOTE = """[this agent's REPL was restarted, so variables and imports from
earlier turns are gone. Re-derive whatever you need before using it.]"""

USER_PROMPT = "Turn {iter_1}/{max_iter}:"

FIRST_TURN_SAFEGUARD = (
    "You have not interacted with the REPL environment yet. Look at INPUTS if any "
    "are bound, otherwise inspect the workspace; do not provide a final answer yet.\n\n"
)


def orchestrator_addendum(flow: Any | None = None) -> str:
    """Orchestrator policy: spawn isolates tokens; ``llm_query`` is one-shot only."""
    context = (
        "Your own context window is small. Do not pull long `INPUTS` into your own "
        "message stream. (If a Python keyword / regex search over `INPUTS` would already "
        "pin the answer, or if a single visible passage already contains it, just read "
        "it directly.) Long REPL stdout pollutes history the same way raw `INPUTS` does: "
        "`print` only bounded samples. Aggregate the small results back in the REPL. "
        "A child costs tokens in its own window, not yours. "
        "`launch_subagent(..., inputs={...})` puts bulky text in the child's context; "
        "you only see the small result. Printing or grepping a long INPUT through your "
        "own turns re-pays that text on every later message. Spawn when a branch would "
        "otherwise live in your history — multi-step search, isolated code, or a dossier "
        "the parent should not keep printing."
    )
    if flow is not None and getattr(flow, "use_llm_query", False):
        context += (
            " Use `llm_query` / `llm_query_batched` only for a one-shot extract or "
            "classify over a chunk — those calls have no REPL. They are not a substitute "
            "for a child that needs several turns."
        )
    return "\n\n".join(
        [
            "As an RLM, you should act as an orchestrator, not a solver.",
            (
                "Directly after you probe the `INPUTS` and understand your task, pause and plan: "
                "state explicitly how the task decomposes into sub-LLM / REPL steps, and sketch "
                "the concrete sequence of turns — what each turn computes and which sub-LLM call "
                "(if any) it issues — like a condensed trajectory, before you execute them. "
                "Then execute one turn at a time: after each step `print` a small sample of the "
                "result, verify it looks right, and only call `finish(...)` once you have "
                "actually printed the candidate answer. If you are running out of turns "
                "without a confirmed answer, submit your best inference rather than letting the "
                "rollout terminate unsubmitted."
            ),
            context,
            (
                "Reserve your own tokens for high-level decisions: what to ask next, how to "
                "combine results, when to finalize. Delegate everything else."
            ),
        ]
    )


ORCHESTRATOR_ADDENDUM = orchestrator_addendum()

LAST_TURN_ADDENDUM = (
    "This is your last turn. Call `finish(...)` now with your best inference rather "
    "than letting the rollout terminate unsubmitted."
)

#: Last sentence of the inherited protocol; FinalQuery strips it so the last
#: turn does not say "don't finish on turn 1."
TURN_ONE_FINISH_SENTENCE = "Do not call `finish(...)` on turn 1 without first inspecting `INPUTS`."


__all__ = [
    "COLD_REPL_NOTE",
    "FINAL_ANSWER_ACTION",
    "FIRST_TURN_SAFEGUARD",
    "LAST_TURN_ADDENDUM",
    "ORCHESTRATOR_ADDENDUM",
    "TRUNCATION_SUMMARY",
    "TURN_ONE_FINISH_SENTENCE",
    "USER_PROMPT",
    "orchestrator_addendum",
]
