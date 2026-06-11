"""Engine-level configuration.

Lives inside :mod:`rflow.engine` so engine helpers can import it
without ever reaching back up into :mod:`rflow.flow` (which would
re-introduce the same circular-import shape this package was
restructured to avoid).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _default_max_concurrency() -> int:
    """Default to full parallelism across runnable agents.

    Uses the host's CPU count (or 1 if it can't be determined).
    Agent work is mostly LLM I/O — there's no real upside to gating
    below the thread count by default. Users with rate-limit
    concerns or single-flight requirements should set this
    explicitly.
    """
    return os.cpu_count() or 1


@dataclass
class FlowConfig:
    """Engine-level knobs for scheduling, prompting, budgets, and repair loops.

    The config is copied into each fresh :class:`~rflow.graph.Graph` so graph
    snapshots can explain the policy they were created under. The root agent
    uses ``max_iterations`` directly; child agents inherit a child-specific
    ``max_iterations`` derived from ``child_max_iterations`` so a delegated
    branch cannot consume the same unbounded budget as the supervisor unless
    the caller chooses that explicitly.
    """

    # Maximum delegation depth from the root. ``0`` is baseline mode: no child
    # agents are allowed, so all work must happen in the root REPL.
    max_depth: int = 5

    # Maximum LLM turns per agent. ``None`` means uncapped. This limit applies
    # independently per agent, not globally across the whole tree.
    max_iterations: int | None = None

    # Maximum captured stdout/stderr text from a REPL execution before the
    # engine truncates it into the next observation. Prevents huge prints from
    # overwhelming the prompt.
    max_output_length: int = 12000

    # Maximum projected chat messages sent to the model for one agent. When set,
    # old messages are summarized/truncated so very long local histories do not
    # exceed provider context limits.
    max_messages: int | None = None

    # Maximum number of runnable agent transitions executed concurrently by the
    # engine scheduler. Defaults to CPU count because most work is I/O-bound LLM
    # or runtime waiting.
    max_concurrency: int | None = field(default_factory=_default_max_concurrency)

    # Separate cap for concurrent LLM requests across all agents and
    # ``llm_query_batched`` calls. ``None`` falls back to ``max_concurrency``.
    # Use this for provider rate limits without serializing non-LLM work.
    llm_max_concurrency: int | None = None

    # Per-request timeout in seconds for calls routed through the shared LLM
    # channel. ``None`` disables the channel-level timeout.
    llm_request_timeout: float | None = 600

    # Iteration cap assigned to newly spawned child agents. This deliberately
    # differs from the root's ``max_iterations`` so fanout stays bounded by
    # default even when the root is uncapped.
    child_max_iterations: int | None = 20

    # If true, the scheduler keeps children moving as soon as they are runnable,
    # even while the parent is still able to take more steps. If false, children
    # run primarily when the parent is supervising/waiting.
    eager_children: bool = False

    # Prompt/extraction policy for LLM replies. True means expect one executable
    # REPL block per turn; this keeps state transitions simple and repairable.
    single_block: bool = True

    # If true, the default prompt advertises structured-output affordances
    # (``output_schema`` for root runs, child agents, and ``llm_query_batched``).
    # Runtime APIs still exist when false; this only hides the hints/examples.
    enable_structured_output: bool = True

    # Full system prompt override. When set, the default prompt builder is
    # bypassed completely; callers are responsible for including the REPL,
    # delegation, and done(...) protocol in their custom prompt.
    system_prompt: str | None = None

    # Optional token budget for an agent. When the accumulated usage crosses
    # this budget, the engine terminates that agent with a budget-exceeded
    # result instead of continuing to ask for more LLM turns.
    max_budget: int | None = None


__all__ = ["FlowConfig"]
