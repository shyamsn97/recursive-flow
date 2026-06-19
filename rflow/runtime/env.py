"""Public process environment keys exposed to agent code.

These keys are safe, string-valued metadata that an agent can read via
``os.environ``. Private control state such as output schemas and done results
lives in :class:`rflow.runtime.context.EngineContext` instead.
"""

from __future__ import annotations

#: Current agent id (for example ``"root"`` or ``"root.worker"``).
RFLOW_AGENT_ID = "RFLOW_AGENT_ID"
#: Current agent depth in the spawn tree (``"0"`` for root).
RFLOW_DEPTH = "RFLOW_DEPTH"
#: Parent agent id, or ``""`` for root.
RFLOW_PARENT_AGENT_ID = "RFLOW_PARENT_AGENT_ID"
#: Recursion bound configured on the flow.
RFLOW_MAX_DEPTH = "RFLOW_MAX_DEPTH"
#: ``"1"`` for root, ``"0"`` otherwise.
RFLOW_IS_ROOT = "RFLOW_IS_ROOT"


def agent_process_env(
    *,
    agent_id: str,
    depth: int,
    parent_agent_id: str | None,
    max_depth: int,
) -> dict[str, str]:
    """Return public ``RFLOW_*`` environment variables for one agent."""
    return {
        RFLOW_AGENT_ID: agent_id,
        RFLOW_DEPTH: str(depth),
        RFLOW_PARENT_AGENT_ID: parent_agent_id or "",
        RFLOW_MAX_DEPTH: str(max_depth),
        RFLOW_IS_ROOT: "1" if depth == 0 else "0",
    }


__all__ = [
    "RFLOW_AGENT_ID",
    "RFLOW_DEPTH",
    "RFLOW_IS_ROOT",
    "RFLOW_MAX_DEPTH",
    "RFLOW_PARENT_AGENT_ID",
    "agent_process_env",
]
