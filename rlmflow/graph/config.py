"""Per-agent limits and identity for a run."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

DEFAULT_QUERY = """Please read through the provided INPUTS if present and answer any
queries or respond to any instructions contained within it."""
#: Official RLM truncates each REPL block at 20K characters (`format_iteration`).
DEFAULT_MAX_OUTPUT_CHARS = 20_000
DEFAULT_MAX_QUERY_CHARS = 20_000


def validate_agent_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or not name.isascii()
        or any(not (char.isalnum() or char in "_-") for char in name)
    ):
        raise ValueError(f"invalid child name {name!r}")


@dataclass
class AgentConfig:
    name: str = "root"
    path: str = "root"
    depth: int = 0
    model: str = "default"
    prompt_profile: str = "default"
    inputs: dict[str, str] = field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    #: Place a child in its caller's live Python worker.
    reuse_repl: bool = False
    #: Stable ordinal of the launch call that created this child.
    launch_call_id: int | None = None
    max_depth: int = 1
    #: How many model turns this agent may take. ``None`` means unbounded.
    max_iters: int | None = None
    child_max_iters: int | None = None
    #: Whole-run billed tokens (input + output). ``None`` means unbounded.
    max_budget: int | None = None
    #: How many of the agent's own transcript turns a prompt carries. ``None``
    #: keeps the full history. The system message and the truncation notice sit
    #: on top of this count when it is set.
    keep_n_messages: int | None = None
    max_output_length: int = DEFAULT_MAX_OUTPUT_CHARS
    max_query_chars: int = DEFAULT_MAX_QUERY_CHARS

    def child(self, name: str, **overrides: Any) -> AgentConfig:
        validate_agent_name(name)
        values = {
            "name": name,
            "path": f"{self.path}.{name}",
            "depth": self.depth + 1,
            "inputs": {},
            "output_schema": None,
            "reuse_repl": False,
            "launch_call_id": None,
            "max_iters": self.child_max_iters or self.max_iters,
        }
        return replace(self, **{**values, **overrides})


def can_spawn(agent: Any) -> bool:
    """Whether this agent can create another recursion level."""
    if agent is None:
        return False
    config = getattr(agent, "config", None)
    if config is None:
        return False
    return config.max_depth > 0 and config.depth < config.max_depth


__all__ = [
    "DEFAULT_MAX_OUTPUT_CHARS",
    "DEFAULT_MAX_QUERY_CHARS",
    "DEFAULT_QUERY",
    "AgentConfig",
    "can_spawn",
    "validate_agent_name",
]
