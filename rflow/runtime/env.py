"""REPL ``env`` keys shared between the engine and the tool closures.

These are the single source of truth (defined in :mod:`rflow.tools.builtins`)
re-exported under ``rflow.runtime`` for callers reasoning about the backend
boundary. ``DONE_RESULT`` is the one the engine reads back after every execution
to discover a finished agent's answer.
"""

from __future__ import annotations

from rflow.tools.builtins import ENV_AGENT_ID, ENV_DONE_RESULT, ENV_OUTPUT_SCHEMA

#: Where ``done(...)`` stashes the final answer for the engine to read back.
DONE_RESULT = ENV_DONE_RESULT
#: The spawning agent's id, read by ``flow_delegate`` at call time.
AGENT_ID = ENV_AGENT_ID
#: The agent's output schema (or ``None``), read by ``done`` at call time.
OUTPUT_SCHEMA = ENV_OUTPUT_SCHEMA

__all__ = ["AGENT_ID", "DONE_RESULT", "OUTPUT_SCHEMA"]
