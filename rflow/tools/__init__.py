"""Tool decorator, metadata, and bundled tool collections.

``@tool`` marks a REPL callable as discoverable and attaches a human-readable
description; :func:`get_tool_metadata` reads it back (through bound methods),
and :func:`format_tool_line` renders the one-line entry the system prompt shows.
The control tools (``done``/``flow_wait``/``flow_delegate``/``launch_subagents``/
``llm_query_batched``) live in :mod:`rflow.tools.builtins`; the filesystem tools
in :mod:`rflow.tools.filesystem`.
"""

from rflow.tools.filesystem import FILE_TOOLS
from rflow.tools.tools import (
    ToolMetadata,
    format_tool_line,
    get_tool_metadata,
    tool,
)

__all__ = [
    "FILE_TOOLS",
    "ToolMetadata",
    "format_tool_line",
    "get_tool_metadata",
    "tool",
]
