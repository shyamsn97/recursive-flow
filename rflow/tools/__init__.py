"""Tool decorator, metadata, and bundled tool collections.

``@tool`` marks a REPL callable as discoverable and attaches a human-readable
description; :func:`get_tool_metadata` reads it back (through bound methods),
and :func:`format_tool_line` renders the one-line entry the system prompt shows.
The control tools (``done``/``flow_wait``/``flow_delegate``/``launch_subagents``/
``llm_query_batched``) live in :mod:`rflow.tools.builtins`; the filesystem tools
in :mod:`rflow.tools.filesystem`.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    description: str
    #: Where the tool runs under a remote runtime. ``False`` (default) → the tool
    #: is shipped into the sandbox and runs there (it touches the sandbox's own
    #: state, e.g. its working directory). ``True`` → it runs on the host and its
    #: calls round-trip from the sandbox (it touches host-only state — the live
    #: ``Graph``, the LLM client). In-process runtimes ignore the flag.
    proxy: bool = False


def _default_tool_name(name: str) -> str:
    return name[5:] if name.startswith("tool_") else name


def tool(description: str, *, name: str | None = None, proxy: bool = False) -> Callable:
    """Mark a function as a discoverable tool.

    ``proxy=True`` marks a host-bound tool (see :class:`ToolMetadata.proxy`); the
    default ships the tool into the sandbox to run there.
    """

    def decorator(fn):
        fn._tool_meta = ToolMetadata(
            name=name or _default_tool_name(fn.__name__),
            description=description.strip(),
            proxy=proxy,
        )
        return fn

    return decorator


def get_tool_metadata(fn: Any) -> ToolMetadata | None:
    """Return tool metadata for a function or bound method, if present."""
    target = getattr(fn, "__func__", fn)
    return getattr(target, "_tool_meta", None)


def format_tool_line(fn: Callable) -> str:
    """Render ``- `name(sig)`: description`` for a decorated tool (or ``""``)."""
    meta = get_tool_metadata(fn)
    if meta is None:
        return ""
    try:
        sig = str(inspect.signature(fn))
    except (TypeError, ValueError):
        sig = "(...)"
    return f"- `{meta.name}{sig}`: {meta.description}"


from rflow.tools.filesystem import FILE_TOOLS  # noqa: E402

__all__ = [
    "FILE_TOOLS",
    "ToolMetadata",
    "format_tool_line",
    "get_tool_metadata",
    "tool",
]
