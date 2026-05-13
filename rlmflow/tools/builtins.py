"""Engine-bound built-in tools: ``done``, ``wait``, ``delegate``.

Each tool is a Python closure created per-runtime and bound to a specific
``runtime.env`` dict (and, for ``delegate``, the owning :class:`RLMFlow`).
They are registered through the normal :meth:`Runtime.register_tool` path —
``LocalRuntime`` injects them straight into the REPL namespace; remote
runtimes expose proxy stubs that round-trip back to the host closure.

The ``env`` dict captured here is the same object the engine reads back
via ``runtime.env`` after each execution to discover ``DONE_RESULT`` and
``DELEGATED`` agent ids.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rlmflow.graph import ChildHandle, WaitRequest
from rlmflow.tools import tool

if TYPE_CHECKING:
    from rlmflow.rlm import RLMFlow


class DoneSignal(Exception):
    """Internal control-flow signal raised by ``done()`` to stop execution."""


def make_done(env: dict[str, Any]):
    """Closure that records the final answer and stops the current block."""

    @tool("Mark the current agent as finished.")
    def done(message: str) -> str:
        if env.get("DONE_RESULT") is None:
            env["DONE_RESULT"] = str(message).strip()
            print(f"[done] {env['DONE_RESULT']}")
        raise DoneSignal(env["DONE_RESULT"])

    return done


def make_wait():
    """Closure that packages :class:`ChildHandle`s into a :class:`WaitRequest`."""

    @tool("Wait for delegated children. Must be called with `yield`.")
    def wait(*handles: ChildHandle) -> WaitRequest:
        return WaitRequest(agent_ids=[h.agent_id for h in handles])

    return wait


def make_delegate(flow: "RLMFlow", env: dict[str, Any]):
    """Closure that calls :meth:`RLMFlow.spawn_child` and tracks the new id."""

    @tool("Delegate a subtask to a named child agent.")
    def delegate(
        name: str,
        query: str,
        context: str,
        *,
        max_iterations: int | None = None,
        model: str = "default",
    ) -> ChildHandle | str:
        handle = flow.spawn_child(
            env["AGENT_ID"],
            env["PARENT_NODE_ID"],
            name,
            query,
            context,
            max_iterations=max_iterations,
            model=model,
        )
        if isinstance(handle, str):
            return handle
        env.setdefault("DELEGATED", []).append(handle.agent_id)
        return handle

    return delegate


__all__ = ["DoneSignal", "make_delegate", "make_done", "make_wait"]
