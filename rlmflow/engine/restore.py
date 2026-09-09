"""Rebuild REPL namespaces from recorded ``ExecAction``s."""

from __future__ import annotations

from typing import Any

from rlmflow.graph.nodes import AgentStart, ExecAction
from rlmflow.runtime.env import RLMFLOW_REPLAY


async def replay(flow: Any, root: AgentStart) -> None:
    """Rebuild the namespaces of a graph this flow did not run.

    Appends nothing: ``launch_subagent`` resolves children already attached
    to each action and reads finished results from their terminal nodes. A block
    that failed the first time fails the same way here, leaving the same partial
    bindings, which is why output and errors are discarded.
    """
    actions = [
        node
        for node in root.walk()
        if isinstance(node, ExecAction) and node is not node.parent_agent.frontier
    ]
    actions.sort(
        key=lambda node: (
            node.repl_execution_order is None,
            node.repl_execution_order or 0,
            node.created_at,
        )
    )
    for node in actions:
        agent = node.parent_agent
        if agent.terminal and not agent.config.reuse_repl:
            continue  # it answered; nothing will run in this namespace again
        repl = flow.runtime.repl_for(agent)
        repl.structured_output = agent.config.output_schema is not None
        repl.seed(flow.build_tools(node), agent.config.inputs)
        repl.update_env({RLMFLOW_REPLAY: "1"})
        try:
            await flow.runtime.execute(node, node.code)
        finally:
            repl.update_env({RLMFLOW_REPLAY: "0"})


async def ensure_replayed(flow: Any, agent: AgentStart) -> None:
    """Restore an unfinished namespace immediately before its first execution."""
    if id(agent) in flow._restored_agents:
        return
    async with flow._restore_lock:
        if id(agent) in flow._restored_agents:
            return
        if flow.runtime.get(agent) is not None:
            flow._restored_agents.add(id(agent))
            return

        root = agent
        while root.config.reuse_repl and root.parent is not None:
            parent = root.parent.parent_agent
            if parent is None or flow.runtime.get(parent) is not None:
                break
            root = parent
        await flow.replay(root)
        flow._restored_agents.update(
            id(node) for node in root.walk() if isinstance(node, AgentStart)
        )


__all__ = ["ensure_replayed", "replay"]
