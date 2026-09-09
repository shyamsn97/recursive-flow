"""Reserved REPL builtins as a prompt-owning toolset.

``finish`` and ``launch_subagent`` are per-node closures Flow already injects.
The one-shot query tools are bound only when their corresponding Flow flags are
enabled. This class is the prompt source for those names, not their implementation:
every ``@tool`` method is ``inject=False``.
"""

from __future__ import annotations

import textwrap
from typing import Any

from rlmflow.graph.config import can_spawn
from rlmflow.tools.agents import AgentHandle
from rlmflow.tools.tools import format_tool_line, tool, toolset, toolset_members

AGENTS_TEXT = """\
`AGENTS` is a read-only snapshot of this run's recursive agent tree:

- `AGENTS.get()` returns you; `AGENTS.get(id_or_path_or_name)` finds another agent.
- `AGENTS.get_parent()`, `.get_siblings()`, and `.get_children()` return
  `AgentInfo` objects relative to you (or pass an agent selector).
- `AgentInfo.status` is `running`, `waiting`, `idle`, or `completed`.
  `.result()` returns an already-completed result and raises while one is still
  in flight; `await agent.wait_for_result()` waits automatically.
- `AGENTS.print_graph(show_results=True)` prints the whole tree with statuses and
  bounded result previews.

The snapshot is refreshed before each REPL action. It does not wait, message,
cancel, or steer agents, and repeated queries within one action are not fresher.
"""

STRUCTURED_OUTPUT_OPTION_TEXT = """\
Child results are JSON-compatible Python values by default. Pass `output_schema`
only when a validated typed result is needed; use explicit `properties`,
`required`, and `additionalProperties: false`.
"""


def _agent(node: Any) -> Any:
    if node is None:
        return None
    parent = getattr(node, "parent_agent", None)
    if parent is not None:
        return parent
    return node if getattr(node, "config", None) is not None else None


def _repl(code: str) -> str:
    return f"```repl\n{textwrap.dedent(code).strip()}\n```"


def _use_llm_query(flow: Any) -> bool:
    return flow is None or bool(getattr(flow, "use_llm_query", False))


def _use_llm_query_batched(flow: Any) -> bool:
    if not _use_llm_query(flow):
        return False
    return flow is None or bool(getattr(flow, "use_llm_query_batched", False))


def _delegated_model(flow: Any) -> str:
    name = getattr(flow, "delegation_model", None) if flow is not None else None
    return name or "default"


@toolset("REPL and Delegation", placement="section")
class BuiltIns:
    @tool(
        "Submit the final answer and end the run immediately.",
        inject=False,
    )
    def finish(self, value: object) -> None:
        raise RuntimeError("finish is bound by Flow, not by BuiltIns")

    @tool(
        "Recursive RLM sub-call with its own persistent REPL. The parent only pays "
        "the small result. Use when a subtask needs multi-step reasoning or code, "
        "not a one-shot answer.",
        inject=False,
    )
    async def launch_subagent(
        self,
        goal: str,
        model: str,
        name: str | None = None,
        inputs: dict[str, str] | None = None,
        output_schema: object | None = None,
        prompt_profile: str | None = None,
        reuse_repl: bool = False,
    ) -> AgentHandle:
        raise RuntimeError("launch_subagent is bound by Flow, not by BuiltIns")

    @tool(
        "A single sub-LLM completion (no REPL, no iteration). Use for extraction, "
        "summarization, or Q&A over a chunk of text.",
        inject=False,
    )
    async def llm_query(
        self,
        prompt: str,
        *,
        model: str = "default",
        output_schema: object | None = None,
    ) -> object:
        raise RuntimeError("llm_query is bound by Flow(use_llm_query=True)")

    @tool(
        "Concurrently call several LLM calls in parallel over a list of prompts; "
        "same order out as in.",
        inject=False,
    )
    async def llm_query_batched(
        self,
        prompts: list[str],
        *,
        model: str = "default",
        output_schema: object | None = None,
    ) -> list:
        raise RuntimeError(
            "llm_query_batched is bound by Flow(" "use_llm_query=True, use_llm_query_batched=True)"
        )

    @tool("List every variable currently in the REPL.", inject=False, name="SHOW_VARS")
    def show_vars(self) -> str:
        raise RuntimeError("SHOW_VARS is bound by the runtime")

    def description(self, flow: Any, node: Any) -> str:
        members = dict(toolset_members(self))
        agent = _agent(node)
        parts: list[str] = []

        core = [
            format_tool_line(members["SHOW_VARS"]),
            format_tool_line(members["finish"]),
            "",
            "Inspect this turn; call `finish(...)` on a later turn:",
            _repl(
                'print({name: value[:500] for name, value in INPUTS.items()} or "(empty INPUTS)")'
            ),
        ]
        parts.append("\n".join(core))

        if flow is None or can_spawn(agent):
            model = _delegated_model(flow)
            launch = [
                format_tool_line(members["launch_subagent"]),
                "",
                "Needs its own REPL (search, compute, iterate). Not the same extract over many chunks:",
                _repl(
                    f"""\
                    chunk = next(iter(INPUTS.values()))
                    handle = await launch_subagent(
                        "Search this dossier, compute the answer, return only that value.",
                        model="{model}",
                        inputs={{"dossier": chunk}},
                    )
                    print(await handle.wait_for_result())
                    """
                ),
            ]
            parts.append("\n".join(launch))

        if _use_llm_query(flow):
            query = [format_tool_line(members["llm_query"])]
            if _use_llm_query_batched(flow):
                query.append(format_tool_line(members["llm_query_batched"]))
            query.extend(
                [
                    "",
                    "One-shot extract or classify over a chunk — no REPL:",
                    _repl(
                        """\
                        chunk = next(iter(INPUTS.values()))[:8000]
                        print(await llm_query("Extract the answer.\\n\\n" + chunk))
                        """
                    ),
                ]
            )
            if _use_llm_query_batched(flow):
                query.extend(
                    [
                        "",
                        "Same one-shot over many independent chunks — batch, don't spawn:",
                        _repl(
                            """\
                            text = next(iter(INPUTS.values()))
                            chunks = [text[i:i+8000] for i in range(0, len(text), 8000)]
                            prompts = ["Return matching IDs only.\\n\\n" + c for c in chunks]
                            print(await llm_query_batched(prompts))
                            """
                        ),
                    ]
                )
            parts.append("\n".join(query))

        if flow is not None and getattr(flow, "use_agent_tree", False):
            parts.append(
                "\n\n".join(
                    [
                        AGENTS_TEXT.strip(),
                        "Query the snapshot, then wait on a child:\n"
                        + _repl(
                            """\
                            AGENTS.print_graph(show_results=True)
                            child = AGENTS.get_children()[0]
                            print(await child.wait_for_result())
                            """
                        ),
                    ]
                )
            )

        return "\n\n".join(part for part in parts if part)

    def child_schema_note(self, flow: Any, node: Any) -> str:
        """Child `output_schema` guidance; a separate category from the tool list."""
        if flow is None or not getattr(flow, "enable_structured_output", False):
            return ""
        if not can_spawn(_agent(node)):
            return ""
        model = _delegated_model(flow)
        return "\n\n".join(
            [
                STRUCTURED_OUTPUT_OPTION_TEXT.strip(),
                _repl(
                    f"""\
                    handle = await launch_subagent(
                        "Count items",
                        model="{model}",
                        output_schema={{
                            "type": "object",
                            "properties": {{"n": {{"type": "number"}}}},
                            "required": ["n"],
                            "additionalProperties": False,
                        }},
                    )
                    print((await handle.wait_for_result())["n"])
                    """
                ),
            ]
        )


__all__ = ["AGENTS_TEXT", "BuiltIns", "STRUCTURED_OUTPUT_OPTION_TEXT"]
