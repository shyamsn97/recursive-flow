"""RLMFlow — one recursive interpreter over typed agent states.

The engine reads and writes through :class:`~rlmflow.workspace.session.Session`
(per-agent invariants + per-turn states) and returns
:class:`~rlmflow.graph.Graph` after every step. A :class:`Graph` is one
agent, frozen, with all its per-run invariants as flat fields plus its
``states`` trajectory and ``children`` sub-Graphs. The engine itself
holds no graph state — the session is the source of truth.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from rlmflow.graph import (
    ActionNode,
    ChildHandle,
    ErrorNode,
    Graph,
    Node,
    ObservationNode,
    QueryNode,
    ResultNode,
    ResumeNode,
    RuntimeRef,
    SupervisingNode,
)
from rlmflow.llm import LLMClient, LLMUsage
from rlmflow.pool import CallablePool, Pool, SequentialPool, ThreadPool
from rlmflow.prompts.default import BASELINE_BUILDER, DEFAULT_BUILDER
from rlmflow.prompts.messages import (
    CONTEXT_HINT_ABSENT,
    CONTEXT_HINT_PRESENT,
    CONTINUE_ACTION,
    DEFAULT_QUERY,
    EXECUTION_OUTPUT,
    FINAL_ANSWER_ACTION,
    FIRST_ACTION,
    NO_CODE_BLOCK,
    ORPHANED_DELEGATES,
    STATUS_DEPTH_MID,
    STATUS_DEPTH_NEAR_MAX,
    STATUS_DEPTH_ROOT,
    TRUNCATION_SESSION_HINT,
    TRUNCATION_SUMMARY,
)
from rlmflow.runtime import Runtime
from rlmflow.tools.builtins import make_delegate, make_done, make_wait
from rlmflow.utils import OrphanedDelegatesError, check_yield_errors, find_code_blocks
from rlmflow.workspace import (
    Context,
    ContextVariable,
    InMemoryContext,
    InMemorySession,
    Session,
    SessionVariable,
    Workspace,
)

ROOT_RUNTIME_ID = "root"


@dataclass
class RLMConfig:
    """Engine-level knobs."""

    max_depth: int = 5
    max_iterations: int = 30
    max_output_length: int = 12000
    max_messages: int | None = None
    max_concurrency: int | None = None
    child_max_iterations: int | None = None
    single_block: bool = True
    system_prompt: str | None = None
    max_budget: int | None = None


def append_node(session: Session, graph: Graph, node: Node) -> Node:
    """Append ``node`` to ``graph``'s trajectory with the next ``seq``.

    Single source of truth for sequence numbering. Call sites never set
    ``agent_id`` or ``seq`` themselves — pass a node with the payload fields
    populated and this helper assigns identity.
    """
    next_seq = (graph.states[-1].seq + 1) if graph.states else 0
    fields_dict = node.model_dump(exclude={"id", "agent_id", "seq"}, mode="python")
    fixed = node.__class__(agent_id=graph.agent_id, seq=next_seq, **fields_dict)
    session.write_state(fixed)
    return fixed


def unique_child_id(parent_aid: str, name: str, existing: set[str]) -> str:
    base = f"{parent_aid}.{name}"
    if base not in existing:
        return base
    i = 1
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


def create_pool(config: RLMConfig, pool: Pool | Callable | None = None) -> Pool:
    if pool is not None:
        return pool if hasattr(pool, "execute") else CallablePool(pool)
    if config.max_concurrency is not None:
        return ThreadPool(config.max_concurrency)
    return SequentialPool()


# ── scheduler ────────────────────────────────────────────────────────


class NodeScheduler:
    """Pick the agents that can take a step right now.

    Walks the graph top-down from root. A supervising agent is
    "runnable" iff all the children it is ``waiting_on`` are terminal;
    otherwise the scheduler recurses into those still-running children.
    """

    def __init__(self, pool: Pool | None = None) -> None:
        self.pool = pool

    def runnable_agents(self, graph: Graph) -> list[str]:
        runnable: list[str] = []

        def visit(aid: str) -> None:
            agent = graph.agents[aid]
            if agent.finished:
                return
            cur = agent.current()
            if cur is None:
                return
            if isinstance(cur, SupervisingNode):
                waiting = [
                    graph.agents[child_aid]
                    for child_aid in cur.waiting_on
                    if child_aid in graph.agents
                ]
                if all(child.finished for child in waiting):
                    runnable.append(aid)
                    return
                for child in waiting:
                    if not child.finished:
                        visit(child.agent_id)
                return
            runnable.append(aid)

        visit(graph.agent_id)
        return runnable


# ── engine ───────────────────────────────────────────────────────────


class RLMFlow(LLMClient):
    """Recursive language-model flow engine.

    Holds the prompt builder, runtime sessions, pool, and persistence
    handles. The execution graph itself lives in the session — every step
    reloads it through :meth:`~rlmflow.workspace.session.Session.load_graph`.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        runtime: Runtime | None = None,
        config: RLMConfig | None = None,
        runtime_factory: Callable[[], Runtime] | None = None,
        llm_clients: dict[str, dict] | None = None,
        pool: Any = None,
        prompt_builder: Any = None,
        *,
        workspace: Workspace | None = None,
        node_scheduler: NodeScheduler | None = None,
    ) -> None:
        if workspace is not None and runtime is None:
            runtime = workspace.materialize_runtime()
        if runtime is None:
            raise ValueError("RLMFlow requires either runtime= or workspace=.")
        if workspace is None:
            runtime_workspace = getattr(runtime, "workspace", None)
            if runtime_workspace is not None:
                runtime_root = Path(runtime_workspace).resolve()
                if runtime_root != Path.cwd().resolve():
                    workspace = Workspace.create(runtime_root)

        self.llm_client = llm_client
        self.runtime = runtime
        self.workspace = workspace
        self.session: Session = workspace.session if workspace else InMemorySession()
        self.context: Context = workspace.context if workspace else InMemoryContext()
        self.config = config or RLMConfig()
        self.runtime_factory = runtime_factory
        default_builder = (
            BASELINE_BUILDER if self.config.max_depth == 0 else DEFAULT_BUILDER
        )
        self.prompt_builder = prompt_builder or default_builder
        self.pool = create_pool(self.config, pool)
        self.node_scheduler = node_scheduler or NodeScheduler()

        self.llm_clients: dict[str, LLMClient] = {}
        self.model_descriptions: dict[str, str] = {}
        for key, entry in (llm_clients or {}).items():
            self.llm_clients[key] = entry["model"]
            if "description" in entry:
                self.model_descriptions[key] = entry["description"]
        if "default" not in self.llm_clients:
            self.llm_clients["default"] = self.llm_client

        self._runtime_sessions: dict[str, Runtime] = {ROOT_RUNTIME_ID: runtime}
        self._terminate_requested: set[str] = set()
        self.register_tools(runtime)

    # ── lifecycle ────────────────────────────────────────────────────

    def start(
        self,
        query: str | None = None,
        *,
        context: str | None = None,
        contexts: dict[str, str] | None = None,
        context_metadata: dict[str, Any] | None = None,
        agent_id: str = "root",
    ) -> Graph:
        query = query or DEFAULT_QUERY

        self.context.write(
            "context",
            context if context is not None else "",
            agent_id=agent_id,
            metadata=context_metadata,
        )
        for key, value in (contexts or {}).items():
            self.context.write(key, value, agent_id=agent_id)

        context_hint = CONTEXT_HINT_PRESENT if context else CONTEXT_HINT_ABSENT
        root = Graph(
            agent_id=agent_id,
            branch_id=self.workspace.branch_id if self.workspace else "main",
            depth=0,
            query=query,
            system_prompt=self.build_system_prompt_for(
                query=query, agent_id=agent_id, depth=0
            ),
            config=self.node_config(),
            workspace=self.workspace.ref() if self.workspace else None,
            runtime=RuntimeRef(id=ROOT_RUNTIME_ID),
        )
        self.session.write_agent(root)
        append_node(
            self.session,
            root,
            QueryNode(
                content=FIRST_ACTION.format(query=query, context_hint=context_hint)
            ),
        )
        return self.session.load_graph()

    def run(self, query: str | None = None, **kwargs) -> str:
        graph = self.start(query, **kwargs)
        while not graph.finished:
            graph = self.step(graph)
        return graph.result()

    def chat(self, messages: list[dict[str, str]], *args, **kwargs) -> str:
        query = next(
            (
                m.get("content", "")
                for m in reversed(messages)
                if m.get("role") == "user"
            ),
            "",
        )
        return self.run(query)

    def step(self, graph: Graph) -> Graph:
        """Advance the run by one synchronized batch.

        All currently-runnable agents are stepped (in parallel if a
        :class:`Pool` is configured). Returns a freshly-loaded
        :class:`Graph` snapshot.
        """
        runnable = self.node_scheduler.runnable_agents(graph)
        if not runnable:
            return graph
        tasks = [(aid, (lambda aid=aid: self._step_agent(aid))) for aid in runnable]
        self.pool.execute(tasks)
        return self.session.load_graph()

    def terminate(self, graph: Graph) -> Graph:
        """Mark every still-running agent for a final-answer turn.

        Equivalent to giving every agent one last chance to emit ``done()``.
        The engine then drives those agents to terminal states as normal.
        """
        for aid in graph.agents:
            if not graph.agents[aid].finished:
                self._terminate_requested.add(aid)
        return self.session.load_graph()

    # ── per-agent step dispatcher ────────────────────────────────────

    def _step_agent(self, agent_id: str) -> None:
        full = self.session.load_graph()
        if agent_id not in full.agents:
            return
        graph = full.agents[agent_id]
        if graph.finished:
            return

        over = self._budget_exceeded(full)
        if over is not None:
            append_node(
                self.session,
                graph,
                ResultNode(result=f"[budget exceeded: {over} tokens]"),
            )
            return

        cur = graph.current()
        if cur is None:
            return
        if isinstance(cur, SupervisingNode):
            self._step_supervising(graph, cur)
            return
        if isinstance(cur, ObservationNode):
            self._step_observation(graph, cur)
            return
        raise TypeError(
            f"Unexpected current state type for agent {agent_id!r}: "
            f"{type(cur).__name__}"
        )

    # ── observation → action → next state ────────────────────────────

    def _step_observation(self, graph: Graph, last: ObservationNode) -> None:
        iteration = self._iteration_count(graph)
        max_iter = graph.config.get("max_iterations", self.config.max_iterations)
        terminate = iteration >= max_iter or graph.agent_id in self._terminate_requested

        action_or_error = self._reply_to(graph, last, force_final=terminate)
        action_state = append_node(self.session, graph, action_or_error)
        if isinstance(action_state, ErrorNode):
            return
        # Run the freshly-recorded ActionNode immediately so the agent
        # observes the result in the same step.
        full = self.session.load_graph()
        graph = full.agents[graph.agent_id]
        self._step_action(graph, action_state)

    def _reply_to(
        self,
        graph: Graph,
        last: ObservationNode,
        *,
        force_final: bool,
    ) -> ActionNode | ErrorNode:
        messages = self.build_messages(graph, force_final=force_final)
        client = self.llm_client_for(graph)
        raw = self.call_llm(messages, client=client)
        usage = client.last_usage or LLMUsage()
        code = self.extract_code(raw)
        next_seq = last.seq + 1
        if not code:
            return ErrorNode(
                agent_id=graph.agent_id,
                seq=next_seq,
                content=NO_CODE_BLOCK,
                error="no_code_block",
            )
        return ActionNode(
            agent_id=graph.agent_id,
            seq=next_seq,
            reply=raw,
            code=code,
            model=getattr(client, "model", None),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )

    def _step_action(self, graph: Graph, action: ActionNode) -> None:
        err = check_yield_errors(action.code)
        if err:
            append_node(
                self.session,
                graph,
                ErrorNode(code=action.code, content=err, error="invalid_yield"),
            )
            return

        runtime = self.inject_env(graph, action)
        suspended, raw = runtime.start_code(action.code)
        raw = self._truncate(raw)
        env = runtime.env
        delegated = list(env.get("DELEGATED") or [])
        done_result = env.get("DONE_RESULT")

        if delegated and not suspended and done_result is None:
            msg = ORPHANED_DELEGATES.format(names=", ".join(delegated))
            base = raw if isinstance(raw, str) else ""
            output = self._truncate(
                runtime.execute(f"raise OrphanedDelegatesError({msg!r})")
            )
            content = (base + "\n\n" + output).strip()
            append_node(
                self.session,
                graph,
                ErrorNode(
                    code=action.code,
                    content=self.format_exec_output(content),
                    error="orphaned_delegates",
                ),
            )
            return

        if done_result is not None:
            append_node(self.session, graph, ResultNode(result=done_result.strip()))
            return

        if suspended:
            request, pre_output = raw
            append_node(
                self.session,
                graph,
                SupervisingNode(
                    code=action.code,
                    output=pre_output,
                    waiting_on=list(request.agent_ids),
                ),
            )
            return

        output = raw if isinstance(raw, str) else ""
        if not output.strip():
            output = "(no output)"
        append_node(
            self.session,
            graph,
            ObservationNode(
                code=action.code,
                output=output,
                content=self.format_exec_output(output),
            ),
        )

    # ── supervising → resume / step children ─────────────────────────

    def _step_supervising(self, graph: Graph, last: SupervisingNode) -> None:
        if not self._can_resume(graph, last):
            # Children still need to advance. The scheduler picks them up
            # on the next outer step; nothing for this agent to do now.
            return

        children = [graph.agents[aid] for aid in last.waiting_on if aid in graph.agents]
        results = [
            child.result() if isinstance(child.current(), ResultNode) else ""
            for child in children
        ]

        runtime = self.inject_env(graph, last)
        suspended, raw = runtime.resume_code(results)
        raw = self._truncate(raw)
        env = runtime.env
        done_result = env.get("DONE_RESULT")

        if suspended:
            request, output = raw
        else:
            output = raw if isinstance(raw, str) else ""
        if not output.strip():
            output = "(no output)"

        append_node(
            self.session,
            graph,
            ResumeNode(
                code=last.code,
                output=output,
                content=self.format_exec_output(output),
                resumed_from=list(last.waiting_on),
            ),
        )
        graph = self.session.load_graph().agents[graph.agent_id]

        if done_result is not None:
            append_node(self.session, graph, ResultNode(result=done_result.strip()))
            return

        if suspended:
            append_node(
                self.session,
                graph,
                SupervisingNode(
                    code=last.code,
                    output=output,
                    waiting_on=list(request.agent_ids),
                ),
            )
            return

    # ── recording helpers ────────────────────────────────────────────

    def _iteration_count(self, graph: Graph) -> int:
        return sum(isinstance(s, ActionNode) for s in graph.states)

    def _budget_exceeded(self, graph: Graph) -> int | None:
        """Return total tokens if the run is over budget, else ``None``."""
        if self.config.max_budget is None:
            return None
        total = graph.total_tokens()
        return total if total >= self.config.max_budget else None

    def _can_resume(self, graph: Graph, supervising: SupervisingNode) -> bool:
        """``graph`` is the supervising agent's sub-graph (children are inside)."""
        if not supervising.waiting_on:
            return False
        for aid in supervising.waiting_on:
            if aid not in graph.agents:
                return False
            if not isinstance(graph.agents[aid].current(), ResultNode):
                return False
        return True

    # ── runtime sessions ─────────────────────────────────────────────

    def runtime_for(self, ref: RuntimeRef | None) -> Runtime:
        session_id = ref.id if ref is not None else ROOT_RUNTIME_ID
        return self._runtime_sessions[session_id]

    def create_runtime_session(
        self, parent_runtime: Runtime, *, agent_id: str
    ) -> RuntimeRef:
        session_id = f"{agent_id}:{uuid4().hex[:8]}"
        runtime = (
            self.runtime_factory() if self.runtime_factory else parent_runtime.clone()
        )
        self._runtime_sessions[session_id] = runtime
        self.register_tools(runtime)
        return RuntimeRef(id=session_id)

    def inject_env(self, graph: Graph, node: Node) -> Runtime:
        """Reset per-execution state on the runtime and inject env-style vars
        into the REPL namespace before running an action or resume.

        ``runtime.env`` is the host-side dict shared with ``done`` / ``delegate``
        closures (cleared + seeded each call). The same per-agent facts plus
        ``CONTEXT`` / ``SESSION`` are also pushed into the REPL namespace so
        user code can reference them by bare name.
        """
        runtime = self.runtime_for(graph.runtime)
        facts = {
            "AGENT_ID": graph.agent_id,
            "DEPTH": graph.depth,
            "MAX_DEPTH": self.config.max_depth,
            "PARENT_NODE_ID": node.id,
        }
        runtime.env.clear()
        runtime.env.update({**facts, "DONE_RESULT": None, "DELEGATED": []})

        repl_vars = {
            **facts,
            "OrphanedDelegatesError": OrphanedDelegatesError,
            "SESSION": SessionVariable(
                self.session,
                agent_id=graph.agent_id,
                node_id=node.id,
                branch_id=graph.branch_id,
            ),
            "CONTEXT": ContextVariable(self.context, agent_id=graph.agent_id),
        }
        for name, value in repl_vars.items():
            runtime.inject(name, value)
        return runtime

    def _truncate(self, raw: object) -> object:
        """Cap REPL output at ``max_output_length`` chars; passthrough non-strings."""
        if isinstance(raw, str) and len(raw) > self.config.max_output_length:
            return raw[: self.config.max_output_length] + "\n...<truncated>"
        return raw

    # ── LLM / messages ───────────────────────────────────────────────

    def build_messages(
        self,
        graph: Graph,
        *,
        force_final: bool = False,
    ) -> list[dict[str, str]]:
        system_content = graph.system_prompt or self.build_system_prompt_for(
            query=graph.query,
            agent_id=graph.agent_id,
            depth=graph.depth,
        )
        system = {"role": "system", "content": system_content}

        try:
            payload = self.context.read("context", agent_id=graph.agent_id)
        except KeyError:
            payload = ""
        context_hint = CONTEXT_HINT_PRESENT if payload else CONTEXT_HINT_ABSENT

        msgs: list[dict[str, str]] = []
        for state in graph.states:
            if isinstance(state, (ResultNode, SupervisingNode)):
                continue
            if isinstance(state, ObservationNode):
                msgs.append({"role": "user", "content": state.content})
            elif isinstance(state, ActionNode):
                msgs.append({"role": "assistant", "content": state.reply})

        cap = self.config.max_messages
        if cap and len(msgs) > cap:
            msgs = [
                {
                    "role": "user",
                    "content": TRUNCATION_SUMMARY.format(
                        query=graph.query,
                        total=len(msgs),
                        cap=cap,
                        session_hint=TRUNCATION_SESSION_HINT,
                    ),
                }
            ] + msgs[-cap:]

        if force_final:
            msgs.append({"role": "user", "content": FINAL_ANSWER_ACTION})
        elif self._iteration_count(graph) > 0:
            msgs.append(
                {
                    "role": "user",
                    "content": CONTINUE_ACTION.format(
                        query=graph.query, context_hint=context_hint
                    ),
                }
            )
        return [system] + msgs

    def llm_client_for(self, graph: Graph) -> LLMClient:
        model = graph.config.get("model", "default")
        return self.llm_clients.get(model, self.llm_client)

    def call_llm(
        self,
        messages: list[dict[str, str]],
        *,
        client: LLMClient | None = None,
    ) -> str:
        active_client = client or self.llm_client
        result = "".join(active_client.stream(messages))
        self.last_usage = active_client.last_usage
        return result

    def extract_code(self, text: str) -> str | None:
        blocks = find_code_blocks(text)
        if not blocks:
            return None
        return blocks[0] if self.config.single_block else "\n\n".join(blocks)

    def format_exec_output(self, output: str) -> str:
        return EXECUTION_OUTPUT.format(output=output or "(no output)")

    # ── prompt building ──────────────────────────────────────────────

    def build_system_prompt_for(
        self,
        *,
        query: str,
        agent_id: str,
        depth: int,
        config: dict[str, Any] | None = None,
    ) -> str:
        stub = Graph(
            agent_id=agent_id,
            depth=depth,
            query=query,
            config=config or self.node_config(),
        )
        return self.build_system_prompt(stub)

    def build_system_prompt(self, graph: Graph) -> str:
        """Render the system prompt for the agent rooted in ``graph``."""
        if self.config.system_prompt:
            return self.config.system_prompt
        return self.prompt_builder.build(
            tools=self.build_tools_section(),
            status=self.build_status_section(graph),
        )

    def build_tools_section(self) -> str:
        baseline = self.config.max_depth == 0
        tool_defs = self.runtime.get_tool_defs()
        if baseline:
            tool_defs = [t for t in tool_defs if t.name not in ("delegate", "wait")]
        lines = [
            f"- `{tool_def.name}{tool_def.signature}`: {tool_def.description}"
            for tool_def in tool_defs
        ]
        if len(self.llm_clients) > 1 and not baseline:
            lines.append("\nAvailable models for `delegate(model=...)`:")
            for key in sorted(self.llm_clients):
                desc = self.model_descriptions.get(key)
                lines.append(f"- `{key}`: {desc}" if desc else f"- `{key}`")
        modules = self.runtime.available_modules()
        if modules:
            lines.append(f"\nPre-imported: `{'`, `'.join(modules)}`")
        return "\n".join(lines)

    def build_status_section(self, graph: Graph) -> str:
        max_depth = graph.config.get("max_depth", self.config.max_depth)
        if max_depth == 0:
            return "Baseline mode: no sub-agents available. Do all work directly in this REPL."
        note = f"You are at recursion depth **{graph.depth}** of max **{max_depth}**."
        if graph.depth == 0:
            note += STATUS_DEPTH_ROOT
        elif graph.depth >= max_depth - 1:
            note += STATUS_DEPTH_NEAR_MAX
        elif graph.depth > 0:
            note += STATUS_DEPTH_MID
        return note

    def node_config(self) -> dict[str, Any]:
        return {
            "model": "default",
            "max_depth": self.config.max_depth,
            "max_iterations": self.config.max_iterations,
            "max_output_length": self.config.max_output_length,
            "max_messages": self.config.max_messages,
            "child_max_iterations": self.config.child_max_iterations,
            "single_block": self.config.single_block,
            "max_budget": self.config.max_budget,
        }

    def child_config(
        self,
        parent: Graph,
        max_iterations: int | None,
    ) -> dict[str, Any]:
        child_iters = (
            max_iterations
            or self.config.child_max_iterations
            or max(
                1,
                parent.config.get("max_iterations", self.config.max_iterations) // 3,
            )
        )
        return {**parent.config, "max_iterations": child_iters}

    # ── core tools ───────────────────────────────────────────────────

    def register_tools(self, runtime: Runtime | None = None) -> None:
        """Bind ``done`` / ``wait`` / ``delegate`` closures to ``runtime.env``.

        Closures live in :mod:`rlmflow.tools.builtins` and capture the
        same ``env`` dict the engine reads back after each execution
        (so ``DONE_RESULT`` / ``DELEGATED`` round-trip cleanly).
        """
        runtime = runtime or self.runtime
        runtime.inject("OrphanedDelegatesError", OrphanedDelegatesError)
        runtime.register_tool(make_done(runtime.env), core=True)
        runtime.register_tool(make_wait(), core=True)
        runtime.register_tool(make_delegate(self, runtime.env), core=True)

    def spawn_child(
        self,
        parent_agent_id: str,
        parent_node_id: str,
        name: str,
        query: str,
        context: str,
        *,
        max_iterations: int | None = None,
        model: str = "default",
    ) -> ChildHandle | str:
        """Spawn a child agent under ``parent_agent_id`` and return its handle.

        Public, overridable seam: subclass :class:`RLMFlow` and override this
        method to customize how delegation creates child agents. The default
        creates a child :class:`Graph`, allocates a new runtime session,
        writes the initial :class:`QueryNode`, and returns a
        :class:`ChildHandle`. Returns a refusal string instead of a handle if
        the child cannot be created (max depth reached, unknown model, …).
        """
        parent = self.session.load_graph().agents[parent_agent_id]
        if parent.depth >= self.config.max_depth:
            return f"[refused: max depth {self.config.max_depth}] Do this directly."
        if model not in self.llm_clients:
            keys = ", ".join(sorted(self.llm_clients))
            return f"[error: unknown model {model!r}. available: {keys}]"

        child_aid = unique_child_id(parent_agent_id, name, set(parent.children))
        self.context.write("context", context, agent_id=child_aid)
        runtime_ref = self.create_runtime_session(
            self.runtime_for(parent.runtime), agent_id=child_aid
        )
        context_hint = CONTEXT_HINT_PRESENT if context else CONTEXT_HINT_ABSENT

        child_config = {**self.child_config(parent, max_iterations), "model": model}
        child_graph = Graph(
            agent_id=child_aid,
            branch_id=parent.branch_id,
            depth=parent.depth + 1,
            query=query,
            system_prompt=self.build_system_prompt_for(
                query=query,
                agent_id=child_aid,
                depth=parent.depth + 1,
                config=child_config,
            ),
            config=child_config,
            workspace=parent.workspace,
            runtime=runtime_ref,
            model=None,
            parent_agent_id=parent.agent_id,
            parent_node_id=parent_node_id,
        )
        self.session.write_agent(child_graph)
        append_node(
            self.session,
            child_graph,
            QueryNode(
                content=FIRST_ACTION.format(query=query, context_hint=context_hint)
            ),
        )
        return ChildHandle(child_aid)


__all__ = ["NodeScheduler", "RLMConfig", "RLMFlow"]
