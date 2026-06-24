"""The :class:`Flow` engine — a minimal recursive language-model loop.

One agent is an LLM in a loop with a stateful REPL: ask the model, run the
```repl block it emits, feed stdout back, stop on ``done(...)``. Agents
recurse: ``await launch_subagents([...])`` spawns child agents and suspends
the parent's REPL coroutine until they finish, then resumes it with their
results.

The whole tree advances by *synchronized steps*. Each :meth:`Flow.step`:

1. asks the scheduler which agents can move right now (leaves, plus
   supervisors whose awaited children have all finished);
2. advances each of them by one action in parallel through an execution pool.

A "leaf" advance is one of: call the LLM, run the emitted code, or resume a
suspended coroutine. The graph is held in memory — no persistence, no
cold-start replay.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterable

import rflow.prompts.projection as prompt_projection
from rflow.base import BaseFlow
from rflow.clients.llm import LLMClient, LLMUsage
from rflow.clients.llm_channel import LLMChannel
from rflow.code import check_wait_syntax, find_code_blocks
from rflow.graph import (
    ChildHandle,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    Graph,
    LLMAction,
    LLMOutput,
    Node,
    ResumeAction,
    SupervisingOutput,
    UserQuery,
)
from rflow.graph.actions import Action, ActionPlan, CallLLM, Exec, Recover, Resume
from rflow.integrations.structured import (
    Schema,
    StructuredOutputParser,
    json_schema_for,
)
from rflow.prompts import (
    DEFAULT_BUILDER,
    SYSTEM_PROMPT,
    PromptBuilder,
    messages,
)
from rflow.runtime.context import EngineContext
from rflow.runtime.env import agent_process_env
from rflow.runtime.runtime import LocalRuntime, RemoteRepl, ReplBackend, Runtime
from rflow.tools import tool
from rflow.tools.builtins import (
    DEFAULT_MAX_QUERY_CHARS,
    make_done,
    make_get_subagent_result,
    make_launch_subagents,
    make_show_vars,
    make_spawn_child,
)
from rflow.utils.pool import ThreadPool, create_pool


class ResumeError(RuntimeError):
    """Raised when an adopted graph has a suspended agent that can't resume.

    A suspended agent (its current node is :class:`SupervisingOutput`) parks a
    live coroutine on ``flow.repls`` waiting for child results. That coroutine
    is never serialized and never crosses a process boundary, so a graph
    ``load()``-ed from disk — or forked/adopted by a flow that didn't create the
    coroutine — cannot be resumed bit-for-bit. ``step()`` raises this by default
    instead of silently diverging or wedging. Pass ``salvage=True`` to truncate
    the stranded agents back to their delegating turn and re-run it
    deterministically (re-launching their children) instead.
    """

    def __init__(self, agent_ids: "Iterable[str]") -> None:
        self.agent_ids = list(agent_ids)
        joined = ", ".join(self.agent_ids)
        super().__init__(
            f"cannot resume suspended agent(s) [{joined}]: their live REPL "
            "coroutine did not survive into this process (it is never "
            "serialized). Re-run from the start, or pass salvage=True to "
            "truncate the stranded agents and re-delegate deterministically."
        )


class Flow(BaseFlow):
    """A minimal, stateless rlmflow engine.

    A :class:`Flow` holds only configuration — the LLM client and the core
    knobs — and never stores the run it's driving. All per-run state (the
    trajectory, each agent's live REPL, the step counter, the mutation lock)
    lives on the :class:`~rflow.graph.Graph`, so one engine can step any graph
    and several graphs can be driven independently.

    Parameters mirror the only knobs that matter for the core loop:
    ``max_depth`` (recursion bound), ``max_iters`` (per-agent LLM-turn cap
    before it's forced to finish), and ``max_concurrency`` (thread-pool size
    for advancing runnable agents in parallel).

    A :class:`Flow` owns one run at a time, but :meth:`step` is functional:
    ``graph = agent.step(graph)`` deep-copies the graph you pass in (so it stays
    a frozen snapshot) and returns the advanced copy. :meth:`start` builds the
    root graph and resets run state (the per-agent REPLs, the step counter, the
    mutation lock); ``step(query=...)`` on a fresh flow is ``start`` + one tick,
    and ``step(query=...)`` on a running graph adds a follow-up turn.

    The text/projection seams below are public on purpose: subclass and
    override ``CONTINUE`` / ``FINAL`` (nudge text), :meth:`format_exec_output`,
    :meth:`first_prompt`, :meth:`followup_prompt`, or :meth:`build_messages` to
    customize what the model sees without touching the loop.
    """

    #: Nudge appended when the latest turn left no pending user message.
    CONTINUE: str = messages.CONTINUE_NUDGE
    #: Nudge appended once the agent has exhausted its iteration budget.
    FINAL: str = messages.FINAL_ANSWER_ACTION

    def __init__(
        self,
        llm: LLMClient,
        *,
        llm_clients: dict[str, LLMClient] | None = None,
        llm_max_concurrency: int | None = None,
        llm_request_timeout: float | None = 600,
        llm_thread_safe: dict[str, bool] | None = None,
        max_depth: int = 3,
        max_iters: int | None = 20,
        child_max_iters: int | None = 20,
        max_concurrency: int = 8,
        max_output_length: int = 4_000,
        max_budget: int | None = None,
        max_messages: int | None = None,
        max_query_chars: int = DEFAULT_MAX_QUERY_CHARS,
        eager_children: bool = False,
        pool: object | None = None,
        system_prompt: str | None = None,
        prompt_builder: PromptBuilder | None = None,
        show_vars: bool = False,
        runtime: Runtime | None = None,
        enable_structured_output: bool = True,
        include_llm_query: bool = False,
    ) -> None:
        self.llm = llm
        self.max_depth = max_depth
        self.max_iters = max_iters
        self.child_max_iters = child_max_iters
        self.max_concurrency = max_concurrency
        self.max_output_length = max_output_length
        self.max_budget = max_budget
        self.max_messages = max_messages
        self.max_query_chars = max_query_chars
        self.eager_children = eager_children
        self.enable_structured_output = enable_structured_output
        self.include_llm_query = include_llm_query
        # ``system_prompt=None`` renders the live prompt builder per turn (the
        # default). Pass a string to force a fixed system prompt (escape hatch).
        self.system_prompt = system_prompt
        self.prompt_builder = prompt_builder or DEFAULT_BUILDER
        self.show_vars = show_vars
        # Where each agent's code runs. Defaults to the in-process
        # :class:`LocalRuntime`; pass a ``DockerRuntime`` / sandbox runtime (see
        # rflow.runtime) to sandbox execution, or one carrying registered tools /
        # a working directory. The runtime mints one backend per agent, lazily on
        # first execution.
        self.runtime = runtime or LocalRuntime()
        # LLM routing: "default" is the primary client; named lanes select
        # alternate models per agent (Graph.model) or optional llm_query_batched calls.
        # A single bounded channel owns the HTTP thread pool for every lane,
        # kept separate from the per-step agent execution pool.
        self._llm_clients: dict[str, LLMClient] = {
            "default": llm,
            **(llm_clients or {}),
        }
        self._llm_channel = LLMChannel(
            self._llm_clients,
            max_concurrency=llm_max_concurrency or max_concurrency,
            request_timeout=llm_request_timeout,
            thread_safe=llm_thread_safe,
        )
        self.output_parser = StructuredOutputParser()
        self.last_usage: LLMUsage | None = None
        self.pool = create_pool(pool, max_concurrency=max_concurrency)
        # Run state, (re)set by start(). One run per Flow at a time.
        self.graph: Graph | None = None
        self.repls: dict[str, ReplBackend] = {}
        self._step = 0
        self._terminate_requested: set[str] = set()
        self._lock = threading.RLock()

    @property
    def llm_clients(self) -> dict[str, LLMClient]:
        """Named LLM clients available for model routing."""
        return self._llm_clients

    def close(self) -> None:
        """Release the LLM channel's HTTP thread pool.

        Optional but recommended for long-lived processes / notebooks to avoid
        leaking idle worker threads. Also tears down the agent execution pool
        and any remote REPL backends (Docker containers, cloud sandboxes)
        created during the run. The flow is single-use afterwards.
        """
        for repl in self.repls.values():
            try:
                repl.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
        self.repls = {}
        try:
            self.runtime.close()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
        if isinstance(self.pool, ThreadPool):
            self.pool.shutdown()
        self._llm_channel.shutdown()

    # ── lifecycle ─────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        inputs: dict[str, str] | None = None,
        *,
        output_schema: Schema | None = None,
    ) -> str:
        """Start a run and step it to completion; return the root result."""
        graph = self.start(query, inputs, output_schema=output_schema)
        while not graph.finished:
            self.step()
        return graph.result()

    def tui(
        self,
        *,
        salvage: bool = False,
        max_steps_per_turn: int | None = None,
    ) -> Graph | None:
        """Open the optional full-screen terminal chat UI for this flow."""

        from rflow.tui import run_tui

        return run_tui(
            self,
            salvage=salvage,
            max_steps_per_turn=max_steps_per_turn,
        )

    # ── drop-in LLMClient: treat a whole Flow as one "model" ──────────
    #: A run mutates ``self.graph``; serialize when used as a shared client.
    thread_safe: bool = False

    def chat(self, messages: list[dict[str, str]], *args, **kwargs) -> str:
        """Run the flow on the last user message and return the root result.

        Lets a :class:`Flow` stand in anywhere an :class:`~rflow.clients.LLMClient`
        is expected (e.g. as the ``llm`` of another flow, or a DSPy LM): the
        last user turn becomes the query and the run's answer is the reply.
        """
        text, _usage = self.completion(messages, *args, **kwargs)
        return text

    def completion(
        self, messages: list[dict[str, str]], *args, **kwargs
    ) -> tuple[str, LLMUsage]:
        query = next(
            (
                m.get("content", "")
                for m in reversed(messages)
                if m.get("role") == "user"
            ),
            "",
        )
        result = self.run(query)
        inp, out = self.graph.tokens() if self.graph is not None else (0, 0)
        usage = LLMUsage(input_tokens=inp, output_tokens=out)
        self.last_usage = usage
        return result, usage

    def start(
        self,
        query: str,
        inputs: dict[str, str] | None = None,
        *,
        output_schema: Schema | None = None,
    ) -> Graph:
        """Begin a new run: build the root graph, reset run state, return it.

        Pass ``output_schema`` (a Pydantic model/TypeAdapter, JSON-schema dict,
        or JSON-schema string) to require the root agent's ``done(...)`` value
        to validate against it; the schema is shown to the model and enforced.
        """
        inputs = self._validate_inputs(inputs)
        self.graph = Graph(
            agent_id="root",
            depth=0,
            query=query,
            inputs=inputs,
            output_schema=(
                json_schema_for(output_schema) if output_schema is not None else None
            ),
        )
        self.repls = {}
        self.runtime.clear_graph_sync_cache()
        self._step = 0
        self._terminate_requested = set()
        # Store the rendered prompt on the graph so traces are self-describing.
        # Later schema edits are reconciled from graph state before stepping.
        # The agent's REPL is NOT created here: that stays lazy (first execution,
        # in a worker thread) so heavy backends (Docker/Modal) only boot when the
        # agent actually runs. tools_section reads tool *metadata* via a throwaway
        # build, which creates no REPL/sandbox.
        self._ensure_system_prompt_current(self.graph)
        self._append(
            self.graph, UserQuery(content=self.first_prompt(query, inputs, depth=0))
        )
        return self.graph

    def step(
        self,
        graph: Graph | None = None,
        query: str | None = None,
        inputs: dict[str, str] | None = None,
        *,
        output_schema: Schema | None = None,
        salvage: bool = False,
    ) -> Graph:
        """Advance the run by one tick and return the graph.

        ``graph = agent.step(graph)``: the graph you pass is deep-copied (so it
        stays a frozen snapshot / checkpoint) and the advanced copy is returned.
        Omit ``graph`` to advance the flow's own graph in place. ``query=``
        starts a run (no current graph) or appends a follow-up user turn (merged
        into a trailing user node so two user messages never sit in a row).
        ``output_schema=`` on an existing graph updates the root agent's
        structured-output contract before the next transition.

        A graph loaded/forked while paused mid-delegation (tip
        ``SupervisingOutput``) has no live coroutine to resume, so this raises
        :class:`ResumeError` — unless ``salvage=True`` truncates it back to the
        delegating turn and re-runs it.
        """
        if graph is not None:
            self._adopt(graph)

        if self.graph is None:
            if query is None:
                raise RuntimeError(
                    "step() needs a query to start or a graph to advance"
                )
            self.start(query, inputs, output_schema=output_schema)
        else:
            if output_schema is not None:
                self._update_root_output_schema(output_schema)
            if query is not None:
                self._add_user_turn(query, inputs)

        self.sync_graph_state()
        self._check_resumable(salvage)
        self._step += 1
        plan = self.plan(self.graph)
        if plan:
            self.run_plan(plan)
        return self.graph

    def _adopt(self, graph: Graph) -> None:
        """Take ``graph`` as run state, deep-copied so the original stays frozen.

        If the adopted graph is not the exact same history already bound to
        this flow, live REPLs are discarded and recreated lazily. A graph can be
        serialized or edited, but a Python namespace, sandbox process, or
        suspended coroutine cannot be safely diffed onto a different history.
        """
        same_history = (
            self.graph is not None and graph.to_dict() == self.graph.to_dict()
        )
        if not same_history:
            for aid in list(self.repls):
                self.runtime.discard_repl(self.repls, aid)
            self._terminate_requested = set()
        self.graph = graph.copy(deep=True)
        self.runtime.clear_graph_sync_cache()
        self._step = self.graph.max_global_step() or 0

    def _update_root_output_schema(self, output_schema: Schema) -> None:
        """Set the root graph's structured-output contract."""
        assert self.graph is not None
        self.graph.output_schema = json_schema_for(output_schema)

    def sync_graph_state(self) -> Graph:
        """Refresh runtime mirrors from the current graph without advancing.

        ``Graph`` is the durable source of truth. Cached REPL state exists only
        so tools and code execution have fast/local access to graph metadata.
        """
        if self.graph is None:
            raise RuntimeError("sync_graph_state() needs a graph")
        agents = self.graph.agents
        for agent in agents.values():
            self._ensure_system_prompt_current(agent)
        for aid in list(self.repls):
            agent = agents.get(aid)
            if agent is None:
                self.runtime.discard_repl(self.repls, aid)
            else:
                self._sync_repl_with_graph(self.repls[aid], agent)
        return self.graph

    def _add_user_turn(self, query: str, inputs: dict[str, str] | None) -> None:
        """Append a follow-up user turn to the root.

        Two adjacent ``UserQuery`` nodes are fine — ``Graph.messages()`` coalesces
        same-role blocks at projection time, so the model never sees two user turns
        in a row regardless of how the trajectory is stored.
        """
        root = self.graph
        extra = self._validate_inputs(inputs)
        root.query = query
        if extra:
            root.inputs = {**root.inputs, **extra}
        self._append(
            root, UserQuery(content=self.followup_prompt(query, depth=root.depth))
        )

    def _check_resumable(self, salvage: bool) -> None:
        """Stop (or salvage) if an adopted graph is paused with no live REPL."""
        stranded = [
            a.agent_id
            for a in self.graph.walk()
            if isinstance(a.current(), SupervisingOutput)
            and a.agent_id not in self.repls
            and not getattr(a.current(), "launch_id", None)
        ]
        if not stranded:
            return
        if not salvage:
            raise ResumeError(stranded)
        for aid in stranded:
            if aid in self.graph.agents:
                self._salvage(aid)

    def _salvage(self, agent_id: str) -> None:
        """Truncate a stranded supervisor back to its delegating turn to re-run it."""
        from rflow.graph.truncation import prune_unreachable_children

        agent = self.graph[agent_id]
        keep = next(
            (
                i
                for i in reversed(range(len(agent.nodes)))
                if isinstance(agent.nodes[i], LLMOutput)
            ),
            None,
        )
        if keep is not None:
            agent.nodes = agent.nodes[: keep + 1]
            prune_unreachable_children(agent)

    def terminate(self, agent_ids: "Iterable[str] | None" = None) -> Graph:
        """Force the named agents (or all) to finish on their next LLM turn.

        Flagged agents get ``force_final=True`` on their next ``CallLLM`` (the
        engine appends the FINAL nudge), so they wrap up instead of continuing.
        Flags persist until the agent reaches a terminal ``done(...)``.
        """
        assert self.graph is not None, "call start() before terminate()"
        targets = agent_ids if agent_ids is not None else list(self.graph.agents)
        for aid in targets:
            agent = self.graph.agents.get(aid)
            if agent is not None and not agent.finished:
                self._terminate_requested.add(aid)
        return self.graph

    # ── execution: how a step's plan is carried out (override seams) ──

    def run_plan(self, plan: ActionPlan) -> None:
        """Carry out one step's plan by running each action.

        Each action is independent — it touches only its own agent's slice of
        the graph (mutations are guarded by ``self._lock``) — so the pool can
        fan them out. With ``eager_children=True`` the pool runs work-conserving:
        as each task finishes it pulls in a waiting supervisor's newly-runnable
        descendants. Override to add ordering, batching, per-agent rate limits,
        etc.
        """
        tasks = [(aid, lambda a=action: self.act(a)) for aid, action in plan.items()]
        if self.eager_children:
            self.pool.run_until_idle(tasks, self._refill_eager_children)
            return
        self.pool.execute(tasks)

    def _refill_eager_children(
        self, _done_id: str, _result: object, active_ids: set[str]
    ) -> "list[tuple[str, Callable[[], None]]]":
        """Schedule a waiting supervisor's runnable descendants as work frees up."""
        tasks: list[tuple[str, Callable[[], None]]] = []
        scheduled: set[str] = set(active_ids)
        for supervisor in list(self.graph.walk()):
            cur = supervisor.current()
            if not isinstance(cur, SupervisingOutput):
                continue
            runnable = (
                [supervisor.agent_id]
                if self._can_resume(cur)
                else supervisor.runnable_descendants()
            )
            runnable = [aid for aid in runnable if aid not in scheduled]
            if not runnable:
                continue
            for aid, action in self.plan_for(self.graph, runnable).items():
                if aid in scheduled:
                    continue
                scheduled.add(aid)
                tasks.append((aid, lambda a=action: self.act(a)))
        return tasks

    # ── policy: decide what each agent does next (pure) ───────────────

    def plan(self, graph: Graph) -> ActionPlan:
        """Project the runnable agents into ``{agent_id: Action}``.

        Pure: reads the graph and engine config, writes nothing. Pairs with
        :meth:`Graph.get_runnable_nodes` (*who* may run) to answer *what* each
        runnable agent should do this step.
        """
        return self.plan_for(graph, graph.get_runnable_nodes())

    def plan_for(self, graph: Graph, agent_ids: "Iterable[str]") -> ActionPlan:
        """Project a specific set of agent ids into ``{agent_id: Action}``."""
        agents = graph.agents
        plan: ActionPlan = {}
        for aid in agent_ids:
            agent = agents.get(aid)
            if agent is None:
                continue
            action = self.plan_one(agent)
            if action is not None:
                plan[aid] = action
        return plan

    def plan_one(self, agent: Graph) -> Action | None:
        """The next :class:`Action` for one agent, or ``None`` if it can't move.

        Override to change routing or the force-final policy.
        """
        cur = agent.current()
        if cur is None or cur.terminal:
            return None
        if isinstance(cur, SupervisingOutput):
            if self._has_live_coroutine(agent.agent_id):
                return Resume(agent.agent_id)
            if cur.launch_id:
                return Recover(agent.agent_id, cur.launch_id)
            return Resume(agent.agent_id)
        if isinstance(cur, LLMOutput):
            return Exec(agent.agent_id)
        if isinstance(cur, (UserQuery, ExecOutput, ErrorOutput)):
            last_terminal = next(
                (
                    i
                    for i in range(len(agent.nodes) - 1, -1, -1)
                    if agent.nodes[i].terminal
                ),
                -1,
            )
            iters = sum(
                isinstance(n, LLMAction) for n in agent.nodes[last_terminal + 1 :]
            )
            max_iter = (
                agent.max_iters if agent.max_iters is not None else self.max_iters
            )
            force_final = (max_iter is not None and iters >= max_iter) or (
                agent.agent_id in self._terminate_requested
            )
            return CallLLM(agent.agent_id, force_final=force_final)
        # A bare ActionNode as current means a half-written step; skip it.
        return None

    # ── transition handlers (public override seams) ───────────────────
    #
    # ``act`` materializes one planned :class:`Action`; each handler
    # appends exactly one observation (via :meth:`record_observation`):
    #
    #   CallLLM -> step_llm
    #   Exec    -> step_exec -> run_exec
    #   Resume  -> step_after_supervising
    #
    # Override any one to customize a phase without touching the loop.

    def act(self, action: Action) -> None:
        """Execute one planned action against the current run."""
        agent = self._find(action.agent_id)
        cur = agent.current()
        if cur is None or cur.terminal:
            return
        over = self._budget_exceeded(agent)
        if over is not None:
            self._append(
                agent,
                DoneOutput(
                    result=f"[budget exceeded: {over} tokens]", output="", content=""
                ),
            )
            return
        if isinstance(action, CallLLM) and isinstance(
            cur, (UserQuery, ExecOutput, ErrorOutput)
        ):
            self.step_llm(agent, force_final=action.force_final)
        elif isinstance(action, Exec) and isinstance(cur, LLMOutput):
            self.step_exec(agent, cur)
        elif isinstance(action, Resume) and isinstance(cur, SupervisingOutput):
            self.step_after_supervising(agent, cur)
        elif isinstance(action, Recover) and isinstance(cur, SupervisingOutput):
            self.step_recover_supervising(agent, cur)

    # ── LLM access (override seams) ───────────────────────────────────

    def llm_client_for(self, agent: Graph, *, model: str | None = None) -> LLMClient:
        """The :class:`LLMClient` backing ``agent`` (or an explicit ``model``)."""
        return self._llm_clients.get(model or agent.model or "default", self.llm)

    def call_llm(
        self, messages: list[dict[str, str]], *, model: str | None = None, **llm_kwargs
    ) -> tuple[str, LLMUsage]:
        """Route one chat completion through the bounded :class:`LLMChannel`.

        Override to add caching, request logging, or a different transport.
        Returns ``(text, usage)``.
        """
        return self._llm_channel.call(model or "default", messages, **llm_kwargs)

    def record_usage(self, usage: LLMUsage) -> None:
        """Record token usage from the most recent LLM call.

        Default keeps only the last call's usage on ``self.last_usage``.
        Override to accumulate totals, attribute cost per agent, etc.
        """
        self.last_usage = usage

    def step_llm(self, agent: Graph, *, force_final: bool) -> None:
        """LLM half of a turn: append ``LLMAction`` then the model's ``LLMOutput``."""
        model_key = agent.model or "default"
        self._append(agent, LLMAction(model=model_key))
        messages = self.build_messages(agent, force_final=force_final)
        try:
            reply, usage = self.call_llm(messages, model=model_key)
        except Exception as exc:  # noqa: BLE001
            self._append(
                agent,
                ErrorOutput(
                    error="llm_exception",
                    content=f"LLM call failed: {type(exc).__name__}: {exc}",
                ),
            )
            return
        self.record_usage(usage)
        client = self.llm_client_for(agent, model=model_key)
        blocks = find_code_blocks(reply)
        self._append(
            agent,
            LLMOutput(
                reply=reply,
                code=blocks[0] if blocks else "",
                model=getattr(client, "model", model_key),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            ),
        )

    def step_exec(self, agent: Graph, llm_output: LLMOutput) -> None:
        """Exec half of a turn: append ``ExecAction``, then run the code block."""
        code = llm_output.code
        self._append(agent, ExecAction(code=code))
        if not code:
            self._append(
                agent,
                ErrorOutput(
                    error="no_code_block", content=self.no_code_block_message()
                ),
            )
            return
        self.run_exec(agent, code)

    def no_code_block_message(self) -> str:
        """The error shown when a reply has no ```repl``` block. Override seam."""
        return messages.NO_CODE_BLOCK

    def run_exec(self, agent: Graph, code: str) -> None:
        """Validate and run a fresh code block; record the observation it yields."""
        err = self.validate_code(code)
        if err is not None:
            self._append(agent, ErrorOutput(error="invalid_wait", content=err))
            return
        repl = self.repl_for(agent)
        repl.engine_context.done_result = None
        suspended, payload = repl.start(code)
        self.record_observation(agent, repl, suspended, payload)

    def step_after_supervising(self, agent: Graph, sup: SupervisingOutput) -> None:
        """Resume half: append ``ResumeAction``, then resume the paused coroutine."""
        if not self._can_resume(sup):
            return
        results = [self._child_result(aid) for aid in sup.waiting_on]
        self._append(agent, ResumeAction(resumed_from=list(sup.waiting_on)))
        repl = self.repl_for(agent)
        repl.engine_context.done_result = None
        suspended, payload = repl.resume(results)
        self.record_observation(
            agent, repl, suspended, payload, resumed_from=list(sup.waiting_on)
        )

    def step_recover_supervising(self, agent: Graph, sup: SupervisingOutput) -> None:
        """Append a recovery observation for a stranded supervisor."""
        if not self._can_resume(sup):
            return
        self._append(
            agent,
            ExecOutput(
                output="",
                content=self.recovery_prompt(agent, sup),
                resumed_from=list(sup.waiting_on),
            ),
        )

    def record_observation(
        self,
        agent: Graph,
        repl: ReplBackend,
        suspended: bool,
        payload: object,
        *,
        resumed_from: list[str] | None = None,
    ) -> None:
        """Turn a REPL outcome into the single observation it produced.

        Captured stdout is run through :meth:`truncate_output` before it is
        recorded so a runaway ``print`` can't blow the next prompt.
        """
        rf = list(resumed_from or [])
        done = repl.engine_context.done_result
        if done is not None:
            out = self.truncate_output(payload if isinstance(payload, str) else "")
            self._append(
                agent,
                DoneOutput(result=done, output=out, content=out, resumed_from=rf),
            )
            return
        if suspended:
            request, pre = payload  # type: ignore[misc]
            pre = self.truncate_output(pre)
            self._append(
                agent,
                SupervisingOutput(
                    output=pre,
                    content=self.format_exec_output(pre) if pre.strip() else "",
                    waiting_on=list(request.agent_ids),
                    launch_id=request.launch_id,
                    launch_specs=list(request.launch_specs),
                    launch_names=list(request.launch_names),
                    resumed_from=rf,
                ),
            )
            return
        out = self.truncate_output(payload if isinstance(payload, str) else "")
        out = out if out.strip() else "(no output)"
        if repl.errored:
            self._append(
                agent,
                ErrorOutput(
                    error="exec_exception",
                    output=out,
                    content=self.format_exec_output(out),
                    resumed_from=rf,
                ),
            )
            return
        self._append(
            agent,
            ExecOutput(
                output=out, content=self.format_exec_output(out), resumed_from=rf
            ),
        )

    def validate_code(self, code: str) -> str | None:
        """Static pre-flight check on a code block.

        Return an error string to reject the block before it runs (recorded as
        an ``invalid_wait`` error the model can recover from), or ``None`` to
        allow it. Defaults to checking ``await`` usage; override to add your own
        policy (banned imports, size limits, …).
        """
        return check_wait_syntax(code)

    def truncate_output(self, output: str) -> str:
        """Cap REPL stdout at ``max_output_length`` chars (0 disables the cap)."""
        limit = self.max_output_length
        if limit and len(output) > limit:
            omitted = len(output) - limit
            return (
                output[:limit]
                + f"\n...<truncated {omitted} chars; keep full data in variables>"
            )
        return output

    def _budget_exceeded(self, agent: Graph) -> int | None:
        """Tokens spent by ``agent``'s subtree if it's over ``max_budget``, else None."""
        if self.max_budget is None:
            return None
        used = agent.total_tokens()
        return used if used >= self.max_budget else None

    def _can_resume(self, sup: SupervisingOutput) -> bool:
        agents = self.graph.agents
        return all(aid in agents and agents[aid].finished for aid in sup.waiting_on)

    def _has_live_coroutine(self, agent_id: str) -> bool:
        """Whether this Flow can resume ``agent_id``'s suspended Python frame."""
        repl = self.repls.get(agent_id)
        if repl is None:
            return False
        # Local REPLs expose ``coro``; remote backends do not, but if the backend
        # survived in ``self.repls`` it owns the paused state.
        return getattr(repl, "coro", True) is not None

    def recovery_prompt(self, agent: Graph, sup: SupervisingOutput) -> str:
        """Prompt shown when a stranded supervisor must recover from graph state."""
        launch_id = sup.launch_id or sup.id
        children = "\n".join(f"- `{aid}`" for aid in sup.waiting_on)
        return "\n\n".join(
            [
                "You are recovering this agent after a delegated subagent call.",
                (
                    "The original Python coroutine is unavailable, so execution "
                    "cannot resume at the line after `await launch_subagents(...)`."
                ),
                (
                    f"Delegation `{launch_id}` is complete. Its immediate child "
                    "results are available in original launch order."
                ),
                f"Immediate children:\n{children or '- <none>'}",
                (
                    "Call:\n"
                    f"```repl\nresults = get_subagent_result({launch_id!r})\n```"
                ),
                (
                    "Then continue this agent's task from those graph-backed "
                    "results. Do not relaunch these children unless you decide "
                    "their results are invalid."
                ),
            ]
        )

    def _child_result(self, agent_id: str) -> object:
        """Value handed back to a parent's coroutine for one finished child.

        A child with an ``output_schema`` stored its answer as a JSON string in
        ``DoneOutput.result`` (see ``make_done``); parse it back into the
        validated JSON value the parent expects. Schemaless children return
        their plain string result.
        """
        child = self.graph.agents.get(agent_id)
        if not (child and child.finished):
            return ""
        result = child.result()
        if child.output_schema is not None and result:
            try:
                return json.loads(result)
            except (json.JSONDecodeError, TypeError):
                return result
        return result

    # ── child spawning (called from inside the REPL) ──────────────────

    def spawn_child(
        self,
        parent_agent_id: str,
        name: str,
        query: str,
        inputs: dict[str, str] | None = None,
        model: str = "default",
        output_schema: Schema | None = None,
        strict_name: bool = False,
    ) -> ChildHandle | str:
        """Create a child agent under ``parent_agent_id`` in the current run.

        Invoked by the ``launch_subagents`` launcher while the parent's code
        runs. Returns a :class:`ChildHandle`, or a refusal string when the
        depth bound is hit (which the launcher surfaces as a child result).
        """
        inputs = self._validate_inputs(inputs)
        if len(query) > self.max_query_chars:
            return (
                f"[refused: query too long ({len(query)} chars > "
                f"{self.max_query_chars})] keep query a short instruction and "
                "move bulk payloads into inputs"
            )
        with self._lock:
            parent = self.graph[parent_agent_id]
            if parent.depth >= self.max_depth:
                return f"[refused: max depth {self.max_depth}] do this inline"
            base_child_id = f"{parent_agent_id}.{name}"
            if strict_name and base_child_id in parent.children:
                raise ValueError(
                    f"child id {base_child_id!r} already exists; choose a unique "
                    "`name` for this launch_subagents(...) spec"
                )
            child_id = (
                base_child_id
                if strict_name
                else self._unique_id(parent_agent_id, name, parent)
            )
            parent_node = parent.current()
            child = Graph(
                agent_id=child_id,
                depth=parent.depth + 1,
                query=query,
                inputs=inputs,
                model=None if model == "default" else model,
                max_iters=self.child_max_iters,
                output_schema=(
                    json_schema_for(output_schema)
                    if output_schema is not None
                    else None
                ),
                parent_agent_id=parent_agent_id,
                parent_node_id=parent_node.id if parent_node else None,
            )
            # Render + store the child's prompt now (cheap), but leave its REPL
            # lazy — a heavy backend must not boot under this lock; it boots when
            # the child is first stepped, in a worker thread.
            self._ensure_system_prompt_current(child)
            self._append(
                child,
                UserQuery(
                    content=self.first_prompt(query, inputs, depth=parent.depth + 1)
                ),
            )
            parent.children[child_id] = child
        return ChildHandle(child_id)

    @staticmethod
    def _unique_id(parent_id: str, name: str, parent: Graph) -> str:
        base = f"{parent_id}.{name}"
        if base not in parent.children:
            return base
        i = 1
        while f"{base}_{i}" in parent.children:
            i += 1
        return f"{base}_{i}"

    # ── REPL setup ────────────────────────────────────────────────────

    def seed_agent_context(self, repl: ReplBackend, agent: Graph) -> None:
        """Seed per-agent engine context and public process env.

        Called once when a backend is first created for ``agent``. Override to
        customize either the trusted host-side context used by tool closures or
        the public ``RFLOW_*`` process environment visible to agent code.
        """
        repl.engine_context = EngineContext(
            agent_id=agent.agent_id,
            output_schema=agent.output_schema,
        )
        repl.process_env.update(self._agent_process_env(agent))

    def repl_for(self, agent: Graph) -> ReplBackend:
        """Get (creating lazily, on first execution) this agent's REPL backend.

        Backends are held by agent id on ``self.repls`` for the current run and
        are created the first time an agent actually runs code — never at spawn
        time — so heavy backends (Docker/Modal) only boot when work arrives, in
        the worker thread that steps the agent rather than under the engine lock.
        :meth:`make_repl` picks the backend (the runtime mints it);
        :meth:`seed_agent_context` seeds per-agent context;
        this method binds tools and inputs. Override any seam to customize.
        """
        repl = self.repls.get(agent.agent_id)
        if repl is None:
            repl = self.make_repl(agent)
            self.seed_agent_context(repl, agent)
            repl_inputs = agent.repl_inputs()
            tools = self.build_tools(repl.engine_context)
            if isinstance(repl, RemoteRepl):
                repl.seed(tools, repl_inputs, max_query_chars=self.max_query_chars)
            else:
                repl.namespace.update(tools)
                if self.show_vars:
                    repl.namespace["SHOW_VARS"] = make_show_vars(repl.namespace)
                # Inputs live under a single ``INPUTS`` dict (read as
                # ``INPUTS["key"]``) rather than as top-level names, so an input
                # key can never shadow a real REPL variable, tool, or import.
                repl.namespace["INPUTS"] = repl_inputs
            self.repls[agent.agent_id] = repl
            self.runtime.repl_env_cache[agent.agent_id] = dict(repl.process_env)
            self.runtime.repl_inputs_cache[agent.agent_id] = agent.repl_inputs()
        else:
            self._sync_repl_with_graph(repl, agent)
        return repl

    def make_repl(self, agent: Graph) -> ReplBackend:
        """Mint the REPL backend for ``agent`` from ``self.runtime``.

        The runtime decides where code runs (in-process :class:`LocalRuntime` by
        default, or a ``DockerRuntime`` / sandbox runtime). Called once per
        agent, in the worker thread that first steps it. Override only for
        bespoke per-agent backend selection; otherwise configure the runtime.
        """
        return self.runtime.open(agent)

    def _agent_process_env(self, agent: Graph) -> dict[str, str]:
        """Public ``RFLOW_*`` environment, derived from graph + flow config."""
        return agent_process_env(
            agent_id=agent.agent_id,
            depth=agent.depth,
            parent_agent_id=agent.parent_agent_id,
            max_depth=self.max_depth,
        )

    # ── graph-derived prompts ─────────────────────────────────────────

    def _ensure_system_prompt_current(self, agent: Graph) -> None:
        """Render ``agent.system_prompt`` if its graph projection is stale."""
        fp = json.dumps(
            {
                "output_schema": agent.output_schema,
                "depth": agent.depth,
                "max_depth": self.max_depth,
                "enable_structured_output": self.enable_structured_output,
                "show_vars": self.show_vars,
                "runtime_tools": sorted(self.runtime.tools),
                "prompt_builder": id(self.prompt_builder),
                "system_prompt": self.system_prompt,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        if self.runtime.prompt_fingerprints.get(agent.agent_id) == fp:
            return
        agent.system_prompt = self.build_system_prompt(agent)
        self.runtime.prompt_fingerprints[agent.agent_id] = fp

    def _sync_repl_with_graph(self, repl: ReplBackend, agent: Graph) -> None:
        """Ask the runtime to mirror graph metadata into one backend."""
        self.runtime.sync_repl(
            repl,
            agent,
            env=self._agent_process_env(agent),
            inputs=agent.repl_inputs(),
        )

    def build_tools(self, engine_context: EngineContext | None = None) -> dict:
        """Assemble the core REPL tools (see :mod:`rflow.tools.builtins`).

        The tools are bound to ``engine_context``, not to a Graph: they read
        per-agent control state — agent id, output schema, done result — from it
        at call time, so this is built once per agent in
        :meth:`repl_for`. For most cases, register tools on the runtime
        (``runtime.register_tools(FILE_TOOLS)``) and they are merged in here.
        Override (typically ``super().build_tools(engine_context) | extra``) only
        when a tool must close over the context or the flow.
        """
        engine_context = engine_context or EngineContext()
        spawn_child = make_spawn_child(self, engine_context)
        tools = {
            "done": make_done(self, engine_context),
            "launch_subagents": make_launch_subagents(
                spawn_child, max_query_chars=self.max_query_chars
            ),
            "_rflow_spawn_child": spawn_child,
            "get_subagent_result": make_get_subagent_result(self, engine_context),
        }
        if self.include_llm_query:
            tools["llm_query_batched"] = self.llm_query_batched
        # Tools registered on the runtime (e.g. ``runtime.register_tools(
        # FILE_TOOLS)``) reach every agent and its children. Core control tools
        # win on a name clash, so a registered tool can never shadow ``done`` etc.
        for name, fn in self.runtime.tools.items():
            tools.setdefault(name, fn)
        return tools

    @tool(
        "Send a list of independent one-shot prompts to the model in parallel "
        "and get back a list of replies, in order. Use for simple fanout (no "
        "tools, no REPL). Pass output_schema (a JSON Schema dict) to validate "
        "each reply into a JSON-compatible value.",
        proxy=True,
    )
    def llm_query_batched(
        self,
        prompts: list[str],
        *,
        model: str = "default",
        output_schema: Schema | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> list:
        """Fan out one-shot LLM prompts through the shared channel (REPL tool).

        ``prompts`` is a ``list[str]``; results come back in the same order.
        Pass ``output_schema`` to parse each reply into a validated object. The
        channel bounds concurrency and serializes non-thread-safe clients, so
        agents can batch independent queries without managing threads.
        """
        if not isinstance(prompts, list) or not all(
            isinstance(p, str) for p in prompts
        ):
            raise TypeError("llm_query_batched(prompts) takes a list[str]")
        if model not in self._llm_clients:
            keys = ", ".join(sorted(self._llm_clients))
            raise ValueError(f"unknown model {model!r}. available: {keys}")
        schema = json_schema_for(output_schema) if output_schema is not None else None
        sent = prompts
        if schema is not None:
            hint = self._schema_instruction(schema)
            sent = [f"{p}\n\n{hint}" for p in prompts]
        llm_kwargs = {
            key: value
            for key, value in (
                ("temperature", temperature),
                ("top_p", top_p),
                ("max_tokens", max_tokens),
                ("stop", stop),
            )
            if value is not None
        }
        pairs = self._llm_channel.batch(model, sent, **llm_kwargs)
        self.record_usage(
            LLMUsage(
                input_tokens=sum(u.input_tokens for _, u in pairs),
                output_tokens=sum(u.output_tokens for _, u in pairs),
            )
        )
        texts = [text for text, _ in pairs]
        if schema is not None:
            return [self.output_parser(text, schema) for text in texts]
        return texts

    # ── message / output projection (public override seams) ───────────

    def build_messages(
        self, graph: Graph, *, force_final: bool
    ) -> list[dict[str, str]]:
        """Project a graph trajectory into chat messages for the LLM call.

        The graph's system prompt (rendered once at creation and stored on the
        graph) plus its trajectory projection and the engine's continue/final
        nudge. Override to reshape what the model sees (ordering,
        truncation/windowing, extra steering turns, …).
        """
        self._ensure_system_prompt_current(graph)
        system_prompt = graph.system_prompt or self.build_system_prompt(graph)
        return prompt_projection.build_messages(
            graph,
            system_prompt=system_prompt,
            max_messages=self.max_messages,
            continue_nudge=self.CONTINUE,
            final_nudge=self.FINAL,
            force_final=force_final,
        )

    def build_system_prompt(self, graph: Graph) -> str:
        """The system prompt for one agent's LLM call.

        Default renders the live :attr:`prompt_builder` (``DEFAULT_BUILDER``)
        against this flow + graph, so tool docs, recursion status, and any
        structured-output schema reflect the current run. If ``system_prompt``
        was set explicitly, that fixed string is used instead (plus a schema
        instruction when the graph has an ``output_schema``). Override to swap
        in a custom builder or templating.
        """
        if self.system_prompt is not None:
            schema = None
            if graph.output_schema is not None and self.enable_structured_output:
                schema = self._schema_instruction(graph.output_schema)
            return prompt_projection.build_system_prompt(
                self.system_prompt,
                schema_instruction=schema,
            )
        return self.prompt_builder.build(self, graph)

    def _schema_instruction(self, schema: Schema) -> str:
        """Render the structured-output instruction block for a schema."""
        hint = self.output_parser.system_prompt_hint(schema)
        return prompt_projection.schema_instruction(hint)

    def format_exec_output(self, output: str) -> str:
        """Wrap captured REPL stdout for the model's next user turn.

        Override to change how execution output is presented (framing,
        labels, syntax fences, …).
        """
        return prompt_projection.format_exec_output(output)

    def first_prompt(
        self, query: str, inputs: dict[str, str], *, depth: int = 0
    ) -> str:
        """Build an agent's bootstrap user message (see :mod:`rflow.prompts.messages`).

        Override to change the bootstrap framing.
        """
        return prompt_projection.first_prompt(
            query,
            inputs,
            depth=depth,
            max_depth=self.max_depth,
        )

    def followup_prompt(self, query: str, *, depth: int = 0) -> str:
        """Build a follow-up user message for an existing agent trajectory."""
        return prompt_projection.followup_prompt(
            query,
            depth=depth,
            max_depth=self.max_depth,
        )

    _RESERVED = frozenset(
        {
            "done",
            "launch_subagents",
            "get_subagent_result",
            "_rflow_spawn_child",
            "llm_query_batched",
            "query",
            "HISTORY",
            "SHOW_VARS",
        }
    )

    @classmethod
    def _validate_inputs(cls, inputs: dict[str, str] | None) -> dict[str, str]:
        """Coerce/validate agent inputs to a ``{identifier: str}`` namespace."""
        if inputs is None:
            return {}
        if not isinstance(inputs, dict):
            raise TypeError("inputs must be a dict of {str: str}")
        clean: dict[str, str] = {}
        for name, value in inputs.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(f"input name {name!r} must be a valid identifier")
            if name in cls._RESERVED:
                raise ValueError(f"input name {name!r} is reserved")
            if not isinstance(value, str):
                raise TypeError(
                    f"input {name!r} must be a str "
                    f"(JSON-encode structured values); got {type(value).__name__}"
                )
            clean[name] = value
        return clean

    # ── internal graph mutation (locked) ──────────────────────────────

    def _append(self, agent: Graph, node: Node) -> Node:
        with self._lock:
            prev = agent.nodes[-1] if agent.nodes else None
            seq = prev.seq + 1 if prev else 0
            stamped = node.update(
                agent_id=agent.agent_id, seq=seq, global_step=self._step
            )
            agent.nodes.append(stamped)
        return stamped

    def _find(self, agent_id: str) -> Graph:
        with self._lock:
            return self.graph[agent_id]


__all__ = ["Flow", "ResumeError", "find_code_blocks", "SYSTEM_PROMPT"]
