"""RLMFlow — the recursive language-model orchestrator.

This module holds :class:`RLMFlow`, the engine. Every piece of
behavior a user might want to customize is a method on this class —
override what you want, call ``super()`` for default behavior.

Pure helpers live under :mod:`rlmflow.engine`:

- :mod:`rlmflow.engine.actions` — :class:`Action` types and the pure
  projection ``Graph -> ActionPlan``.
- :mod:`rlmflow.engine.replay` — cold-start replay-of-one for
  rebuilding a suspended coroutine after a fork or process restart.
- :mod:`rlmflow.engine.scheduling` — implementation of the outer
  ``step`` loop and async-child refill policy.
- :mod:`rlmflow.engine.transitions` — implementation of action-to-state
  transition handlers.
- :mod:`rlmflow.engine.helpers` — tiny shared helpers (node appends,
  output truncation, the pool factory).
- :mod:`rlmflow.engine.config` — :class:`RLMConfig` (pure data).

Nothing under ``engine/`` holds engine state; ``RLMFlow`` does.

The class is grouped:

1. Construction
2. Lifecycle           — ``start`` / ``run`` / ``chat`` / ``step`` / ``terminate``
3. Per-step transitions — ``apply_one`` and the three half-step
                          handlers (LLM / exec / resume-after-supervising)
4. LLM half-step       — ``reply_to`` / ``call_llm`` / ``llm_client_for`` /
                          ``extract_code`` (+ private transcript writer)
5. Messages / prompt   — ``build_messages`` / ``build_system_prompt`` /
                          ``build_tools_section`` / ``build_status_section``
6. Runtime / env       — ``runtime_for`` / ``create_runtime_session`` /
                          ``inject_env`` / ``register_tools`` /
                          ``format_exec_output``
7. Child spawning      — ``spawn_child``
8. Bookkeeping         — ``record_usage`` / ``node_config``
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from rlmflow.engine import scheduling, transitions
from rlmflow.engine.actions import Action
from rlmflow.engine.config import RLMConfig
from rlmflow.engine.helpers import (
    ROOT_RUNTIME_ID,
    create_pool,
    format_exec_output,
    prepare_node_for_append,
    unique_child_id,
)
from rlmflow.engine.scheduler import NodeScheduler
from rlmflow.engine.transcript import TranscriptRecorder
from rlmflow.graph import (
    ChildHandle,
    ExecAction,
    Graph,
    LLMOutput,
    Node,
    RuntimeRef,
    SupervisingOutput,
    UserQuery,
    is_llm_output,
)
from rlmflow.integrations.structured import (
    Schema,
    StructuredOutputParser,
    json_schema_for,
)
from rlmflow.llm import LLMClient, LLMUsage
from rlmflow.llm_channel import LLMChannel
from rlmflow.prompts.default import DEFAULT_BUILDER, status_section, tools_section
from rlmflow.prompts.messages import (
    DEFAULT_QUERY,
    FINAL_ANSWER_ACTION,
    TRUNCATION_SESSION_HINT,
    TRUNCATION_SUMMARY,
    build_user_prompt,
)
from rlmflow.prompts.projection import (
    append_message,
    coalesce_messages,
    project_state_messages,
)
from rlmflow.runtime import LocalRuntime, Runtime
from rlmflow.runtime.env import (
    DONE_OUTPUT_SCHEMA,
    execution_facts,
    seed_execution_env,
)
from rlmflow.tools import tool
from rlmflow.tools.builtins import (
    SHOW_VARS,
    make_delegate,
    make_done,
    make_launch_subagents,
    make_wait,
)
from rlmflow.utils import find_code_blocks
from rlmflow.workspace import (
    Context,
    ContextVariable,
    InMemoryContext,
    InMemorySession,
    Session,
    SessionVariable,
    Workspace,
)


def _child_config(
    parent: Graph,
    *,
    child_max_iterations: int | None,
) -> dict[str, Any]:
    """Derive the per-child config dict from ``parent.config``.

    Child iteration caps are engine policy, not model-controlled tool args.
    Children do not inherit a fraction of the parent's iteration limit; root
    and child budgets are independent knobs.
    """
    return {**parent.config, "max_iterations": child_max_iterations}


class RLMFlow(LLMClient):
    """Recursive language-model flow engine.

    Holds the prompt builder, runtime sessions, pool, and persistence
    handles. The execution graph itself lives in the session — every
    step reloads it through
    :meth:`~rlmflow.workspace.Session.load_graph`.

    Every method below is an extension seam. Subclass and override
    what you want; the default implementations call ``super()`` paths
    or pure helpers from :mod:`rlmflow.engine`.

    Overridable surface:

    - **Lifecycle:** :meth:`start` / :meth:`run` / :meth:`chat` /
      :meth:`step` / :meth:`terminate`
    - **Per-step transitions:** :meth:`apply_one` / :meth:`step_llm` /
      :meth:`step_exec` / :meth:`step_after_supervising`
    - **LLM half-step:** :meth:`reply_to` / :meth:`call_llm` /
      :meth:`llm_client_for` / :meth:`extract_code`
    - **Messages / prompt:** :meth:`build_messages` /
      :meth:`build_system_prompt` / :meth:`build_tools_section` /
      :meth:`build_status_section`
    - **Runtime / env:** :meth:`runtime_for` /
      :meth:`create_runtime_session` / :meth:`inject_env` /
      :meth:`register_tools` / :meth:`format_exec_output`
    - **Child spawning:** :meth:`spawn_child`
    - **Bookkeeping:** :meth:`record_usage` / :meth:`node_config`
    """

    # ── construction ─────────────────────────────────────────────────

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
        output_parser: Callable[[str, Schema], Any] | None = None,
    ) -> None:
        if workspace is None and runtime is None:
            raise ValueError("RLMFlow requires either runtime= or workspace=.")
        if workspace is not None and runtime is None:
            runtime = LocalRuntime(workspace=workspace)
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
        self.transcript_recorder = TranscriptRecorder(self.session)
        self.config = config or RLMConfig()
        self.runtime_factory = runtime_factory
        self.prompt_builder = prompt_builder or DEFAULT_BUILDER
        self.pool = create_pool(self.config, pool)
        self.node_scheduler = node_scheduler or NodeScheduler()
        self.output_parser = output_parser or StructuredOutputParser()

        self.llm_clients: dict[str, LLMClient] = {}
        self.model_descriptions: dict[str, str] = {}
        llm_thread_safe: dict[str, bool] = {}
        for key, entry in (llm_clients or {}).items():
            self.llm_clients[key] = entry["model"]
            if "description" in entry:
                self.model_descriptions[key] = entry["description"]
            if "thread_safe" in entry:
                llm_thread_safe[key] = bool(entry["thread_safe"])
        if "default" not in self.llm_clients:
            self.llm_clients["default"] = self.llm_client
        self.llm_channel = LLMChannel(
            self.llm_clients,
            max_concurrency=(
                self.config.llm_max_concurrency
                if self.config.llm_max_concurrency is not None
                else self.config.max_concurrency
            ),
            request_timeout=self.config.llm_request_timeout,
            thread_safe=llm_thread_safe,
        )

        self.runtime_sessions: dict[str, Runtime] = {ROOT_RUNTIME_ID: runtime}
        self.terminate_requested: set[str] = set()
        self.last_usage: LLMUsage | None = None
        self.register_tools(runtime)

    # ── lifecycle ────────────────────────────────────────────────────

    def start(
        self,
        query: str | None = None,
        *,
        graph: Graph | None = None,
        context: str | None = None,
        contexts: dict[str, str] | None = None,
        context_metadata: dict[str, Any] | None = None,
        agent_id: str = "root",
        output_schema: Schema | None = None,
    ) -> Graph:
        query = query or DEFAULT_QUERY
        if graph is not None:
            return self._start_from_graph(
                graph,
                query=query,
                context=context,
                contexts=contexts,
                context_metadata=context_metadata,
                output_schema=output_schema,
            )

        durable_output_schema = self._register_output_schema(agent_id, output_schema)

        self.context.write(
            "context",
            context if context is not None else "",
            agent_id=agent_id,
            metadata=context_metadata,
        )
        for key, value in (contexts or {}).items():
            self.context.write(key, value, agent_id=agent_id)

        context_hint = self.context.list_contexts(agent_id=agent_id)
        context_info = self._context_info(agent_id)
        root = Graph(
            agent_id=agent_id,
            branch_id=self.workspace.branch_id if self.workspace else "main",
            depth=0,
            query=query,
            config=self.node_config(),
            workspace=self.workspace.ref() if self.workspace else None,
            runtime=RuntimeRef(id=ROOT_RUNTIME_ID),
        )
        initial_query = prepare_node_for_append(
            root,
            UserQuery(
                output_schema=durable_output_schema,
                content=build_user_prompt(
                    query=query,
                    iteration=0,
                    depth=0,
                    max_depth=self.config.max_depth,
                    context_keys=context_hint,
                    context_info=context_info,
                ),
            ),
        )
        root.nodes.append(initial_query)
        root.system_prompt = self.build_system_prompt(root)
        self.session.rewrite_graph(root)
        graph = self.session.load_graph()
        if self.workspace is not None:
            self.workspace.mark_graph_synced(graph)
        return graph

    def run(
        self,
        query: str | None = None,
        *,
        graph: Graph | None = None,
        output_schema: Schema | None = None,
        **kwargs,
    ) -> Any:
        schema = output_schema
        graph = self.start(query, graph=graph, output_schema=output_schema, **kwargs)
        while not graph.finished:
            graph = self.step(graph)
        result = graph.result()
        if schema is None:
            return result
        return self.output_parser(self._json_content(result), schema)

    def _start_from_graph(
        self,
        graph: Graph,
        *,
        query: str,
        context: str | None = None,
        contexts: dict[str, str] | None = None,
        context_metadata: dict[str, Any] | None = None,
        output_schema: Schema | None = None,
    ) -> Graph:
        engine = self.for_graph(graph)
        if engine is not self:
            return engine._start_from_graph(
                graph,
                query=query,
                context=context,
                contexts=contexts,
                context_metadata=context_metadata,
                output_schema=output_schema,
            )

        committed = self.commit_graph(graph)
        if not committed.finished:
            raise ValueError(
                "start(..., graph=...) requires a finished graph. "
                "Use step(graph) to continue unfinished work."
            )

        agent_id = committed.agent_id
        durable_output_schema = self._register_output_schema(agent_id, output_schema)

        if context is not None:
            self.context.write(
                "context",
                context,
                agent_id=agent_id,
                metadata=context_metadata,
            )
        for key, value in (contexts or {}).items():
            self.context.write(key, value, agent_id=agent_id)

        context_hint = self.context.list_contexts(agent_id=agent_id)
        context_info = self._context_info(agent_id)
        next_query = prepare_node_for_append(
            committed,
            UserQuery(
                output_schema=durable_output_schema,
                content=build_user_prompt(
                    query=query,
                    iteration=0,
                    depth=committed.depth,
                    max_depth=committed.config.get("max_depth", self.config.max_depth),
                    context_keys=context_hint,
                    context_info=context_info,
                ),
            ),
            inherit_output_schema=False,
        )
        committed.nodes.append(next_query)
        committed.system_prompt = self.build_system_prompt(committed)
        self.session.rewrite_graph(committed)
        graph = self.session.load_graph()
        if self.workspace is not None:
            self.workspace.mark_graph_synced(graph)
        return graph

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

        Two phases:

        1. **Plan** — :func:`rlmflow.engine.actions.act` projects
           every runnable agent's current observation into an
           :class:`~rlmflow.engine.actions.Action` (pure, no I/O).
        2. **Apply** — every action is materialized in parallel via
           :meth:`apply_one`, which writes the resulting
           ``(ActionNode, ObservationNode)`` pair through the session.

        Returns a freshly-loaded :class:`Graph` snapshot.
        """
        return scheduling.step(self, graph)

    def for_graph(self, graph: Graph) -> RLMFlow:
        """Return an engine bound to ``graph``'s workspace, if any."""

        ref = graph.workspace
        if ref is None:
            return self
        if (
            self.workspace is not None
            and Path(ref.root).resolve() == self.workspace.root
        ):
            return self
        return self.with_workspace(Workspace.create(ref.root, branch_id=ref.branch_id))

    def with_workspace(self, workspace: Workspace) -> RLMFlow:
        """Return a sibling engine using ``workspace`` as durable state."""

        if self.workspace is not None and workspace.root == self.workspace.root:
            return self
        llm_clients = {
            key: {
                "model": client,
                **(
                    {"description": self.model_descriptions[key]}
                    if key in self.model_descriptions
                    else {}
                ),
            }
            for key, client in self.llm_clients.items()
            if key != "default" or client is not self.llm_client
        }
        return type(self)(
            self.llm_client,
            config=self.config,
            runtime_factory=self.runtime_factory,
            llm_clients=llm_clients,
            pool=self.pool,
            prompt_builder=self.prompt_builder,
            workspace=workspace,
            node_scheduler=self.node_scheduler,
        )

    def commit_graph(
        self,
        graph: Graph,
        *,
        fork: bool = False,
        new_location: str | Path | None = None,
        branch_id: str | None = None,
    ) -> Graph:
        """Rewrite the bound session so ``graph`` is the durable state."""

        engine = self.for_graph(graph)
        if fork:
            if engine.workspace is None:
                raise ValueError("commit_graph(..., fork=True) requires a workspace")
            if new_location is None:
                raise TypeError("commit_graph(..., fork=True) requires new_location")
            workspace = engine.workspace.fork(
                new_location=new_location,
                new_branch_id=branch_id,
            )
            engine = engine.with_workspace(workspace)

        if engine.workspace is not None:
            return engine.workspace.sync_graph(graph)

        engine.session.rewrite_graph(graph)
        return engine.session.load_graph()

    def _refill_eager_children(
        self,
        _done_id: str,
        _result: object,
        active_ids: set[str],
    ) -> list[tuple[str, Callable[[], None]]]:
        """Return newly runnable eager-child tasks after one task completes."""
        return scheduling.refill_eager_children(self, _done_id, _result, active_ids)

    def terminate(self, graph: Graph) -> Graph:
        """Mark every still-running agent for a final-answer turn.

        Equivalent to giving every agent one last chance to emit ``done()``.
        The engine then drives those agents to terminal states as normal.
        """
        for aid in graph.agents:
            if not graph.agents[aid].finished:
                self.terminate_requested.add(aid)
        return self.session.load_graph()

    # ── per-step transitions ─────────────────────────────────────────

    def apply_one(self, action: Action) -> None:
        """Materialize one :class:`Action` against the persisted graph.

        Reloads the graph from ``self.session``, enforces the global
        token budget, and dispatches to the half-step handler keyed
        by action type. The dispatch logic itself lives in
        :func:`rlmflow.engine.actions.act_one`; this method does no
        re-decisioning.
        """
        return transitions.apply_one(self, action)

    def step_llm(
        self,
        graph: Graph,
        last: Node,
        *,
        force_final: bool,
        model: str | None = None,
    ) -> None:
        """LLM half of one turn: write ``LLMAction → LLMOutput``.

        ``last`` is the observation the LLM is replying to (a
        :class:`UserQuery`, :class:`ExecOutput`, or :class:`ErrorOutput`).
        ``force_final`` is the policy decision (computed by
        :func:`~rlmflow.engine.actions.act_one`) to force a terminal
        answer this turn. ``model`` optionally overrides
        ``graph.config['model']`` for this single call.

        The next :meth:`apply_one` round will see :class:`LLMOutput`
        as the current state and run :meth:`step_exec` against it.
        """
        return transitions.step_llm(
            self,
            graph,
            last,
            force_final=force_final,
            model=model,
        )

    def step_exec(self, graph: Graph, llm_output: LLMOutput) -> None:
        """Exec half of one turn: write ``ExecAction → CodeObservation``.

        Reads the code from ``llm_output`` (the assistant's reply
        rendered as a code block), runs it through the runtime, and
        persists the resulting :class:`CodeObservation` (one of
        :class:`ExecOutput` / :class:`SupervisingOutput` /
        :class:`ErrorOutput` / :class:`DoneOutput`).
        """
        return transitions.step_exec(self, graph, llm_output)

    def _run_exec(
        self,
        graph: Graph,
        exec_action: ExecAction,
        code: str,
    ) -> None:
        return transitions.run_exec(self, graph, exec_action, code)

    def step_after_supervising(
        self,
        graph: Graph,
        last: SupervisingOutput,
    ) -> None:
        """Resume half: write ``ResumeAction → CodeObservation``.

        Drives the supervising agent forward after its waited-on
        children have settled. On a cold start (process restart or
        fork), the live coroutine is gone — we replay the action code
        with ``rlm_delegate`` in replay mode so it pauses at the same
        await before the regular resume path takes over.
        """
        return transitions.step_after_supervising(self, graph, last)

    # ── LLM half-step ────────────────────────────────────────────────

    def reply_to(
        self,
        graph: Graph,
        last: Node,
        *,
        force_final: bool,
    ) -> tuple[LLMOutput, LLMUsage]:
        """Ask the LLM for the next turn; return ``(LLMOutput, LLMUsage)``.

        Always returns an :class:`LLMOutput`, even when the reply has
        no parseable code block (in which case ``LLMOutput.code`` is
        ``""``). The caller is responsible for handling the empty-code
        case by appending a follow-up :class:`ErrorOutput` (with
        ``error="no_code_block"``).

        The returned ``LLMUsage`` is the per-call usage; the caller
        (typically :meth:`step_llm`) decides whether to cache it as
        ``self.last_usage`` via :meth:`record_usage`.
        """
        messages = self.build_messages(graph, force_final=force_final)
        model = getattr(last, "model", None) or graph.config.get("model", "default")
        client = self.llm_client_for(graph, model=model)
        t0 = time.time()
        raw, usage = self.call_llm(messages, model=model)
        elapsed_s = round(time.time() - t0, 3)
        code = self.extract_code(raw)
        self.transcript_recorder.record_turn(
            graph=graph,
            last=last,
            messages=messages,
            client=client,
            force_final=force_final,
            raw=raw,
            usage=usage,
            elapsed_s=elapsed_s,
        )
        output = LLMOutput(
            agent_id=graph.agent_id,
            seq=last.seq + 1,
            reply=raw,
            code=code or "",
            model=getattr(client, "model", model),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
        return output, usage

    def call_llm(
        self,
        messages: list[dict[str, str]],
        *,
        client: LLMClient | None = None,
        model: str | None = None,
        **llm_kwargs,
    ) -> tuple[str, LLMUsage]:
        """Stream a chat completion and return ``(text, usage)``.

        Override to add retries, caching, mocking, etc. Defaults to
        routing through the shared LLM channel and returning per-call
        usage.
        """
        if model is not None:
            return self.llm_channel.call(model, messages, **llm_kwargs)
        if client is None:
            return self.llm_channel.call("default", messages, **llm_kwargs)
        for key, candidate in self.llm_clients.items():
            if candidate is client:
                return self.llm_channel.call(key, messages, **llm_kwargs)
        return client.completion(messages, **llm_kwargs)

    @tool(
        "Run multiple independent one-shot LLM prompts concurrently. "
        "Returns results in the same order as the prompts. Set output_schema "
        "to validate each response as structured JSON."
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
        stop: str | list[str] | None = None,
    ) -> list[Any]:
        """Run one-shot LLM prompts concurrently without spawning child agents."""

        if isinstance(prompts, str) or not isinstance(prompts, list):
            raise TypeError("llm_query_batched() requires a list[str] of prompts")
        if not all(isinstance(prompt, str) for prompt in prompts):
            raise TypeError("llm_query_batched() requires a list[str] of prompts")
        if model not in self.llm_clients:
            keys = ", ".join(sorted(self.llm_clients))
            raise ValueError(f"unknown model {model!r}. available: {keys}")
        if not prompts:
            return []

        request_prompts = prompts
        if output_schema is not None:
            json_schema_for(output_schema)
            request_prompts = [
                self._with_output_schema_hint(prompt, output_schema)
                for prompt in prompts
            ]

        llm_kwargs = {
            key: value
            for key, value in {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stop": stop,
            }.items()
            if value is not None
        }
        pairs = self.llm_channel.batch(model, request_prompts, **llm_kwargs)

        total_usage = LLMUsage(
            input_tokens=sum(usage.input_tokens for _, usage in pairs),
            output_tokens=sum(usage.output_tokens for _, usage in pairs),
        )
        self.record_usage(total_usage)
        if output_schema is None:
            return [text for text, _ in pairs]
        return [self.output_parser(text, output_schema) for text, _ in pairs]

    def llm_client_for(self, graph: Graph, *, model: str | None = None) -> LLMClient:
        """Pick the per-agent LLM client.

        The agent's ``config["model"]`` is the lookup key into
        ``self.llm_clients``; when missing, fall back to
        ``self.llm_client``. Override to add per-graph routing.
        """
        key = model or graph.config.get("model", "default")
        return self.llm_clients.get(key, self.llm_client)

    def extract_code(self, text: str) -> str | None:
        """Pull the first (or merged) ```repl block from an LLM reply.

        Override to recognize different fence syntax, inject a
        preamble, or filter blocks before they reach the runtime.
        """
        blocks = find_code_blocks(text)
        if not blocks:
            return None
        return blocks[0] if self.config.single_block else "\n\n".join(blocks)

    # ── messages / system prompt ─────────────────────────────────────

    def build_messages(
        self,
        graph: Graph,
        *,
        force_final: bool = False,
    ) -> list[dict[str, str]]:
        """Render ``graph``'s trajectory as a chat-message list."""
        system_content = graph.system_prompt or self.build_system_prompt(graph)
        system = {"role": "system", "content": system_content}

        context_keys = self.context.list_contexts(agent_id=graph.agent_id)
        max_depth = (graph.config or {}).get("max_depth", self.config.max_depth)

        msgs = project_state_messages(graph.nodes)

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
            msgs = coalesce_messages(msgs)

        # Gate on LLMOutput count — not LLMAction count — so we don't
        # double up the user prompt on the very first turn. The
        # transition writes the paired ``LLMAction`` *before* calling
        # ``build_messages``, so the action for the in-progress turn is
        # already in ``graph.nodes`` here. ``LLMOutput``s only exist
        # for *completed* prior turns, which is what should gate the
        # continuation nudge.
        prior_turns = sum(1 for s in graph.nodes if is_llm_output(s))
        has_prior_turn = prior_turns > 0
        nudge: str | None = None
        if force_final:
            nudge = FINAL_ANSWER_ACTION
        elif has_prior_turn:
            nudge = build_user_prompt(
                query=graph.query,
                iteration=prior_turns,
                depth=graph.depth,
                max_depth=max_depth,
                context_keys=context_keys,
            )

        if nudge is not None:
            # `append_message` coalesces consecutive user messages so providers
            # that reject adjacent user blocks still receive a valid projection.
            append_message(msgs, "user", nudge)
        return [system] + msgs

    def build_system_prompt(self, graph: Graph) -> str:
        """Render the system prompt for the agent rooted in ``graph``."""
        if self.config.system_prompt:
            return self.config.system_prompt
        return self.prompt_builder.build(self, graph)

    def build_tools_section(self) -> str:
        """Render the tools section that lands inside the system prompt."""
        return tools_section(self, None)

    def build_status_section(self, graph: Graph) -> str:
        """Render the depth/status note that lands inside the system prompt."""
        return status_section(self, graph)

    # ── runtime / env ────────────────────────────────────────────────

    def runtime_for(self, ref: RuntimeRef | None) -> Runtime:
        """Return the runtime session bound to ``ref``, restoring lazily.

        On a fresh engine attached to a forked or reloaded workspace,
        ``self.runtime_sessions`` only holds the ``ROOT_RUNTIME_ID``
        runtime. Any other agent ``RuntimeRef`` would otherwise
        ``KeyError``. Instead, we materialize a fresh runtime via
        ``runtime_factory`` (or by cloning the root) and call
        :meth:`register_tools` against it. The REPL namespace and any
        suspended generator are *not* restored — callers that need a
        paused generator (the supervising transition) ask for
        replay-of-one separately.
        """
        session_id = ref.id if ref is not None else ROOT_RUNTIME_ID
        runtime = self.runtime_sessions.get(session_id)
        if runtime is None:
            runtime = (
                self.runtime_factory() if self.runtime_factory else self.runtime.clone()
            )
            self.runtime_sessions[session_id] = runtime
            self.register_tools(runtime)
        return runtime

    def create_runtime_session(
        self, parent_runtime: Runtime, *, agent_id: str
    ) -> RuntimeRef:
        """Allocate a fresh runtime session for a child agent."""
        session_id = f"{agent_id}:{uuid4().hex[:8]}"
        runtime = (
            self.runtime_factory() if self.runtime_factory else parent_runtime.clone()
        )
        self.runtime_sessions[session_id] = runtime
        self.register_tools(runtime)
        return RuntimeRef(id=session_id)

    def inject_env(self, graph: Graph, node: Node) -> Runtime:
        """Reset per-execution state on the runtime and seed env-style vars.

        ``runtime.env`` is the host-side dict shared with ``done`` /
        ``rlm_delegate`` closures (cleared + seeded each call). ``rlm_delegate``
        / ``rlm_wait`` are the internal primitives the ``launch_subagents``
        REPL launcher composes over. The same
        per-agent facts plus ``CONTEXT`` / ``SESSION`` are also pushed
        into the REPL namespace so user code can reference them by
        bare name.
        """
        runtime = self.runtime_for(graph.runtime)
        facts = execution_facts(
            agent_id=graph.agent_id,
            depth=graph.depth,
            max_depth=self.config.max_depth,
            parent_node_id=node.id,
        )
        seed_execution_env(runtime.env, facts)
        schema = graph.active_output_schema(node)
        if schema is not None:
            runtime.env[DONE_OUTPUT_SCHEMA] = schema
        preserve_suspension = runtime.suspended
        if preserve_suspension:
            runtime.prepare_for_resume()
            for name, value in facts.items():
                runtime.inject(name, value)
            return runtime

        runtime.prepare_for_execution()

        repl_vars = {
            **facts,
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

    def register_tools(self, runtime: Runtime | None = None) -> None:
        """Bind ``done`` / ``rlm_wait`` / ``rlm_delegate`` closures to ``runtime.env``.

        The ``rlm_delegate`` tool needs a way to spawn child agents — we
        pass :meth:`spawn_child` (bound to ``self``) so the tool can
        call back into engine state.

        Closures live in :mod:`rlmflow.tools.builtins` and capture the
        same ``env`` dict the engine reads back after each execution
        (so ``DONE_RESULT`` round-trips cleanly).
        """
        runtime = runtime or self.runtime
        runtime.register_tool(SHOW_VARS, core=True)
        runtime.register_tool(make_done(runtime.env, self.output_parser), core=True)
        rlm_wait = make_wait()
        runtime.register_tool(rlm_wait, core=True, hidden=True)
        runtime.register_tool(self.llm_query_batched, core=True)
        rlm_delegate = make_delegate(self.spawn_child, runtime.env)
        runtime.register_tool(rlm_delegate, core=True, hidden=True)
        runtime.register_tool(make_launch_subagents(rlm_delegate, rlm_wait), core=True)

    def format_exec_output(self, output: str) -> str:
        """Wrap REPL stdout for inclusion in the next user message."""
        return format_exec_output(output)

    # ── child spawning ───────────────────────────────────────────────

    def spawn_child(
        self,
        parent_agent_id: str,
        parent_node_id: str,
        name: str,
        query: str,
        context: str,
        *,
        model: str = "default",
        output_schema: Schema | None = None,
    ) -> ChildHandle | str:
        """Spawn a child agent under ``parent_agent_id``.

        Public seam invoked by the ``rlm_delegate(...)`` REPL closure.
        Creates a child :class:`~rlmflow.graph.Graph`, allocates a new
        runtime session, writes the initial seed action, and returns
        a :class:`~rlmflow.graph.ChildHandle`. Returns a refusal
        string instead of a handle if the child cannot be created
        (max depth reached, unknown model, …).
        """
        parent = self.session.load_graph().agents[parent_agent_id]
        if parent.depth >= self.config.max_depth:
            return f"[refused: max depth {self.config.max_depth}] Do this directly."
        if model not in self.llm_clients:
            keys = ", ".join(sorted(self.llm_clients))
            return f"[error: unknown model {model!r}. available: {keys}]"

        child_aid = unique_child_id(parent_agent_id, name, set(parent.children))
        durable_output_schema = self._register_output_schema(child_aid, output_schema)
        self.context.write("context", context, agent_id=child_aid)

        parent_runtime = self.runtime_for(parent.runtime)
        runtime_ref = self.create_runtime_session(parent_runtime, agent_id=child_aid)

        cfg = {
            **_child_config(
                parent,
                child_max_iterations=self.config.child_max_iterations,
            ),
            "model": model,
        }
        context_keys = self.context.list_contexts(agent_id=child_aid)
        context_info = self._context_info(child_aid)
        child_graph = Graph(
            agent_id=child_aid,
            branch_id=parent.branch_id,
            depth=parent.depth + 1,
            query=query,
            config=cfg,
            workspace=parent.workspace,
            runtime=runtime_ref,
            model=None,
            parent_agent_id=parent.agent_id,
            parent_node_id=parent_node_id,
        )
        initial_query = prepare_node_for_append(
            child_graph,
            UserQuery(
                output_schema=durable_output_schema,
                content=build_user_prompt(
                    query=query,
                    iteration=0,
                    depth=parent.depth + 1,
                    max_depth=cfg.get("max_depth", self.config.max_depth),
                    context_keys=context_keys,
                    context_info=context_info,
                ),
            ),
        )
        child_graph.nodes.append(initial_query)
        child_graph.system_prompt = self.build_system_prompt(child_graph)
        self.session.write_agent(child_graph)
        self.session.write_state(initial_query)
        return ChildHandle(child_aid)

    # ── bookkeeping ──────────────────────────────────────────────────

    def record_usage(self, usage: LLMUsage) -> None:
        """Cache the most recent ``LLMUsage``. Override for metrics."""
        self.last_usage = usage

    def _register_output_schema(
        self,
        _agent_id: str,
        output_schema: Schema | None,
    ) -> dict[str, Any] | None:
        if output_schema is None:
            return None
        return json_schema_for(output_schema)

    def structured_output_hint(self, schema: Schema) -> str:
        render_hint = getattr(self.output_parser, "system_prompt_hint", None)
        if callable(render_hint):
            return render_hint(schema)
        return json.dumps(json_schema_for(schema), indent=2)

    def _with_output_schema_hint(self, prompt: str, schema: Schema) -> str:
        return (
            f"{prompt}\n\n"
            "Return only a JSON value matching this JSON Schema. "
            "Do not include Markdown fences or explanatory text.\n"
            f"```json\n{self.structured_output_hint(schema)}\n```"
        )

    def _json_content(self, value: Any) -> str:
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)

    def _context_info(self, agent_id: str) -> dict[str, Any] | None:
        """Best-effort size/shape signal for the agent's primary ``CONTEXT``.

        Returns ``None`` when the agent has no `context` blob written —
        avoids lying about a 0-char chunk for the on-disk haystack case.
        """
        try:
            return self.context.info("context", agent_id=agent_id)
        except KeyError:
            return None

    def node_config(self) -> dict[str, Any]:
        """The default config dict written onto every fresh :class:`Graph`."""
        return {
            "model": "default",
            "max_depth": self.config.max_depth,
            "max_iterations": self.config.max_iterations,
            "max_output_length": self.config.max_output_length,
            "max_messages": self.config.max_messages,
            "child_max_iterations": self.config.child_max_iterations,
            "eager_children": self.config.eager_children,
            "single_block": self.config.single_block,
            "enable_structured_output": self.config.enable_structured_output,
            "max_budget": self.config.max_budget,
        }


__all__ = ["NodeScheduler", "RLMConfig", "RLMFlow", "create_pool"]
