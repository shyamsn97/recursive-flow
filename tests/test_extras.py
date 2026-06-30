"""Engine extras + graph persistence.

terminate, budget early-stop, per-child iteration cap, max_messages windowing,
single_block, eager_children scheduling, token rollups, the drop-in
``Flow.chat`` adapter, and ``Graph.save``/``load`` round-trips.
"""

from __future__ import annotations

import json
import threading
import time

from rflow import DoneOutput, Flow, Graph, LLMClient, LLMUsage, UserQuery
from rflow.utils.pool import CallablePool, SequentialPool, ThreadPool, create_pool
from tests.helpers import ScriptedLLM, StubLLM, make_flow, run_to_completion


class UsageLLM(ScriptedLLM):
    """Reports fixed per-call usage; replies with a plain print loop unless final."""

    def __init__(self, per_call: int = 10) -> None:
        super().__init__(self._reply)
        self._per_call = per_call

    def _reply(self, messages):
        self.last_usage = LLMUsage(
            input_tokens=self._per_call, output_tokens=self._per_call
        )
        if any("full iteration budget" in (m.get("content") or "") for m in messages):
            return '```repl\ndone("forced")\n```'
        return "```repl\nprint('thinking')\n```"


# ── terminate ────────────────────────────────────────────────────────


def test_terminate_forces_unfinished_agent_to_finish():
    flow = Flow(UsageLLM(), max_iters=100, max_depth=1)
    graph = flow.start("go")
    flow.step()
    flow.step()
    assert not graph.finished
    flow.terminate()
    while not graph.finished:
        flow.step()
    assert graph.result() == "forced"


def test_terminate_ignores_finished_agents():
    flow = make_flow('```repl\ndone("ok")\n```')
    run_to_completion(flow, "go")
    # Re-flagging a finished agent is a no-op and doesn't error.
    flow.terminate(["root"])
    assert "root" not in flow._terminate_requested


# ── budget early-stop ────────────────────────────────────────────────


def test_budget_stops_with_done_observation_and_no_more_calls():
    flow = Flow(UsageLLM(per_call=10), max_iters=100, max_budget=15)
    graph = run_to_completion(flow, "go")
    assert "budget exceeded" in graph.result()


def test_no_budget_means_no_early_stop():
    flow = Flow(UsageLLM(per_call=10), max_iters=3, max_budget=None)
    graph = run_to_completion(flow, "go")
    assert graph.result() == "forced"  # forced by max_iters, not budget


# ── per-child iteration cap ──────────────────────────────────────────


def test_child_max_iters_stamped_on_spawned_children():
    flow = Flow(StubLLM(), child_max_iters=3)
    flow.start("go")
    child = flow.spawn_child("root", "kid", "do work")
    assert flow.graph[child.agent_id].max_iters == 3


def test_plan_one_uses_per_agent_max_iters_over_flow_default():
    flow = Flow(StubLLM(), max_iters=99)
    flow.start("go")
    agent = flow.graph
    agent.max_iters = 1
    # one LLMAction already counts toward the cap after the first turn
    from rflow import LLMAction

    agent.nodes.append(LLMAction(agent_id="root", seq=1))
    agent.nodes.append(UserQuery(agent_id="root", seq=2, content="next"))
    action = flow.plan_one(agent)
    assert action.force_final is True


# ── max_messages windowing ───────────────────────────────────────────


def test_max_messages_windows_history():
    from rflow import ExecOutput, LLMOutput

    flow = Flow(StubLLM(), max_messages=4)
    graph = Graph(agent_id="root", system_prompt="SYS")
    # Alternating assistant/user turns so they don't coalesce into one block.
    graph.nodes = [UserQuery(agent_id="root", seq=0, content="turn-0")]
    seq = 1
    for i in range(1, 7):
        graph.nodes.append(LLMOutput(agent_id="root", seq=seq, reply=f"reply-{i}"))
        seq += 1
        graph.nodes.append(
            ExecOutput(agent_id="root", seq=seq, output=f"o{i}", content=f"out-{i}")
        )
        seq += 1
    msgs = flow.build_messages(graph, force_final=False)
    assert msgs[0]["role"] == "system"
    assert any("omitted" in m["content"] for m in msgs)
    # window keeps it bounded near max_messages (+nudge)
    assert len(msgs) <= 5


# ── eager_children + pool ────────────────────────────────────────────


def test_eager_children_runs_to_completion():
    flow = Flow(StubLLM(), eager_children=True, max_concurrency=4, max_depth=2)
    graph = run_to_completion(flow, "go")
    assert graph.finished
    assert isinstance(flow.pool, ThreadPool)


class TimelineLLM(LLMClient):
    """Deterministic LLM that exposes whether child turns overlap."""

    thread_safe = True

    def __init__(self) -> None:
        self.started_at = time.perf_counter()
        self.events: list[tuple[float, str]] = []
        self._lock = threading.Lock()

    def mark(self, label: str) -> None:
        with self._lock:
            self.events.append((time.perf_counter() - self.started_at, label))

    def chat(self, messages, *args, **kwargs):
        self.last_usage = LLMUsage(input_tokens=1, output_tokens=1)
        convo = "\n".join((m.get("content") or "").lower() for m in messages)
        if "child a slow task" in convo:
            self.mark("childa.task_1 start")
            time.sleep(0.25)
            self.mark("childa.task_1 finish")
            return '```repl\ndone("A done")\n```'
        if "child b two-step task" in convo:
            if "childb task_1 exec" not in convo:
                self.mark("childb.task_1 start")
                self.mark("childb.task_1 finish")
                return '```repl\nprint("childb task_1 exec")\n```'
            self.mark("childb.task_2 start")
            self.mark("childb.task_2 finish")
            return '```repl\ndone("B done")\n```'
        return (
            "```repl\n"
            "results = await launch_subagents([\n"
            '    {"name": "childa", "query": "Child A slow task"},\n'
            '    {"name": "childb", "query": "Child B two-step task"},\n'
            "])\n"
            'done(" | ".join(results))\n'
            "```"
        )


def _run_timeline_flow(*, eager_children: bool) -> TimelineLLM:
    llm = TimelineLLM()
    flow = Flow(
        llm,
        eager_children=eager_children,
        max_concurrency=2,
        max_depth=2,
        max_iters=8,
    )
    graph = run_to_completion(flow, "Show eager child scheduling.")
    assert graph.result() == "A done | B done"
    return llm


def test_eager_children_refills_fast_child_before_slow_sibling_finishes():
    llm = _run_timeline_flow(eager_children=True)
    events = {label: t for t, label in llm.events}

    assert events["childb.task_2 start"] < events["childa.task_1 finish"]


def test_non_eager_children_waits_for_barrier_before_fast_child_continues():
    llm = _run_timeline_flow(eager_children=False)
    events = {label: t for t, label in llm.events}

    assert events["childb.task_2 start"] > events["childa.task_1 finish"]


def test_thread_pool_run_until_idle_refills_before_active_sibling_finishes():
    pool = ThreadPool(max_concurrency=2)
    started_at = time.perf_counter()
    events: list[tuple[float, str]] = []
    lock = threading.Lock()

    def mark(label: str) -> None:
        with lock:
            events.append((time.perf_counter() - started_at, label))

    def slow() -> str:
        mark("slow start")
        time.sleep(0.25)
        mark("slow finish")
        return "slow"

    def fast() -> str:
        mark("fast start")
        mark("fast finish")
        return "fast"

    def refill(task_id, _result, active_ids):
        if task_id == "fast":
            assert active_ids == {"slow"}

            def fast_next() -> str:
                mark("fast next start")
                mark("fast next finish")
                return "fast next"

            return [("fast.next", fast_next)]
        return []

    try:
        results = pool.run_until_idle([("slow", slow), ("fast", fast)], refill)
    finally:
        pool.shutdown()

    times = {label: t for t, label in events}
    assert results == {"fast": "fast", "fast.next": "fast next", "slow": "slow"}
    assert times["fast next start"] < times["slow finish"]


def test_custom_pool_runs_normal_steps_without_eager_children():
    seen: list[list[str]] = []

    def execute(tasks):
        seen.append([task_id for task_id, _fn in tasks])
        return {task_id: fn() for task_id, fn in tasks}

    flow = Flow(StubLLM(), pool=execute, max_depth=0)
    flow.start("go")
    flow.step()

    assert seen == [["root"]]


def test_thread_pool_executes_single_task_inline():
    pool = ThreadPool()
    main_thread_id = threading.get_ident()
    seen: list[int] = []
    try:
        pool.execute([("root", lambda: seen.append(threading.get_ident()))])
    finally:
        pool.shutdown()

    assert seen == [main_thread_id]


def test_create_pool_resolves_variants():
    assert isinstance(create_pool(None, max_concurrency=4), ThreadPool)
    assert isinstance(create_pool(None, max_concurrency=1), SequentialPool)
    sentinel = SequentialPool()
    assert create_pool(sentinel, max_concurrency=4) is sentinel
    assert isinstance(create_pool(lambda tasks: {}, max_concurrency=4), CallablePool)


def test_sequential_pool_run_until_idle_drains_refill():
    pool = SequentialPool()
    seen: list[str] = []

    def refill(task_id, _result, _active):
        if task_id == "a":
            return [("b", lambda: seen.append("b"))]
        return []

    pool.run_until_idle([("a", lambda: seen.append("a"))], refill)
    assert seen == ["a", "b"]


# ── token rollups ────────────────────────────────────────────────────


def test_tokens_sum_over_subtree():
    flow = Flow(UsageLLM(per_call=7), max_iters=2)
    graph = run_to_completion(flow, "go")
    inp, out = graph.tokens()
    assert inp > 0 and out > 0
    assert graph.total_tokens() == inp + out


# ── drop-in Flow.chat ────────────────────────────────────────────────


def test_flow_chat_runs_query_and_reports_usage():
    inner = make_flow('```repl\ndone("42")\n```')
    text, usage = inner.completion([{"role": "user", "content": "what is the answer"}])
    assert text == "42"
    assert isinstance(usage, LLMUsage)
    assert inner.chat([{"role": "user", "content": "again"}]) == "42"
    assert inner.thread_safe is False


# ── persistence ──────────────────────────────────────────────────────


def test_graph_save_load_single_file_roundtrip(tmp_path):
    g = Graph(
        agent_id="root",
        query="hi",
        nodes=[
            UserQuery(agent_id="root", seq=0, content="hi"),
            DoneOutput(agent_id="root", seq=1, result="hello"),
        ],
    )
    path = g.save(tmp_path / "graph.json")
    assert path.name == "graph.json"
    loaded = Graph.load(path)
    assert loaded.to_dict() == g.to_dict()
    assert loaded.result() == "hello"


def test_graph_save_to_directory_creates_run_layout(tmp_path):
    g = Graph(agent_id="root", nodes=[DoneOutput(agent_id="root", seq=0, result="x")])
    run_dir = g.save(tmp_path / "run")
    assert run_dir == tmp_path / "run"
    manifest = json.loads((run_dir / "graph.json").read_text())
    assert "root_agent_id" in manifest
    assert "nodes" not in manifest
    assert (run_dir / "agents" / "root" / "session.jsonl").is_file()
    assert Graph.load(tmp_path / "run").result() == "x"


def test_max_iters_survives_serialization_roundtrip():
    g = Graph(agent_id="root", max_iters=5)
    assert Graph.from_dict(g.to_dict()).max_iters == 5
