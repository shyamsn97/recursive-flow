"""Fork-and-resume from a ``SupervisingOutput``.

Covers the scenario where a parent agent has yielded inside ``rlm_wait(...)``,
its children have run to completion, the workspace is forked (or the
process restarted), and a fresh ``RLMFlow`` is asked to continue. The
new engine has no live generator on its runtime — it has to replay the
parent's action code with ``rlm_delegate`` in replay mode to rebuild the
suspended generator at the right yield, then drop into the normal
resume path.
"""

from __future__ import annotations

from pathlib import Path

from rlmflow import (
    DoneOutput,
    Graph,
    LLMClient,
    LLMUsage,
    RLMConfig,
    RLMFlow,
    Workspace,
    is_supervising,
)


class _ScriptedLLM(LLMClient):
    """Returns a fixed reply per ``(prompt-substring → reply)`` rule."""

    def __init__(self, rules: list[tuple[str, str]]) -> None:
        self.rules = list(rules)

    def chat(self, messages, *args, **kwargs):
        self.last_usage = LLMUsage(input_tokens=10, output_tokens=5)
        prompt = messages[-1]["content"]
        for needle, reply in self.rules:
            if needle in prompt:
                return reply
        raise AssertionError(
            f"No scripted reply matched the prompt:\n{prompt[:200]}"
        )


def _step_until(agent: RLMFlow, graph: Graph, predicate) -> Graph:
    """Step the engine until ``predicate(graph)`` is true. Safety bound."""
    for _ in range(50):
        if predicate(graph):
            return graph
        graph = agent.step(graph)
    raise AssertionError("predicate never became true within 50 steps")


def _parent_supervising_with_terminal_children(graph: Graph) -> bool:
    parent = graph.agents.get("root")
    if parent is None:
        return False
    cur = parent.current()
    if not is_supervising(cur):
        return False
    waiting = [graph.agents[aid] for aid in cur.waiting_on if aid in graph.agents]
    return bool(waiting) and all(c.finished for c in waiting)


# ── single-yield fork-resume ─────────────────────────────────────────


PARENT_REPLY_SINGLE = (
    "```repl\n"
    'h = rlm_delegate(name="worker", query="do thing", context="")\n'
    "results = await rlm_wait(h)\n"
    'done("got: " + results[0])\n'
    "```"
)
WORKER_REPLY = '```repl\ndone("hello")\n```'


def _scripted() -> _ScriptedLLM:
    return _ScriptedLLM(
        [
            ('"do thing"', WORKER_REPLY),
            ("", PARENT_REPLY_SINGLE),  # default for the parent's first turn
        ]
    )


def test_fork_resumes_supervising_with_terminal_children(tmp_path: Path):
    source = Workspace.create(tmp_path / "main")
    src_engine = RLMFlow(
        llm_client=_scripted(),
        workspace=source,
        config=RLMConfig(max_depth=2, max_iterations=5),
    )

    graph = src_engine.start("parent task")
    graph = _step_until(src_engine, graph, _parent_supervising_with_terminal_children)

    # Sanity: parent is sitting at SupervisingOutput, child finished.
    assert is_supervising(graph.agents["root"].current())
    assert graph.agents["root.worker"].finished
    assert not graph.finished

    # Fork the workspace. New engine: brand-new runtime, no live generator,
    # no REPL namespace. The forked engine must rebuild via replay-of-one.
    forked = source.fork(new_dir=tmp_path / "b2")
    new_engine = RLMFlow(
        llm_client=_scripted(),
        workspace=forked,
        config=RLMConfig(max_depth=2, max_iterations=5),
    )

    forked_graph = forked.session.load_graph()
    assert is_supervising(forked_graph.agents["root"].current())

    # First step on the fresh engine must replay-of-one and resume.
    forked_graph = new_engine.step(forked_graph)
    while not forked_graph.finished:
        forked_graph = new_engine.step(forked_graph)

    assert forked_graph.result() == "got: hello"

    # Source workspace was not touched by the fork's resume.
    src_graph = source.session.load_graph()
    assert is_supervising(src_graph.agents["root"].current())


def test_fork_lets_us_swap_a_child_result_and_re_resume(tmp_path: Path):
    """The headline use case: branch a run, replace a child's result, continue."""
    source = Workspace.create(tmp_path / "main")
    src_engine = RLMFlow(
        llm_client=_scripted(),
        workspace=source,
        config=RLMConfig(max_depth=2, max_iterations=5),
    )

    graph = src_engine.start("parent task")
    graph = _step_until(src_engine, graph, _parent_supervising_with_terminal_children)

    forked = source.fork(new_dir=tmp_path / "alt")

    # Manually rewrite the child's terminal DoneOutput in the forked
    # session so that resume sees a different result.
    child_session_path = (
        forked.root / "session" / "root.worker" / "session.jsonl"
    )
    lines = child_session_path.read_text().splitlines()
    import json

    rewritten = []
    for line in lines:
        rec = json.loads(line)
        if rec.get("type") == "done_output":
            rec["result"] = "swapped"
            rec["content"] = "swapped"
        rewritten.append(json.dumps(rec))
    child_session_path.write_text("\n".join(rewritten) + "\n")

    new_engine = RLMFlow(
        llm_client=_scripted(),
        workspace=forked,
        config=RLMConfig(max_depth=2, max_iterations=5),
    )

    forked_graph = forked.session.load_graph()
    while not forked_graph.finished:
        forked_graph = new_engine.step(forked_graph)

    assert forked_graph.result() == "got: swapped"


# ── multi-yield fork-resume ──────────────────────────────────────────


PARENT_REPLY_MULTI = (
    "```repl\n"
    'h = rlm_delegate(name="a", query="step a", context="")\n'
    "first = await rlm_wait(h)\n"
    'v = rlm_delegate(name="b", query="step b", context="")\n'
    "second = await rlm_wait(v)\n"
    'done("p:" + first[0] + "+" + second[0])\n'
    "```"
)


def _multi_scripted() -> _ScriptedLLM:
    return _ScriptedLLM(
        [
            ('"step a"', '```repl\ndone("A")\n```'),
            ('"step b"', '```repl\ndone("B")\n```'),
            ("", PARENT_REPLY_MULTI),
        ]
    )


def _parent_at_second_supervise(graph: Graph) -> bool:
    parent = graph.agents.get("root")
    if parent is None:
        return False
    supervises = [s for s in parent.nodes if is_supervising(s)]
    if len(supervises) < 2:
        return False
    cur = parent.current()
    if not is_supervising(cur):
        return False
    if cur is not supervises[-1]:
        return False
    waiting = [graph.agents[aid] for aid in cur.waiting_on if aid in graph.agents]
    return bool(waiting) and all(c.finished for c in waiting)


def test_fork_resume_replays_through_multiple_yields(tmp_path: Path):
    source = Workspace.create(tmp_path / "main")
    src_engine = RLMFlow(
        llm_client=_multi_scripted(),
        workspace=source,
        config=RLMConfig(max_depth=2, max_iterations=8),
    )

    graph = src_engine.start("multi yield")
    graph = _step_until(src_engine, graph, _parent_at_second_supervise)

    forked = source.fork(new_dir=tmp_path / "b2")
    new_engine = RLMFlow(
        llm_client=_multi_scripted(),
        workspace=forked,
        config=RLMConfig(max_depth=2, max_iterations=8),
    )

    forked_graph = forked.session.load_graph()
    while not forked_graph.finished:
        forked_graph = new_engine.step(forked_graph)

    assert forked_graph.result() == "p:A+B"


# ── nested supervising tree fork-resume ───────────────────────────────


ROOT_REPLY_NESTED = (
    "```repl\n"
    "results = await launch_subagents([\n"
    '    {"name": "mid", "query": "mid task"},\n'
    "])\n"
    'done("root saw " + results[0])\n'
    "```"
)
MID_REPLY_NESTED = (
    "```repl\n"
    "results = await launch_subagents([\n"
    '    {"name": "leaf", "query": "leaf task"},\n'
    "])\n"
    'done("mid saw " + results[0])\n'
    "```"
)
LEAF_REPLY_NESTED = '```repl\ndone("leaf done")\n```'


def _nested_scripted() -> _ScriptedLLM:
    return _ScriptedLLM(
        [
            ('"leaf task"', LEAF_REPLY_NESTED),
            ('"mid task"', MID_REPLY_NESTED),
            ("", ROOT_REPLY_NESTED),
        ]
    )


def _nested_mid_supervising_with_leaf_done(graph: Graph) -> bool:
    if not {"root", "root.mid", "root.mid.leaf"} <= set(graph.agents):
        return False
    root = graph.agents["root"]
    mid = graph.agents["root.mid"]
    leaf = graph.agents["root.mid.leaf"]
    root_cur = root.current()
    mid_cur = mid.current()
    return (
        is_supervising(root_cur)
        and root_cur.waiting_on == ["root.mid"]
        and is_supervising(mid_cur)
        and mid_cur.waiting_on == ["root.mid.leaf"]
        and leaf.finished
        and leaf.result() == "leaf done"
    )


def _nested_source_at_leaf_done(tmp_path: Path) -> tuple[Workspace, Graph]:
    source = Workspace.create(tmp_path / "nested-source")
    engine = RLMFlow(
        llm_client=_nested_scripted(),
        workspace=source,
        config=RLMConfig(max_depth=3, max_iterations=8),
    )
    graph = engine.start("nested replay")
    graph = _step_until(engine, graph, _nested_mid_supervising_with_leaf_done)
    return source, graph


def test_fresh_engine_resumes_nested_child_supervisor_without_deleting_leaf(
    tmp_path: Path,
):
    source, graph = _nested_source_at_leaf_done(tmp_path)
    assert _nested_mid_supervising_with_leaf_done(graph)

    forked = source.fork(new_dir=tmp_path / "nested-fork")
    forked_graph = forked.session.load_graph()
    original_agents = list(forked_graph.agents)
    original_leaf_state_count = len(forked_graph.agents["root.mid.leaf"].nodes)

    fresh_engine = RLMFlow(
        llm_client=_nested_scripted(),
        workspace=forked,
        config=RLMConfig(max_depth=3, max_iterations=8),
    )

    forked_graph = fresh_engine.step(forked_graph)

    assert list(forked_graph.agents) == original_agents
    assert len(forked_graph.agents["root.mid.leaf"].nodes) == original_leaf_state_count
    assert forked_graph.agents["root.mid.leaf"].result() == "leaf done"
    assert forked_graph.agents["root.mid"].result() == "mid saw leaf done"
    assert is_supervising(forked_graph.agents["root"].current())

    source_graph = source.session.load_graph()
    assert is_supervising(source_graph.agents["root.mid"].current())


def test_fresh_engine_resumes_nested_child_then_root_supervisor(tmp_path: Path):
    source, graph = _nested_source_at_leaf_done(tmp_path)
    assert _nested_mid_supervising_with_leaf_done(graph)

    forked = source.fork(new_dir=tmp_path / "nested-fork")
    fresh_engine = RLMFlow(
        llm_client=_nested_scripted(),
        workspace=forked,
        config=RLMConfig(max_depth=3, max_iterations=8),
    )

    forked_graph = forked.session.load_graph()
    forked_graph = fresh_engine.step(forked_graph)
    assert forked_graph.agents["root.mid"].result() == "mid saw leaf done"
    assert is_supervising(forked_graph.agents["root"].current())

    forked_graph = fresh_engine.step(forked_graph)

    assert forked_graph.finished
    assert forked_graph.result() == "root saw mid saw leaf done"
    assert list(forked_graph.agents) == ["root", "root.mid", "root.mid.leaf"]

    source_graph = source.session.load_graph()
    assert is_supervising(source_graph.agents["root"].current())
    assert is_supervising(source_graph.agents["root.mid"].current())


# ── nested injection repair + cold-start replay ───────────────────────


ROOT_REPLY_REPAIR = (
    "```repl\n"
    "results = await launch_subagents([\n"
    '    {"name": "planner", "query": "planner task"},\n'
    "])\n"
    'done("root accepted: " + results[0])\n'
    "```"
)
PLANNER_REPLY_REPAIR = (
    "```repl\n"
    "results = await launch_subagents([\n"
    '    {"name": "worker", "query": "worker task"},\n'
    "])\n"
    'done("planner accepted: " + results[0])\n'
    "```"
)
WORKER_REPLY_REPAIR = (
    "```repl\n"
    'print("worker saw malformed payload")\n'
    'raise KeyError("missing field: answer")\n'
    "```"
)


def _repair_scripted() -> _ScriptedLLM:
    return _ScriptedLLM(
        [
            ('"worker task"', WORKER_REPLY_REPAIR),
            ('"planner task"', PLANNER_REPLY_REPAIR),
            ("", ROOT_REPLY_REPAIR),
        ]
    )


def _nested_worker_error_reached(graph: Graph) -> bool:
    if not {"root", "root.planner", "root.planner.worker"} <= set(graph.agents):
        return False
    root = graph.agents["root"].current()
    planner = graph.agents["root.planner"].current()
    worker = graph.agents["root.planner.worker"].current()
    return (
        is_supervising(root)
        and root.waiting_on == ["root.planner"]
        and is_supervising(planner)
        and planner.waiting_on == ["root.planner.worker"]
        and worker is not None
        and worker.type == "error_output"
    )


def test_injected_worker_fix_resumes_nested_supervisors_on_fresh_engine(
    tmp_path: Path,
):
    source = Workspace.create(tmp_path / "repair-source")
    source_engine = RLMFlow(
        llm_client=_repair_scripted(),
        workspace=source,
        config=RLMConfig(max_depth=3, max_iterations=8),
    )

    graph = source_engine.start("nested repair")
    graph = _step_until(source_engine, graph, _nested_worker_error_reached)

    forked = source.fork(new_dir=tmp_path / "repair-fork")
    forked_graph = forked.session.load_graph()
    original_agents = list(forked_graph.agents)
    original_worker_state_count = len(
        forked_graph.agents["root.planner.worker"].nodes
    )

    forked_graph = forked_graph.inject(
        target="root.planner.worker",
        node=DoneOutput(
            result="fixed worker result",
            content="Injected repair: fixed worker result",
            output="operator injected a fixed worker result",
        ),
    )
    forked.session.write_state(forked_graph.agents["root.planner.worker"].current())

    fresh_engine = RLMFlow(
        llm_client=_repair_scripted(),
        workspace=forked,
        config=RLMConfig(max_depth=3, max_iterations=8),
    )

    forked_graph = forked.session.load_graph()
    forked_graph = fresh_engine.step(forked_graph)
    assert forked_graph.agents["root.planner"].result() == (
        "planner accepted: fixed worker result"
    )
    assert is_supervising(forked_graph.agents["root"].current())

    forked_graph = fresh_engine.step(forked_graph)
    assert forked_graph.finished
    assert forked_graph.result() == (
        "root accepted: planner accepted: fixed worker result"
    )
    assert list(forked_graph.agents) == original_agents
    assert (
        len(forked_graph.agents["root.planner.worker"].nodes)
        == original_worker_state_count + 1
    )

    source_graph = source.session.load_graph()
    assert source_graph.agents["root.planner.worker"].current().type == "error_output"
    assert is_supervising(source_graph.agents["root"].current())
    assert is_supervising(source_graph.agents["root.planner"].current())
