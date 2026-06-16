"""Phase 5 trajectory editing: replace / truncate / inject / retrace_steps.

Ported and adapted from the legacy ``test_graph_surgery`` and ``test_timeline``
suites, minus the removed ``output_schema``/``branch_id``/``Workspace`` surface.
"""

from __future__ import annotations

import re

from rflow import (
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    Flow,
    Graph,
    LLMAction,
    LLMOutput,
    LLMUsage,
    ResumeAction,
    SupervisingOutput,
    UserQuery,
    retrace_steps,
)
from rflow.graph.timeline import _execution_ticks
from tests.helpers import ScriptedLLM, run_to_completion


# ── fixtures ──────────────────────────────────────────────────────────


def _graph_with_child() -> Graph:
    action = ExecAction(agent_id="root", seq=3, code="bad()")
    child = Graph(
        agent_id="root.worker",
        depth=1,
        parent_agent_id="root",
        parent_node_id=action.id,
        nodes=[UserQuery(agent_id="root.worker", seq=0, content="work")],
    )
    return Graph(
        agent_id="root",
        nodes=[
            UserQuery(agent_id="root", seq=0, content="q"),
            LLMAction(agent_id="root", seq=1),
            LLMOutput(agent_id="root", seq=2, reply="bad", code="bad()"),
            action,
            ErrorOutput(agent_id="root", seq=4, error="exec_exception"),
        ],
        children={child.agent_id: child},
    )


def _graph_waiting_on_children() -> Graph:
    spawn = ExecAction(agent_id="root", seq=3, code="spawn()")
    waited = SupervisingOutput(
        agent_id="root",
        seq=4,
        waiting_on=["root.rows", "root.cols", "root.missing"],
    )
    rows_grandchild = Graph(
        agent_id="root.rows.audit",
        depth=2,
        parent_agent_id="root.rows",
        parent_node_id="rows-audit-spawn",
        nodes=[UserQuery(agent_id="root.rows.audit", seq=0, content="audit")],
    )
    rows = Graph(
        agent_id="root.rows",
        depth=1,
        parent_agent_id="root",
        parent_node_id=spawn.id,
        nodes=[
            UserQuery(agent_id="root.rows", seq=0, content="rows"),
            ExecAction(agent_id="root.rows", id="rows-audit-spawn", seq=1, code="audit()"),
        ],
        children={rows_grandchild.agent_id: rows_grandchild},
    )
    cols = Graph(
        agent_id="root.cols",
        depth=1,
        parent_agent_id="root",
        parent_node_id=spawn.id,
        nodes=[UserQuery(agent_id="root.cols", seq=0, content="cols")],
    )
    unrelated = Graph(
        agent_id="root.unrelated",
        depth=1,
        parent_agent_id="root",
        parent_node_id=spawn.id,
        nodes=[UserQuery(agent_id="root.unrelated", seq=0, content="keep")],
    )
    return Graph(
        agent_id="root",
        nodes=[
            UserQuery(agent_id="root", seq=0, content="q"),
            LLMAction(agent_id="root", seq=1),
            LLMOutput(agent_id="root", seq=2, reply="spawn", code="spawn()"),
            spawn,
            waited,
        ],
        children={rows.agent_id: rows, cols.agent_id: cols, unrelated.agent_id: unrelated},
    )


# ── replace ─────────────────────────────────────────────────────────


def test_replace_last_action_defaults_to_descendant_truncation():
    graph = _graph_with_child()
    old_action = graph.last_action("root")
    edited = graph.replace_last_action("root", ExecAction(code="fixed()"))

    new_action = edited.last_action("root")
    assert new_action is not None and old_action is not None
    assert new_action.id != old_action.id
    assert new_action.agent_id == "root" and new_action.seq == old_action.seq
    assert new_action.code == "fixed()"
    assert [n.type for n in edited.nodes] == [
        "user_query",
        "llm_action",
        "llm_output",
        "exec_action",
    ]
    assert "root.worker" not in edited.agents
    # original is untouched (pure edit)
    assert graph.current().type == "error_output"


def test_replace_node_none_keeps_local_future_and_children():
    graph = _graph_with_child()
    old_action = graph.last_action("root")
    edited = graph.replace_node(old_action.id, ExecAction(code="meta()"), truncate="none")

    assert [n.type for n in edited.nodes] == [
        "user_query",
        "llm_action",
        "llm_output",
        "exec_action",
        "error_output",
    ]
    assert edited.nodes[3].code == "meta()"
    assert "root.worker" in edited.agents


def test_replace_supervising_node_prunes_waited_children_by_default():
    graph = _graph_waiting_on_children()
    supervising = graph.last_observation("root")
    edited = graph.replace_node(supervising.id, ExecOutput(output="another route"))

    assert [n.type for n in edited.nodes] == [
        "user_query",
        "llm_action",
        "llm_output",
        "exec_action",
        "exec_output",
    ]
    assert "root.rows" not in edited.agents
    assert "root.rows.audit" not in edited.agents
    assert "root.cols" not in edited.agents
    assert "root.unrelated" in edited.agents


def test_replace_supervising_after_keeps_waited_children():
    graph = _graph_waiting_on_children()
    supervising = graph.last_observation("root")
    edited = graph.replace_node(
        supervising.id, ExecOutput(output="another route"), truncate="after"
    )
    assert "root.rows" in edited.agents and "root.cols" in edited.agents


def test_replace_observation_gets_fresh_global_step():
    # Replacing an observation stamps a fresh next-step (the inherit fast-path
    # only applies when an action is replaced by an observation in one tick).
    graph = Graph(
        agent_id="root",
        nodes=[
            UserQuery(agent_id="root", seq=0, global_step=0, content="q"),
            LLMAction(agent_id="root", seq=1, global_step=1),
            LLMOutput(agent_id="root", seq=2, global_step=1, code="old"),
        ],
    )
    edited = graph.replace_last_observation("root", LLMOutput(code="new"), truncate="none")
    assert edited.nodes[2].global_step == 2
    assert edited.nodes[2].code == "new"


# ── truncate ────────────────────────────────────────────────────────


def test_truncate_after_supervising_keeps_waited_children():
    graph = _graph_waiting_on_children()
    supervising = graph.last_observation("root")
    edited = graph.truncate_after(supervising.id, descendants=True)
    assert {"root.rows", "root.cols", "root.unrelated"} <= set(edited.agents)


def test_truncate_agent_prunes_children_spawned_after_kept_states():
    first = ExecAction(agent_id="root", seq=1, code="first")
    second = ExecAction(agent_id="root", seq=3, code="second")
    early = Graph(
        agent_id="root.early",
        parent_agent_id="root",
        parent_node_id=first.id,
        nodes=[UserQuery(agent_id="root.early", seq=0, content="early")],
    )
    late = Graph(
        agent_id="root.late",
        parent_agent_id="root",
        parent_node_id=second.id,
        nodes=[UserQuery(agent_id="root.late", seq=0, content="late")],
    )
    graph = Graph(
        agent_id="root",
        nodes=[
            UserQuery(agent_id="root", seq=0, content="q"),
            first,
            ExecOutput(agent_id="root", seq=2, output="first"),
            second,
            ExecOutput(agent_id="root", seq=4, output="second"),
        ],
        children={early.agent_id: early, late.agent_id: late},
    )
    edited = graph.truncate_agent("root", after_seq=2)
    assert "root.early" in edited.agents and "root.late" not in edited.agents


def test_prune_descendants_spawned_after_keeps_boundary_children():
    first = ExecAction(agent_id="root", seq=1, code="first")
    second = ExecAction(agent_id="root", seq=3, code="second")
    early = Graph(
        agent_id="root.early",
        parent_agent_id="root",
        parent_node_id=first.id,
        nodes=[UserQuery(agent_id="root.early", seq=0, content="early")],
    )
    late = Graph(
        agent_id="root.late",
        parent_agent_id="root",
        parent_node_id=second.id,
        nodes=[UserQuery(agent_id="root.late", seq=0, content="late")],
    )
    graph = Graph(
        agent_id="root",
        nodes=[UserQuery(agent_id="root", seq=0, content="q"), first, second],
        children={early.agent_id: early, late.agent_id: late},
    )
    edited = graph.prune_descendants_spawned_after("root", seq=1)
    assert "root.early" in edited.agents and "root.late" not in edited.agents


# ── child edits invalidate ancestor resume paths ────────────────────


def test_child_edit_truncates_parent_resume_and_done_states():
    spawn = ExecAction(agent_id="root", seq=3, code="spawn()")
    child = Graph(
        agent_id="root.child",
        depth=1,
        parent_agent_id="root",
        parent_node_id=spawn.id,
        nodes=[
            UserQuery(agent_id="root.child", seq=0, content="child"),
            DoneOutput(agent_id="root.child", seq=1, result="old child"),
        ],
    )
    graph = Graph(
        agent_id="root",
        nodes=[
            UserQuery(agent_id="root", seq=0, content="q"),
            LLMAction(agent_id="root", seq=1),
            LLMOutput(agent_id="root", seq=2, reply="spawn", code="spawn()"),
            spawn,
            SupervisingOutput(agent_id="root", seq=4, waiting_on=["root.child"]),
            ResumeAction(agent_id="root", seq=5, resumed_from=["root.child"]),
            DoneOutput(agent_id="root", seq=6, result="old root"),
        ],
        children={child.agent_id: child},
    )
    edited = graph.replace_last_observation("root.child", ExecOutput(output="try again"))

    assert [n.type for n in edited.nodes] == [
        "user_query",
        "llm_action",
        "llm_output",
        "exec_action",
        "supervising_output",
    ]
    assert edited.agents["root.child"].current().type == "exec_output"
    assert not edited.finished


def test_child_edit_with_multiple_supervisors_truncates_to_last_wait():
    spawn = ExecAction(agent_id="root", seq=1, code="spawn()")
    child = Graph(
        agent_id="root.child",
        parent_agent_id="root",
        parent_node_id=spawn.id,
        nodes=[
            UserQuery(agent_id="root.child", seq=0, content="child"),
            DoneOutput(agent_id="root.child", seq=1, result="old child"),
        ],
    )
    graph = Graph(
        agent_id="root",
        nodes=[
            UserQuery(agent_id="root", seq=0, content="q"),
            spawn,
            SupervisingOutput(agent_id="root", seq=2, waiting_on=["root.child"]),
            ResumeAction(agent_id="root", seq=3, resumed_from=["root.child"]),
            ExecOutput(agent_id="root", seq=4, output="first resume"),
            SupervisingOutput(agent_id="root", seq=5, waiting_on=["root.child"]),
            ResumeAction(agent_id="root", seq=6, resumed_from=["root.child"]),
            DoneOutput(agent_id="root", seq=7, result="old root"),
        ],
        children={child.agent_id: child},
    )
    edited = graph.replace_last_observation("root.child", ExecOutput(output="try again"))
    assert edited.current().type == "supervising_output"
    assert edited.current().seq == 5


# ── injection ───────────────────────────────────────────────────────


def test_inject_output_appends_after_observation():
    graph = Graph(
        agent_id="root",
        nodes=[UserQuery(agent_id="root", seq=0, content="q")],
    )
    edited = graph.inject_output(target="root", output="steer here")
    assert [n.type for n in edited["root"].nodes] == ["user_query", "exec_output"]
    assert edited["root"].nodes[-1].output == "steer here"
    assert graph["root"].nodes[-1].type == "user_query"  # original untouched


def test_inject_into_finished_agent_raises():
    graph = Graph(
        agent_id="root",
        nodes=[DoneOutput(agent_id="root", seq=0, result="done")],
    )
    try:
        graph.inject_output(target="root", output="late")
    except ValueError as exc:
        assert "finished" in str(exc)
    else:
        raise AssertionError("expected ValueError injecting into finished agent")


def test_inject_pending_action_when_action_current_raises():
    graph = Graph(
        agent_id="root",
        nodes=[
            UserQuery(agent_id="root", seq=0, content="q"),
            LLMAction(agent_id="root", seq=1),
        ],
    )
    try:
        graph.inject(target="root", node=ExecAction(code="x()"))
    except ValueError as exc:
        assert "pending" in str(exc)
    else:
        raise AssertionError("expected ValueError queueing a second action")


def test_inject_regex_fans_across_matching_agents():
    child_a = Graph(
        agent_id="root.a",
        parent_agent_id="root",
        nodes=[UserQuery(agent_id="root.a", seq=0, content="a")],
    )
    child_b = Graph(
        agent_id="root.b",
        parent_agent_id="root",
        nodes=[UserQuery(agent_id="root.b", seq=0, content="b")],
    )
    graph = Graph(
        agent_id="root",
        nodes=[UserQuery(agent_id="root", seq=0, content="q")],
        children={"root.a": child_a, "root.b": child_b},
    )
    edited = graph.inject(target=re.compile(r"root\."), node=ExecOutput(output="hi"))
    assert edited["root.a"].nodes[-1].type == "exec_output"
    assert edited["root.b"].nodes[-1].type == "exec_output"
    assert edited["root"].nodes[-1].type == "user_query"  # didn't match


# ── retrace_steps ───────────────────────────────────────────────────


class _OneChild(ScriptedLLM):
    """root delegates one child, child returns immediately."""

    ROOT = (
        "```repl\n"
        "h = flow_delegate(name='child', query='do thing')\n"
        "results = await flow_wait(h)\n"
        "done('root:' + results[0])\n"
        "```"
    )
    CHILD = "```repl\ndone('child-answer')\n```"

    def __init__(self) -> None:
        super().__init__(self._reply)

    def _reply(self, messages):
        self.last_usage = LLMUsage(input_tokens=1, output_tokens=1)
        if any("do thing" in (m.get("content") or "") for m in messages):
            return self.CHILD
        return self.ROOT


def _state_count(graph: Graph) -> int:
    return sum(1 for _ in graph.all_nodes)


def test_retrace_steps_singleton_graph_returns_self():
    g = Graph(agent_id="root")
    assert retrace_steps(g) == [g]


def test_retrace_steps_counts_strictly_increasing_and_final_matches():
    flow = Flow(_OneChild(), max_depth=2)
    graph = run_to_completion(flow, "kick off")
    steps = retrace_steps(graph)

    counts = [_state_count(s) for s in steps]
    assert counts == sorted(counts)
    for prev, nxt in zip(counts, counts[1:]):
        assert nxt > prev
    assert counts[-1] == _state_count(graph)
    assert steps[-1].result() == graph.result()


def test_retrace_steps_first_snapshot_is_root_user_query():
    flow = Flow(_OneChild(), max_depth=2)
    graph = run_to_completion(flow, "kick off")
    first = retrace_steps(graph)[0]
    assert _state_count(first) == 1
    assert next(iter(first.all_nodes)).type == "user_query"


def test_retrace_steps_respects_spawn_dependency():
    flow = Flow(_OneChild(), max_depth=2)
    graph = run_to_completion(flow, "kick off")
    steps = retrace_steps(graph)
    child_id = next(iter(graph.children))
    for snap in steps:
        child = snap.children.get(child_id)
        if child and child.nodes:
            assert any(s.type == "supervising_output" for s in snap.nodes)
            return
    raise AssertionError("no snapshot contained a child state")


def test_retrace_steps_legacy_fallback_without_global_steps():
    graph = Graph(agent_id="root")
    graph.nodes = [
        UserQuery(agent_id="root", seq=0, content="kick"),
        LLMAction(agent_id="root", seq=1, model="x"),
        LLMOutput(agent_id="root", seq=2, code="done('ok')"),
        ExecAction(agent_id="root", seq=3, code="done('ok')"),
        DoneOutput(agent_id="root", seq=4, result="ok"),
    ]
    steps = retrace_steps(graph)
    assert [n.global_step for n in graph.nodes] == [None] * 5
    assert [_state_count(s) for s in steps] == [1, 3, 5]


def test_no_idle_ticks_when_agents_are_ready():
    flow = Flow(_OneChild(), max_depth=2)
    graph = run_to_completion(flow, "kick off")
    for i, tick in enumerate(_execution_ticks(graph)):
        assert len(tick) >= 1, f"tick {i} was empty"
