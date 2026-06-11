from __future__ import annotations

from rflow import (
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    Graph,
    LLMAction,
    LLMOutput,
    FlowConfig,
    RecursiveFlow,
    ResumeAction,
    SupervisingOutput,
    UserQuery,
    Workspace,
)
from rflow import NodeScheduler
from rflow.engine.replay import can_resume
from tests.helpers import StaticLLM


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}

OTHER_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer"}},
    "required": ["score"],
}


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


def test_replace_last_action_defaults_to_descendant_truncation():
    graph = _graph_with_child()
    old_action = graph.last_action("root")

    edited = graph.replace_last_action("root", ExecAction(code="fixed()"))

    new_action = edited.last_action("root")
    assert new_action is not None
    assert old_action is not None
    assert new_action.id != old_action.id
    assert new_action.agent_id == "root"
    assert new_action.seq == old_action.seq
    assert new_action.code == "fixed()"
    assert [node.type for node in edited.nodes] == [
        "user_query",
        "llm_action",
        "llm_output",
        "exec_action",
    ]
    assert "root.worker" not in edited.agents
    assert graph.current().type == "error_output"


def test_replace_node_defaults_to_pruning_descendants_spawned_by_removed_action():
    graph = _graph_with_child()
    old_action = graph.last_action("root")
    assert old_action is not None

    edited = graph.replace_node(
        old_action.id,
        ExecAction(code="fixed()"),
    )

    assert "root.worker" not in edited.agents


def test_replace_node_none_keeps_local_future_and_children():
    graph = _graph_with_child()
    old_action = graph.last_action("root")
    assert old_action is not None

    edited = graph.replace_node(
        old_action.id,
        ExecAction(code="metadata-only()"),
        truncate="none",
    )

    assert [node.type for node in edited.nodes] == [
        "user_query",
        "llm_action",
        "llm_output",
        "exec_action",
        "error_output",
    ]
    assert edited.nodes[3].code == "metadata-only()"
    assert "root.worker" in edited.agents


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
        children={
            rows.agent_id: rows,
            cols.agent_id: cols,
            unrelated.agent_id: unrelated,
        },
    )


def test_replace_supervising_node_prunes_waited_children_by_default():
    graph = _graph_waiting_on_children()
    supervising = graph.last_observation("root")
    assert isinstance(supervising, SupervisingOutput)

    edited = graph.replace_node(
        supervising.id,
        ExecOutput(output="try another route"),
    )

    assert [node.type for node in edited.nodes] == [
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


def test_replace_supervising_node_after_truncate_keeps_waited_children():
    graph = _graph_waiting_on_children()
    supervising = graph.last_observation("root")
    assert isinstance(supervising, SupervisingOutput)

    edited = graph.replace_node(
        supervising.id,
        ExecOutput(output="try another route"),
        truncate="after",
    )

    assert "root.rows" in edited.agents
    assert "root.cols" in edited.agents


def test_truncate_after_supervising_keeps_waited_children_by_spawn_policy():
    graph = _graph_waiting_on_children()
    supervising = graph.last_observation("root")
    assert isinstance(supervising, SupervisingOutput)

    edited = graph.truncate_after(supervising.id, descendants=True)

    assert "root.rows" in edited.agents
    assert "root.cols" in edited.agents
    assert "root.unrelated" in edited.agents


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

    assert "root.early" in edited.agents
    assert "root.late" not in edited.agents


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

    assert "root.early" in edited.agents
    assert "root.late" not in edited.agents


def test_step_commits_replaced_supervising_node_and_pruned_children(tmp_path):
    workspace = Workspace.create(tmp_path)
    agent = RecursiveFlow(
        StaticLLM('```repl\ndone("ok")\n```'),
        workspace=workspace,
        config=FlowConfig(max_iterations=3),
    )
    graph = _graph_waiting_on_children()
    workspace.session.rewrite_graph(graph)
    workspace.context.write("context", "old rows context", agent_id="root.rows")
    workspace.context.write("context", "old cols context", agent_id="root.cols")
    supervising = graph.last_observation("root")
    assert isinstance(supervising, SupervisingOutput)

    edited = graph.replace_node(
        supervising.id,
        ExecOutput(
            output="try another route",
            content="REPL output for previous block:\ntry another route",
        ),
        truncate="descendants",
    )
    stepped = agent.step(edited)
    persisted = workspace.session.load_graph()

    assert "root.rows" not in persisted.agents
    assert "root.cols" not in persisted.agents
    assert not (workspace.root / "session" / "root.rows").exists()
    assert not (workspace.root / "session" / "root.cols").exists()
    assert not (workspace.root / "context" / "root.rows").exists()
    assert not (workspace.root / "context" / "root.cols").exists()
    assert persisted.current().type == "llm_output"
    assert stepped.current().type == "llm_output"


def test_parent_finished_requires_all_descendants_finished():
    child = Graph(
        agent_id="root.child",
        parent_agent_id="root",
        parent_node_id="spawn",
        nodes=[UserQuery(agent_id="root.child", seq=0, content="unfinished")],
    )
    graph = Graph(
        agent_id="root",
        nodes=[DoneOutput(agent_id="root", seq=0, result="root done")],
        children={child.agent_id: child},
    )

    assert not graph.finished

    child.nodes.append(DoneOutput(agent_id="root.child", seq=1, result="child done"))

    assert graph.finished


def test_step_can_advance_unfinished_descendant_under_terminal_parent(tmp_path):
    child = Graph(
        agent_id="root.child",
        parent_agent_id="root",
        parent_node_id="spawn",
        nodes=[UserQuery(agent_id="root.child", seq=0, content="finish child")],
    )
    graph = Graph(
        agent_id="root",
        nodes=[DoneOutput(agent_id="root", seq=0, result="root done")],
        children={child.agent_id: child},
    )
    workspace = Workspace.create(tmp_path)
    workspace.session.rewrite_graph(graph)
    agent = RecursiveFlow(
        StaticLLM('```repl\ndone("child done")\n```'),
        workspace=workspace,
        config=FlowConfig(max_iterations=3),
    )

    stepped = agent.step(graph)

    assert stepped.agents["root.child"].current().type == "llm_output"
    assert not stepped.finished


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

    edited = graph.replace_last_observation(
        "root.child",
        ExecOutput(output="try again"),
    )

    assert [node.type for node in edited.nodes] == [
        "user_query",
        "llm_action",
        "llm_output",
        "exec_action",
        "supervising_output",
    ]
    assert edited.current().type == "supervising_output"
    assert edited.agents["root.child"].current().type == "exec_output"
    assert not edited.finished


def test_deep_child_edit_truncates_all_stale_ancestor_resume_states():
    root_spawn = ExecAction(agent_id="root", seq=3, code="spawn parent")
    parent_spawn = ExecAction(agent_id="root.parent", seq=3, code="spawn child")
    child = Graph(
        agent_id="root.parent.child",
        depth=2,
        parent_agent_id="root.parent",
        parent_node_id=parent_spawn.id,
        nodes=[
            UserQuery(agent_id="root.parent.child", seq=0, content="child"),
            DoneOutput(agent_id="root.parent.child", seq=1, result="old child"),
        ],
    )
    parent = Graph(
        agent_id="root.parent",
        depth=1,
        parent_agent_id="root",
        parent_node_id=root_spawn.id,
        nodes=[
            UserQuery(agent_id="root.parent", seq=0, content="parent"),
            LLMAction(agent_id="root.parent", seq=1),
            LLMOutput(agent_id="root.parent", seq=2, reply="spawn", code="spawn child"),
            parent_spawn,
            SupervisingOutput(
                agent_id="root.parent",
                seq=4,
                waiting_on=["root.parent.child"],
            ),
            ResumeAction(
                agent_id="root.parent",
                seq=5,
                resumed_from=["root.parent.child"],
            ),
            DoneOutput(agent_id="root.parent", seq=6, result="old parent"),
        ],
        children={child.agent_id: child},
    )
    graph = Graph(
        agent_id="root",
        nodes=[
            UserQuery(agent_id="root", seq=0, content="q"),
            LLMAction(agent_id="root", seq=1),
            LLMOutput(agent_id="root", seq=2, reply="spawn", code="spawn parent"),
            root_spawn,
            SupervisingOutput(agent_id="root", seq=4, waiting_on=["root.parent"]),
            ResumeAction(agent_id="root", seq=5, resumed_from=["root.parent"]),
            DoneOutput(agent_id="root", seq=6, result="old root"),
        ],
        children={parent.agent_id: parent},
    )

    edited = graph.replace_last_observation(
        "root.parent.child",
        ExecOutput(output="try again"),
    )

    assert edited.current().type == "supervising_output"
    assert edited.agents["root.parent"].current().type == "supervising_output"
    assert edited.agents["root.parent.child"].current().type == "exec_output"
    assert [node.type for node in edited.nodes][-2:] == [
        "exec_action",
        "supervising_output",
    ]
    assert [node.type for node in edited.agents["root.parent"].nodes][-2:] == [
        "exec_action",
        "supervising_output",
    ]
    assert not edited.finished


def test_child_edit_without_matching_supervisor_leaves_ancestors_untouched():
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
            DoneOutput(agent_id="root", seq=2, result="old root"),
        ],
        children={child.agent_id: child},
    )

    edited = graph.replace_last_observation(
        "root.child",
        ExecOutput(output="try again"),
    )

    assert edited.current().type == "done_output"
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

    edited = graph.replace_last_observation(
        "root.child",
        ExecOutput(output="try again"),
    )

    assert edited.current().type == "supervising_output"
    assert edited.current().seq == 5
    assert [node.seq for node in edited.nodes] == [0, 1, 2, 3, 4, 5]


def test_scheduler_does_not_resume_supervisor_with_missing_waited_child():
    graph = Graph(
        agent_id="root",
        nodes=[
            UserQuery(agent_id="root", seq=0, content="q"),
            SupervisingOutput(
                agent_id="root",
                seq=1,
                waiting_on=["root.missing"],
            ),
        ],
    )

    assert NodeScheduler().runnable_agents(graph) == []


def test_scheduler_uses_recursive_child_finished_for_supervisor_readiness():
    grandchild = Graph(
        agent_id="root.child.grand",
        parent_agent_id="root.child",
        parent_node_id="grand-spawn",
        nodes=[UserQuery(agent_id="root.child.grand", seq=0, content="grand")],
    )
    child = Graph(
        agent_id="root.child",
        parent_agent_id="root",
        parent_node_id="child-spawn",
        nodes=[DoneOutput(agent_id="root.child", seq=0, result="child done")],
        children={grandchild.agent_id: grandchild},
    )
    graph = Graph(
        agent_id="root",
        nodes=[
            UserQuery(agent_id="root", seq=0, content="q"),
            SupervisingOutput(agent_id="root", seq=1, waiting_on=["root.child"]),
        ],
        children={child.agent_id: child},
    )

    assert NodeScheduler().runnable_agents(graph) == ["root.child.grand"]


def test_can_resume_uses_recursive_child_finished():
    grandchild = Graph(
        agent_id="root.child.grand",
        parent_agent_id="root.child",
        parent_node_id="grand-spawn",
        nodes=[UserQuery(agent_id="root.child.grand", seq=0, content="grand")],
    )
    child = Graph(
        agent_id="root.child",
        parent_agent_id="root",
        parent_node_id="child-spawn",
        nodes=[DoneOutput(agent_id="root.child", seq=0, result="child done")],
        children={grandchild.agent_id: grandchild},
    )
    supervisor = SupervisingOutput(
        agent_id="root",
        seq=1,
        waiting_on=["root.child"],
    )
    graph = Graph(
        agent_id="root",
        nodes=[UserQuery(agent_id="root", seq=0, content="q"), supervisor],
        children={child.agent_id: child},
    )

    assert not can_resume(graph, supervisor)

    grandchild.nodes.append(
        DoneOutput(agent_id="root.child.grand", seq=1, result="grand done")
    )

    assert can_resume(graph, supervisor)


def test_step_commits_structural_graph_edits_before_planning(tmp_path):
    workspace = Workspace.create(tmp_path)
    agent = RecursiveFlow(
        StaticLLM('```repl\ndone("ok")\n```'),
        workspace=workspace,
        config=FlowConfig(max_iterations=3),
    )
    graph = agent.start("old query")

    edited = graph.replace_last_observation(
        "root",
        UserQuery(content="edited query"),
    )
    stepped = agent.step(edited)

    persisted = workspace.session.load_graph()
    assert persisted.nodes[0].content == "edited query"
    assert stepped.nodes[0].content == "edited query"
    assert [node.type for node in stepped.nodes] == [
        "user_query",
        "llm_action",
        "llm_output",
    ]


def test_replace_node_inherits_active_output_schema_by_default():
    graph = Graph(
        agent_id="root",
        nodes=[
            UserQuery(agent_id="root", seq=0, content="q", output_schema=OUTPUT_SCHEMA),
            ExecOutput(agent_id="root", seq=1, output="old", output_schema=OUTPUT_SCHEMA),
        ],
    )

    edited = graph.replace_last_observation("root", ExecOutput(output="new"))

    assert edited.current().output_schema == OUTPUT_SCHEMA


def test_replace_node_can_clear_active_output_schema():
    graph = Graph(
        agent_id="root",
        nodes=[
            UserQuery(agent_id="root", seq=0, content="q", output_schema=OUTPUT_SCHEMA),
            ExecOutput(agent_id="root", seq=1, output="old", output_schema=OUTPUT_SCHEMA),
        ],
    )

    edited = graph.replace_last_observation(
        "root",
        ExecOutput(output="new"),
        inherit_output_schema=False,
    )

    assert edited.current().output_schema is None


def test_replace_node_can_override_active_output_schema():
    graph = Graph(
        agent_id="root",
        nodes=[
            UserQuery(agent_id="root", seq=0, content="q", output_schema=OUTPUT_SCHEMA),
            ExecOutput(agent_id="root", seq=1, output="old", output_schema=OUTPUT_SCHEMA),
        ],
    )

    edited = graph.replace_last_observation(
        "root",
        ExecOutput(output="new"),
        output_schema=OTHER_OUTPUT_SCHEMA,
    )

    assert edited.current().output_schema == OTHER_OUTPUT_SCHEMA


def test_inject_inherits_active_output_schema_by_default():
    graph = Graph(
        agent_id="root",
        nodes=[UserQuery(agent_id="root", seq=0, content="q", output_schema=OUTPUT_SCHEMA)],
    )

    edited = graph.inject(target="root", node=ExecOutput(output="injected"))

    assert edited.current().output_schema == OUTPUT_SCHEMA


def test_inject_can_clear_active_output_schema():
    graph = Graph(
        agent_id="root",
        nodes=[UserQuery(agent_id="root", seq=0, content="q", output_schema=OUTPUT_SCHEMA)],
    )

    edited = graph.inject(
        target="root",
        node=ExecOutput(output="injected"),
        inherit_output_schema=False,
    )

    assert edited.current().output_schema is None


def test_commit_graph_can_fork_workspace(tmp_path):
    workspace = Workspace.create(tmp_path / "base")
    agent = RecursiveFlow(
        StaticLLM('```repl\ndone("ok")\n```'),
        workspace=workspace,
        config=FlowConfig(max_iterations=3),
    )
    graph = agent.start("old query")
    edited = graph.replace_last_observation(
        "root",
        UserQuery(content="forked query"),
    )

    committed = agent.commit_graph(
        edited,
        fork=True,
        new_location=tmp_path / "fork",
    )

    assert committed.nodes[0].content == "forked query"
    base_query = workspace.session.load_graph().nodes[0].content
    assert "old query" in base_query
    assert "forked query" not in base_query
