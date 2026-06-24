from __future__ import annotations

import json

from rflow import (
    DoneOutput,
    Graph,
    UserQuery,
)
from rflow.graph.run_layout import (
    AGENTS_DIRNAME,
    is_graph_snapshot,
    is_run_manifest,
    resolve_agent_dir,
)


def _delegated_graph() -> Graph:
    from rflow import Flow

    from tests.helpers import ScriptedLLM, run_to_completion

    def reply(messages):
        task = next((m["content"] for m in messages if m["role"] == "user"), "")
        if "depth 1" in task:
            return '```repl\ndone("c")\n```'
        return (
            "```repl\n"
            'results = await launch_subagents([{"name": "child", "query": "child task"}])\n'
            'done("p:" + results[0])\n'
            "```"
        )

    flow = Flow(ScriptedLLM(reply), max_depth=1, max_iters=5)
    return run_to_completion(flow, "parent")


def test_save_directory_writes_manifest_not_nested_tree(tmp_path):
    g = _delegated_graph()
    run_dir = g.save(tmp_path / "run")

    manifest = json.loads((run_dir / "graph.json").read_text())
    assert is_run_manifest(manifest)
    assert manifest["root_agent_id"] == "root"
    assert "root" in manifest["agents"]
    assert "root.child" in manifest["agents"]
    assert "nodes" not in manifest
    assert (run_dir / "graph.json").stat().st_size < 2000
    # nested layout: child sits under its parent's directory
    assert (run_dir / AGENTS_DIRNAME / "root" / "agent.json").is_file()
    assert (run_dir / AGENTS_DIRNAME / "root" / "child" / "agent.json").is_file()
    assert not (run_dir / AGENTS_DIRNAME / "root.child").exists()


def test_save_system_prompt_once_per_agent(tmp_path):
    g = _delegated_graph()
    g.system_prompt = "ROOT_PROMPT_UNIQUE"
    g["root.child"].system_prompt = "CHILD_PROMPT_UNIQUE"

    run_dir = g.save(tmp_path / "run")
    root_meta = resolve_agent_dir(run_dir, "root").joinpath("agent.json").read_text()
    child_meta = (
        resolve_agent_dir(run_dir, "root.child").joinpath("agent.json").read_text()
    )

    assert root_meta.count("ROOT_PROMPT_UNIQUE") == 1
    assert child_meta.count("CHILD_PROMPT_UNIQUE") == 1
    assert "ROOT_PROMPT_UNIQUE" not in child_meta


def test_save_session_jsonl_has_no_system_prompt(tmp_path):
    g = Graph(
        agent_id="root",
        system_prompt="SYS",
        nodes=[
            UserQuery(agent_id="root", seq=0, content="hi"),
            DoneOutput(agent_id="root", seq=1, result="ok"),
        ],
    )
    run_dir = g.save(tmp_path / "run")
    session = resolve_agent_dir(run_dir, "root").joinpath("session.jsonl").read_text()
    assert "SYS" not in session
    assert "user_query" in session


def test_load_round_trips_delegated_run_directory(tmp_path):
    g = _delegated_graph()
    run_dir = g.save(tmp_path / "run")
    restored = Graph.load(run_dir)

    assert restored.to_dict() == g.to_dict()
    assert restored.result() == g.result() == "p:c"
    assert list(restored.children) == list(g.children)


def test_save_json_file_writes_monolithic_snapshot(tmp_path):
    g = _delegated_graph()
    path = g.save(tmp_path / "snap.json")
    data = json.loads(path.read_text())
    assert is_graph_snapshot(data)
    assert Graph.load(path).to_dict() == g.to_dict()


def test_load_skips_agents_without_meta(tmp_path):
    g = Graph(
        agent_id="root",
        nodes=[UserQuery(agent_id="root", seq=0, content="hi")],
    )
    run_dir = g.save(tmp_path / "run")
    resolve_agent_dir(run_dir, "root").joinpath("agent.json").unlink()

    restored = Graph.load(run_dir)
    assert restored.agent_id == "root"
    assert restored.nodes == []


def test_spawn_edges_survive_roundtrip(tmp_path):
    g = _delegated_graph()
    run_dir = g.save(tmp_path / "run")
    restored = Graph.load(run_dir)
    spawn = restored.edges.spawns()
    assert len(spawn) == 1
    assert restored["root.child"].parent_agent_id == "root"


def test_trace_load_run_directory(tmp_path):
    g = _delegated_graph()
    g.save(tmp_path / "run")

    from rflow.utils.trace import Trace

    trace = Trace.load(tmp_path / "run")
    assert trace.latest.result() == g.result()
    assert len(trace.graphs) > 1


def test_latest_json_written(tmp_path):
    g = Graph(
        agent_id="root",
        nodes=[
            UserQuery(agent_id="root", seq=0, content="hi"),
            DoneOutput(agent_id="root", seq=1, result="done"),
        ],
    )
    run_dir = g.save(tmp_path / "run")
    latest = json.loads(
        resolve_agent_dir(run_dir, "root").joinpath("latest.json").read_text()
    )
    assert latest["type"] == "done_output"
    assert latest["terminal"] is True
    assert latest["result"] == "done"


def test_backward_compat_monolithic_snapshot_in_dir(tmp_path):
    g = _delegated_graph()
    g.save(tmp_path / "legacy" / "graph.json")
    restored = Graph.load(tmp_path / "legacy")
    assert restored.to_dict() == g.to_dict()


def test_nested_child_dir_uses_local_name(tmp_path):
    g = _delegated_graph()
    run_dir = g.save(tmp_path / "run")
    child_dir = resolve_agent_dir(run_dir, "root.child")
    # the child folder is named by its local segment and lives under root/
    assert child_dir.name == "child"
    assert child_dir.parent.name == "root"


def test_load_legacy_flat_agent_dirs(tmp_path):
    """A pre-nesting flat layout (agents/<full-id>/) still loads."""
    g = _delegated_graph()
    run_dir = tmp_path / "legacy-flat"
    agents = run_dir / AGENTS_DIRNAME
    agents.mkdir(parents=True)
    (run_dir / "graph.json").write_text(
        json.dumps({"root_agent_id": "root", "agents": ["root", "root.child"]})
    )
    for agent in g.walk():
        d = agents / agent.agent_id  # flat, full-id directory name
        d.mkdir()
        from rflow.graph.run_layout import agent_meta_dict

        (d / "agent.json").write_text(json.dumps(agent_meta_dict(agent)))
        (d / "session.jsonl").write_text(
            "\n".join(json.dumps(n.to_dict()) for n in agent.nodes) + "\n"
        )

    restored = Graph.load(run_dir)
    assert restored.to_dict() == g.to_dict()


def test_resave_prunes_removed_agents(tmp_path):
    g = _delegated_graph()
    run_dir = g.save(tmp_path / "run")
    # a stale child dir left over from an earlier save
    stale = resolve_agent_dir(run_dir, "root").joinpath("stale")
    stale.mkdir()
    (stale / "agent.json").write_text('{"agent_id":"root.stale"}')
    # a stale top-level flat agent dir from a previous layout
    flat = run_dir / AGENTS_DIRNAME / "root.child"
    flat.mkdir(exist_ok=True)
    (flat / "agent.json").write_text('{"agent_id":"root.child"}')

    g.save(run_dir)
    assert not stale.exists()
    assert not flat.exists()
    assert resolve_agent_dir(run_dir, "root.child").name == "child"
