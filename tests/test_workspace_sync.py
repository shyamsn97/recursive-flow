from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rflow.graph import Graph
from rflow.llm import LLMClient
from rflow.flow import RecursiveFlow
from rflow.workspace import ArtifactStore, InMemoryWorkspace, Workspace


class FakeFileRuntime:
    def __init__(self, root: Path):
        self.root = root

    def upload_file(self, local_path: str | Path, remote_path: str) -> None:
        dst = self.root / remote_path.strip("/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dst)

    def download_file(self, remote_path: str, local_path: str | Path) -> None:
        src = self.root / remote_path.strip("/")
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    def remove_path(self, remote_path: str, *, recursive: bool = False) -> None:
        path = self.root / remote_path.strip("/")
        if not path.exists():
            return
        if path.is_dir():
            shutil.rmtree(path) if recursive else path.rmdir()
        else:
            path.unlink()

    def list_files(self, remote_root: str) -> list[str]:
        root = self.root / remote_root.strip("/")
        if not root.exists():
            return []
        return sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        )


class DummyLLM(LLMClient):
    def chat(self, messages: list[dict[str, str]], *args, **kwargs) -> str:
        return '```repl\ndone("ok")\n```'


def test_workspace_materialize_and_commit_are_local_noops(tmp_path):
    workspace = Workspace.create(tmp_path / "workspace")

    assert workspace.materialize() == workspace.root
    workspace.commit()
    assert workspace.root.exists()
    assert isinstance(workspace.artifacts, ArtifactStore)


def test_workspace_from_graph_creates_synced_workspace(tmp_path):
    graph = Graph(agent_id="root")
    graph.set_context("portable context", metadata={"source": "test"})

    workspace = Workspace.from_graph(
        graph,
        tmp_path / "variant",
    )
    synced = workspace.load_graph()

    assert workspace.root == (tmp_path / "variant").resolve()
    assert synced.agent_id == "root"
    assert list(synced.agents) == ["root"]
    assert synced.context.text == "portable context"
    assert synced.context.metadata == {"source": "test"}
    assert workspace.context.read("context", agent_id="root") == "portable context"
    assert workspace.context.info("context", agent_id="root")["chars"] == len(
        "portable context"
    )


def test_graph_context_round_trips_through_json(tmp_path):
    graph = Graph(agent_id="root").set_context(
        "portable context",
        metadata={"source": "test"},
    )

    loaded = Graph.load(graph.save(tmp_path / "graph.json"))

    assert loaded.context.text == "portable context"
    assert loaded.context.metadata == {"source": "test"}


def test_workspace_from_graph_replaces_existing_engine_state(tmp_path):
    existing = Workspace.create(tmp_path / "variant")
    stale = Graph(agent_id="root.stale")
    existing.sync_graph(stale)
    existing.artifacts.write_text("notes.txt", "stale")

    workspace = Workspace.from_graph(
        Graph(agent_id="root"),
        tmp_path / "variant",
    )
    synced = workspace.load_graph()

    assert list(synced.agents) == ["root"]
    assert list(workspace.load_graph().agents) == ["root"]
    assert not workspace.artifacts.exists("notes.txt")


def test_fresh_workspace_gets_graph_context_when_syncing_edited_graph(tmp_path):
    source = Workspace.create(tmp_path / "source")
    agent = RecursiveFlow(DummyLLM()).attach_workspace(source)
    graph = agent.start("read the context", context="root payload")
    edited = graph.inject_output(target="root", output="controller note")

    variant = Workspace.create(tmp_path / "variant")
    synced = variant.sync_graph_if_changed(edited)

    assert [node.type for node in synced.nodes] == [node.type for node in edited.nodes]
    assert synced.context.text == "root payload"
    assert variant.context.read("context", agent_id="root") == "root payload"
    assert variant.context.info("context", agent_id="root")["chars"] == len(
        "root payload"
    )


def test_sync_graph_if_changed_creates_missing_empty_context(tmp_path):
    workspace = Workspace.create(tmp_path / "variant")
    graph = Graph(agent_id="root")

    workspace.sync_graph_if_changed(graph)

    assert workspace.context.read("context", agent_id="root") == ""
    assert workspace.context.info("context", agent_id="root")["chars"] == 0


def test_sync_graph_if_changed_repairs_missing_context_for_synced_graph(tmp_path):
    workspace = Workspace.from_graph(Graph(agent_id="root"), tmp_path / "variant")
    shutil.rmtree(workspace.root / "context")

    workspace.sync_graph_if_changed(workspace.load_graph())

    assert workspace.context.read("context", agent_id="root") == ""


def test_workspace_from_graph_requires_path_for_filesystem_workspace():
    with pytest.raises(TypeError, match="requires a workspace path"):
        Workspace.from_graph(Graph(agent_id="root"))


def test_in_memory_workspace_from_graph_needs_no_path():
    workspace = InMemoryWorkspace.from_graph(Graph(agent_id="root"))
    synced = workspace.load_graph()

    assert workspace.load_graph().agent_id == "root"
    assert synced.agent_id == "root"
    assert workspace.context.read("context", agent_id="root") == ""


def test_in_memory_workspace_fork_artifacts_are_opt_in():
    workspace = InMemoryWorkspace.create()
    workspace.artifacts.write_text("skills/review/SKILL.md", "review carefully")
    workspace.context.write("context", "ctx")

    core_only = workspace.fork()
    with_artifacts = workspace.fork(include_artifacts=True)

    assert not core_only.artifacts.exists("skills/review/SKILL.md")
    assert core_only.context.read("context") == "ctx"
    assert with_artifacts.artifacts.read_text("skills/review/SKILL.md") == (
        "review carefully"
    )


def test_recursive_flow_attach_workspace_binds_in_place(tmp_path):
    workspace = Workspace.create(tmp_path / "workspace")
    agent = RecursiveFlow(DummyLLM())

    result = agent.attach_workspace(workspace)

    assert result is agent
    assert agent.workspace is workspace
    assert agent.session is workspace.session
    assert agent.context is workspace.context
    assert agent.runtime is not None


def test_recursive_flow_defaults_to_local_runtime_without_workspace():
    agent = RecursiveFlow(DummyLLM())

    assert agent.runtime is not None
    assert agent.workspace is None


def test_recursive_flow_clone_rejects_unknown_overrides(tmp_path):
    workspace = Workspace.create(tmp_path / "workspace")
    agent = RecursiveFlow(DummyLLM()).attach_workspace(workspace)

    with pytest.raises(TypeError, match="unknown RecursiveFlow.clone"):
        agent.clone(workspaec=workspace)


def test_workspace_artifacts_use_user_chosen_flat_paths(tmp_path):
    workspace = Workspace.create(tmp_path / "workspace")

    workspace.artifacts.write_text("openclaw/soul.md", "be kind\n")
    workspace.artifacts.append_text("openclaw/soul.md", "be sharp\n")
    workspace.artifacts.write_json("reports/summary.json", {"ok": True})

    assert workspace.artifacts.exists("openclaw/soul.md")
    assert workspace.artifacts.read_text("openclaw/soul.md") == "be kind\nbe sharp\n"
    assert workspace.artifacts.read_json("reports/summary.json") == {"ok": True}
    assert workspace.artifacts.list("openclaw") == ["openclaw/soul.md"]
    assert workspace.artifacts.list("reports") == ["reports/summary.json"]


def test_workspace_artifacts_hide_and_protect_engine_state(tmp_path):
    workspace = Workspace.create(tmp_path / "workspace")
    workspace.artifacts.write_text("skills/review/SKILL.md", "review carefully")
    (workspace.root / "graph.json").write_text("{}")
    (workspace.root / "session" / "root").mkdir(parents=True)
    (workspace.root / "session" / "root" / "session.jsonl").write_text("{}\n")
    (workspace.root / "context" / "root").mkdir(parents=True)
    (workspace.root / "context" / "root" / "context.txt").write_text("ctx")

    assert workspace.artifacts.list() == ["skills/review/SKILL.md"]
    for path in [
        "/tmp/out.md",
        "../out.md",
        "safe/../out.md",
        "graph.json",
        "session/root/session.jsonl",
        "context/root/context.txt",
        "trace/events.jsonl",
        "checkpoint.json",
    ]:
        with pytest.raises(ValueError):
            workspace.artifacts.write_text(path, "nope")


def test_workspace_fork_defaults_to_core_state_only(tmp_path):
    workspace = Workspace.create(tmp_path / "workspace")
    workspace.artifacts.write_text("openclaw/soul.md", "original")
    workspace.context.write("context", "ctx")

    forked = workspace.fork(tmp_path / "forked")

    assert not forked.artifacts.exists("openclaw/soul.md")
    assert forked.context.read("context") == "ctx"


def test_workspace_fork_can_include_artifacts(tmp_path):
    workspace = Workspace.create(tmp_path / "workspace")
    workspace.artifacts.write_text("openclaw/soul.md", "original")
    workspace.context.write("context", "ctx")

    forked = workspace.fork(tmp_path / "forked", include_artifacts=True)
    forked.artifacts.write_text("openclaw/soul.md", "fork")

    assert workspace.artifacts.read_text("openclaw/soul.md") == "original"
    assert forked.artifacts.read_text("openclaw/soul.md") == "fork"
    assert forked.context.read("context") == "ctx"


def test_workspace_push_and_pull_use_runtime_file_primitives(tmp_path):
    workspace = Workspace.create(tmp_path / "workspace")
    runtime = FakeFileRuntime(tmp_path / "runtime")

    (workspace.root / "graph.json").write_text('{"nodes": []}')
    (workspace.root / "session").mkdir()
    (workspace.root / "session" / "root.jsonl").write_text("{}\n")
    (workspace.root / "context").mkdir()
    (workspace.root / "context" / "payload.txt").write_text("ctx")
    (workspace.root / "src").mkdir()
    (workspace.root / "src" / "app.py").write_text("print('hi')\n")
    (workspace.root / ".ruff_cache").mkdir()
    (workspace.root / ".ruff_cache" / "ignored").write_text("cache")

    workspace.push_to(runtime, "/workspace")

    assert (runtime.root / "workspace" / "graph.json").read_text() == '{"nodes": []}'
    assert (runtime.root / "workspace" / "session" / "root.jsonl").read_text() == "{}\n"
    assert (runtime.root / "workspace" / "context" / "payload.txt").read_text() == "ctx"
    assert (
        runtime.root / "workspace" / "src" / "app.py"
    ).read_text() == "print('hi')\n"
    assert not (runtime.root / "workspace" / ".ruff_cache").exists()

    (runtime.root / "workspace" / "remote.txt").write_text("from runtime")
    (workspace.root / "stale.txt").write_text("old")

    workspace.pull_from(runtime, "/workspace")

    assert not (workspace.root / "stale.txt").exists()
    assert (workspace.root / "remote.txt").read_text() == "from runtime"


def test_workspace_pull_can_skip_coordinator_engine_state(tmp_path):
    workspace = Workspace.create(tmp_path / "workspace")
    runtime = FakeFileRuntime(tmp_path / "runtime")
    remote_root = runtime.root / "workspace"

    (workspace.root / "graph.json").write_text("host graph")
    (workspace.root / "session" / "root").mkdir(parents=True)
    (workspace.root / "session" / "root" / "session.jsonl").write_text("host session")
    (workspace.root / "context" / "root").mkdir(parents=True)
    (workspace.root / "context" / "root" / "context.txt").write_text("host context")

    (remote_root / "session" / "root").mkdir(parents=True)
    (remote_root / "session" / "root" / "session.jsonl").write_text("remote session")
    (remote_root / "context" / "root").mkdir(parents=True)
    (remote_root / "context" / "root" / "context.txt").write_text("remote context")
    (remote_root / "graph.json").write_text("remote graph")
    (remote_root / "output").mkdir()
    (remote_root / "output" / "app.txt").write_text("artifact")

    workspace.pull_from(
        runtime,
        "/workspace",
        merge=True,
        skip_engine_state=True,
    )

    assert (workspace.root / "graph.json").read_text() == "host graph"
    assert (
        workspace.root / "session" / "root" / "session.jsonl"
    ).read_text() == "host session"
    assert (
        workspace.root / "context" / "root" / "context.txt"
    ).read_text() == "host context"
    assert (workspace.root / "output" / "app.txt").read_text() == "artifact"
