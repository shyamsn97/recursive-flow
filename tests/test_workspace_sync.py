from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rlmflow.workspace import ArtifactStore, Workspace


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


def test_workspace_materialize_and_commit_are_local_noops(tmp_path):
    workspace = Workspace.create(tmp_path / "workspace")

    assert workspace.materialize() == workspace.root
    workspace.commit()
    assert workspace.root.exists()
    assert isinstance(workspace.artifacts, ArtifactStore)


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


def test_workspace_fork_copies_artifacts_without_engine_state_collision(tmp_path):
    workspace = Workspace.create(tmp_path / "workspace")
    workspace.artifacts.write_text("openclaw/soul.md", "original")
    workspace.context.write("context", "ctx")

    forked = workspace.fork(tmp_path / "forked", new_branch_id="forked")
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
    assert (runtime.root / "workspace" / "src" / "app.py").read_text() == "print('hi')\n"
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
