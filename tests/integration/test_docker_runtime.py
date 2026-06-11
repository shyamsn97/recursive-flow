"""Integration test: DockerRuntime end-to-end against the shipped image.

Covers the three smoke paths that regressed during the 0.1.0 cycle:

1. ``inject`` on a non-literal object exposes its methods over the wire.
2. ``from X import *`` works inside a code block that also awaits.
3. Proxied tool writes resolve to the host workspace directory.

Gated on ``RECURSIVE_FLOW_DOCKER_TEST=1`` plus a running docker daemon.  Build
the image once before running:

    docker build -t recursive-flow:local .
    RECURSIVE_FLOW_DOCKER_TEST=1 pytest tests/integration/test_docker_runtime.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from rflow.graph import ChildHandle, WaitRequest
from rflow.runtime.docker import DockerRuntime
from rflow.workspace import FileContext


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5, check=False
        )
        return r.returncode == 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    os.environ.get("RECURSIVE_FLOW_DOCKER_TEST") != "1" or not _docker_available(),
    reason="set RECURSIVE_FLOW_DOCKER_TEST=1 with a working docker daemon",
)


IMAGE = os.environ.get("RECURSIVE_FLOW_DOCKER_TEST_IMAGE", "recursive-flow:local")


@pytest.fixture
def runtime(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    rt = DockerRuntime(
        IMAGE,
        mounts={str(ws): "/workspace"},
        workdir="/workspace",
        workspace=ws,
        network="none",
    )
    try:
        yield rt, ws
    finally:
        rt.close()


def test_object_proxy_round_trips_file_context_methods(runtime, tmp_path: Path):
    """Injected objects expose public methods as callable proxies."""
    rt, _ = runtime
    context = FileContext(tmp_path / "workspace")
    rt.inject("STORE", context)

    rt.execute("STORE.write('context', 'hello from container')")
    assert context.read("context") == "hello from container"

    out = rt.execute("print(STORE.read('context'))")
    assert out == "hello from container"


def test_star_import_with_await(runtime):
    rt, _ = runtime

    def flow_delegate(
        *,
        name: str,
        query: str,
        context: str,
        model: str = "default",
        output_schema=None,
    ) -> ChildHandle:
        return ChildHandle(agent_id="c")

    def flow_wait(*handles):
        return WaitRequest(agent_ids=[h.agent_id for h in handles])

    rt.inject("flow_delegate", flow_delegate)
    rt.inject("flow_wait", flow_wait)
    rt.inject_launcher("launch_subagents")

    suspended, _, _ = rt.start_code(
        "from math import *\n"
        "await launch_subagents([{'name': 'c', 'query': 'q'}])\n"
        "print(int(pi * 100))\n"
    )
    assert suspended is True
    suspended, out, _ = rt.resume_code(["done"])
    assert suspended is False
    assert "314" in out


def test_proxied_writes_land_in_host_workspace(runtime):
    """A host-side tool writing a relative path must resolve inside the workspace."""
    rt, ws = runtime

    def write_rel(path: str, content: str) -> str:
        Path(path).write_text(content)
        return "ok"

    rt.inject("write_rel", write_rel)
    rt.execute("write_rel('hello.txt', 'hi')")

    assert (ws / "hello.txt").read_text() == "hi"


def test_end_to_end_delegate_wait():
    """Spawn a fresh container; exercise execute, inject, and await suspension."""
    rt = DockerRuntime(IMAGE, network="none")
    try:
        assert rt.execute("print('hi from container')") == "hi from container"

        def flow_delegate(
            *,
            name: str,
            query: str,
            context: str,
            model: str = "default",
            output_schema=None,
        ) -> ChildHandle:
            return ChildHandle(agent_id=f"child-{name}")

        def flow_wait(*handles):
            return WaitRequest(agent_ids=[h.agent_id for h in handles])

        rt.inject("flow_delegate", flow_delegate)
        rt.inject("flow_wait", flow_wait)
        rt.inject_launcher("launch_subagents")

        suspended, payload, _ = rt.start_code(
            "results = await launch_subagents([{'name': 'q1', 'query': 'q1'}])\n"
            "print('after:', results)\n"
        )
        assert suspended is True
        request, _ = payload
        assert request.agent_ids == ["child-q1"]

        suspended, out, _ = rt.resume_code(["answer"])
        assert suspended is False
        assert "after: ['answer']" in out
    finally:
        rt.close()
