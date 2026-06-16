"""Persisting a Graph: run directory and monolithic snapshots.

Two clean ways to put a run on disk now that the engine holds the whole run in
one ``Graph`` (no Workspace / session logs):

1. ``graph.save(dir)`` / ``Graph.load(dir)`` — manifest ``graph.json`` plus
   per-agent ``agent.json``, ``session.jsonl``, and ``latest.json`` (see
   ``docs/internal/run-layout.md``).
2. ``graph.save("graph.json")`` / ``Graph.load("graph.json")`` — one nested
   blob for portable fixtures (e.g. injection baselines).
3. ``rflow.save_trace(graphs, dir)`` / ``rflow.load_trace(dir)`` — a sequence of
   snapshots written as ``trace.json`` inside a run directory. This is what the
   viewer/CLI read when you point them at a folder.

Inputs are embedded in the graph (``inputs``), so there's nothing else to
persist — the JSON round-trips identically.

Run:
    python examples/graph/04_save_load.py
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
from pathlib import Path

import rflow


class DummyLLM(rflow.LLMClient):
    def chat(self, messages, *args, **kwargs):
        self.last_usage = rflow.LLMUsage(input_tokens=10, output_tokens=5)
        return '```repl\ndone("ok")\n```'


def banner(title: str) -> None:
    print("\n" + "─" * 60)
    print(title)
    print("─" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "_runs" / "save-load-demo"),
        help="where to write the demo artifacts (default: examples/_runs/save-load-demo/)",
    )
    args = parser.parse_args()

    tmp_path = Path(args.out_dir).resolve()
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)
    with contextlib.nullcontext():
        banner("1. Graph.save / Graph.load — run directory")
        g = rflow.Graph.from_meta_dict(
            {"agent_id": "root", "depth": 0, "query": "hi"},
            nodes=[
                rflow.UserQuery(agent_id="root", seq=0, content="hi"),
                rflow.DoneOutput(agent_id="root", seq=1, result="hello"),
            ],
        )
        run_dir = g.save(tmp_path / "run-layout")
        print(f"wrote run directory {run_dir.relative_to(tmp_path)}")
        for p in sorted(run_dir.rglob("*")):
            if p.is_file():
                print(f"  {p.relative_to(run_dir)}  ({p.stat().st_size} bytes)")
        loaded = rflow.Graph.load(run_dir)
        print(f"reloaded: agents={list(loaded.agents)} result={loaded.result()!r}")
        print(f"identical roundtrip: {g.to_dict() == loaded.to_dict()}")

        manifest = json.loads((run_dir / "graph.json").read_text())
        print("\nmanifest keys:", sorted(manifest.keys()))
        print(f"  agents: {manifest['agents']}")

        banner("2. Graph.save / Graph.load — monolithic JSON (fixtures)")
        snap_path = g.save(tmp_path / "graph.json")
        print(f"wrote {snap_path.relative_to(tmp_path)} ({snap_path.stat().st_size} bytes)")
        snap_loaded = rflow.Graph.load(snap_path)
        print(f"reloaded snapshot: result={snap_loaded.result()!r}")

        print("\nfirst-node JSON keys (snapshot):")
        first = json.loads(snap_path.read_text())["nodes"][0]
        for k, v in first.items():
            print(f"  {k:<10} {v!r}")

        banner("3. save_trace / load_trace — a run directory of snapshots")
        flow = rflow.Flow(DummyLLM(), max_iters=2)
        graph = flow.start("hello run")
        snapshots = [graph]
        while not graph.finished:
            graph = flow.step(graph)
            snapshots.append(graph)

        run_dir = tmp_path / "run"
        trace_path = rflow.save_trace(snapshots, run_dir)
        print(f"wrote {trace_path.relative_to(tmp_path)} ({len(snapshots)} snapshots)")
        for p in sorted(run_dir.rglob("*")):
            if p.is_file():
                print(f"  {p.relative_to(run_dir)}  ({p.stat().st_size} bytes)")

        trace = rflow.load_trace(run_dir)
        final = trace.graphs[-1]
        print(f"\nload_trace(): {len(trace.graphs)} snapshots, result={final.result()!r}")
        print(
            "nodes match in-memory: "
            f"{[n.type for n in final.all_nodes] == [n.type for n in flow.graph.all_nodes]}"
        )


if __name__ == "__main__":
    main()
