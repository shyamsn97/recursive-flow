"""Reconstruct timeline snapshots with ``retrace_steps(graph)``.

This is visualization/time-travel retrace, not cold-start runtime replay.
It does not run Python code, call the LLM, or resume suspended coroutines.
It walks the already-recorded final graph and reconstructs stable snapshots
for viewers/exporters.

This script:
  1. runs the real engine end-to-end with a tiny scripted LLM,
  2. prints the final persisted node log,
  3. calls ``retrace_steps(graph)`` and shows the inferred snapshots.

For supervisor injection and graph surgery on a saved run, see
``examples/control/injection/inject_variants.py``.

Run:
    python examples/graph/05_timeline.py
"""

from __future__ import annotations

from pathlib import Path

import rflow


def _example_run_dir(source_file: str | Path, name: str) -> Path:
    source = Path(source_file).resolve()
    for parent in (source.parent, *source.parents):
        if parent.name == "examples":
            return parent / "_runs" / name
    return source.parent / "_runs" / name


def _save_example_graph(
    graph,
    source_file: str | Path,
    name: str,
    *,
    out_dir: str | Path | None = None,
    label: str = "Graph saved to",
) -> Path:
    path = graph.save(
        Path(out_dir) if out_dir is not None else _example_run_dir(source_file, name)
    )
    print(f"{label} {path}")
    return path


ROOT_SPLIT = (
    "```repl\n"
    "results = await launch_subagents([\n"
    '    {"name": "a", "query": "do A"},\n'
    '    {"name": "b", "query": "do B"},\n'
    "])\n"
    'done("/".join(results))\n'
    "```"
)


class ScriptedLLM(rflow.LLMClient):
    """Tiny deterministic LLM: root delegates, each child returns its name."""

    def chat(self, messages, *args, **kwargs):
        self.last_usage = rflow.LLMUsage(input_tokens=10, output_tokens=5)
        last = messages[-1]["content"]
        if "do A" in last:
            return '```repl\ndone("A done")\n```'
        if "do B" in last:
            return '```repl\ndone("B done")\n```'
        return ROOT_SPLIT


def banner(title: str) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def current_type(subgraph) -> str:
    current = subgraph.current()
    return current.type if current else "empty"


def main() -> None:
    flow = rflow.Flow(
        ScriptedLLM(), max_depth=1, max_iters=5, max_concurrency=2
    )

    banner("running the engine - one tick per step()")
    graph = flow.start("split into A and B")
    tick = 0
    while not graph.finished:
        tick += 1
        graph = flow.step(graph)
        print(
            f"step {tick}: agents="
            + ", ".join(
                f"{aid}:{current_type(sub)}" for aid, sub in graph.agents.items()
            )
        )
    print(f"\nfinal result: {graph.result()!r}")

    banner("final node log")
    for n in graph.all_nodes:
        tag = (
            getattr(n, "result", None)
            or getattr(n, "content", None)
            or getattr(n, "reply", None)
            or ""
        )
        preview = tag.splitlines()[0][:50] if tag else ""
        print(f"  {n.agent_id:<7} seq={n.seq:<2} {n.type:<18} {preview!r}")

    banner("retrace_steps(graph) - one snapshot per stable transition")
    snapshots = rflow.retrace_steps(graph)
    print(f"{len(snapshots)} snapshots reconstructed from {tick} engine ticks\n")
    for i, snap in enumerate(snapshots, start=1):
        agents = ", ".join(
            f"{aid}:{current_type(sub)}" for aid, sub in snap.agents.items()
        )
        print(f"snapshot {i}  nodes={len(snap.all_nodes)}  ({agents})")

    banner("the parallel snapshot - both children advance together")
    for snap in snapshots:
        kids = [s for aid, s in snap.agents.items() if aid != "root"]
        if kids and all(k.finished for k in kids):
            print(snap.tree())
            break
    _save_example_graph(graph, __file__, "graph-timeline")


if __name__ == "__main__":
    main()
