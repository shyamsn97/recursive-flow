"""Forking a run: copy a Graph, diverge it, compare outcomes.

There's no Workspace anymore — a whole run *is* a ``Graph``, so "forking" is
just ``graph.copy(deep=True)`` followed by an out-of-band edit. Each fork is an
independent value you can edit, re-run, save, or throw away without touching the
original.

Use it for:

- repair variants (try fix A vs fix B from the same starting point)
- best-of-N exploration (fan out a partial run multiple ways)
- speculative edits without disturbing the canonical run

This script:
  1. runs a base graph to completion with a deterministic mock LLM,
  2. forks it twice and rewrites each fork's result differently
     (``replace_last_observation``),
  3. saves each fork to its own ``graph.json`` and shows independence.

Run:
    python examples/graph/06_fork.py
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
from pathlib import Path

import rflow


class ScriptedLLM(rflow.LLMClient):
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def chat(self, messages, *args, **kwargs):
        self.last_usage = rflow.LLMUsage(input_tokens=10, output_tokens=5)
        return self.reply


def banner(title: str) -> None:
    print("\n" + "─" * 60)
    print(title)
    print("─" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "_runs" / "graph-fork"),
        help="where to save the forks (default: examples/_runs/graph-fork/)",
    )
    args = parser.parse_args()

    root = Path(args.out_dir).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    with contextlib.nullcontext():
        banner("seed: run a base graph to completion")
        flow = rflow.Flow(ScriptedLLM('```repl\ndone("seeded result")\n```'), max_iters=3)
        base = flow.run("do the thing")  # returns the result string
        base_graph = flow.graph
        print(f"base result: {base!r}")
        print(f"base nodes : {[n.type for n in base_graph.all_nodes]}")

        banner("fork twice — copy(deep=True) gives independent graphs")
        fork_a = base_graph.copy(deep=True)
        fork_b = base_graph.copy(deep=True)

        # Diverge each fork by rewriting its terminal answer out of band.
        fork_a = fork_a.replace_last_observation(
            "root", rflow.DoneOutput(result="fork A: the careful path"), truncate="none"
        )
        fork_b = fork_b.replace_last_observation(
            "root", rflow.DoneOutput(result="fork B: the bold path"), truncate="none"
        )

        banner("save each fork to its own graph.json")
        for name, g in (("base", base_graph), ("fork_a", fork_a), ("fork_b", fork_b)):
            path = g.save(root / name)
            print(f"  {name:<7} -> {path.relative_to(root)}  result={g.result()!r}")

        banner("the original is unchanged; forks are independent")
        reloaded = rflow.Graph.load(root / "base")
        print(f"base result still : {reloaded.result()!r}")
        for name in ("base", "fork_a", "fork_b"):
            g = rflow.Graph.load(root / name)
            print(f"  {name:<7} nodes={len(g.all_nodes):>2} result={g.result()!r}")


if __name__ == "__main__":
    main()
