"""Showcase the Graph-centric Flow API.

This walks through the pieces that matter in the engine:

1. Step-by-step execution that advances a single live ``Graph``.
2. Persisting a run with ``graph.save()`` / ``rflow.Graph.load()``.
3. Latest-state inspection across agents.
4. In-process history by keeping graph snapshots.
5. Graph summary helpers (``graph.tree()``, ``graph.tokens()``).
6. Gym-style stepping with a scalar reward.

Usage:
    python examples/showcase.py
    python examples/showcase.py --no-viz
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import rflow
from rflow.tools import FILE_TOOLS

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RESET = "\033[0m"


class DemoLLM(rflow.LLMClient):
    """Deterministic LLM for an offline showcase."""

    def chat(self, messages, *args, **kwargs) -> str:
        self.last_usage = rflow.LLMUsage(input_tokens=80, output_tokens=20)
        prompt = messages[-1]["content"].lower()
        if "hello.py" in prompt and "goodbye.py" in prompt:
            return (
                "```repl\n"
                "results = await launch_subagents([\n"
                '    {"name": "hello", "query": "Create hello.py"},\n'
                '    {"name": "goodbye", "query": "Create goodbye.py"},\n'
                "])\n"
                'done("\\n".join(results))\n'
                "```"
            )
        if "hello.py" in prompt:
            return '```repl\nwrite_file("hello.py", "print(\\"hello\\")\\n")\ndone("hello.py")\n```'
        if "goodbye.py" in prompt:
            return '```repl\nwrite_file("goodbye.py", "print(\\"goodbye\\")\\n")\ndone("goodbye.py")\n```'
        if "haiku" in prompt:
            return '```repl\nwrite_file("haiku.txt", "Calls fold into calls\\nNodes branch, wait, and then resume\\nFlow returns a leaf\\n")\ndone("wrote haiku.txt")\n```'
        return '```repl\ndone("ok")\n```'


def file_flow(workdir: Path, **kwargs) -> rflow.Flow:
    """A Flow whose agents get the filesystem tools, running inside ``workdir``."""
    runtime = rflow.LocalRuntime(working_directory=workdir)
    runtime.register_tools(FILE_TOOLS)
    return rflow.Flow(DemoLLM(), runtime=runtime, **kwargs)


def banner(msg: str) -> None:
    print(f"\n{BOLD}{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}{RESET}\n")


def run(flow: rflow.Flow, graph: rflow.Graph, no_viz: bool) -> list[rflow.Graph]:
    if no_viz:
        history = [graph]
        step = 0
        while not graph.finished:
            graph = flow.step(graph)
            step += 1
            history.append(graph)
            print(f"-- step {step} --")
            print(graph.tree())
        return history
    from rflow.utils.viz import live

    return live(flow, graph)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-iters", type=int, default=8)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "_runs" / "showcase"),
        help="working dir + saved run (default: examples/_runs/showcase/)",
    )
    args = parser.parse_args()

    workdir = Path(args.out_dir).resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    flow = file_flow(workdir, max_depth=args.max_depth, max_iters=args.max_iters)

    banner("1. Step-by-step execution")
    graph = flow.start("Create hello.py and goodbye.py. Delegate each file.")
    history = run(flow, graph, args.no_viz)
    final = history[-1]
    print(f"\n{GREEN}Result:{RESET} {final.result()}")

    banner("2. Persistence — graph.save() / Graph.load()")
    path = final.save(workdir / "run")
    loaded = rflow.Graph.load(path)
    print(
        f"Saved + reloaded {len(loaded.agents)} agents and "
        f"{len(loaded.all_nodes)} states from {path}"
    )
    print(loaded.tree())

    banner("3. Latest state per agent")
    for aid, sub in loaded.agents.items():
        current = sub.current()
        label = current.type if current else "(empty)"
        print(f"  {aid}: {label}")

    banner("4. Time travel — kept snapshots")
    for idx, snapshot in enumerate(history):
        current = snapshot.current()
        kind = current.type if current else "empty"
        print(
            f"{CYAN}step {idx}{RESET}: root [{kind}]  "
            f"agents={len(snapshot.agents)}"
        )

    banner("5. Graph summary")
    inp, out = final.tokens()
    print(f"Agents:  {len(final.agents)}")
    print(f"States:  {len(final.all_nodes)}")
    print(f"Tokens:  {inp + out:,} ({inp:,} in / {out:,} out)")
    print(f"Final:   {final.current().type if final.current() else '(empty)'}")

    banner("6. Gym-style loop")
    flow3 = file_flow(workdir, max_depth=0, max_iters=args.max_iters)
    graph3 = flow3.start("Write a haiku about recursion to haiku.txt")
    rewards: list[float] = []
    step = 0
    while not graph3.finished:
        graph3 = flow3.step(graph3)
        step += 1
        current = graph3.current()
        reward = 1.0 if graph3.finished else 0.0
        rewards.append(reward)
        kind = current.type if current else "empty"
        print(f"step {step}: state={kind} reward={reward}")
    print(f"{GREEN}Result:{RESET} {graph3.result()}")
    print(f"Total reward: {sum(rewards):.1f}")
    gym_path = graph3.save(workdir / "gym-run")
    print(f"Gym run saved to {gym_path}")

    banner("Done")


if __name__ == "__main__":
    main()
