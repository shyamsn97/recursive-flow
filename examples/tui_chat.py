"""Open the live Flow TUI with a real model-backed Flow.

Requires OpenAI credentials and the TUI optional dependency:

    export OPENAI_API_KEY=...
    pip install -e ".[openai,tui]"
    python examples/tui_chat.py

Type or paste a query directly in the TUI, add optional supporting context in the
context box, and press Ctrl+Enter to send. Both inputs are multi-line. Press
Ctrl+C to quit; the latest graph is saved under ``examples/_runs/tui-chat`` by
default.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import rflow
from rflow.tools import FILE_TOOLS


def _example_run_dir(source_file: str | Path, name: str) -> Path:
    source = Path(source_file).resolve()
    for parent in (source.parent, *source.parents):
        if parent.name == "examples":
            return parent / "_runs" / name
    return source.parent / "_runs" / name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a real Flow in the live TUI.")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument(
        "--max-iters",
        type=int,
        default=30,
        help="Max LLM turns for the root agent before it is forced to finalize.",
    )
    parser.add_argument(
        "--child-max-iters",
        type=int,
        default=20,
        help="Max LLM turns for each child agent before it is forced to finalize.",
    )
    parser.add_argument(
        "--max-steps-per-turn",
        type=int,
        default=240,
        help="Safety cap for each submitted prompt before returning control to the TUI.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_example_run_dir(__file__, "tui-chat"),
        help="Directory for the workdir and saved graph.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this real-model example.")

    out_dir = args.out_dir.resolve()
    workdir = out_dir / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    runtime = rflow.LocalRuntime(working_directory=workdir)
    runtime.register_tools(FILE_TOOLS)

    flow = rflow.Flow(
        rflow.OpenAIClient(model=args.model),
        runtime=runtime,
        max_depth=args.max_depth,
        max_iters=args.max_iters,
        child_max_iters=args.child_max_iters,
    )

    graph = None
    try:
        graph = flow.tui(
            max_steps_per_turn=args.max_steps_per_turn,
        )
    finally:
        try:
            latest = graph if graph is not None else flow.graph
            if latest is not None:
                path = latest.save(out_dir / "graph")
                print(f"Graph saved to {path}")
                if latest.result():
                    print(f"Result: {latest.result()}")
        finally:
            flow.close()


if __name__ == "__main__":
    main()
