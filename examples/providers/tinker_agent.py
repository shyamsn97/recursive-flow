"""Run RecursiveFlow with Tinker inference.

Requires Tinker credentials and optional dependencies:

    export TINKER_API_KEY=...
    pip install -e ".[tinker]"
    python examples/providers/tinker_agent.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import rflow
from rflow.runtime.local import LocalRuntime
from rflow.utils.viz import live


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a tiny RecursiveFlow task with Tinker."
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-8B",
        help="Tinker base model for inference.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional saved Tinker weights path, e.g. tinker://run/weights/checkpoint.",
    )
    parser.add_argument(
        "--renderer",
        default="qwen3",
        help="Tinker cookbook renderer name matching the model family.",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument(
        "--query",
        default="Use Python to compute 17 * 23, then call done() with the answer.",
    )
    args = parser.parse_args()

    examples_root = Path(__file__).resolve().parents[1]
    workspace = rflow.Workspace.create(
        examples_root / "_runs" / "example-workspaces" / "tinker-workspace"
    )
    llm = rflow.TinkerClient(
        base_model=None if args.model_path else args.base_model,
        model_path=args.model_path,
        renderer=args.renderer,
        max_tokens=args.max_tokens,
    )
    agent = rflow.RecursiveFlow(
        llm_client=llm,
        runtime=LocalRuntime(workspace=workspace),
        config=rflow.FlowConfig(max_iterations=args.max_iterations),
    )
    print(f"Query: {args.query}\n")
    graph = agent.start(args.query)
    result = live(agent, graph)[-1].result()
    print(result)


if __name__ == "__main__":
    main()
