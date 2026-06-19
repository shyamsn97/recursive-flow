"""Run a platformer-building Flow task inside a Daytona Sandbox.

Each agent's code runs remotely in a Daytona Sandbox (via a
:class:`DaytonaRuntime`); files are written with plain Python inside the sandbox
working directory.

Setup:
    pip install -e ".[openai,daytona]"
    export OPENAI_API_KEY=...
    export DAYTONA_API_KEY=...

Run:
    python examples/sandboxes/daytona_agent.py --model gpt-5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rflow  # noqa: E402
from rflow.runtime.sandbox.daytona import DaytonaRuntime  # noqa: E402
from rflow.utils.example_runs import save_example_graph  # noqa: E402

PLATFORMER_QUERY = """\
Build a simple 2D side-scrolling platformer in plain HTML/CSS/JS under output/.
No build tools, no libraries, no ES modules. Write files with plain Python
(e.g. `open(path, "w").write(...)`) in the sandbox.

Files:
- output/index.html
- output/styles.css
- output/scripts/engine.js   — state, input, physics
- output/scripts/main.js     — level, render, requestAnimationFrame loop

index.html loads engine.js then main.js. Canvas with left/right movement, jump,
gravity, platform collision, scrolling camera, and restart.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Flow inside Daytona.")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument(
        "--fast-model",
        default="gpt-5-mini",
        help="Cheaper model exposed to delegates as `model='fast'`.",
    )
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument("--snapshot", help="Daytona snapshot name or ID.")
    parser.add_argument("--create-timeout", type=float, default=60)
    parser.add_argument("--repl-timeout", type=float, default=30)
    parser.add_argument("--remote-workdir", default="/workspace")
    parser.add_argument(
        "--setup-command",
        action="append",
        help=(
            "Command to run before starting the REPL. Repeat for multiple commands. "
            "Defaults to installing recursive-flow from PyPI."
        ),
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip setup commands, useful for snapshots with recursive-flow preinstalled.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "examples" / "_runs" / "sandbox-daytona"),
        help="Save the final run here (default: examples/_runs/sandbox-daytona/).",
    )
    return parser.parse_args()


def create_params(snapshot: str | None) -> Any:
    if snapshot is None:
        return None
    from daytona import CreateSandboxFromSnapshotParams

    return CreateSandboxFromSnapshotParams(snapshot=snapshot, language="python")


def main() -> None:
    args = parse_args()
    setup_commands = [] if args.skip_setup else args.setup_command
    params = create_params(args.snapshot)

    # One sandbox per agent; created lazily when the agent first runs code.
    runtime = DaytonaRuntime(
        create_params=params,
        create_timeout=args.create_timeout,
        remote_workdir=args.remote_workdir,
        repl_timeout=args.repl_timeout,
        setup_commands=setup_commands,
    )

    flow = rflow.Flow(
        rflow.OpenAIClient(model=args.model),
        llm_clients={"fast": rflow.OpenAIClient(model=args.fast_model)},
        runtime=runtime,
        max_depth=1,
        max_iters=args.max_iters,
    )
    try:
        graph = flow.start(PLATFORMER_QUERY)
        while not graph.finished:
            graph = flow.step(graph)
        print(graph.result())
        save_example_graph(graph, __file__, "sandbox-daytona", out_dir=args.out_dir)
    finally:
        flow.close()


if __name__ == "__main__":
    main()
