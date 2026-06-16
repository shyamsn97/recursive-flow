"""Run a platformer-building Flow task inside an E2B Sandbox.

Each agent's code runs remotely in an E2B Sandbox (via an :class:`E2BRuntime`);
files are written with plain Python inside the sandbox working directory.

Setup:
    pip install -e ".[openai,e2b]"
    export OPENAI_API_KEY=...
    export E2B_API_KEY=...

Run:
    python examples/sandboxes/e2b_agent.py --model gpt-5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rflow  # noqa: E402
from rflow.runtime.sandbox.e2b import E2BRuntime  # noqa: E402

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
    parser = argparse.ArgumentParser(description="Run Flow inside E2B.")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument(
        "--fast-model",
        default="gpt-5-mini",
        help="Cheaper model exposed to delegates as `model='fast'`.",
    )
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument(
        "--max-depth",
        type=int,
        default=1,
        help="Recursive sub-agent depth. Defaults to 1 so delegation is enabled.",
    )
    parser.add_argument("--template", help="E2B template name or ID.")
    parser.add_argument("--sandbox-timeout", type=int, default=300)
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
        help="Skip setup commands, useful for templates with recursive-flow preinstalled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_commands = [] if args.skip_setup else args.setup_command

    # One sandbox per agent; created lazily when the agent first runs code.
    runtime = E2BRuntime(
        template=args.template,
        timeout=args.sandbox_timeout,
        remote_workdir=args.remote_workdir,
        repl_timeout=args.repl_timeout,
        setup_commands=setup_commands,
    )

    flow = rflow.Flow(
        rflow.OpenAIClient(model=args.model),
        llm_clients={"fast": rflow.OpenAIClient(model=args.fast_model)},
        runtime=runtime,
        max_depth=args.max_depth,
        max_iters=args.max_iters,
    )
    try:
        print(flow.run(PLATFORMER_QUERY))
    finally:
        flow.close()


if __name__ == "__main__":
    main()
