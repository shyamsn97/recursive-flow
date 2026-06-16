"""Needle in a haystack across many files.

Generates many files of random noise. One file contains a magic string. The
agent uses the standard file tools to find it, delegating the search in
parallel across batches.

Usage:
    python examples/needle/filesystem.py
    python examples/needle/filesystem.py --no-viz
    python examples/needle/filesystem.py --viewer
    python examples/needle/filesystem.py --docker-image recursive-flow:local
"""

from __future__ import annotations

import argparse
import random
import string
import tempfile
from pathlib import Path

import rflow
from rflow.tools import FILE_TOOLS


def generate_haystack(
    directory: Path, num_files: int = 500, lines_per_file: int = 200
) -> str:
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    answer = "".join(random.choices(string.digits, k=7))
    needle_file = random.randint(0, num_files - 1)
    needle_line = random.randint(0, lines_per_file - 1)

    for i in range(num_files):
        lines = []
        for j in range(lines_per_file):
            if i == needle_file and j == needle_line:
                lines.append(f"The magic number is {answer}")
            else:
                n = random.randint(3, 8)
                lines.append(" ".join(random.choice(words) for _ in range(n)))
        (directory / f"file_{i:04d}.txt").write_text("\n".join(lines))

    print(f"Needle in file_{needle_file:04d}.txt line {needle_line}")
    return answer


def build_llm(model: str):
    return (
        rflow.AnthropicClient(model)
        if model.startswith("claude")
        else rflow.OpenAIClient(model)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Needle in a haystack across many files"
    )
    parser.add_argument("--num-files", type=int, default=500)
    parser.add_argument(
        "--viewer", action="store_true", help="Open the state viewer after finishing"
    )
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--fast-model", default="gpt-5-nano")
    parser.add_argument(
        "--docker-image",
        default=None,
        help="If set, run agent code inside this Docker image (e.g. recursive-flow:local).",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Directory to hold haystack/ and run in (default: a temp dir).",
    )
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-iters", type=int, default=15)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    if args.docker_image:
        print(f">>> DOCKER RUNTIME  image={args.docker_image}")
    else:
        print(">>> LOCAL RUNTIME")

    tmp = None
    if args.workdir is None:
        tmp = tempfile.TemporaryDirectory()
        workdir = Path(tmp.name)
    else:
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)

    try:
        haystack_path = workdir / "haystack"
        haystack_path.mkdir(parents=True, exist_ok=True)
        for stale in haystack_path.glob("*.txt"):
            stale.unlink()
        answer = generate_haystack(haystack_path, num_files=args.num_files)
        print(f"Generated {args.num_files} files in {haystack_path}")

        if args.docker_image:
            runtime = rflow.DockerRuntime(args.docker_image, working_directory=workdir)
        else:
            runtime = rflow.LocalRuntime(working_directory=workdir)
        runtime.register_tools(FILE_TOOLS)

        llm_clients = None
        if args.fast_model:
            llm_clients = {"fast": build_llm(args.fast_model)}

        flow = rflow.Flow(
            build_llm(args.model),
            llm_clients=llm_clients,
            runtime=runtime,
            max_depth=args.max_depth,
            max_iters=args.max_iters,
        )

        graph = flow.start(
            f"There are {args.num_files} text files in haystack/. "
            "Exactly one line in one file matches the pattern "
            "`The magic number is <number>`. Find and return the number. "
            "There are too many files to search manually, so split the work "
            "into batches and delegate."
        )

        if args.no_viz:
            while not graph.finished:
                graph = flow.step(graph)
                print(graph.tree())
        else:
            from rflow.utils.viz import live

            graph = live(flow, graph)[-1]

        print(f"\n{'=' * 40}")
        print(f"Actual answer:  {answer}")
        print(f"Correct:        {answer in graph.result()}")

        if args.viewer:
            from rflow.utils.viewer import open_viewer

            open_viewer([graph])

        flow.close()
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    main()
