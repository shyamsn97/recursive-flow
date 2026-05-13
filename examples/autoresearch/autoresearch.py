"""Karpathy-style autoresearch loop on top of RLMFlow.

Mirrors the loop from https://github.com/karpathy/autoresearch:

    edit train.py  →  run for budget_s  →  read val_bpb  →
        if better:  git commit (keep)
        else:       git reset --hard (discard)
    repeat

The human authors `program.md` (the agent's operating manual) and a
target directory holding `train.py` (mutable) and `prepare.py` (fixed).
This script wires that directory into RLMFlow with two extra tools — a
timeboxed experiment runner and a small `git` shell — so any RLMFlow
agent (Anthropic / OpenAI) can drive the loop.

Children are a natural fit: each one tries an independent mutation
(branch / commit / measure / report `val_bpb`) and the parent keeps the
best diff. Use `--branches N` to fan that out.

Usage:
    # point at a directory containing train.py + prepare.py + program.md
    python examples/autoresearch.py --target ../autoresearch
    python examples/autoresearch.py --target ../autoresearch --budget-s 300 --rounds 6
    python examples/autoresearch.py --target ../autoresearch --branches 4
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

from rlmflow import RLMConfig, RLMFlow, Workspace
from rlmflow.llm import AnthropicClient, OpenAIClient
from rlmflow.prompts import DEFAULT_BUILDER
from rlmflow.runtime.local import LocalRuntime
from rlmflow.tools import tool


METRIC_RE = re.compile(r"val_bpb\s*[:=]\s*([0-9]+\.?[0-9]*)", re.IGNORECASE)


AUTORESEARCH_RULES = """\
**You are running an autoresearch hill-climb on `train.py`.**

- Read `program.md` first — it is the human's operating manual for this
  research org. Follow whatever protocol it sets.
- The mutable surface is `train.py`. `prepare.py` and the eval harness
  are immutable.
- Every experiment runs through `run_experiment(budget_s=...)` which
  executes `python train.py` with a wall-clock timeout and returns
  `{"val_bpb", "returncode", "stdout_tail", "stderr_tail", "elapsed_s"}`.
  Lower `val_bpb` is better. **If `returncode != 0`, ALWAYS read
  `stderr_tail` — that's where the real error is. Never report a failed
  run without quoting `stderr_tail`.**
- Use `git_op("status" | "diff" | "commit -am '<msg>'" | "reset --hard")`
  for memory: commit improvements, reset failures.
- When you delegate a child to try a mutation, give it the contract
  "edit train.py, run an experiment, return JSON
  `{\"val_bpb\": float, \"diff\": str, \"notes\": str}`". The parent
  keeps the best diff and discards the rest.
- Never invent numbers — every reported val_bpb comes from
  `run_experiment` output.
"""


def _safe_join(root: Path, path: str) -> Path:
    """Resolve ``path`` against ``root``; reject anything that escapes.

    Tools rebound here are meant to operate strictly inside the target
    repo. ``..`` traversal or absolute paths outside ``root`` raise.
    """
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"path escapes target dir: {path}") from exc
    return candidate


def _make_file_tools(target: Path):
    """File tools rebound to ``target`` (the repo the agent is mutating).

    The default ``FILE_TOOLS`` resolve relative paths against the
    runtime's *workspace* (the rlmflow run dir), which is the right
    sandbox for most examples but the wrong cwd for autoresearch — the
    agent needs to read/edit ``train.py`` in the **target repo**, not in
    its own state dir. These mirrors keep the same names so prompts and
    examples don't change.
    """
    target = target.resolve()

    @tool("Read a file from the target repo and return its contents.")
    def read_file(path: str) -> str:
        return _safe_join(target, path).read_text()

    @tool("Write content to a file in the target repo, creating directories if needed.")
    def write_file(path: str, content: str) -> str:
        p = _safe_join(target, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"

    @tool("Append content to a file in the target repo.")
    def append_file(path: str, content: str) -> str:
        p = _safe_join(target, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(content)
        return f"Appended {len(content)} bytes to {path}"

    @tool("Find-and-replace edits in a target file. Each edit is (old, new).")
    def edit_file(path: str, *edits: tuple[str, str]) -> str:
        p = _safe_join(target, path)
        text = p.read_text()
        count = 0
        for old, new in edits:
            if old in text:
                text = text.replace(old, new, 1)
                count += 1
        p.write_text(text)
        return f"Applied {count}/{len(edits)} edits to {path}"

    @tool("List files and directories in the target repo.")
    def ls(path: str = ".") -> list[str]:
        p = _safe_join(target, path)
        if p.is_file():
            return [p.name]
        return sorted(entry.name for entry in p.iterdir())

    @tool("Read lines start:end (0-indexed, exclusive) from a target file.")
    def read_lines(path: str, start: int, end: int) -> str:
        return "\n".join(_safe_join(target, path).read_text().splitlines()[start:end])

    @tool("Count the number of lines in a target file.")
    def line_count(path: str) -> int:
        return len(_safe_join(target, path).read_text().splitlines())

    @tool("List files in the target repo matching a glob pattern.")
    def list_files(pattern: str = "*.py") -> list[str]:
        return sorted(str(p.relative_to(target)) for p in target.glob(pattern))

    @tool("Search target files for lines matching a regex pattern.")
    def grep(pattern: str, path: str = ".", *, max_results: int = 50) -> str:
        regex = re.compile(pattern)
        root = _safe_join(target, path)
        matches: list[str] = []
        files = [root] if root.is_file() else sorted(root.rglob("*"))
        for f in files:
            if not f.is_file():
                continue
            try:
                for i, line in enumerate(f.read_text().splitlines(), 1):
                    if regex.search(line):
                        matches.append(f"{f.relative_to(target)}:{i}: {line}")
                        if len(matches) >= max_results:
                            return "\n".join(matches)
            except (UnicodeDecodeError, PermissionError):
                continue
        return "\n".join(matches)

    return [
        read_file,
        write_file,
        append_file,
        edit_file,
        ls,
        read_lines,
        line_count,
        list_files,
        grep,
    ]


def _make_run_experiment(target: Path):
    @tool(
        "Run `python train.py` from the target directory under a wall-clock "
        "timeout. Returns JSON: {val_bpb, elapsed_s, returncode, stdout_tail, "
        "stderr_tail}. val_bpb is parsed from stdout (`val_bpb: <float>`); "
        "missing if the run crashed or didn't print it. **If returncode != 0, "
        "the real error is almost always in `stderr_tail`, not `stdout_tail`.**"
    )
    def run_experiment(budget_s: int = 300) -> str:
        budget_s = max(10, min(int(budget_s), 3600))
        try:
            proc = subprocess.run(
                [sys.executable, "train.py"],
                cwd=str(target),
                capture_output=True,
                text=True,
                timeout=budget_s,
            )
            stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
            elapsed = None
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = (exc.stderr or "") + f"\n[timed out after {budget_s}s]"
            rc = -1
            elapsed = budget_s
        match = METRIC_RE.search(stdout)
        return json.dumps(
            {
                "val_bpb": float(match.group(1)) if match else None,
                "elapsed_s": elapsed,
                "returncode": rc,
                "stdout_tail": stdout[-2000:],
                "stderr_tail": stderr[-1000:],
            },
            indent=2,
        )

    return run_experiment


def _make_git_op(target: Path):
    @tool(
        "Run `git <args>` inside the target directory. Returns JSON: "
        "{returncode, stdout, stderr}. Use this for status / diff / commit / "
        "reset / log to manage the experiment journal."
    )
    def git_op(args: str) -> str:
        proc = subprocess.run(
            ["git", *shlex.split(args)],
            cwd=str(target),
            capture_output=True,
            text=True,
        )
        return json.dumps(
            {
                "returncode": proc.returncode,
                "stdout": proc.stdout[-3000:],
                "stderr": proc.stderr[-1000:],
            },
            indent=2,
        )

    return git_op


def build_prompt_builder():
    return DEFAULT_BUILDER.section(
        "autoresearch",
        AUTORESEARCH_RULES,
        title="Autoresearch Rules",
        after="recursion",
    )


def make_llm(model: str):
    return AnthropicClient(model) if model.startswith("claude") else OpenAIClient(model)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Karpathy-style autoresearch hill-climb wired through RLMFlow."
    )
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Directory containing train.py + prepare.py + program.md (a checkout of "
        "karpathy/autoresearch or a compatible fork).",
    )
    parser.add_argument(
        "--budget-s",
        type=int,
        default=300,
        help="Wall-clock seconds per `run_experiment` call (default: 300 = 5 min).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=6,
        help="Iteration budget the parent agent gets (one round = one outer LLM turn).",
    )
    parser.add_argument(
        "--branches",
        type=int,
        default=4,
        help="Hint to the parent for how many parallel children to fan out per round.",
    )
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--fast-model", default="gpt-5-mini")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("./runs/autoresearch"),
    )
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    target = args.target.resolve()
    if not (target / "train.py").exists():
        raise SystemExit(f"autoresearch: {target}/train.py not found")
    if not (target / "program.md").exists():
        raise SystemExit(f"autoresearch: {target}/program.md not found")

    workspace = Workspace.create(args.workspace)
    runtime = LocalRuntime(workspace=workspace)
    runtime.register_tools(
        [
            *_make_file_tools(target),
            _make_run_experiment(target),
            _make_git_op(target),
        ]
    )

    llm_clients = None
    if args.fast_model:
        llm_clients = {
            "fast": {
                "model": make_llm(args.fast_model),
                "description": "Cheaper/faster model for scoped child mutations.",
            }
        }

    agent = RLMFlow(
        llm_client=make_llm(args.model),
        runtime=runtime,
        workspace=workspace,
        llm_clients=llm_clients,
        config=RLMConfig(
            max_depth=args.max_depth,
            max_iterations=args.rounds,
            max_concurrency=args.max_concurrency,
        ),
        prompt_builder=build_prompt_builder(),
    )

    program_md = (target / "program.md").read_text()
    query = (
        "Run an autoresearch hill-climb on `train.py` in the target repo "
        f"(absolute path: {target}). Target ~{args.branches} parallel mutations per "
        "round, keep the best diff via `git_op('commit -am ...')`, discard the rest "
        "via `git_op('reset --hard')`. Iterate until `done(best_val_bpb)`.\n\n"
        "----- program.md -----\n"
        f"{program_md}"
    )

    graph = agent.start(query)
    while not graph.finished:
        graph = agent.step(graph)
        print(graph.tree())

    result = graph.result()
    print("\n" + "=" * 80)
    print(result or "(no result)")
    print(f"\nWorkspace saved to {workspace.root}")

    if not args.no_viewer:
        try:
            from rlmflow.utils.viewer import save_html

            save_html(workspace, args.workspace / "viewer.html")
            print(f"Viewer saved to {args.workspace / 'viewer.html'}")
        except ImportError as exc:
            print(f"Viewer not saved: {exc}")


if __name__ == "__main__":
    main()
