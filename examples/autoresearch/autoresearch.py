"""Karpathy-style autoresearch loop on top of RLMFlow.

Mirrors the loop from https://github.com/karpathy/autoresearch:

    edit train.py  →  run for budget_s  →  read val_bpb  →
        if better:  git commit (keep)
        else:       git reset --hard (discard)
    repeat

The human authors `program.md` (the agent's operating manual) and a
source target directory holding `train.py` (mutable) and `prepare.py`
(fixed). This script copies that target into the RLMFlow workspace as
`target/`, gives children isolated copies under `trials/<name>/`, and
wires in two extra tools — a timeboxed experiment runner and a small
`git` shell — so any RLMFlow agent (Anthropic / OpenAI) can drive the loop.

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
import shutil
import subprocess
import sys
from pathlib import Path

from rlmflow import RLMConfig, RLMFlow, Workspace
from rlmflow.llm import AnthropicClient, OpenAIClient
from rlmflow.prompts import DEFAULT_BUILDER
from rlmflow.runtime.local import LocalRuntime
from rlmflow.tools import FILE_TOOLS, tool


METRIC_RE = re.compile(r"val_bpb\s*[:=]\s*([0-9]+\.?[0-9]*)", re.IGNORECASE)


AUTORESEARCH_RULES = """\
**You are running an autoresearch hill-climb on `train.py`.**

- The live experiment is inside the RLMFlow workspace, not the original
  `--target` directory:
  - `target/` is the parent working copy.
  - `trials/<name>/` are isolated child working copies.
- File tools are workspace-rooted. Read/edit `target/train.py` in the
  parent, and `trials/<name>/train.py` inside child tasks.
- Every experiment runs through `run_experiment(path=..., budget_s=...)`
  which executes `python train.py` in that path and returns
  `{"val_bpb", "returncode", "stdout_tail", "stderr_tail", "elapsed_s"}`.
  Lower `val_bpb` is better. **If `returncode != 0`, ALWAYS read
  `stderr_tail` — that's where the real error is. Never report a failed
  run without quoting `stderr_tail`.**
- Use `git_op("status" | "diff" | "commit -am '<msg>'" | "reset --hard", path=...)`
  for memory inside each working copy.
- For each delegated child:
  1. create a trial with `trial = create_trial("<short_name>")`;
  2. pass the exact `trial` path in the query;
  3. tell the child to edit only `{trial}/train.py`;
  4. tell it to run `run_experiment(path=trial, budget_s=...)`;
  5. require JSON: `{"success": bool, "val_bpb": float|null,
     "train_py": str, "diff": str, "notes": str, "stderr_tail": str}`.
- Do not put executable driver code in `CONTEXT`; context is data, not code.
- The parent applies only the best child by copying its returned `train_py`
  into `target/train.py` and committing in `target/`.
- Never invent numbers — every reported val_bpb comes from
  `run_experiment` output.
"""


def _safe_workspace_path(root: Path, path: str) -> Path:
    """Resolve a relative workspace path and reject escapes."""
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"path escapes workspace: {path}") from exc
    return candidate


def _ignore_generated(_: str, names: list[str]) -> set[str]:
    ignored = {
        ".git",
        ".DS_Store",
        "__pycache__",
        ".ipynb_checkpoints",
        "runs",
        "workspace",
        "workspaces",
    }
    return {name for name in names if name in ignored}


def _copy_target_tree(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest, symlinks=True, ignore=_ignore_generated)


def _init_experiment_repo(path: Path) -> None:
    """Create a tiny local git journal for the mutable experiment copy.

    Only `train.py` is guaranteed tracked. Data and cache files stay present
    but untracked, so `git reset --hard` is fast and won't delete datasets.
    """
    subprocess.run(["git", "init"], cwd=path, capture_output=True, text=True)
    tracked = [
        name
        for name in ("train.py", "program.md", "prepare.py")
        if (path / name).exists()
    ]
    if tracked:
        subprocess.run(["git", "add", *tracked], cwd=path, capture_output=True, text=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=rlmflow",
                "-c",
                "user.email=rlmflow@example.invalid",
                "commit",
                "-m",
                "baseline",
            ],
            cwd=path,
            capture_output=True,
            text=True,
        )


def prepare_workspace_target(source: Path, workspace_root: Path) -> Path:
    """Copy the user's target into the RLMFlow workspace as `target/`."""
    target_copy = workspace_root / "target"
    trials = workspace_root / "trials"
    _copy_target_tree(source, target_copy)
    if trials.exists():
        shutil.rmtree(trials)
    trials.mkdir(parents=True, exist_ok=True)
    _init_experiment_repo(target_copy)
    return target_copy


def _make_create_trial(workspace_root: Path):
    @tool(
        "Create an isolated trial copy under `trials/<name>` from a workspace "
        "source path (default `target`). Returns the relative trial path. "
        "Use this before delegating a child experiment."
    )
    def create_trial(name: str, source: str = "target") -> str:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-")
        if not slug:
            raise ValueError("trial name must contain at least one safe character")
        src = _safe_workspace_path(workspace_root, source)
        if not src.is_dir():
            raise FileNotFoundError(f"trial source is not a directory: {source}")
        rel = f"trials/{slug}"
        dst = _safe_workspace_path(workspace_root, rel)
        _copy_target_tree(src, dst)
        _init_experiment_repo(dst)
        return rel

    return create_trial


def _make_run_experiment(workspace_root: Path):
    @tool(
        "Run `python train.py` from a workspace path under a wall-clock "
        "timeout. Returns JSON: {val_bpb, elapsed_s, returncode, stdout_tail, "
        "stderr_tail}. val_bpb is parsed from stdout (`val_bpb: <float>`); "
        "`path` defaults to `target`; children should pass their `trials/<name>` path. "
        "missing if the run crashed or didn't print it. **If returncode != 0, "
        "the real error is almost always in `stderr_tail`, not `stdout_tail`.**"
    )
    def run_experiment(path: str = "target", budget_s: int = 300) -> str:
        budget_s = max(10, min(int(budget_s), 3600))
        workdir = _safe_workspace_path(workspace_root, path)
        try:
            proc = subprocess.run(
                [sys.executable, "train.py"],
                cwd=str(workdir),
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


def _make_git_op(workspace_root: Path):
    @tool(
        "Run `git <args>` inside a workspace path. Returns JSON: "
        "{returncode, stdout, stderr}. Use this for status / diff / commit / "
        "reset / log to manage the experiment journal. `path` defaults to `target`; "
        "children should pass their `trials/<name>` path."
    )
    def git_op(args: str, path: str = "target") -> str:
        workdir = _safe_workspace_path(workspace_root, path)
        proc = subprocess.run(
            [
                "git",
                "-c",
                "user.name=rlmflow",
                "-c",
                "user.email=rlmflow@example.invalid",
                *shlex.split(args),
            ],
            cwd=str(workdir),
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
    workspace_target = prepare_workspace_target(target, workspace.root)
    runtime = LocalRuntime(workspace=workspace)
    runtime.register_tools(
        [
            *FILE_TOOLS,
            _make_create_trial(workspace.root),
            _make_run_experiment(workspace.root),
            _make_git_op(workspace.root),
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

    program_md = (workspace_target / "program.md").read_text()
    query = (
        "Run an autoresearch hill-climb inside this RLMFlow workspace. "
        f"The original source target was copied from {target} into `target/`; "
        "`target/train.py` is the parent working copy. "
        f"Target ~{args.branches} parallel mutations per round. For each child, "
        "first call `create_trial('<short_name>')`, pass the returned "
        "`trials/<short_name>` path to the child, and have the child edit only "
        "that trial's `train.py` and run `run_experiment(path=trial_path, ...)`. "
        "The parent keeps the best child by copying its returned `train_py` into "
        "`target/train.py` and committing with `git_op(..., path='target')`. "
        "Discard the rest. Iterate until `done(best_val_bpb)`.\n\n"
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
    print(f"Parent working copy: {workspace.root / 'target'}")
    print(f"Trial copies: {workspace.root / 'trials'}")

    if not args.no_viewer:
        try:
            from rlmflow.utils.viewer import save_html

            save_html(workspace, args.workspace / "viewer.html")
            print(f"Viewer saved to {args.workspace / 'viewer.html'}")
        except ImportError as exc:
            print(f"Viewer not saved: {exc}")


if __name__ == "__main__":
    main()
