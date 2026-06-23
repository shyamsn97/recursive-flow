"""Minimal autoresearch runner.

Usage:
    python examples/autoresearch/run.py --model gpt-5 --gpu L4 --parallel 4
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Protocol

import rflow
from rflow.prompts import DEFAULT_BUILDER
from rflow.tools import tool

try:  # Allow both `python examples/autoresearch/run.py` and package imports.
    from .modal_runner import ModalConfig, classify_result, collect, preflight, submit
except ImportError:  # pragma: no cover - exercised when run as a script
    from modal_runner import ModalConfig, classify_result, collect, preflight, submit


AUTORESEARCH_RECURSION_TEXT = """
Run the autoresearch controller. The actual research task is in
`INPUTS["task_instructions"]`; treat it as the source of truth.

Parent loop:
1. Call `refresh_results()`.
2. Inspect `best_run()`, `get_runs()`, `list_runs()`, and `submission_status()`.
3. If the baseline has not been submitted, call `run_baseline()`.
4. If submissions remain, choose diverse, idea-named slugs that do not already
   appear in `get_runs()`.
5. Launch up to the remaining submission budget with `launch_subagents`, one
   hypothesis per child.
6. Repeat until the submission budget is exhausted or the scored results clearly
   plateau.

Child workflow:
1. Read `INPUTS["task_instructions"]` and `train.py`; read archived sources only
   if the parent points to them.
2. Produce one complete runnable `train.py` source string.
3. Submit with `run_experiment(source, slug, hypothesis)`.
4. If submitted, report slug/status/source_path and stop.
5. For obvious syntax/import/runtime bugs, make at most one targeted fix with a
   fresh slug like `<slug>_fix1`. Do not retry unchanged slow/OOM ideas.

Good first idea families:
- learning-rate schedule changes,
- optimizer settings,
- depth/width tradeoffs,
- batch-size or gradient-accumulation changes,
- attention or positional-embedding changes,
- normalization, activation, dropout, and weight decay tweaks,
- simplifications that keep or improve `val_bpb`.

Guardrails:
- Only `train.py` changes. Submit complete source strings; do not edit files.
- Preserve the fixed evaluation contract: final output includes `val_bpb:`.
- Do not add dependencies, network calls beyond the existing data cache,
  subprocesses, alternate CLIs, or result-file protocols.
- Use `best_run()` for the best scored run. A submitted/crashed run is never
  best.

Do not use shell, git, commits, or filesystem writes. The tools are the ledger.
"""


def build_prompt_builder():
    return DEFAULT_BUILDER.section(
        "autoresearch_recursion",
        AUTORESEARCH_RECURSION_TEXT,
        title="Autoresearch",
        after="format",
    )


class TrialSubmitter(Protocol):
    """Submits one archived candidate `train.py` and returns job data."""

    def __call__(
        self,
        *,
        path: Path,
        train_budget_s: int,
        slug: str,
        n: int,
        run_id: str,
        seed: int | None = None,
    ) -> dict[str, Any]: ...


class TrialCollector(Protocol):
    """Collects one submitted job result if it is ready."""

    def __call__(self, *, job_id: str, timeout_s: float = 0) -> dict[str, Any]: ...


class ExperimentCrashed(RuntimeError):
    """Raised for candidate-code failures; `row` is already in the ledger."""

    def __init__(self, row: dict[str, Any]) -> None:
        self.row = row
        tail = (row.get("stderr_tail") or "").strip()
        super().__init__(f"{row.get('status')} slug={row.get('slug')!r}\n{tail}")


class SubmissionError(RuntimeError):
    """Raised when the experiment submission cap is exhausted."""


class AutoresearchState:
    """Host-side state and tools for one autoresearch run."""

    def __init__(
        self,
        *,
        example_dir: Path,
        out_dir: Path,
        submitter: TrialSubmitter,
        collector: TrialCollector,
        train_budget_s: int,
        max_submissions: int | None,
        gpu: str,
        run_id: str | None = None,
    ) -> None:
        self.example_dir = example_dir.resolve()
        self.out_dir = out_dir.resolve()
        self.history_dir = self.out_dir / "history"
        self.ledger_path = self.history_dir / "ledger.jsonl"
        self.submitter = submitter
        self.collector = collector
        self.train_budget_s = train_budget_s
        self.max_submissions = max_submissions
        self.gpu = gpu
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
        self.lock = threading.RLock()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    @property
    def baseline_source(self) -> str:
        return (self.example_dir / "train.py").read_text()

    def allowed_read(self, path: str) -> str:
        """Read only the agent-facing files and archived trial files."""

        candidate = Path(path)
        if not candidate.is_absolute():
            root_first = (self.example_dir / candidate).resolve()
            out_first = (self.out_dir / candidate).resolve()
            candidate = root_first if root_first.exists() else out_first
        candidate = candidate.resolve()
        allowed_names = {"README.md", "program.md", "train.py", "pyproject.toml"}
        allowed_example_files = {
            (self.example_dir / name).resolve()
            for name in allowed_names
            if (self.example_dir / name).exists()
        }
        if candidate not in allowed_example_files and not _is_relative_to(
            candidate, self.history_dir
        ):
            raise ValueError(
                "read_file may only read README.md, program.md, train.py, "
                "pyproject.toml, or history"
            )
        if not candidate.is_file():
            raise FileNotFoundError(str(candidate))
        return candidate.read_text()

    def run_baseline(self) -> dict[str, Any]:
        """Run the original train.py once. Baseline is idempotent."""

        with self.lock:
            baseline = self._latest_by_slug_locked().get("baseline")
            if baseline:
                return _agent_row(baseline)
        return self._run_trial(
            source=self.baseline_source,
            slug="baseline",
            hypothesis="Original train.py baseline.",
            is_baseline=True,
        )

    def run_experiment(
        self, source: str, slug: str, hypothesis: str = "", seed: int | None = None
    ) -> dict[str, Any]:
        """Submit one candidate source string as an experiment."""

        return self._run_trial(
            source=source,
            slug=slug,
            hypothesis=hypothesis,
            is_baseline=False,
            seed=seed,
        )

    def list_runs(self) -> list[dict[str, Any]]:
        """Compact best-first ledger view."""

        rows = list(self._latest_by_n().values())
        rows.sort(key=_rank_key)
        return [_summary_row(row) for row in rows]

    def best_run(self) -> dict[str, Any] | None:
        """Best successful/scored run, or None if no experiment has scored yet."""

        scored = [
            row
            for row in self._latest_by_n().values()
            if row.get("status") == "succeeded" and row.get("val_bpb") is not None
        ]
        if not scored:
            return None
        return _agent_row(min(scored, key=lambda row: float(row["val_bpb"])))

    def get_runs(self) -> dict[str, dict[str, Any]]:
        """Latest ledger rows keyed by slug."""

        rows = sorted(self._latest_by_n().values(), key=lambda row: int(row["n"]))
        return {str(row["slug"]): _agent_row(row) for row in rows}

    def get_run(self, n: int) -> dict[str, Any] | None:
        """Full latest row for trial number `n`."""

        row = self._latest_by_n().get(n)
        return _agent_row(row) if row else None

    def latest_run(self) -> dict[str, Any] | None:
        """Most recent latest row by trial number."""

        rows = self._latest_by_n()
        return _agent_row(rows[max(rows)]) if rows else None

    def submission_status(self) -> dict[str, int | None]:
        """Submission budget, excluding the idempotent baseline."""

        with self.lock:
            used = self._submission_count_locked()
        remaining = (
            None
            if self.max_submissions is None
            else max(0, self.max_submissions - used)
        )
        return {
            "max_submissions": self.max_submissions,
            "used_submissions": used,
            "remaining_submissions": remaining,
        }

    def refresh_results(self, timeout_s: float = 0) -> list[dict[str, Any]]:
        """Collect completed submitted jobs and append completed ledger rows."""

        updated = []
        for row in list(self._latest_by_n().values()):
            if row.get("status") != "submitted" or not row.get("job_id"):
                continue
            result = self.collector(job_id=str(row["job_id"]), timeout_s=timeout_s)
            if result.get("status") == "submitted":
                continue
            final_row = self._finalize(
                n=int(row["n"]),
                slug=str(row["slug"]),
                hypothesis=str(row.get("hypothesis") or ""),
                trial_dir=Path(str(row["trial_dir"])),
                result=result,
                is_baseline=bool(row.get("is_baseline") or row.get("slug") == "baseline"),
            )
            updated.append(_agent_row(final_row))
        return updated

    def submitted_runs(self) -> list[dict[str, Any]]:
        """Latest rows that are still waiting for job results."""

        return [
            _agent_row(row)
            for row in self._latest_by_n().values()
            if row.get("status") == "submitted" and row.get("job_id")
        ]

    def _run_trial(
        self,
        *,
        source: str,
        slug: str,
        hypothesis: str,
        is_baseline: bool,
        seed: int | None = None,
    ) -> dict[str, Any]:
        self._validate_source(source)
        slug = _slugify(slug)
        n, trial_dir = self._reserve(source, slug, hypothesis, is_baseline)
        t0 = time.monotonic()
        try:
            result = self.submitter(
                path=trial_dir / "train.py",
                train_budget_s=self.train_budget_s,
                slug=slug,
                n=n,
                run_id=self.run_id,
                seed=seed,
            )
        except Exception as exc:  # noqa: BLE001 - host infra failure, not research
            self._finalize(
                n=n,
                slug=slug,
                hypothesis=hypothesis,
                trial_dir=trial_dir,
                result={
                    "status": "infra_error",
                    "val_bpb": None,
                    "stdout": "",
                    "stderr": f"{type(exc).__name__}: {exc}",
                    "returncode": None,
                    "elapsed_s": time.monotonic() - t0,
                },
                is_baseline=is_baseline,
            )
            raise RuntimeError(
                "Experiment infrastructure failed after startup preflight. "
                "Stop this run and fix the host runner; this is not a research result."
            ) from exc
        row = self._finalize(
            n=n,
            slug=slug,
            hypothesis=hypothesis,
            trial_dir=trial_dir,
            result=result,
            is_baseline=is_baseline,
        )
        if row["status"] in {"crashed", "oom", "timeout"}:
            raise ExperimentCrashed(_agent_row(row))
        return _agent_row(row)

    def _reserve(
        self, source: str, slug: str, hypothesis: str, is_baseline: bool
    ) -> tuple[int, Path]:
        with self.lock:
            by_slug = self._latest_by_slug_locked()
            if is_baseline and "baseline" in by_slug:
                row = by_slug["baseline"]
                return int(row["n"]), Path(row["trial_dir"])
            if not is_baseline and slug in by_slug:
                raise ValueError(f"slug already exists in ledger: {slug}")
            if not is_baseline and self.max_submissions is not None:
                used = self._submission_count_locked()
                if used >= self.max_submissions:
                    raise SubmissionError(
                        f"too many submissions: max_submissions={self.max_submissions}"
                    )
            n = self._next_n_locked()
            trial_dir = self.history_dir / f"{n:03d}_{slug}"
            trial_dir.mkdir(parents=True, exist_ok=False)
            source_path = trial_dir / "train.py"
            source_path.write_text(source)
            row = {
                "n": n,
                "slug": slug,
                "hypothesis": hypothesis,
                "status": "submitted",
                "trial_dir": str(trial_dir),
                "source_path": str(source_path),
                "gpu": self.gpu,
                "ts": time.time(),
            }
            self._append_locked(row)
            return n, trial_dir

    def _finalize(
        self,
        *,
        n: int,
        slug: str,
        hypothesis: str,
        trial_dir: Path,
        result: dict[str, Any],
        is_baseline: bool,
    ) -> dict[str, Any]:
        stdout = str(result.get("stdout") or "")
        stderr = str(result.get("stderr") or "")
        stdout_path = trial_dir / "stdout.txt"
        stderr_path = trial_dir / "stderr.txt"
        stdout_path.write_text(stdout)
        stderr_path.write_text(stderr)
        val_bpb = result.get("val_bpb")
        if val_bpb is not None:
            val_bpb = float(val_bpb)
        status = result.get("status") or classify_result(
            int(result.get("returncode") or 0), stderr, val_bpb
        )
        if status == "finished":
            status = classify_result(int(result.get("returncode") or 0), stderr, val_bpb)
        row = {
            "n": n,
            "slug": slug,
            "hypothesis": hypothesis,
            "status": status,
            "val_bpb": val_bpb,
            "score": -val_bpb if val_bpb is not None else None,
            "elapsed_s": float(result.get("elapsed_s") or 0.0),
            "gpu": result.get("gpu") or self.gpu,
            "source_path": str(trial_dir / "train.py"),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
            "job_id": result.get("job_id"),
            "returncode": result.get("returncode"),
            "trial_dir": str(trial_dir),
            "is_baseline": is_baseline,
            "ts": time.time(),
        }
        with self.lock:
            self._append_locked(row)
        return row

    def _validate_source(self, source: str) -> None:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty complete train.py string")
        compile(source, "train.py", "exec")

    def _read_rows_locked(self) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        rows = []
        for line in self.ledger_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if _is_current_ledger_row(row):
                rows.append(row)
        return rows

    def _append_locked(self, row: dict[str, Any]) -> None:
        with self.ledger_path.open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    def _latest_by_n(self) -> dict[int, dict[str, Any]]:
        with self.lock:
            rows = self._read_rows_locked()
        latest: dict[int, dict[str, Any]] = {}
        for row in rows:
            latest[int(row.get("n", 0))] = row
        return latest

    def _latest_by_slug_locked(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for row in self._read_rows_locked():
            latest[str(row.get("slug"))] = row
        return latest

    def _next_n_locked(self) -> int:
        rows = self._read_rows_locked()
        return max((int(row.get("n", -1)) for row in rows), default=-1) + 1

    def _submission_count_locked(self) -> int:
        return sum(
            1
            for row in self._latest_by_slug_locked().values()
            if row.get("slug") != "baseline"
        )


def build_tools(state: AutoresearchState) -> list[Any]:
    """Create the tiny tool set exposed to researcher agents."""

    @tool(
        "Read README.md, train.py, pyproject.toml, or archived history files.",
        proxy=True,
    )
    def read_file(path: str) -> str:
        return state.allowed_read(path)

    @tool("Run the original train.py baseline once. Idempotent.", proxy=True)
    def run_baseline() -> dict[str, Any]:
        return state.run_baseline()

    @tool(
        "Submit one complete train.py source string as an experiment. "
        "Slug must be unique. Returns a submitted ledger row.",
        proxy=True,
    )
    def run_experiment(
        source: str, slug: str, hypothesis: str = "", seed: int | None = None
    ) -> dict[str, Any]:
        return state.run_experiment(source, slug, hypothesis, seed)

    @tool("Compact best-first ledger view.")
    def list_runs() -> list[dict[str, Any]]:
        return state.list_runs()

    @tool("Best successful/scored run only. Returns None if every run failed.")
    def best_run() -> dict[str, Any] | None:
        return state.best_run()

    @tool("Latest ledger rows keyed by slug. Use this to avoid duplicate slugs.")
    def get_runs() -> dict[str, dict[str, Any]]:
        return state.get_runs()

    @tool("Full latest ledger row for trial number n, including log tails.")
    def get_run(n: int) -> dict[str, Any] | None:
        return state.get_run(n)

    @tool("Most recent latest ledger row by trial number.")
    def latest_run() -> dict[str, Any] | None:
        return state.latest_run()

    @tool("Submission budget, excluding the idempotent baseline.")
    def submission_status() -> dict[str, int | None]:
        return state.submission_status()

    @tool("Refresh submitted jobs and update the ledger with completed results.")
    def refresh_results() -> list[dict[str, Any]]:
        return state.refresh_results()

    return [
        ExperimentCrashed,
        SubmissionError,
        read_file,
        run_baseline,
        run_experiment,
        list_runs,
        best_run,
        get_runs,
        get_run,
        latest_run,
        submission_status,
        refresh_results,
    ]


def build_llm(model: str):
    return (
        rflow.AnthropicClient(model)
        if model.startswith("claude")
        else rflow.OpenAIClient(model)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal autoresearch.")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--gpu", default="L4")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument(
        "--job-timeout-s",
        "--train-budget-s",
        dest="job_timeout_s",
        type=int,
        default=420,
        help="Wall-clock timeout for one submitted train.py job. train.py keeps its own fixed 5-minute training budget.",
    )
    parser.add_argument("--max-submissions", type=int, default=16)
    parser.add_argument("--app-name", default="rlmflow-autoresearch")
    parser.add_argument("--modal-timeout-s", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Run directory. Defaults to examples/_runs/autoresearch/<timestamp>.",
    )
    parser.add_argument("--max-iters", type=int, default=40)
    parser.add_argument("--child-iters", type=int, default=6)
    parser.add_argument("--no-live", action="store_true", help="Disable the live tree view.")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only refresh submitted job results for an existing --out run directory.",
    )
    parser.add_argument(
        "--collect-timeout-s",
        type=float,
        default=0.0,
        help="Seconds to wait per submitted job while collecting results.",
    )
    parser.add_argument(
        "--no-wait-for-results",
        action="store_true",
        help="Do not wait for submitted jobs before writing the final report.",
    )
    parser.add_argument(
        "--report-timeout-s",
        type=float,
        default=None,
        help="Maximum total seconds to wait for submitted jobs before reporting.",
    )
    parser.add_argument(
        "--report-poll-s",
        type=float,
        default=30.0,
        help="Seconds between submitted-job result polls.",
    )
    args = parser.parse_args()

    if args.max_submissions < 0:
        raise SystemExit("--max-submissions must be >= 0")
    if args.parallel < 1:
        raise SystemExit("--parallel must be >= 1")
    if args.collect_only and args.out is None:
        raise SystemExit("--collect-only requires --out")
    if args.out is None:
        args.out = (
            Path(__file__).resolve().parents[1]
            / "_runs"
            / "autoresearch"
            / time.strftime("%Y%m%d-%H%M%S")
        )

    example_dir = Path(__file__).resolve().parent
    timeout_s = args.modal_timeout_s or args.job_timeout_s + 120
    default_report_timeout_s = timeout_s * max(
        1, (args.max_submissions + args.parallel - 1) // args.parallel
    )
    report_timeout_s = (
        0.0
        if args.no_wait_for_results
        else (
            args.report_timeout_s
            if args.report_timeout_s is not None
            else default_report_timeout_s
        )
    )
    modal_config = ModalConfig(
        app_name=args.app_name,
        gpu=args.gpu,
        parallel=args.parallel,
        timeout_s=timeout_s,
    )

    def submit_trial(**kwargs: Any) -> dict[str, Any]:
        return submit(modal_config, **kwargs)

    def collect_trial(**kwargs: Any) -> dict[str, Any]:
        return collect(modal_config, **kwargs)

    state = AutoresearchState(
        example_dir=example_dir,
        out_dir=args.out,
        submitter=submit_trial,
        collector=collect_trial,
        train_budget_s=args.job_timeout_s,
        max_submissions=args.max_submissions,
        gpu=args.gpu,
    )

    if args.collect_only:
        write_run_report(
            state,
            args.out,
            collect_timeout_s=args.collect_timeout_s,
            wait_timeout_s=report_timeout_s,
            poll_interval_s=args.report_poll_s,
        )
        return

    print("[autoresearch] preflighting experiment image...", flush=True)
    try:
        preflight_result = preflight(modal_config)
    except Exception as exc:  # noqa: BLE001 - fail before the LLM sees anything
        raise SystemExit(f"autoresearch infrastructure preflight failed: {exc}") from exc

    config = {
        "model": args.model,
        "gpu": args.gpu,
        "parallel": args.parallel,
        "job_timeout_s": args.job_timeout_s,
        "max_submissions": args.max_submissions,
        "app_name": args.app_name,
        "modal_timeout_s": timeout_s,
        "run_id": state.run_id,
        "preflight": preflight_result,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    task_instructions = (example_dir / "program.md").read_text()

    runtime = rflow.LocalRuntime(working_directory=args.out)
    runtime.register_tools(build_tools(state))
    flow = rflow.Flow(
        build_llm(args.model),
        runtime=runtime,
        max_depth=1,
        max_iters=args.max_iters,
        child_max_iters=args.child_iters,
        max_concurrency=args.parallel,
        prompt_builder=build_prompt_builder(),
    )

    query = f"""\
Kick off the autoresearch loop for up to {args.max_submissions} submissions.

Use INPUTS["task_instructions"] as the research task. Start with the baseline,
then loop: refresh results, inspect the ledger, and launch up to {args.parallel}
child trials per batch until the submission budget is exhausted.
"""
    print(f"[autoresearch] out={args.out}", flush=True)
    print(f"[autoresearch] model={args.model} gpu={args.gpu}", flush=True)
    graph = flow.start(query, inputs={"task_instructions": task_instructions})
    if args.no_live:
        while not graph.finished:
            graph = flow.step(graph)
            print(graph.tree(), flush=True)
    else:
        from rflow.utils.viz import live

        graph = live(flow, graph)[-1]
    graph.save(args.out / "graph")
    print(graph.result())
    write_run_report(
        state,
        args.out,
        collect_timeout_s=args.collect_timeout_s,
        wait_timeout_s=report_timeout_s,
        poll_interval_s=args.report_poll_s,
    )


def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return (value.strip("_") or "trial")[:80]


def _tail(value: str, limit: int = 4000) -> str:
    return value[-limit:]


def _rank_key(row: dict[str, Any]) -> tuple[int, float, int]:
    score = row.get("score")
    if score is not None:
        return (0, -float(score), int(row.get("n", 0)))
    if row.get("status") == "submitted":
        return (1, 0.0, int(row.get("n", 0)))
    return (2, 0.0, int(row.get("n", 0)))


def _summary_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "n",
        "slug",
        "status",
        "val_bpb",
        "score",
        "elapsed_s",
        "source_path",
        "stdout_path",
        "stderr_path",
        "job_id",
        "hypothesis",
    )
    out = {key: row.get(key) for key in keys}
    if row.get("status") != "succeeded":
        out["stderr_tail"] = _tail(str(row.get("stderr_tail") or ""), 1200)
        out["stdout_tail"] = _tail(str(row.get("stdout_tail") or ""), 800)
    return out


def wait_for_results(
    state: AutoresearchState,
    *,
    timeout_s: float,
    poll_interval_s: float,
    collect_timeout_s: float = 0,
) -> int:
    """Poll submitted jobs until they finish or the report deadline expires."""

    if timeout_s <= 0:
        return len(state.refresh_results(timeout_s=collect_timeout_s))

    deadline = time.monotonic() + timeout_s
    refreshed = 0
    while True:
        updated = state.refresh_results(timeout_s=collect_timeout_s)
        refreshed += len(updated)
        for row in updated:
            print(
                "[autoresearch] result "
                f"n={row.get('n')} slug={row.get('slug')} "
                f"status={row.get('status')} val_bpb={row.get('val_bpb')}",
                flush=True,
            )

        pending = state.submitted_runs()
        if not pending:
            return refreshed

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(
                f"[autoresearch] report timeout with {len(pending)} jobs still submitted",
                flush=True,
            )
            return refreshed

        print(
            f"[autoresearch] waiting for {len(pending)} submitted jobs...",
            flush=True,
        )
        time.sleep(max(1.0, min(poll_interval_s, remaining)))


def write_run_report(
    state: AutoresearchState,
    out_dir: Path,
    *,
    collect_timeout_s: float = 0,
    wait_timeout_s: float = 0,
    poll_interval_s: float = 30,
) -> None:
    """Write and print a host-side ledger report with log paths."""

    refreshed = wait_for_results(
        state,
        timeout_s=wait_timeout_s,
        poll_interval_s=poll_interval_s,
        collect_timeout_s=collect_timeout_s,
    )
    rows = state.list_runs()
    best = state.best_run()
    report = {"best": best, "runs": rows}
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    markdown_path = out_dir / "README.md"
    markdown_path.write_text(_markdown_report(best, rows, state.ledger_path, report_path))

    print("\n[autoresearch] run report")
    print(f"[autoresearch] report={report_path}")
    print(f"[autoresearch] readme={markdown_path}")
    print(f"[autoresearch] ledger={state.ledger_path}")
    if refreshed:
        print(f"[autoresearch] refreshed={refreshed}")
    if best is None:
        print("[autoresearch] best=None (no successful val_bpb yet)")
    else:
        print(
            "[autoresearch] best="
            f"{best.get('slug')} val_bpb={best.get('val_bpb')} "
            f"source={best.get('source_path')}"
        )
    for row in rows:
        status = row.get("status")
        print(
            "[autoresearch] run "
            f"n={row.get('n')} slug={row.get('slug')} status={status} "
            f"val_bpb={row.get('val_bpb')}"
        )
        if row.get("job_id"):
            print(f"  job_id: {row.get('job_id')}")
        print(f"  source: {row.get('source_path')}")
        print(f"  stdout: {row.get('stdout_path')}")
        print(f"  stderr: {row.get('stderr_path')}")
        if status != "succeeded" and row.get("stderr_tail"):
            print("  stderr_tail:")
            for line in str(row["stderr_tail"]).splitlines()[-20:]:
                print(f"    {line}")


def _markdown_report(
    best: dict[str, Any] | None,
    rows: list[dict[str, Any]],
    ledger_path: Path,
    report_path: Path,
) -> str:
    lines = [
        "# Autoresearch Run Report",
        "",
        f"- Ledger: `{ledger_path}`",
        f"- JSON report: `{report_path}`",
        f"- Runs: {len(rows)}",
    ]
    if best is None:
        lines.append("- Best: none yet")
    else:
        lines.append(
            "- Best: "
            f"`{best.get('slug')}` val_bpb={_format_value(best.get('val_bpb'))} "
            f"source=`{best.get('source_path')}`"
        )

    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| n | slug | status | val_bpb | score | source |",
            "|---:|---|---|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row.get('n')} | "
            f"`{row.get('slug')}` | "
            f"{row.get('status')} | "
            f"{_format_value(row.get('val_bpb'))} | "
            f"{_format_value(row.get('score'))} | "
            f"`{row.get('source_path')}` |"
        )

    failed = [
        row
        for row in rows
        if row.get("status") not in {"succeeded", "submitted"}
    ]
    if failed:
        lines.extend(["", "## Failures", ""])
        for row in failed:
            lines.append(
                f"- `n={row.get('n')}` `{row.get('slug')}` "
                f"status={row.get('status')} stderr=`{row.get('stderr_path')}`"
            )
            tail = str(row.get("stderr_tail") or "").strip()
            if tail:
                lines.extend(["", "```text", _tail(tail, 1200), "```", ""])

    submitted = [row for row in rows if row.get("status") == "submitted"]
    if submitted:
        lines.extend(["", "## Still Submitted", ""])
        for row in submitted:
            lines.append(f"- `n={row.get('n')}` `{row.get('slug')}`")

    return "\n".join(lines).rstrip() + "\n"


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _is_current_ledger_row(row: object) -> bool:
    """Return True for rows written by this runner's ledger schema."""

    if not isinstance(row, dict):
        return False
    return isinstance(row.get("n"), int) and isinstance(row.get("slug"), str)


def _agent_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return only research-facing row fields."""

    keys = {
        "n",
        "slug",
        "hypothesis",
        "status",
        "val_bpb",
        "score",
        "elapsed_s",
        "source_path",
        "stdout_path",
        "stderr_path",
        "stdout_tail",
        "stderr_tail",
        "returncode",
        "job_id",
        "is_baseline",
        "ts",
    }
    return {key: row.get(key) for key in keys if key in row}


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
