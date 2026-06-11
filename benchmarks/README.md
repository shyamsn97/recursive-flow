# benchmarks/

Runnable benchmark harnesses for rflow. Each subdirectory is a
self-contained driver for one public benchmark, with its own README,
runner, and scoring script.

## Conventions (shared across all benchmarks)

- **Runtime.** Default to `LocalRuntime` for dev; pass `--docker-image
  recursive-flow:local` for any serious run. Never run third-party benchmark
  prompts under `LocalRuntime` on a trusted machine.
- **Budget.** Every task gets a fixed `max_depth × max_iterations` and
  optional `max_budget` (total tokens). These are declared in the CLI and
  written to the run manifest — do not tune per-task.
- **Manifest.** Every run writes a `manifest.json` with:
  `{model, fast_model, max_depth, max_iterations, max_budget, split, n,
   seed, dataset_sha, runtime, timestamp, recursive-flow_version}`.
- **Results.** Per-task rows go to `results.jsonl`; aggregate metrics to
  `summary.json`; full traces to `traces/<task_id>/` via
  `rflow.utils.trace.save_trace`.
- **Seeds.** Any sampling from a dataset is deterministic given `--seed`
  so partial reruns are reproducible.

## Layout

```
benchmarks/
  README.md              # this file
  comparison/            # recursive-flow vs alexzhang13/rlm synthetic smoke comparison
  oolong/                # RLM paper: long-context aggregation
  ...                    # more suites added over time
```

## Running any benchmark

```
python benchmarks/<name>/run.py --help
```

Each driver shares the example-style flags (`--model`, `--fast-model`,
`--docker-image`, `--max-depth`, `--max-iterations`) plus its own
dataset flags (`--split`, `--n`, `--seed`, ...).

## Why these and not others

Short version: reproduce the RLM paper's quartet first (OOLONG /
OOLONG-Pairs / LongBench-v2 CodeQA / BrowseComp-Plus), then add one
coding and one reasoning anchor for industry credibility. Everything
else is skipped until that story is solid.
