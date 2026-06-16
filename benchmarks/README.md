# benchmarks/

Runnable benchmark harnesses for rflow. The canonical harness is
`benchmarks/eval/`: tasks and runners register into one CLI, with consistent
artifacts, summaries, tqdm progress, and optional W&B logging.

## Conventions

- **Runtime.** Default to `LocalRuntime` for dev. Use sandboxed runtimes for
  untrusted benchmark prompts.
- **Budget.** Every task gets fixed `max_depth` and `max_iters` settings from
  the CLI; do not tune them per task.
- **Config.** Every run writes `config.json` with provider/model, task names,
  runner names, seeds, and task parameters.
- **Results.** Per-task rows go to `results.jsonl`; aggregate metrics go to
  `summary.json`; rflow artifacts go under `artifacts/<runner>/<task>/<task_id>/`.
- **Seeds.** Any sampling from a dataset is deterministic given `--seed`
  so partial reruns are reproducible.

## Layout

```
benchmarks/
  README.md              # this file
  eval/                  # shared task/runner harness with tqdm + W&B logging
```

## Running

```
python -m benchmarks.eval --help
```

```
python -m benchmarks.eval --provider fake --model fake --tasks sniah --runners fake rflow --seeds 0:3
```
