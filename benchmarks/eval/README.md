# Shared Eval Harness

`benchmarks/eval/` is the shared runner for deterministic rflow evaluations.
It is adapted from [`avilum/minrlm/eval`](https://github.com/avilum/minrlm/tree/master/eval):
the task registry, runner registry, normalized result rows, metrics aggregation,
and single CLI shape all come from that design. This version saves rflow
artifacts as `Graph.save()` run directories and records graph-shape metrics.

## Quick Smoke

```bash
make eval-smoke
```

Only `eval-smoke` uses the local deterministic fake client. It writes
`results.jsonl` and `summary.json`, and exercises tqdm progress bars without API
keys.

Equivalent direct command:

```bash
python -m benchmarks.eval --provider fake --model fake --tasks sniah --runners fake vanilla rflow --seeds 0:3
```

## Real RLM-Bench Runs

Install the eval extras for progress bars and W&B:

```bash
pip install -e ".[eval,openai]"
```

Run one real task:

```bash
make eval-run
```

By default this runs:

```text
provider: openai
model: gpt-5-mini
tasks: official_sniah
runners: vanilla rflow official
seeds: 0:10
```

W&B is the same real run with logging enabled:

```bash
make eval-wandb
```

The W&B run records the task, runner, model, and seed list in config. During
the run it logs live `overall/accuracy`, `overall/score`, count, errors, and
per `by_runner_task/<runner>/<task>/...` metrics. At the end it also publishes
`tables/results` for every scored row and `tables/summary` for the aggregate
breakdown.

To reproduce the GPT-5-mini benchmark table shape from
[`avilum/minrlm/eval/BENCHMARK.md`](https://github.com/avilum/minrlm/blob/master/eval/BENCHMARK.md),
run the full official task sweep:

```bash
make eval-benchmark
```

That expands to all official tasks, `rflow vanilla official`, and 50 seeds per
task. It logs W&B metrics and writes `summary.json` with `by_runner`,
`by_task`, `by_runner_task`, and `tasks_won` sections for leaderboard-style
tables. Use `EVAL_MODEL=gpt-5-nano`, `EVAL_MODEL=gpt-5.4-mini`, or
`EVAL_MODEL=gpt-5.2` to reproduce the other model rows.

Run the official RLM-Bench task set ported from
[`avilum/minrlm/eval`](https://github.com/avilum/minrlm/tree/master/eval):

```bash
make eval-run EVAL_TASKS=official EVAL_MODEL=gpt-4o-mini EVAL_SEEDS=0:50
```

The `official` alias expands to:

```text
official_sniah
official_oolong
official_longbench_v2
official_codeqa
official_repoqa
official_browsecomp
official_gdpval
official_aime_2025
official_gpqa_diamond
official_mmlu_pro
official_ifeval
official_livecodebench
official_sudoku_extreme
```

Direct CLI form:

```bash
python -m benchmarks.eval \
  --provider openai \
  --model gpt-4o-mini \
  --tasks official \
  --runners vanilla rflow official \
  --seeds 0:50 \
  --official-data-dir evals/data
```

Rows are written under `benchmarks/eval/runs/<run_id>/results.jsonl`.
Per-run artifacts live under:

```text
benchmarks/eval/runs/<run_id>/artifacts/<runner>/<task>/<task_id>/
```

For the `rflow` runner, `graph/` is rewritten after every step by default, so a
viewer or a crashed run can inspect the latest checkpoint.

Model-oriented reports are also written under `eval-runs/`:

```text
eval-runs/
  <model>/
    index.md
    <benchmark>/
      config.json
      summary.json
      report.md
      <task_id>.json
```

Each `<task_id>.json` stores the prompt, inputs/context, expected answer, and
one solution record per runner. `report.md` summarizes accuracy, score, tokens,
latency, errors, task wins, and links to the problem/solution JSON files. Use
`EVAL_REPORT_DIR=...` or `--report-dir ...` to change the root directory.

Dataset notes:

- Hugging Face datasets are loaded at runtime through `datasets`; local
  `load_from_disk` copies under `evals/data/<dataset>/` are preferred when
  present.
- `official_repoqa` expects local RepoQA JSON/JSONL under `evals/data/repoqa/`.
- `official_gpqa_diamond` is gated; accept the Hugging Face license and log in.
- `official_gdpval` can require reference file parsing dependencies included in
  `.[eval]`.

## Adding Tasks

Tasks are registered through a small Gym-style registry:

```python
from benchmarks.eval.tasks import TASK_REGISTRY

task = TASK_REGISTRY.make("official_sniah", data_dir="evals/data")
all_official = TASK_REGISTRY.expand(["official"])
```

Add a task module under `benchmarks/eval/tasks/` and decorate it:

```python
from benchmarks.eval.tasks import register_task

@register_task("my_task")
class MyTask:
    def generate(self, seed: int, **kwargs) -> TaskInstance: ...
    def score(self, answer: str, expected: object, metadata: dict) -> Score: ...
```

`TaskInstance.inputs` is passed directly into `Flow.start(..., inputs=...)` for
recursive runs and rendered as plain text for vanilla runs.

The official RLM-Bench task port is split by domain:

```text
benchmarks/eval/tasks/
  registry.py            # TaskRegistry, TaskSpec, aliases
  synthetic.py           # local deterministic smoke task
  common.py              # shared dataset loading + scoring helpers
  long_context.py        # S-NIAH, OOLONG, LongBench-v2, RepoQA, BrowseComp+
  reasoning.py           # AIME, GPQA, MMLU-Pro, IFEval
  code.py                # LiveCodeBench, Sudoku Extreme
  work.py                # GDP Val
```

## Adding Runners

Add a runner under `benchmarks/eval/runners/` and decorate it:

```python
@register_runner("my_runner")
class MyRunner:
    def run(self, instance: TaskInstance, *, client, model, out_dir, max_iters, max_depth, live_save):
        ...
```

Return a `RunResult`; the CLI handles task scoring, JSONL output, summaries, and
W&B logging.

Built-in runners:

- `vanilla`: direct single-call LLM baseline.
- `rflow`: this repo's recursive-flow implementation.
- `official`: official RLM implementation from the paper, installed into a
  reusable temp venv from `git+https://github.com/alexzhang13/rlm`.
- `fake`: deterministic local runner for smoke tests.

## Make Targets

```bash
make eval-help
make eval-smoke
make eval-test
make eval-run EVAL_PROVIDER=openai EVAL_MODEL=gpt-4o-mini EVAL_RUNNERS="vanilla rflow official" EVAL_SEEDS=0:50
make eval-run EVAL_TASKS=official EVAL_MODEL=gpt-4o-mini EVAL_SEEDS=0:50
make eval-wandb EVAL_PROVIDER=openai EVAL_MODEL=gpt-4o-mini EVAL_RUNNERS="vanilla rflow official" EVAL_SEEDS=0:50
make eval-benchmark EVAL_MODEL=gpt-5-mini
make eval-clean
```

Common overrides:

```bash
EVAL_TASKS="sniah"
EVAL_RUNNERS="fake vanilla rflow official"
EVAL_SEEDS="0:10"
EVAL_BENCHMARK_TASKS="official"
EVAL_BENCHMARK_RUNNERS="rflow vanilla official"
EVAL_BENCHMARK_SEEDS="0:50"
EVAL_REPORT_DIR="eval-runs"
EVAL_OFFICIAL_DATA_DIR="evals/data"
EVAL_OFFICIAL_MAX_SAMPLES=100
EVAL_OFFICIAL_MAX_CONTEXT_TOKENS=131072
EVAL_BROWSECOMP_MAX_DOCS=20
EVAL_TASK_PARAMS="--task-param records=200 --task-param filler_words=8"
EVAL_ARGS="--run-id my-run --no-live-save"
```
