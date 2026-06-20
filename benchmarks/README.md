# benchmarks/

Runnable benchmark harnesses for recursive-flow.

The canonical harness is `benchmarks/eval/`. It is intentionally small and built
around four components:

- `Dataset` - yields examples and scores predictions.
- `Model` - wraps inference.
- `Runner` - executes an example (`vanilla`, `rflow-local`, `official-rlm`, etc.).
- `Logger` - writes JSONL, console output, reports, or W&B metrics.

Initial datasets:

- `synthetic_needle` - deterministic needle-in-haystack smoke task.
- `oolong` - first real long-context dataset.
- `official_longbench_v2` - LongBench-v2 all-domain multiple-choice/QA.
- `official_livecodebench` - LiveCodeBench code generation with public tests.
- `official_sudoku_extreme` - Sudoku Extreme solution checking.

## Running

```bash
python -m benchmarks.eval --help
```

Smoke:

```bash
make eval-smoke
```

Direct equivalent:

```bash
python -m benchmarks.eval \
  --model fake \
  --dataset synthetic_needle \
  --runner fake vanilla rflow-local \
  --seeds 0:3 \
  --dataset-param synthetic_needle.records=8 \
  --dataset-param synthetic_needle.filler_words=2 \
  --runner-param rflow-local.max_iters=3 \
  --runner-param rflow-local.max_depth=1
```

Real run:

```bash
python -m benchmarks.eval \
  --model openai:gpt-5-mini \
  --dataset oolong official_longbench_v2 official_livecodebench official_sudoku_extreme \
  --runner vanilla rflow-local official-rlm \
  --seeds 0:20 \
  --wandb
```

Modal parallel run:

```bash
python -m benchmarks.eval \
  --model openai:gpt-5-mini \
  --dataset oolong official_longbench_v2 official_livecodebench official_sudoku_extreme \
  --runner vanilla rflow-local official-rlm \
  --seeds 0:50 \
  --executor modal \
  --parallel 10 \
  --best-of-n 1 \
  --modal-cpu 1 \
  --wandb
```

Increase `--best-of-n` to duplicate each logical benchmark row and keep the
best-scoring attempt.

Every run writes:

```text
benchmarks/runs/<run_id>/
  config.json
  rows.jsonl
  summary.json
  report.md
  artifacts/<dataset>/<example_id>/<runner>/
```
