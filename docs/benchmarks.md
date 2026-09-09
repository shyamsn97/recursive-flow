# Benchmarks

Eval is organized around **sets**. A set is a question, not a pile of adapters.

```bash
python -m benchmarks.eval --list-sets
make eval-reasoning EVAL_MODEL=gpt-5-mini
```

Pass a set name to `--dataset`. Runners default to that set’s table. Membership is [`benchmarks/eval/sets.py`](https://github.com/shyamsn97/rlmflow/blob/main/benchmarks/eval/sets.py).

## Pick a set

| Set            | Question                                              | Members                                              | Runners                                  |
| -------------- | ----------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------- |
| `smoke`        | Does the harness run?                                 | `synthetic_needle`                                   | `fake` `vanilla` `rlmflow-local`         |
| `reasoning`    | Hard problems, short context                          | AIME 2025, Sudoku Extreme                            | `vanilla` `rlmflow-local` `official-rlm` |
| `long-context` | Recursive loop vs vanilla on long documents           | S-NIAH, OOLONG, OOLONG-Pairs (32K paper set), CodeQA | `vanilla` `rlmflow-local` `official-rlm` |
| `delegation`   | Cheapest correct route: local / batched query / spawn | `delegation-routing` (8 pinned cases)                | `rlmflow-local` `official-rlm`           |
| `task-graph`   | Frozen 20-problem telemetry                           | the `delegation_*` adapters                          | `rlmflow-local`                          |
| `research`     | Deep-research QA over a fixed collection              | BrowseComp-Plus                                      | `vanilla` `rlmflow-local` `official-rlm` |
| `code`         | Code that passes hidden tests                         | LiveCodeBench                                        | `vanilla` `rlmflow-local` `official-rlm` |

Vanilla should stay close to rlmflow on `reasoning`. `long-context` is where a recursive REPL should pull away. `delegation` grades the factoid; a child on a local or batched-query item is a miss even if the answer is right. Do not mix these questions in one run — that was the `rlm-core` problem.

## Run

```bash
make eval-smoke
make eval-reasoning EVAL_MODEL=gpt-5-mini
make eval-long-context EVAL_MODEL=gpt-5-mini
make eval-delegation EVAL_MODEL=gpt-5-mini
```

```bash
python -m benchmarks.eval \
  --model openai:gpt-5-mini \
  --dataset reasoning \
  --seed 0 \
  --limit 5
```

`--limit N` selects fewer **examples**. It never truncates an example’s prompt or context. `--full` uses every example the adapter exposes. `--runner` overrides the set default.

Every run writes `benchmarks/eval/runs/<run_id>/` (`config.json`, `rows.jsonl`, `summary.json`, `report.md`, `artifacts/`). Re-score without tokens: `python -m benchmarks.eval.regrade RUN`. Per-question grid: `python -m benchmarks.eval.matrix RUN`. Pair two runs: `make eval-compare EVAL_RLMFLOW_RUN=... EVAL_OFFICIAL_RUN=...`.

Modal fans out **rows** `(example, runner, seed)`, not graph steps:

```bash
python -m benchmarks.eval --dataset reasoning --limit 5 --executor modal --parallel 10 --wandb
```

## Notes

- **`oolong-pairs`** is the paper’s 20 queries at 32K. It is not `oolong_pairs` (any length) and not the four 8K queries inside `delegation`.
- **`delegation`** is routing, not spawn-lift. The lift A/B is [Delegation problem set](delegation_problems.md) (`delegation-lift`, not wired yet).
- **`task-graph`** is the frozen 20 (manifest in `benchmarks/eval/delegation/manifest.py`). Use it for launch/wait/join traces.
- **`research`**: download BrowseComp-Plus once to `evals/data/browsecomp_plus`.
- **`code`**: LiveCodeBench scores generated code in a hardened Docker container, never on the host.

Working packs `delegation-iteration-five`, `-ten`, and `delegation-regression-five` still exist as dataset names. They are not sets. `rlm-core`, `needle`, and `delegation-suite` still expand.

## Harness

Dataset / Model / Runner / Logger. `rlmflow-local` does `start(query=example.prompt, inputs=...)` then `flow.run_streaming(root)`. Register with `@dataset` / `@runner` and import from the package `__init__.py`. Put a dataset on a named set by adding it to `SETS` in `sets.py`.
