# Autoresearch Task

You are an autonomous researcher trying to lower `val_bpb` for the training
script in this directory.

The experiment contract is intentionally simple:

1. Read `INPUTS["task_instructions"]`, `README.md`, and `train.py`.
2. Establish the baseline with `run_baseline()`.
3. Submit complete replacement `train.py` files with
   `run_experiment(source, slug, hypothesis)`.
4. Use `refresh_results()`, `list_runs()`, `get_runs()`, `get_run(n)`,
   `latest_run()`, and `best_run()` as the ledger.

## Loop

Repeat until `submission_status()["remaining_submissions"] == 0`:

1. Call `refresh_results()`.
2. Review `best_run()`, `list_runs()`, and `get_runs()`.
3. Pick a small, diverse batch of new ideas with unique slugs.
4. Launch child agents with `launch_subagents`, one idea per child.
5. Each child writes one complete `train.py` candidate and submits it.
6. After a batch, refresh results again and choose the next batch based on what
   scored, crashed, or remained pending.

Do not use git, branches, shell commands, `results.tsv`, or filesystem writes.
The controller archives candidate files and submits training jobs.

## The Only File That Matters

`train.py` is self-contained. It owns:

- TinyStories download/cache setup,
- byte-level tokenization,
- random batch sampling,
- BPB evaluation,
- a small GPT model,
- AdamW and the training loop.

Do not import helper functions from any other local file. A candidate must run as:

```bash
python -u train.py
```

It must print a final summary containing:

```text
val_bpb: <float>
```

Lower `val_bpb` is better.

## What To Change

You may change model architecture, depth, width, attention pattern,
normalization, activation, optimizer settings, learning-rate schedule, batch
size, gradient accumulation, initialization, regularization, and training-loop
details inside `train.py`.

Keep these fixed unless you have a very strong reason:

- The BPB evaluation logic and printed `val_bpb` format.
- The TinyStories data source and cache path.
- The dependency set in `pyproject.toml`.

Do not add dependencies, make new network calls beyond the existing data-cache
setup, call subprocesses, introduce alternate CLIs, or submit unchanged code
under a new slug.

## Running Trials

If `run_experiment` returns `status: "submitted"`, the job was accepted. Report
the slug, hypothesis, and source path, then stop that child.

Before deciding what is best or launching another batch, call
`refresh_results()` to update any completed submitted jobs.

If a completed row later has `status: "crashed"`, `status: "oom"`, or
`status: "timeout"`, inspect `stderr_tail` through `get_run(n)` or
`latest_run()`. Make at most one targeted fix for obvious syntax/import/runtime
bugs. Do not retry slow, oversized, or fundamentally broken ideas unchanged.

## Good First Ideas

- Tune warmup, warmdown, max LR, or final LR fraction.
- Try optimizer variants.
- Trade depth for width, or width for depth.
- Adjust effective batch size or gradient accumulation.
- Try different attention implementations or positional embeddings.
- Test normalization and activation variants.
- Add or remove dropout and weight decay carefully.
- Simplify components if the score stays competitive.

The parent should keep a diverse portfolio. Before launching a batch, call
`submission_status()` and avoid exceeding the remaining submission budget. Use
unique lowercase slugs like `lr_warmdown_tune`, `depth6_wider`, `adamw_only`, or
`window_l_only`.
