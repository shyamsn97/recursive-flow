# Tinker autoresearch — LoRA SFT on TinyStories

You are running an autoresearch hill-climb on `train.py` in this directory. The metric is **`val_bpb`** (bits-per-byte on a held-out TinyStories slice), lower is better.

## Setup

The harness is fixed:

- `prepare.py` — downloads TinyStories, caches `data/train.txt` and `data/val.txt`. Run once: `uv run prepare.py` (or `python prepare.py`). **Do not edit.**
- `train.py` — Tinker LoRA SFT on TinyStories. Prints metrics each step and ends with `val_bpb: <float>` on the last line. **This is the only file you edit.**
- Tinker requires `TINKER_API_KEY` in the environment. Confirm it's set before doing anything else.

## What you can change in `train.py`

- `BASE_MODEL` — any Tinker-supported base. Smaller is faster (`Qwen/Qwen3-0.6B-Base`, `meta-llama/Llama-3.2-1B`).
- `LORA_RANK` — capacity vs speed trade.
- `LR`, `BETA1`, `BETA2`, `EPS` — Adam params.
- `BATCH_SIZE`, `SEQ_LEN`, `MAX_STEPS` — compute knobs.
- LR schedule shape (currently linear-decay).
- Loss masking, sequence packing, dataset filtering.
- The eval loop itself if you have a better val_bpb estimator (just don't fake the number).

## Loop

1. Read the relevant copy's `train.py` (`target/train.py` for the parent, `trials/<name>/train.py` for a child). Form a hypothesis: "X should help because Y."
2. Edit only that copy's `train.py`.
3. `run_experiment(path=..., budget_s=300)` — runs `python train.py` in that copy under a wall-clock timeout. Returns `{val_bpb, returncode, stdout_tail, stderr_tail}`.
4. If `val_bpb` improved over the last commit: `git_op("commit -am '<hypothesis>: <delta>'", path=...)`. If it got worse or crashed: `git_op("reset --hard", path=...)`.
5. Repeat. Keep a brief journal of what worked.

## Rules

- **Never invent numbers.** Every reported `val_bpb` comes from `run_experiment` stdout.
- **Verify the run finished.** If `returncode != 0` or stdout is missing `val_bpb:`, treat the experiment as failed and reset.
- **On failure, read `stderr_tail`.** When `returncode != 0` the actual error message is in `stderr_tail`, not `stdout_tail`. Always quote `stderr_tail` in your journal / `done()` message — otherwise debugging is impossible.
- **Tinker calls cost money.** Keep `MAX_STEPS` honest — don't crank it to 10× to brute-force a win.
- **Parallel children:** each child works in its own workspace trial copy under `trials/<name>/` (don't fight over one `train.py`). The parent merges the best returned `train_py` into `target/train.py`.
- **Commit messages = your journal.** Future-you will read `git_op("log --oneline")` to figure out what's been tried.

## Sanity checklist before delegating

- `python prepare.py` finished and `data/train.txt` exists?
- `TINKER_API_KEY` set?
- A baseline `python train.py` ran end-to-end and printed a `val_bpb`? (Don't start hill-climbing until you have a baseline number.)
