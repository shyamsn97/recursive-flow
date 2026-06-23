# autoresearch

This example gives agents a small, real language-model training loop and lets
them submit variations. The baseline is intentionally boring: a bounded
TinyStories sample, byte-level tokens, a compact GPT, AdamW, and a fixed time
budget.

## How it works

The repo is deliberately kept small and only really has two files that matter:

- **`train.py`** — the single file the agent edits. Contains TinyStories cache setup, byte tokenization, batch sampling, BPB-style evaluation, GPT model, AdamW, and the training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), and ideally a CUDA GPU. CPU works for smoke tests but is slow.

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Manually run a single training experiment (~5 min, plus one-time cache setup)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
train.py        — TinyStories data, eval, model, optimizer, training loop
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single training file.** The agent only touches `train.py`, and `train.py` is self-contained so candidates do not depend on helper modules.
- **Fixed time budget.** Training runs for a fixed wall-clock budget, so experiments are directly comparable.
- **Small baseline.** A bounded TinyStories sample and byte-level tokens keep the example understandable. The model is not meant to be state of the art; it is meant to be easy to improve.

## License

MIT
