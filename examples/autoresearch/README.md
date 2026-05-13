# Autoresearch

A Karpathy-style autoresearch hill-climb wired through RLMFlow. The agent
mutates a `train.py` in a target directory, runs it under a wall-clock
budget, parses `val_bpb` from stdout, and uses git to commit the wins and
reset the losses. Children fan out parallel mutations; the parent keeps
the best diff.

```
edit train.py  →  python train.py (budget_s)  →  read val_bpb  →
    if better:  git_op("commit -am ...")    (keep)
    else:       git_op("reset --hard")      (discard)
repeat
```

This is a faithful port of [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
to the RLMFlow recursion / multi-agent shape: instead of one agent in a
single context window, the lead delegates each mutation to a fresh child
that does the edit + run + measure in isolation. See [Karpathy's tweet](https://x.com/karpathy/status/1953240953555730808)
for the original motivation.

```
examples/autoresearch/
├── README.md              # this file
├── autoresearch.py        # driver — registers run_experiment + git_op tools, runs RLMFlow
└── tinker/                # one example target: Tinker LoRA SFT on TinyStories
    ├── README.md
    ├── program.md
    ├── prepare.py
    └── train.py
```

## How it works

`autoresearch.py` is a generic driver. It takes a `--target` directory
that must contain:

- `train.py` — the only file the agent edits. It must end its stdout
  with `val_bpb: <float>` (lower is better). Anything before that is
  free-form.
- `program.md` — the human's operating manual for the agent (its
  "skill"). The driver reads this and folds it into the root query.
- a `git` repo (the driver uses `git commit`/`reset` for memory).

The driver then registers two tools alongside the standard file tools:

| Tool | What it does |
|---|---|
| `run_experiment(budget_s=300)` | Runs `python train.py` in the target dir under a wall-clock timeout. Returns JSON: `{val_bpb, elapsed_s, returncode, stdout_tail, stderr_tail}`. `val_bpb` is regex-parsed from stdout (`val_bpb: <float>`); missing if the run crashed or didn't print it. |
| `git_op(args)` | Runs `git <args>` in the target dir. Used for `status`, `diff`, `commit -am '<msg>'`, `reset --hard`, `log --oneline`. |

The agent reads `program.md`, makes a hypothesis, edits `train.py`,
calls `run_experiment`, decides to keep or revert, and repeats.

### Driver flags

```bash
python examples/autoresearch/autoresearch.py [flags]
```

| Flag | Default | Meaning |
|---|---|---|
| `--target PATH` | required | Directory containing `train.py` + `program.md` + a git repo. |
| `--budget-s N` | `300` | Wall-clock seconds per `run_experiment` call. |
| `--rounds N` | `6` | Outer LLM-turn budget for the parent agent. |
| `--branches N` | `4` | Hint to the parent for how many parallel children to fan out per round. |
| `--model M` | `gpt-5` | Lead model. `claude*` → Anthropic, else OpenAI. |
| `--fast-model M` | `gpt-5-mini` | Cheap model registered as `"fast"` for child mutations. |
| `--workspace PATH` | `./runs/autoresearch` | Where RLMFlow persists the run (graph, sessions, viewer). |
| `--max-depth N` | `2` | Max delegation depth. |
| `--max-concurrency N` | `4` | Max sibling children stepped in parallel. |
| `--no-viewer` | off | Skip writing `viewer.html` at the end. |

## Quickstart — Karpathy's nanochat target

The original from-scratch pretraining loop. Requires a single NVIDIA GPU
(H100 recommended; smaller GPUs need the tweaks in
[Karpathy's README](https://github.com/karpathy/autoresearch#platform-support)).

```bash
# 1. Clone Karpathy's repo somewhere
git clone https://github.com/karpathy/autoresearch.git /tmp/karpathy-autoresearch

# 2. Set it up per his README (uv install, prepare.py, etc.)
cd /tmp/karpathy-autoresearch
uv sync
uv run prepare.py
uv run train.py    # smoke test — should print val_bpb: <float>

# 3. Point the RLMFlow driver at it
cd /Users/shyam/Code/rlmkit
export OPENAI_API_KEY="..."
python examples/autoresearch/autoresearch.py \
    --target /tmp/karpathy-autoresearch \
    --budget-s 300 --rounds 6 --branches 4
```

## Quickstart — Tinker target (no local GPU)

Hosted LoRA SFT on TinyStories via [Tinker](https://thinkingmachines.ai/tinker/).
No GPU required, but Tinker is paid — every `run_experiment` call costs
real money. Read [`tinker/README.md`](tinker/README.md) for the full
walkthrough; the short version:

```bash
# 1. Tinker SDK + dataset loader
pip install tinker datasets

# 2. API keys
export TINKER_API_KEY="..."        # for the training runs
export OPENAI_API_KEY="..."        # for the driving agent

# 3. Cache TinyStories text
cd examples/autoresearch/tinker
python prepare.py

# 4. Init a git repo so the driver can commit/reset
git init && git add . && git commit -m "tinker autoresearch baseline"

# 5. Smoke-test that the contract holds
python train.py
# should end with: val_bpb: <float>

# 6. Run the autoresearch driver
cd /Users/shyam/Code/rlmkit
python examples/autoresearch/autoresearch.py \
    --target examples/autoresearch/tinker \
    --budget-s 360 --rounds 4 --branches 2
```

Start with `--rounds 2 --branches 1` the first time to dry-run the loop
without burning Tinker quota.

## Bring your own target

Anything that runs in 5-10 minutes and exposes a single scalar metric
works. Three things to do:

1. **Make `train.py` print `val_bpb: <float>` on success.** Lower is
   better. The driver doesn't care whether it's literal bits-per-byte —
   it's just the metric name the regex looks for. If you'd rather
   optimize accuracy or another metric, edit `METRIC_RE` in
   `autoresearch.py` (e.g. `r"val_acc\s*[:=]\s*([0-9]+\.?[0-9]*)"`) and
   tell the agent in `program.md` whether higher or lower is better.
2. **Write a `program.md`.** This is the agent's skill. It should
   describe: what `train.py` does, what's mutable vs fixed, how to
   measure success, anti-patterns to avoid, the loop the agent should
   follow. Look at [`tinker/program.md`](tinker/program.md) for a
   working example.
3. **`git init`** the target directory. The driver uses `git commit` /
   `reset --hard` as the agent's memory; without a repo it can't
   keep wins.

That's it — point `--target` at your directory and run.

## Notes

- **Cost discipline.** Each `run_experiment` is real compute. Use
  `--budget-s` aggressively and start with small `--rounds`/`--branches`
  while you debug your `program.md`.
- **Parallel children.** `--branches N` is just a hint to the parent —
  the actual parallelism is gated by `--max-concurrency` and by whether
  your training setup tolerates concurrent runs (Tinker does; sharing
  one local GPU does not — drop `--branches` to 1 for nanochat-style
  targets).
- **Workspace replay.** Every run is persisted under `--workspace`. Open
  the viewer with `rlmflow view <workspace>` or just open the
  `viewer.html` written at the end. `graph.history()` will replay the
  full run including the parallel mutation rounds.
- **The contract.** `train.py` ↔ driver communication is
  one-directional: the driver hands a wall-clock budget, `train.py`
  prints `val_bpb: <float>` on the last line. Don't add hidden state
  files or sidechannels — they break replay and parallel children.

## References

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — the original.
- [Tinker docs](https://tinker-docs.thinkingmachines.ai/) — for the hosted-fine-tuning target.
- [`tinker/README.md`](tinker/README.md) — Tinker target walkthrough + cost notes.
