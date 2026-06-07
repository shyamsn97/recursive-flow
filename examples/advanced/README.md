# Advanced Examples

These examples exercise durable workspaces, graph surgery, replay, and branch
repair. They are meant to be read alongside the generated workspace files, not
just run once.

## Replay And Graph Surgery

The replay example lives in [`replay/`](replay/):

1. [`replay/sudoku.py`](replay/sudoku.py) creates a real saved Sudoku run under
   `replay/runs/sudoku-naive`.
2. [`replay/replay_resume.py`](replay/replay_resume.py) forks that run and
   replaces real `SupervisingOutput` nodes with prompt-based repairs.

Run:

```bash
python examples/advanced/replay/sudoku.py
python examples/advanced/replay/replay_resume.py
```

Both scripts use live LLM clients. Pass `--model` to choose a model.
The live smoke runner executes the same flow with a temporary workspace:

```bash
python examples/run_examples.py --include-live
```

The important behavior is that edited graphs are synced back to their forked
workspaces before stepping, so stale child sessions and contexts are pruned from
the branch.
