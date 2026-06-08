# Advanced Examples

These examples exercise the graph API, durable workspaces, prompt injection,
graph surgery, and branch repair. They are meant to be read alongside the
generated workspace files, not just run once.

## Graph Feature Tour

The graph feature examples live in [`graph-features/`](graph-features/). They
are offline scripts that walk through graph querying, navigation, mutation,
save/load, timeline retrace, forking, and rendering:

```bash
python examples/advanced/graph-features/01_query.py
python examples/advanced/graph-features/05_timeline.py
python examples/advanced/graph-features/07_render.py
```

## Supervisor Injection

The supervisor injection example lives in [`injection/`](injection/):

1. [`injection/sudoku.py`](injection/sudoku.py) creates a real saved Sudoku run
   under `injection/runs/sudoku-naive` with a structured `SudokuSolution`
   result contract.
2. [`injection/inject_variants.py`](injection/inject_variants.py) forks that run and
   replaces real `SupervisingOutput` nodes with prompt-based repairs that
   continue to the same structured result shape.

Run:

```bash
python examples/advanced/injection/sudoku.py
python examples/advanced/injection/inject_variants.py
```

Both scripts use live LLM clients. Pass `--model` to choose a model.
The live smoke runner executes the same flow with a temporary workspace:

```bash
python examples/run_examples.py --include-live
```

The important behavior is that edited graphs are synced back to their forked
workspaces before stepping, so stale child sessions and contexts are pruned from
the branch. The final validation uses `graph.result()` as structured data rather
than normalizing text out of the model's prose.
