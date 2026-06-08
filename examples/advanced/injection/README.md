# Supervisor Injection Example

This directory demonstrates prompt-based graph surgery on a real saved
`RLMFlow` run. The point is injection: replace a real `SupervisingOutput` node,
truncate the now-obsolete children, sync the forked workspace, and continue.

## Files

- `sudoku.py` generates the baseline run at `runs/sudoku-naive`.
- `inject_variants.py` opens that run, forks it, edits graph nodes, and continues
  each fork with `agent.step(graph)`. The baseline and variants use the same
  structured `SudokuSolution` result shape, so validation checks the typed graph
  result instead of scraping a prose answer.

The generated `runs/` directories are normal workspaces. Inspect them with:

```bash
rlmflow view examples/advanced/injection/runs/sudoku-naive
```

## Flow

1. Generate the baseline trace:

   ```bash
   python examples/advanced/injection/sudoku.py
   ```

2. Inject alternate supervisor outcomes:

   ```bash
   python examples/advanced/injection/inject_variants.py
   ```

`inject_variants.py` creates two forks:

- `runs/sudoku-cols-function`: replaces the `root.cols` verifier route with a
  direct helper-function verifier.
- `runs/sudoku-backtracking`: replaces the root supervisor with a direct
  backtracking solve route.

Both edits are prompt-based. The example does not inject precomputed answers; it
changes the supervisor route and lets the model continue to a structured
`done({"solution": ...})` result.

## What To Look For

- The edited node is a real `SupervisingOutput` from the baseline trace.
- `truncate="descendants"` removes obsolete waited-on child routes.
- `Workspace.sync_graph(...)` makes the forked workspace match the edited graph,
  including pruning stale `session/` and `context/` payloads.
- `live_view()` shows the branch continue from the edited graph.
- `graph.result()` returns the structured Sudoku payload from `DoneOutput`, and
  the example validates it with `SudokuSolution.model_validate(...)`.
