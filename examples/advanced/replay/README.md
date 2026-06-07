# Replay Example

This directory demonstrates replaying and editing a real saved `RLMFlow` run.

## Files

- `sudoku.py` generates the baseline run at `runs/sudoku-naive`.
- `replay_resume.py` opens that run, forks it, edits graph nodes, and continues
  each fork with `agent.step(graph)`.

The generated `runs/` directories are normal workspaces. Inspect them with:

```bash
rlmflow view examples/advanced/replay/runs/sudoku-naive
```

## Flow

1. Generate the baseline trace:

   ```bash
   python examples/advanced/replay/sudoku.py
   ```

2. Replay and edit it:

   ```bash
   python examples/advanced/replay/replay_resume.py
   ```

`replay_resume.py` creates two forks:

- `runs/sudoku-cols-function`: replaces the `root.cols` verifier route with a
  direct helper-function verifier.
- `runs/sudoku-backtracking`: replaces the root supervisor with a direct
  backtracking solve route.

Both edits are prompt-based. The example does not inject precomputed answers.

## What To Look For

- The edited node is a real `SupervisingOutput` from the baseline trace.
- `truncate="descendants"` removes obsolete waited-on child routes.
- `Workspace.sync_graph(...)` makes the forked workspace match the edited graph,
  including pruning stale `session/` and `context/` payloads.
- `live_view()` shows the branch continue from the edited graph.
