# Supervisor Injection Example

This directory demonstrates prompt-based graph surgery on a real saved
`RLMFlow` run. The point is injection: replace a real `SupervisingOutput` node,
truncate the now-obsolete children, sync the forked workspace, and continue.

## Files

- `word_search.py` generates the baseline delegated run for finding `GRAPH` at
  `examples/_runs/word-search-workspace/word-search-baseline`.
- `inject_variants.py` opens that run, forks it, edits graph nodes, and continues
  each fork with `agent.step(graph)`. The baseline and variants use the same
  structured `WordSearchResult` shape, so validation checks the typed graph
  result instead of scraping a prose answer.

The generated workspace directories are normal workspaces. Inspect them with:

```bash
rlmflow view examples/_runs/word-search-workspace/word-search-baseline
```

## Flow

1. Generate the baseline trace:

   ```bash
   python examples/control/injection/word_search.py
   ```

2. Inject alternate supervisor outcomes:

   ```bash
   python examples/control/injection/inject_variants.py
   ```

`inject_variants.py` creates two forks:

- `examples/_runs/word-search-workspace/word-search-cols-direct`: replaces the
  `root.cols` delegated route with one direct column helper function.
- `examples/_runs/word-search-workspace/word-search-direct-scan`: replaces the
  root supervisor with one direct all-direction scanner.

Both edits are prompt-based. The example does not inject precomputed answers; it
changes the supervisor route and lets the model continue to a structured
`done({"found": ..., "missing": ...})` result.

The baseline prompt explicitly forbids a root-level all-direction scan so the
saved trace is structurally different from the direct-scan variant: the original
route is child-driven direction analysis, while the injected root variant swaps
in a direct deterministic scanner.

## What To Look For

- The edited node is a real `SupervisingOutput` from the baseline trace.
- `truncate="descendants"` removes obsolete waited-on child routes.
- `Workspace.sync_graph(...)` makes the forked workspace match the edited graph,
  including pruning stale `session/` and `context/` payloads.
- `live_view()` shows the branch continue from the edited graph.
- `graph.result()` returns the structured word-search payload from `DoneOutput`,
  and the example validates it with `WordSearchResult.model_validate(...)`.
