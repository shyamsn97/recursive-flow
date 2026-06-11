# Supervisor Injection Example

This directory demonstrates prompt-based graph surgery on a real saved
`RLMFlow` run. The point is injection: replace a real `SupervisingOutput` node,
truncate the now-obsolete children, materialize the edited graph as a workspace,
and continue with an explicitly bound agent.

## Files

- `word_search.py` generates the baseline delegated run for finding `AGENT` at
  `examples/_runs/word-search-workspace/word-search-baseline`.
- `inject_variants.py` opens that run, edits graph nodes, creates local variant
  workspaces, syncs the edited graphs, and continues both variants together with
  `parallel_step(...)`. The baseline and variants use
  the same structured `WordSearchResult` shape, so validation checks the typed
  graph result instead of scraping a prose answer.

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

`inject_variants.py` creates two variant workspaces:

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
- `base_agent.clone(workspace=...)` creates one explicitly bound agent per
  variant without putting storage metadata on the graph.
- The first `step(...)` on each bound agent syncs the edited graph into its
  workspace before planning the next action.
- `parallel_step(...)` advances both edited workspaces through one shared
  cross-graph step loop instead of running the variants sequentially.
- `graph.result()` returns the structured word-search payload from `DoneOutput`,
  and the example validates it with `WordSearchResult.model_validate(...)`.
