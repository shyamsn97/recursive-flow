# Graph features

Tiny self-contained scripts that show what a `Graph` can do. No LLM keys
needed — every example builds its own graph by hand or runs a one-line
mock LLM, so they finish in milliseconds and can be read top-to-bottom.

| script | what it shows |
|---|---|
| `01_query.py` | flat views (`graph.all_nodes`, `.agents`, `.edges`), filters (`.where`, `.queries()`, action views, `.errors()`), `find()`, `tokens()`, `result()` |
| `02_navigate.py` | `graph[aid]`, dotted paths, `walk()` / `subtree()`, parent ↔ child links, `len(graph)` |
| `03_mutate.py` | mutating editors (`add_node`, `set_node`, `update_node`, `remove_node`, `add_child`, `remove_child`, `update`) and `graph.copy()` |
| `04_save_load.py` | `Graph.save()` / `Graph.load()` JSON round-trip + `save_trace()` / `load_trace()` |
| `05_timeline.py` | `retrace_steps(graph)` — reconstruct visualization snapshots from the final graph |
| `06_fork.py` | `graph.copy(deep=True)` — branch a run, diverge, compare |
| `07_render.py` | `graph.tree()`, `graph.session()`, `graph.transcript()`, `graph.save_html(...)` |

Run any of them directly:

```bash
python examples/graph/01_query.py
python examples/graph/05_timeline.py
python examples/graph/06_fork.py
```

Most scripts just print to stdout. `07_render.py` writes a viewer HTML
file you can open in your browser.
