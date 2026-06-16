# Control

Examples for steering an agent run after it starts: delegation, branching,
injection, and controller-authored graph edits.

- `controller_injection.py` appends controller-provided nodes with
  `graph.inject(...)`.
- `delegation/eager_children.py` starts child work before the parent blocks on
  it.
- `branching/best_of_n.py` runs independent branches and scores the results.
- `branching/fork_repair.py` compares repair attempts across forked runs.
- `injection/` replaces real supervising nodes in a saved graph, edits copies
  with `graph.replace_node(...)`, and continues each variant with its own
  `Flow`.
