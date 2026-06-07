# Control

Examples that change how an agent run is shaped: delegation, branching,
forking, and graph edits.

- `injections.py` replaces graph nodes and resumes from the edited trace.
- `eager_children.py` starts child work before the parent blocks on it.
- `best_of_n.py` runs independent branches and scores the results.
- `fork_repair.py` compares repair attempts across workspace forks.
