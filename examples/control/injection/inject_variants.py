"""Inject alternate prompts into a real word-search supervisor trace.

Prerequisite:
    python examples/control/injection/word_search.py

That produces ``examples/_runs/word-search-workspace/word-search-baseline`` with
a real root ``SupervisingOutput`` that waited on ``root.rows``, ``root.cols``,
and ``root.diagonals``. This example creates two variant workspaces from edited
graphs and replaces real supervising nodes in the saved trace:

1. replace ``root.cols`` so it scans columns directly instead of delegating each
   column;
2. replace the root supervising node so the parent writes one direct
   all-direction scanner instead of reconciling direction children.

The replacements are operator prompts, not pre-written solution code or mocked
results. Each edited graph is written to its own workspace and then continued by
an explicitly workspace-bound agent clone.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

import rflow


class WordHit(BaseModel):
    """One found word and its inclusive coordinates."""

    word: str
    start_row: int
    start_col: int
    end_row: int
    end_col: int
    direction: Literal["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


class WordSearchResult(BaseModel):
    """All target words found in the grid."""

    found: list[WordHit]
    missing: list[str]


EXPECTED_HITS = {("AGENT", 1, 8, 5, 8, "S")}
EXPECTED_MISSING: set[str] = set()

COLS_FUNCTION_PROMPT = """\
Actually, change the column-agent route.

Instead of delegating each column, search the columns directly with a helper
function.

In the next REPL block:

1. define `find_column_hits(grid: list[str], target_words: list[str]) -> list[dict]`;
2. scan every column in both S and N directions;
2. run the helper and verify each returned coordinate range spells the claimed word;
"""

ROOT_DIRECT_SCAN_PROMPT = """\
Actually, change the root route.

Instead of delegating to sub-agents, write a backtracking algorithm to find the target word yourself.
"""


def client_for_model(model: str) -> rflow.LLMClient:
    return (
        rflow.AnthropicClient(model)
        if model.startswith("claude")
        else rflow.OpenAIClient(model)
    )


def default_source() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "_runs"
        / "word-search-workspace"
        / "word-search-baseline"
    )


def summarize(label: str, graph: rflow.Graph) -> None:
    current = graph.current()
    print(f"\n{label}")
    print("-" * len(label))
    print(f"root current: {current.type if current else '<empty>'}")
    if isinstance(current, rflow.SupervisingOutput):
        print(f"waiting_on: {', '.join(current.waiting_on)}")
    print(f"children: {', '.join(graph.children) or '<none>'}")
    if graph.finished:
        print("result:")
        print(graph.result())


def _hit_key(hit: WordHit) -> tuple[str, int, int, int, int, str]:
    return (
        hit.word,
        hit.start_row,
        hit.start_col,
        hit.end_row,
        hit.end_col,
        hit.direction,
    )


def validate_result(label: str, graph: rflow.Graph) -> None:
    if not graph.finished:
        print(f"\n{label} validation: skipped (graph is not finished)")
        return

    result = WordSearchResult.model_validate(graph.result())
    actual = {_hit_key(hit) for hit in result.found}
    missing = set(result.missing)
    ok = actual == EXPECTED_HITS and missing == EXPECTED_MISSING
    print(f"\n{label} validation: {'PASS' if ok else 'FAIL'}")
    print("found:")
    for hit in sorted(result.found, key=lambda h: (h.word, h.direction)):
        print(
            f"- {hit.word}: ({hit.start_row},{hit.start_col}) -> "
            f"({hit.end_row},{hit.end_col}) {hit.direction}"
        )
    if result.missing:
        print(f"missing: {', '.join(result.missing)}")
    if not ok:
        print("expected:")
        for word, sr, sc, er, ec, direction in sorted(EXPECTED_HITS):
            print(f"- {word}: ({sr},{sc}) -> ({er},{ec}) {direction}")
    assert ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=default_source())
    parser.add_argument("--model", default="gpt-5-mini")
    args = parser.parse_args()

    source_workspace = rflow.Workspace.open_path(args.source.resolve())
    base_agent = rflow.RecursiveFlow(
        client_for_model(args.model),
        config=rflow.FlowConfig(
            max_depth=2, max_iterations=None, child_max_iterations=None
        ),
    )
    graph = source_workspace.load_graph()
    summarize("Loaded real word-search run", graph)

    cols_graph = graph.replace_node(
        graph.filter(lambda n: n.agent_id == "root.cols" and n.type == "supervising_output")[-1],
        rflow.ExecOutput(
            output=COLS_FUNCTION_PROMPT,
            content=f"REPL output for previous block:\n{COLS_FUNCTION_PROMPT}",
        ),
        truncate="descendants",
        branch_id="cols-direct",
    )
    cols_workspace = source_workspace.fork(
        source_workspace.root.parent / "word-search-cols-direct",
    )

    root_graph = graph.replace_node(
        graph.filter(lambda n: n.agent_id == "root" and n.type == "supervising_output")[-1],
        rflow.ExecOutput(
            output=ROOT_DIRECT_SCAN_PROMPT,
            content=f"REPL output for previous block:\n{ROOT_DIRECT_SCAN_PROMPT}",
        ),
        truncate="descendants",
        branch_id="direct-scan",
    )
    root_workspace = source_workspace.fork(
        source_workspace.root.parent / "word-search-direct-scan",
    )

    cols_agent = base_agent.clone(workspace=cols_workspace)
    root_agent = base_agent.clone(workspace=root_workspace)

    while not (cols_graph.finished and root_graph.finished):
        active = []
        if not cols_graph.finished:
            active.append((cols_agent, cols_graph))
        if not root_graph.finished:
            active.append((root_agent, root_graph))

        next_graphs = rflow.parallel_step(active)
        i = 0
        if not cols_graph.finished:
            cols_graph = next_graphs[i]
            i += 1
        if not root_graph.finished:
            root_graph = next_graphs[i]

        cols_state = cols_graph.current().type if cols_graph.current() else "<empty>"
        root_state = root_graph.current().type if root_graph.current() else "<empty>"
        print(f"parallel step: Variation A={cols_state}, Variation B={root_state}")

    summarize("Variation A: prompt root.cols to scan columns directly", cols_graph)
    validate_result("Variation A", cols_graph)

    summarize("Variation B: prompt root to write a direct scanner", root_graph)
    validate_result("Variation B", root_graph)


if __name__ == "__main__":
    main()
