"""Inject alternate prompts into a real Sudoku supervisor trace.

Prerequisite:
    python examples/advanced/injection/sudoku.py

That produces ``runs/sudoku-naive`` with a real root ``SupervisingOutput`` that
waited on ``root.rows``, ``root.cols``, and ``root.boxes``. This example forks
that run twice and replaces two real supervising nodes in the saved trace:

1. replace ``root.cols``'s own supervising node so it writes a reusable column
   helper function instead of returning a precomputed column-candidate JSON map;
2. replace the root supervising node so the parent writes a direct deterministic
   backtracking solver instead of reconciling the row/column/box split.

The replacements are operator prompts, not pre-written solution code or mocked
results. After each edit, the example only calls ``agent.step(graph)``. The
engine is responsible for committing the edited graph before continuing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import BaseModel

from rlmflow import (
    AnthropicClient,
    ExecOutput,
    Graph,
    OpenAIClient,
    RLMConfig,
    RLMFlow,
    SupervisingOutput,
)
from rlmflow.llm import LLMClient
from rlmflow.utils.viz import live_view
from rlmflow.workspace import Workspace

PUZZLE = """\
530070000
600195000
098000060
800060003
400803001
700020006
060000280
000419005
000080079
"""

EXPECTED = """\
534678912
672195348
198342567
859761423
426853791
713924856
961537284
287419635
345286179
"""


ROWS_COLS_BOXES = {"root.rows", "root.cols", "root.boxes"}


class SudokuSolution(BaseModel):
    """The completed Sudoku grid."""

    solution: str


EXPECTED_SOLUTION = SudokuSolution(solution=EXPECTED.strip())


COLS_FUNCTION_PROMPT = """\
Actually, change the column-agent route.

Instead of delegating each column, verify the columns directly with a helper
function.

In the next REPL block:

1. define `invalid_columns(board: str) -> list[int]`;
2. read the solved board from `CONTEXT` after the `BOARD:` line;
3. run the function;
4. verify each column contains digits 1 through 9 exactly once;
"""

ROOT_BACKTRACKING_PROMPT = """\
Actually, change the root route.

Do not reconcile the row/column/box child notes. In the next REPL block, write
and run one deterministic Python backtracking solver for the Sudoku puzzle
directly. Do not delegate.

The solver should:

- parse the original 9-line puzzle;
- use exact Sudoku constraints with MRV/backtracking or equivalent;
- verify the solution preserves all givens;
- verify every row, column, and 3x3 box contains digits 1 through 9.

you should verify if the solution is valid, otherwise you should return a failure message.
"""


def client_for_model(model: str) -> LLMClient:
    return AnthropicClient(model) if model.startswith("claude") else OpenAIClient(model)


def default_source() -> Path:
    return Path(__file__).resolve().parent / "sudoku-workspace" / "sudoku-naive"


def branch_path(source: Path, name: str) -> Path:
    return source.parent / name


def find_root_supervising(graph: Graph) -> SupervisingOutput:
    for node in graph.nodes:
        if isinstance(node, SupervisingOutput) and set(node.waiting_on) == ROWS_COLS_BOXES:
            return node
    raise ValueError("could not find root supervising node for rows/cols/boxes")


def find_latest_supervising(graph: Graph, agent_id: str) -> SupervisingOutput:
    for node in reversed(graph[agent_id].nodes):
        if isinstance(node, SupervisingOutput):
            return node
    raise ValueError(f"could not find supervising node for {agent_id!r}")


def replace_supervising_node(
    graph: Graph,
    supervising: SupervisingOutput,
    prompt: str,
) -> Graph:
    """Replace the actual supervising node and abandon its waited-on children."""

    return graph.replace_node(
        supervising.id,
        ExecOutput(
            output=prompt,
            content=f"REPL output for previous block:\n{prompt}",
        ),
        truncate="descendants",
    )


def edit_cols_child_to_function(graph: Graph) -> Graph:
    """Patch the saved ``root.cols`` route with a prompt-only replacement."""

    return replace_supervising_node(
        graph,
        find_latest_supervising(graph, "root.cols"),
        COLS_FUNCTION_PROMPT,
    )


def edit_root_to_backtracking(graph: Graph) -> Graph:
    """Patch the saved root route with a prompt-only direct solver."""

    return replace_supervising_node(
        graph,
        find_root_supervising(graph),
        ROOT_BACKTRACKING_PROMPT,
    )


def make_agent(workspace: Workspace, model: str) -> RLMFlow:
    return RLMFlow(
        client_for_model(model),
        workspace=workspace,
        config=RLMConfig(max_depth=2, max_iterations=None, child_max_iterations=None),
    )


def step_until(
    agent: RLMFlow,
    graph: Graph,
) -> Graph:
    with live_view() as view:
        view(graph)
        while not graph.finished:
            graph = agent.step(graph)
            view(graph)
    return graph


def fork_edit_and_step(
    source_workspace: Workspace,
    *,
    branch_name: str,
    edit,
    model: str,
) -> Graph:
    branch = source_workspace.fork(
        new_location=branch_path(source_workspace.root, branch_name),
        new_branch_id=branch_name,
    )
    edited = edit(branch.session.load_graph())

    agent = make_agent(branch, model)
    return step_until(agent, edited)

def summarize(label: str, graph: Graph) -> None:
    current = graph.current()
    print(f"\n{label}")
    print("-" * len(label))
    print(f"root current: {current.type if current else '<empty>'}")
    if isinstance(current, SupervisingOutput):
        print(f"waiting_on: {', '.join(current.waiting_on)}")
    print(f"children: {', '.join(graph.children) or '<none>'}")
    if graph.finished:
        print("result:")
        print(graph.result())


def validate_grid(label: str, graph: Graph) -> None:
    if not graph.finished:
        print(f"\n{label} validation: skipped (graph is not finished)")
        return

    result = SudokuSolution.model_validate(graph.result())
    actual = result.solution.strip()
    expected = EXPECTED_SOLUTION.solution.strip()
    ok = actual == expected
    print(f"\n{label} validation: {'PASS' if ok else 'FAIL'}")
    if actual:
        print("normalized grid:")
        print(actual)
    else:
        print("normalized grid: <none found in result>")
    if not ok:
        print("expected:")
        print(expected)
    assert ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=default_source())
    parser.add_argument("--model", default="gpt-5-mini")
    args = parser.parse_args()

    source_workspace = Workspace.open_path(args.source.resolve())
    graph = source_workspace.session.load_graph()
    summarize("Loaded real Sudoku run", graph)

    cols_function = fork_edit_and_step(
        source_workspace,
        branch_name="sudoku-cols-function",
        edit=edit_cols_child_to_function,
        model=args.model,
    )
    summarize("Variation A: prompt root.cols to verify columns via a function", cols_function)
    validate_grid("Variation A", cols_function)

    backtracking = fork_edit_and_step(
        source_workspace,
        branch_name="sudoku-backtracking",
        edit=edit_root_to_backtracking,
        model=args.model,
    )
    summarize("Variation B: prompt root to write a backtracking solver", backtracking)
    validate_grid("Variation B", backtracking)


if __name__ == "__main__":
    main()
