"""Generate the naive Sudoku run used by the supervisor injection example.

This is the "original route": ask a real RLMFlow agent to solve the puzzle by
delegating into exactly three subagents:

- ``rows`` analyzes row constraints;
- ``cols`` analyzes column constraints;
- ``boxes`` analyzes 3x3 box constraints.

That route is intentionally plausible but awkward. It creates a real
``SupervisingOutput`` on the root, which ``inject_variants.py`` can later replace
with a different route.

Run:
    export OPENAI_API_KEY=...
    python examples/advanced/injection/sudoku.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from pydantic import BaseModel
from rlmflow import AnthropicClient, OpenAIClient, RLMConfig, RLMFlow, Workspace
from rlmflow.utils.viz import live_view

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

QUERY = f"""\
Solve this Sudoku puzzle:

{PUZZLE}

You can aproach this problem with the following strategy:
1. Delegate rows, cols, and boxes to child agents;
    - `rows` should analyze row constraints by delegating each row to sub-agents (row.<row_number>);
    - `cols` should analyze column constraints by delegating each column to sub-agents (col.<column_number>);
    - `boxes` should analyze 3x3 box constraints by delegating each box to sub-agents (box.<box_number>);
3. Verify the completed board preserves all givens and every row, column, and 3x3 box contains digits 1 through 9;
"""

class SudokuSolution(BaseModel):
    """The completed Sudoku grid."""

    solution: str

EXPECTED_SOLUTION = SudokuSolution(solution=EXPECTED.strip())


def client_for_model(model: str):
    return AnthropicClient(model) if model.startswith("claude") else OpenAIClient(model)


def normalize_grid(text: str) -> str:
    lines = [
        "".join(ch for ch in line if ch.isdigit())
        for line in text.splitlines()
    ]
    lines = [line for line in lines if len(line) == 9]
    return "\n".join(lines[-9:])


def run(model: str, workspace_path: Path, *, reset: bool) -> None:
    if reset and workspace_path.exists():
        shutil.rmtree(workspace_path)

    workspace = Workspace.create(workspace_path, branch_id="sudoku-naive")
    agent = RLMFlow(
        client_for_model(model),
        workspace=workspace,
        config=RLMConfig(max_depth=2, max_iterations=8, child_max_iterations=4),
    )

    graph = agent.start(QUERY, output_schema=SudokuSolution)
    with live_view() as view:
        view(graph)
        while not graph.finished:
            graph = agent.step(graph)
            view(graph)

    result = SudokuSolution.model_validate(graph.result())
    print("=== RESULT ===")
    print(result)
    if result.solution.strip() != EXPECTED_SOLUTION.solution.strip():
        raise SystemExit("agent returned a grid that does not match EXPECTED")
    else:
        print("agent returned the expected grid!")

    print(f"\nwrote injection workspace: {workspace.root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).with_suffix("").parent / "sudoku-workspace" / "sudoku-naive",
    )
    parser.add_argument("--no-reset", action="store_true")
    args = parser.parse_args()

    run(args.model, args.workspace.resolve(), reset=not args.no_reset)


if __name__ == "__main__":
    main()