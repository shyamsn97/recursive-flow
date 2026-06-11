"""Generate the baseline word-search run used by the injection example.

This is the original route: ask a real RLMFlow agent to find ``AGENT`` by
delegating direction-specific search to three child agents:

- ``rows`` searches rows east/west;
- ``cols`` searches columns north/south;
- ``diagonals`` searches diagonals in all four diagonal directions.

That route is intentionally plausible but more complicated than necessary. It
creates a real root ``SupervisingOutput`` that ``inject_variants.py`` can later
replace with a direct scanner route.

Run:
    export OPENAI_API_KEY=...
    python examples/control/injection/word_search.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from rlmflow import AnthropicClient, OpenAIClient, RLMConfig, RLMFlow, Workspace
from rlmflow.utils.viz import live_view

TARGET_WORD = "AGENT"

CONTEXT = f"""Word Search Grid:

TYPHONQWER
LMNOPQRSAT
ZGXYZLMNGO
ABDDEFGHEI
QRSATUVWNY
JKLMPQRSTF
ABCDEPFGOI
UVWXYZARBC
MNOPQRKSTU
ABCDECARTZ

Target Word:
{TARGET_WORD}

Row and Column indices start at 0.
"""


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


EXPECTED_HITS = {
    ("AGENT", 1, 8, 5, 8, "S"),
}
EXPECTED_MISSING: set[str] = set()

QUERY = """\
Solve the word search puzzle in CONTEXT.

You can aproach this problem with the following strategy:
1. The root should first delegate to three child agents: `rows`, `cols`,
    and `diagonals`.
2. Inside row and column agents, subdelegate the actual line searches:
    - `rows` should search rows for the target words and delegate each row in parallel to its own sub-agent (rows.<row_number>);
    - `cols` should search columns for the target words and delegate each column in parallel to its own sub-agent (cols.<column_number>).
3. `diagonals` should search diagonals by itself directly without delegating.
4. Each child agent should return a list of tuples, each containing the word found and its inclusive coordinates.
"""

def client_for_model(model: str):
    return AnthropicClient(model) if model.startswith("claude") else OpenAIClient(model)


def _hit_key(hit: WordHit) -> tuple[str, int, int, int, int, str]:
    return (
        hit.word,
        hit.start_row,
        hit.start_col,
        hit.end_row,
        hit.end_col,
        hit.direction,
    )


def run(model: str, workspace_path: Path, *, reset: bool) -> None:
    if reset and workspace_path.exists():
        shutil.rmtree(workspace_path)

    workspace = Workspace.create(workspace_path)
    agent = RLMFlow(
        client_for_model(model),
        config=RLMConfig(max_depth=2, child_max_iterations=10),
    ).attach_workspace(workspace)

    graph = agent.start(QUERY, output_schema=WordSearchResult, context=CONTEXT)
    with live_view() as view:
        view(graph)
        while not graph.finished:
            graph = agent.step(graph)
            view(graph)

    result = WordSearchResult.model_validate(graph.result())
    actual = {_hit_key(hit) for hit in result.found}
    missing = set(result.missing)

    print("=== RESULT ===")
    print(result.model_dump_json(indent=2))
    if actual != EXPECTED_HITS or missing != EXPECTED_MISSING:
        raise SystemExit("agent returned word-search hits that do not match EXPECTED")

    print("agent returned the expected word-search hits!")
    print(f"\nwrote injection workspace: {workspace.root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "_runs"
        / "word-search-workspace"
        / "word-search-baseline",
    )
    parser.add_argument("--no-reset", action="store_true")
    args = parser.parse_args()

    run(args.model, args.workspace.resolve(), reset=not args.no_reset)


if __name__ == "__main__":
    main()
