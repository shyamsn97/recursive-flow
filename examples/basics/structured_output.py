"""Structured output for root runs and delegated child agents.

This live example asks a real model to extract facts from a provided trip brief.
The root launches child agents with JSON Schema output contracts, receives their
typed dictionary results, then returns a Pydantic-validated root result.

Run:
    export OPENAI_API_KEY=...
    python examples/basics/structured_output.py --model gpt-5-mini
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from rlmflow import OpenAIClient, RLMConfig, RLMFlow, Workspace
from rlmflow.runtime.local import LocalRuntime


class CityForecast(BaseModel):
    """Forecast facts for one city extracted from the trip brief."""

    city: str
    condition: Literal["rain", "sun", "clouds"]
    high_f: float
    packing_tip: str


class PackingPlan(BaseModel):
    """Packing plan synthesized from the structured city forecasts."""

    destination_count: int
    forecasts: list[CityForecast]
    shared_items: list[str]
    summary: str


TRIP_BRIEF = """\
Trip brief for structured extraction.

Seattle leg:
- City: Seattle
- Forecast condition: rain
- Forecast high: 60.0 F

Austin leg:
- City: Austin
- Forecast condition: sun
- Forecast high: 96.0 F

Denver leg:
- City: Denver
- Forecast condition: clouds
- Forecast high: 72.0 F
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Live structured output example")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--workspace", default=None)
    args = parser.parse_args()

    examples_root = Path(__file__).resolve().parents[1]
    workspace_path = (
        Path(args.workspace)
        if args.workspace
        else examples_root / "_runs" / "example-workspaces" / "structured-output"
    )
    workspace = Workspace.create(workspace_path)

    agent = RLMFlow(
        OpenAIClient(args.model),
        runtime=LocalRuntime(workspace=workspace),
        workspace=workspace,
        config=RLMConfig(max_depth=args.max_depth, max_iterations=args.max_iterations),
    )

    query = """Build a packing plan for my upcoming trip. The important weather info is in CONTEXT. Make sure to delegate each city to child agents.
    """

    graph = agent.start(
        query,
        context=TRIP_BRIEF,
        output_schema=PackingPlan,
    )

    from rlmflow.utils.viz import live_view

    with live_view() as view:
        view(graph)
        while not graph.finished:
            graph = agent.step(graph)
            view(graph)

    plan = PackingPlan.model_validate(graph.result())

    print("Typed result:")
    print(type(plan).__name__)
    print(plan.model_dump_json(indent=2))

    print("\nGraph result:")
    print(graph.result())

    print("\nTree:")
    print(graph.tree())
    print(f"\nWorkspace saved to {workspace.root}")


if __name__ == "__main__":
    main()
