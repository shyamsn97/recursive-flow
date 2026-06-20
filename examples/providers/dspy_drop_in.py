"""Use a Flow agent as the LM behind a DSPy program.

Because `Flow` implements the `LLMClient` protocol, it drops straight into the
DSPy adapter — every DSPy "LM call" becomes a full recursive agent run.

Run with:
    export OPENAI_API_KEY=...
    pip install -e ".[openai,dspy]"
    python examples/providers/dspy_drop_in.py
"""

from __future__ import annotations

from pathlib import Path

import dspy

import rflow
from rflow.integrations.dspy import RecursiveFlowLM


def _example_run_dir(source_file: str | Path, name: str) -> Path:
    source = Path(source_file).resolve()
    for parent in (source.parent, *source.parents):
        if parent.name == "examples":
            return parent / "_runs" / name
    return source.parent / "_runs" / name


def _save_example_graph(
    graph,
    source_file: str | Path,
    name: str,
    *,
    out_dir: str | Path | None = None,
    label: str = "Graph saved to",
) -> Path:
    path = graph.save(
        Path(out_dir) if out_dir is not None else _example_run_dir(source_file, name)
    )
    print(f"{label} {path}")
    return path



def main() -> None:
    flow = rflow.Flow(
        rflow.OpenAIClient(model="gpt-4o-mini"),
        max_depth=1,
        max_iters=5,
    )

    dspy.configure(lm=RecursiveFlowLM(flow, model="recursive-flow/gpt-4o-mini"))

    qa = dspy.ChainOfThought("question -> answer")
    result = qa(question="What is 17 * 23? Show a short calculation.")
    print(result.answer)
    if flow.graph is not None:
        _save_example_graph(flow.graph, __file__, "dspy-drop-in")

    flow.close()


if __name__ == "__main__":
    main()
