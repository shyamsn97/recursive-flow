"""Use an RecursiveFlow agent as the LM behind a DSPy program.

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
from rflow.runtime.local import LocalRuntime


def main() -> None:
    examples_root = Path(__file__).resolve().parents[1]
    workspace = rflow.Workspace.create(
        examples_root / "_runs" / "example-workspaces" / "dspy-workspace"
    )
    agent = rflow.RecursiveFlow(
        llm_client=rflow.OpenAIClient(model="gpt-4o-mini"),
        runtime=LocalRuntime(workspace=workspace),
        config=rflow.FlowConfig(max_depth=1, max_iterations=5),
    )

    dspy.configure(lm=RecursiveFlowLM(agent, model="recursive-flow/gpt-4o-mini"))

    qa = dspy.ChainOfThought("question -> answer")
    result = qa(question="What is 17 * 23? Show a short calculation.")
    print(result.answer)


if __name__ == "__main__":
    main()
