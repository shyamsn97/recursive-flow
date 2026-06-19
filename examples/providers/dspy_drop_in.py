"""Use a Flow agent as the LM behind a DSPy program.

Because `Flow` implements the `LLMClient` protocol, it drops straight into the
DSPy adapter — every DSPy "LM call" becomes a full recursive agent run.

Run with:
    export OPENAI_API_KEY=...
    pip install -e ".[openai,dspy]"
    python examples/providers/dspy_drop_in.py
"""

from __future__ import annotations

import dspy

import rflow
from rflow.integrations.dspy import RecursiveFlowLM
from rflow.utils.example_runs import save_example_graph


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
        save_example_graph(flow.graph, __file__, "dspy-drop-in")

    flow.close()


if __name__ == "__main__":
    main()
