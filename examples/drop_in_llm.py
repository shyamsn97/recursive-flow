"""Flow as a drop-in LLM.

Because `Flow` implements the `LLMClient` protocol (`chat()` / `completion()`),
you can swap it in anywhere you'd use a raw LLM. Calling `flow.chat(messages)`
runs the full recursive agent loop under the hood and returns a plain string —
same signature as any other LLM client.

This enables two patterns:

1. **Replace an LLM with an agent.** Any function that takes an `LLMClient`
   (e.g. a summarization helper, a router, a retrieval pipeline) gets agentic
   behavior for free — no code changes.

2. **Nest agents.** An outer `Flow` can use an inner `Flow` as its `llm`.
   The outer agent's every "LLM call" is itself a full recursive sub-agent run.

Run with:
    export OPENAI_API_KEY=...
    python examples/drop_in_llm.py
"""

from __future__ import annotations

import rflow
from rflow.utils.example_runs import example_run_dir, save_example_graph


def ask(llm: rflow.LLMClient, question: str) -> str:
    """A generic helper that takes any LLMClient. Doesn't know or care
    whether it got a plain OpenAI client or a full recursive agent."""
    reply = llm.chat([{"role": "user", "content": question}])
    usage = llm.last_usage
    tokens = usage.input_tokens + usage.output_tokens if usage else 0
    print(f"[{type(llm).__name__}] tokens={tokens}")
    return reply


def demo_plain_llm():
    print("=== plain OpenAI client ===")
    llm = rflow.OpenAIClient(model="gpt-4o-mini")
    answer = ask(llm, "In one sentence: what is the capital of France?")
    print(answer, "\n")


def demo_flow_as_llm():
    print("=== Flow as LLMClient (drop-in) ===")
    agent = rflow.Flow(
        rflow.OpenAIClient(model="gpt-4o-mini"),
        max_iters=5,
        max_budget=20_000,
    )
    answer = ask(agent, "Compute 17 * 23 using a ```repl``` block, then call done().")
    print(answer, "\n")
    if agent.graph is not None:
        save_example_graph(
            agent.graph,
            __file__,
            "drop-in-llm",
            out_dir=example_run_dir(__file__, "drop-in-llm") / "flow-as-llm",
        )
    agent.close()


def demo_nested_flow():
    print("=== nested Flow (outer agent uses inner agent as its LLM) ===")
    inner = rflow.Flow(
        rflow.OpenAIClient(model="gpt-4o-mini"),
        max_iters=3,
    )
    outer = rflow.Flow(
        inner,
        max_iters=3,
        max_budget=50_000,
    )
    answer = outer.run("What's the 7th Fibonacci number? Use ```repl``` to compute.")
    print(answer)
    if outer.graph is not None:
        save_example_graph(
            outer.graph,
            __file__,
            "drop-in-llm",
            out_dir=example_run_dir(__file__, "drop-in-llm") / "nested-flow",
        )
    outer.close()
    inner.close()


if __name__ == "__main__":
    demo_plain_llm()
    demo_flow_as_llm()
    demo_nested_flow()
