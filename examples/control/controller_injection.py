"""Inject typed nodes into a running graph from a controller.

The controller workflow with the stateful engine:

1. Build a new graph value with ``graph.inject(...)`` (pure: returns a copy).
2. Adopt it by passing it to ``flow.step(injected)`` to react (no ``flow.graph =``).
3. To stop a run early, call ``flow.terminate()`` — the agent is forced to
   ``done(...)`` on its next turn (the supported "finalize now" mechanism).

Run:
    python examples/control/controller_injection.py
"""

from __future__ import annotations

import rflow
from rflow.utils.example_runs import example_run_dir, save_example_graph

OBSERVATION = "Injected controller observation: finalize using this note."


class DemoLLM(rflow.LLMClient):
    """Deterministic model so the example runs offline."""

    def chat(self, messages, *args, **kwargs) -> str:
        self.last_usage = rflow.LLMUsage(input_tokens=80, output_tokens=20)
        convo = "\n".join(m["content"] for m in messages)
        if "Injected controller observation" in convo:
            return '```repl\ndone("used the injected controller observation")\n```'
        if "full iteration budget" in convo:  # FINAL nudge from terminate()
            return '```repl\ndone("controller stopped the run")\n```'
        return '```repl\nprint("waiting for controller input")\n```'


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def state_types(graph) -> list[str]:
    return [state.type for state in graph.nodes]


def assert_types(graph, expected: list[str]) -> None:
    actual = state_types(graph)
    assert actual == expected, f"expected states {expected}, got {actual}"


def print_states(label: str, graph) -> None:
    print(f"\n{label}")
    print("state types:", " -> ".join(state_types(graph)))


def observation_injection() -> None:
    banner("1. Inject an observation and let the LLM react")

    flow = rflow.Flow(DemoLLM(), max_depth=0, max_iters=4)
    graph = flow.start("Wait for a controller note, then finish.")
    assert_types(graph, ["user_query"])

    # inject() is pure — it returns a NEW graph, leaving the live one untouched.
    injected = graph.inject(
        target="root",
        node=rflow.ExecOutput(output=OBSERVATION, content=OBSERVATION),
    )
    assert injected is not graph
    assert_types(graph, ["user_query"])
    assert_types(injected, ["user_query", "exec_output"])

    extra = injected.nodes[-1]
    assert isinstance(extra, rflow.ExecOutput)
    assert "injected" not in set(extra.to_dict())

    print_states("start(): original graph", graph)
    print_states("graph.inject(...): returned a copy with one plain ExecOutput", injected)

    projected = flow.build_messages(injected, force_final=False)[-1]["content"]
    assert OBSERVATION in projected
    print("message projection contains the controller observation.")

    # Adopt the edited graph by passing it to step() — no more flow.graph = ...
    graph = flow.step(injected)  # reacts to the observation -> LLM call
    graph = flow.step(graph)  # executes the LLM's done(...) block
    assert graph.result() == "used the injected controller observation"
    print_states("after adopting + stepping: run reacted and finished", graph)
    print(f"result={graph.result()!r}")
    save_example_graph(
        graph,
        __file__,
        "controller-injection",
        out_dir=example_run_dir(__file__, "controller-injection")
        / "observation-injection",
    )


def terminate_to_finalize() -> None:
    banner("2. terminate() to finalize a run immediately")

    flow = rflow.Flow(DemoLLM(), max_depth=0, max_iters=4)
    graph = flow.start("This run will be stopped by the controller.")
    graph = flow.step(graph)  # one normal turn (print)
    flow.terminate()  # force the agent to wrap up on its next LLM turn
    while not graph.finished:
        graph = flow.step(graph)
    assert graph.result() == "controller stopped the run"
    print_states("after terminate(): forced a clean done(...)", graph)
    print(f"result={graph.result()!r}")
    save_example_graph(
        graph,
        __file__,
        "controller-injection",
        out_dir=example_run_dir(__file__, "controller-injection")
        / "terminate-to-finalize",
    )


def main() -> None:
    observation_injection()
    terminate_to_finalize()


if __name__ == "__main__":
    main()
