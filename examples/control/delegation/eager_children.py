"""Demonstrate `eager_children=True` work-conserving child scheduling.

Run:
    python examples/control/delegation/eager_children.py

The key thing to watch in the output:

- With `eager_children=False`, child B's second LLM step starts only after
  child A's slow first LLM step completes (the scheduler advances in
  synchronized waves).
- With `eager_children=True`, child B's second LLM step starts while child A's
  slow first LLM step is still running (a free worker picks up the next
  runnable agent instead of idling at the wave boundary).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import rflow


@dataclass
class TimelineLLM(rflow.LLMClient):
    # The fake only sleeps/records, so concurrent calls are safe; this lets the
    # bounded LLM channel run sibling calls in parallel instead of serializing.
    thread_safe = True
    started_at: float = field(default_factory=time.perf_counter)
    events: list[tuple[float, str]] = field(default_factory=list)

    def mark(self, label: str) -> None:
        self.events.append((time.perf_counter() - self.started_at, label))

    def chat(self, messages, *args, **kwargs) -> str:
        self.last_usage = rflow.LLMUsage(input_tokens=1, output_tokens=1)
        # An agent's query lives in its system prompt, so scan the whole
        # conversation rather than just the latest "continue" nudge.
        convo = "\n".join(m["content"].lower() for m in messages)

        if "child a slow task" in convo:
            self.mark("childa.task_1 start")
            time.sleep(1.0)
            self.mark("childa.task_1 finish")
            return '```repl\ndone("A done")\n```'

        if "child b two-step task" in convo:
            if "childb task_1 exec" not in convo:
                self.mark("childb.task_1 start")
                self.mark("childb.task_1 finish")
                return '```repl\nprint("childb task_1 exec")\n```'
            self.mark("childb.task_2 start")
            self.mark("childb.task_2 finish")
            return '```repl\ndone("B done")\n```'

        return (
            "```repl\n"
            "results = await launch_subagents([\n"
            '    {"name": "childa", "query": "Child A slow task"},\n'
            '    {"name": "childb", "query": "Child B two-step task"},\n'
            "])\n"
            'done(" | ".join(results))\n'
            "```"
        )


def run_case(*, eager_children: bool) -> None:
    llm = TimelineLLM()
    flow = rflow.Flow(
        llm,
        eager_children=eager_children,
        max_depth=2,
        max_iters=8,
        max_concurrency=2,
    )

    graph = flow.start("Show eager child scheduling.")
    steps = 0
    while not graph.finished:
        graph = flow.step(graph)
        steps += 1

    mode = "eager_children=True" if eager_children else "eager_children=False"
    print(f"\n=== {mode} ===")
    print(f"outer step() calls: {steps}")
    for t, label in llm.events:
        print(f"{t:0.3f}s  {label}")
    print("result:", graph.result())


def main() -> None:
    run_case(eager_children=False)
    run_case(eager_children=True)


if __name__ == "__main__":
    main()
