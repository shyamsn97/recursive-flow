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

from pathlib import Path

import time
from dataclasses import dataclass, field
from threading import Lock

import rflow


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



@dataclass
class TimelineLLM(rflow.LLMClient):
    # The fake only sleeps/records, so concurrent calls are safe; this lets the
    # bounded LLM channel run sibling calls in parallel instead of serializing.
    thread_safe = True
    started_at: float = field(default_factory=time.perf_counter)
    events: list[tuple[float, str]] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)

    def mark(self, label: str) -> None:
        with self.lock:
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


@dataclass
class CaseResult:
    eager_children: bool
    steps: int
    result: str
    events: list[tuple[float, str]]

    def time_of(self, label: str) -> float:
        for t, event_label in self.events:
            if event_label == label:
                return t
        raise KeyError(label)


def run_case(*, eager_children: bool) -> CaseResult:
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
    suffix = "eager-true" if eager_children else "eager-false"
    _save_example_graph(
        graph,
        __file__,
        "eager-children",
        out_dir=_example_run_dir(__file__, "eager-children") / suffix,
    )
    return CaseResult(
        eager_children=eager_children,
        steps=steps,
        result=str(graph.result()),
        events=list(llm.events),
    )


def check_timeline(lazy: CaseResult, eager: CaseResult) -> None:
    """Assert the visible scheduling difference this example is meant to show."""

    lazy_b2 = lazy.time_of("childb.task_2 start")
    lazy_a_done = lazy.time_of("childa.task_1 finish")
    eager_b2 = eager.time_of("childb.task_2 start")
    eager_a_done = eager.time_of("childa.task_1 finish")

    print("\n=== verdict ===")
    print(
        "non-eager barrier:",
        f"child B step 2 starts at {lazy_b2:0.3f}s,",
        f"after child A finishes at {lazy_a_done:0.3f}s",
    )
    print(
        "eager refill:",
        f"child B step 2 starts at {eager_b2:0.3f}s,",
        f"before child A finishes at {eager_a_done:0.3f}s",
    )

    if lazy_b2 <= lazy_a_done:
        raise AssertionError("non-eager mode did not wait for the sibling barrier")
    if eager_b2 >= eager_a_done:
        raise AssertionError("eager mode did not refill ready child work immediately")


def main() -> None:
    lazy = run_case(eager_children=False)
    eager = run_case(eager_children=True)
    check_timeline(lazy, eager)


if __name__ == "__main__":
    main()
