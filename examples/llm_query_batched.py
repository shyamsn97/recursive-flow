"""Exercise ``llm_query_batched(...)`` as a core REPL tool.

Deterministic and offline: ``GuidedLLM`` returns a root REPL block that must
call ``llm_query_batched(...)``, then answers each batched prompt itself.

Run:
    python examples/llm_query_batched.py
"""

from __future__ import annotations

from pathlib import Path

import threading

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


REVIEWS = [
    "The new search UI is fast and surprisingly easy to use.",
    "The export job failed twice and the error message was useless.",
    "The dashboard loads, but I do not feel strongly about it yet.",
]


class GuidedLLM(rflow.LLMClient):
    """Fake model that lets us verify root -> llm_query_batched -> done."""

    def __init__(self) -> None:
        self.batch_prompts: list[str] = []
        self._lock = threading.Lock()

    def chat(self, messages, *args, **kwargs) -> str:
        self.last_usage = rflow.LLMUsage(input_tokens=25, output_tokens=10)
        text = messages[-1]["content"]

        if "Classify this review" in text:
            with self._lock:
                self.batch_prompts.append(text)
            return classify_review(text)

        return root_repl_block()


def classify_review(prompt: str) -> str:
    lower = prompt.lower()
    if "fast" in lower or "easy" in lower:
        return "positive"
    if "failed" in lower or "useless" in lower:
        return "negative"
    return "neutral"


def root_repl_block() -> str:
    return (
        "```repl\n"
        f"reviews = {REVIEWS!r}\n"
        "prompts = [\n"
        "    'Classify this review as positive, negative, or neutral: ' + review\n"
        "    for review in reviews\n"
        "]\n"
        "labels = llm_query_batched(prompts)\n"
        "print('llm_query_batched returned:', labels)\n"
        "done('\\n'.join(f'{label}: {review}' for label, review in zip(labels, reviews)))\n"
        "```"
    )


def main() -> None:
    llm = GuidedLLM()
    flow = rflow.Flow(
        llm,
        max_depth=0,
        max_iters=3,
        max_concurrency=3,
        include_llm_query=True,
    )

    graph = flow.start(
        "Classify the reviews. You must use `llm_query_batched(prompts)` for "
        "the per-review classifications, then call done(...) with one line per review."
    )
    while not graph.finished:
        graph = flow.step(graph)
        print(graph.tree())

    print("\nBatched prompts sent:")
    for prompt in llm.batch_prompts:
        print("-", prompt)

    print("\nFinal answer:")
    print(graph.result())
    _save_example_graph(graph, __file__, "llm-query-batched")


if __name__ == "__main__":
    main()
