"""Let one live rlmflow agent repair another live agent's graph.

This example uses live LLM clients:

    python examples/control/graph_controller_agent.py --worker-model gpt-5-mini --controller-model gpt-5

The controller is a normal :class:`rflow.Flow`. Its only special power is the
example-local ``WorkerPoolControl`` class below, whose host-side tools close
over a list of worker flows and graphs.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import rflow
from rflow.prompts.default import DEFAULT_BUILDER

LIVE_DELAY_SECONDS = 0.35


@dataclass
class WorkerRun:
    name: str
    approach: str
    worker: rflow.Flow
    graph: rflow.Graph


class WorkerPoolControl:
    """Example-local host operations for supervising several worker graphs."""

    def __init__(
        self,
        runs: list[WorkerRun] | None = None,
        *,
        worker_factory: Callable[[], rflow.Flow] | None = None,
        snippet_chars: int = 500,
    ) -> None:
        self.runs = {run.name: run for run in (runs or [])}
        self.worker_factory = worker_factory
        self.snippet_chars = snippet_chars

    def tools(self) -> list[object]:
        """Return the bound methods that should be registered as controller tools."""
        return [
            self.create_worker,
            self.inspect_worker_specs,
            self.inspect_graph_shapes,
            self.inspect_workers,
            self.inspect_worker,
            self.advance_worker,
            self.inject_worker_note,
            self.replace_worker_supervisor,
            self.terminate_worker,
            self.worker_result,
            self.worker_results,
        ]

    def graphs(self) -> list[rflow.Graph]:
        return [run.graph for run in self.runs.values()]

    def _run(self, name: str) -> WorkerRun:
        try:
            return self.runs[name]
        except KeyError as exc:
            available = ", ".join(self.runs)
            raise KeyError(f"unknown worker {name!r}; available: {available}") from exc

    @rflow.tool("Create and start a named worker graph with the requested approach.", proxy=True)
    def create_worker(self, *, name: str, approach: str) -> str:
        if name in self.runs:
            raise ValueError(f"worker {name!r} already exists")
        if self.worker_factory is None:
            raise RuntimeError("this WorkerPoolControl was not configured to create workers")
        worker = self.worker_factory()
        graph = worker.start(worker_prompt(name, approach))
        self.runs[name] = WorkerRun(
            name=name,
            approach=approach,
            worker=worker,
            graph=graph,
        )
        return self.inspect_worker_specs()

    @rflow.tool("Return a bounded summary of every worker graph.", proxy=True)
    def inspect_workers(self) -> str:
        if not self.runs:
            return "<no workers created>"
        return "\n\n".join(
            self._summary(run, title=f"worker={run.name!r} approach={run.approach!r}")
            for run in self.runs.values()
        )

    @rflow.tool("Return worker names, assigned approaches, and root query previews.", proxy=True)
    def inspect_worker_specs(self) -> str:
        if not self.runs:
            return "<no workers created>"
        lines = []
        for run in self.runs.values():
            lines.append(f"worker={run.name!r}")
            lines.append(f"  approach={run.approach!r}")
            self._append_snippet(lines, "query", run.graph.query)
        return "\n".join(lines)

    @rflow.tool("Return compact graph-shape summaries for diversity checks.", proxy=True)
    def inspect_graph_shapes(self) -> str:
        if not self.runs:
            return "<no workers created>"
        lines = []
        for run in self.runs.values():
            graph = run.graph
            node_types = [node.type for node in graph.all_nodes]
            child_edges = [
                f"{child.parent_agent_id}->{child.agent_id}"
                for child in graph.walk()
                if child.parent_agent_id is not None
            ]
            lines.append(f"worker={run.name!r}")
            lines.append(f"  agents={list(graph.agents)}")
            lines.append(f"  node_types={node_types}")
            lines.append(f"  child_edges={child_edges}")
            lines.append(f"  runnable={graph.get_runnable_nodes()}")
            lines.append(f"  finished={graph.finished}")
        return "\n".join(lines)

    @rflow.tool("Return a bounded summary of one worker graph.", proxy=True)
    def inspect_worker(self, *, worker: str) -> str:
        return self._summary(self._run(worker), title=f"worker={worker!r}")

    def _summary(self, run: WorkerRun, *, title: str) -> str:
        graph = run.graph
        lines = [
            title,
            f"finished={graph.finished}",
            f"agents={list(graph.agents)}",
            f"runnable={graph.get_runnable_nodes()}",
            f"tokens={graph.tokens()}",
            f"root_result={graph.result()!r}",
        ]
        for agent_id, subgraph in graph.agents.items():
            current = subgraph.current()
            in_tokens, out_tokens = subgraph.tokens(recursive=False)
            current_type = current.type if current is not None else "<empty>"
            lines.append(
                f"{agent_id}: current={current_type} tokens=({in_tokens}, {out_tokens})"
            )
            if current is None:
                continue
            if isinstance(current, rflow.SupervisingOutput):
                lines.append(f"  waiting_on={current.waiting_on}")
                self._append_snippet(lines, "output", current.output)
            elif isinstance(current, rflow.ErrorOutput):
                self._append_snippet(lines, "error", current.error)
                self._append_snippet(lines, "output", current.output)
            elif isinstance(current, rflow.DoneOutput):
                self._append_snippet(lines, "result", current.result)
            elif isinstance(current, rflow.ExecOutput):
                self._append_snippet(lines, "output", current.output)
            elif isinstance(current, rflow.LLMOutput):
                self._append_snippet(lines, "code", current.code)
        return "\n".join(lines)

    @rflow.tool("Advance one worker graph by one scheduler tick.", proxy=True)
    def advance_worker(self, *, worker: str, salvage: bool = False) -> str:
        run = self._run(worker)
        run.graph = run.worker.step(run.graph, salvage=salvage)
        return self.inspect_worker(worker=worker)

    @rflow.tool("Inject controller feedback into one worker agent.", proxy=True)
    def inject_worker_note(self, *, worker: str, target: str, note: str) -> str:
        run = self._run(worker)
        run.graph = run.graph.inject_output(
            target=target,
            output=f"Controller note: {note}",
            content=f"Controller note: {note}",
        )
        return self.inspect_worker(worker=worker)

    @rflow.tool("Replace the latest supervisor node for one worker agent.", proxy=True)
    def replace_worker_supervisor(self, *, worker: str, target: str, note: str) -> str:
        run = self._run(worker)
        matches = run.graph.all_nodes.where(
            lambda node: node.agent_id == target
            and isinstance(node, rflow.SupervisingOutput)
        )
        if not matches:
            return f"no supervising_output found for {target!r} in worker {worker!r}"
        run.graph = run.graph.replace_node(
            matches[-1],
            rflow.ExecOutput(
                output=f"Controller replacement: {note}",
                content=f"Controller replacement: {note}",
            ),
            truncate="descendants",
        )
        return self.inspect_worker(worker=worker)

    @rflow.tool("Request forced final answers from one worker's agents.", proxy=True)
    def terminate_worker(
        self, *, worker: str, agent_ids: list[str] | None = None
    ) -> str:
        run = self._run(worker)
        run.worker.terminate(agent_ids)
        targets = "all unfinished agents" if agent_ids is None else ", ".join(agent_ids)
        return (
            f"termination requested for {targets} in worker {worker!r}; "
            "call advance_worker(worker=...) to commit it"
        )

    @rflow.tool("Return one worker's final result if it is finished.", proxy=True)
    def worker_result(self, *, worker: str) -> str:
        graph = self._run(worker).graph
        return graph.result() if graph.finished else "<worker not finished>"

    @rflow.tool("Return final results for all finished workers.", proxy=True)
    def worker_results(self) -> dict[str, str]:
        return {
            name: run.graph.result()
            for name, run in self.runs.items()
            if run.graph.finished
        }

    def _append_snippet(self, lines: list[str], label: str, value: str) -> None:
        if not value:
            return
        text = " ".join(value.split())
        if len(text) > self.snippet_chars:
            text = text[: self.snippet_chars] + "..."
        lines.append(f"  {label}={text!r}")


MATH_TASK = "Solve the equation x^2 - 5x + 6 = 0. Return all real solutions."

WORKER_APPROACHES = {
    "one-root": (
        "Risky baseline: delegate first to a child named `one-root` that is "
        "asked to find only one root. This may lose another solution."
    ),
    "factor": "Solve by factoring the quadratic and verify every root by substitution.",
    "formula": "Solve with the quadratic formula and verify every root by substitution.",
}


def worker_prompt(name: str, approach: str) -> str:
    return f"""\
Solve the equation x^2 - 5x + 6 = 0.

Assigned worker: {name}
Assigned approach: {approach}
"""


CONTROLLER_POLICY_TEXT = """\
You are a graph controller supervising a pool of separate worker rlmflow runs.
Do not solve the worker's task yourself. Your only job is to inspect and edit
the worker graph through the provided host-side tools.

Available control loop:
1. First create diversified workers with create_worker(name=..., approach=...).
2. Then call inspect_worker_specs() to verify the intended approaches differ.
3. Use inspect_graph_shapes() while running to verify the graphs actually diverge
   (different child edges, node types, or finished states), not just the labels.
4. Use inspect_workers() before decisions that need current outputs/errors.
5. Advance diversified workers with advance_worker(worker="...").
6. If an agent is stuck, looping, or missing a key instruction, call
   inject_worker_note(worker=..., target=..., note=...) with a concise,
   evidence-based note.
7. If a supervisor delegated down a bad route, call replace_worker_supervisor(
   worker=..., target=..., note=...) and then use
   advance_worker(worker=..., salvage=True) on the next tick.
8. If the worker has done enough and should stop, call terminate_worker() and
   then advance_worker() until the forced final answer lands.
9. When enough independent workers agree on all roots, call done(...) with the
   selected result and short evidence.

Intervention rules:
- Prefer advancing over editing when the worker is making useful progress.
- Make at most one graph edit before advancing again.
- Every injected note should name the concrete evidence from inspect_worker().
- Keep controller messages short; the worker graph remains the source of truth.

Example moves:
- Create the pool up front:
  create_worker(name="one-root", approach="delegate first to a child asked to find only one root")
  create_worker(name="factor", approach="solve by factoring and verify each root")
  create_worker(name="formula", approach="solve with the quadratic formula and verify each root")
- If inspect_worker_specs() shows duplicate approaches, inject notes or prioritize
  workers so the pool covers genuinely different methods.
- If inspect_graph_shapes() shows every worker has only root and the same current
  type, advance different named workers before deciding.
- If worker="one-root" is waiting on a child asked for only one root, keep
  advancing until the child result is visible.
- If worker="one-root" would resume with one root only, call
  replace_worker_supervisor(worker="one-root", target="root", note="...") to
  turn the bad wait into a controller observation, then call
  advance_worker(worker="one-root", salvage=True).
- Once two approaches agree on x = 2 and x = 3, call done(...) with the answer
  and name the agreeing workers.
"""


CONTROLLER_PROMPT_BUILDER = DEFAULT_BUILDER.section(
    "graph_controller",
    CONTROLLER_POLICY_TEXT,
    title="Graph Controller",
    after="role",
)


CONTROLLER_TASK = """\
Supervise the worker until it gives all solutions to x^2 - 5x + 6 = 0.

Create multiple workers with deliberately different approaches, then advance and
compare them. Use these initial workers:
- name: one-root; approach: delegate first to a child asked to find only one root
- name: factor; approach: solve by factoring and verify each root
- name: formula; approach: solve with the quadratic formula and verify each root

If the `one-root` worker takes an incomplete route, repair that worker graph
instead of letting it resume with a partial answer. Finish only when the pool
has a complete answer with independent agreement.

Before advancing, inspect both the worker specs and graph shapes so you can tell
whether the pool is actually diversified by query and by resulting graph
structure.
"""


def build_llm(model: str) -> rflow.LLMClient:
    return (
        rflow.AnthropicClient(model)
        if model.startswith("claude")
        else rflow.OpenAIClient(model)
    )


def make_worker(
    llm: rflow.LLMClient, *, max_depth: int = 2, max_iters: int = 12
) -> rflow.Flow:
    return rflow.Flow(llm, max_depth=max_depth, max_iters=max_iters)


def make_controller(
    control: WorkerPoolControl,
    llm: rflow.LLMClient,
    *,
    max_iters: int = 30,
) -> rflow.Flow:
    runtime = rflow.LocalRuntime()
    runtime.register_tools(control.tools())
    return rflow.Flow(
        llm,
        runtime=runtime,
        max_depth=0,
        max_iters=max_iters,
        prompt_builder=CONTROLLER_PROMPT_BUILDER,
    )


def _record_snapshot(
    snapshots: list[tuple[str, list[WorkerRun]]],
    label: str,
    control: WorkerPoolControl,
) -> None:
    signature = tuple((run.name, run.graph.tree()) for run in control.runs.values())
    if snapshots:
        previous = tuple((run.name, run.graph.tree()) for run in snapshots[-1][1])
        if previous == signature:
            return
    snapshots.append(
        (
            label,
            [
                WorkerRun(
                    name=run.name,
                    approach=run.approach,
                    worker=run.worker,
                    graph=run.graph.copy(),
                )
                for run in control.runs.values()
            ],
        )
    )


def _worker_pool_text(control: WorkerPoolControl) -> str:
    return "\n\n".join(
        f"[{run.name}] {run.approach}\n{run.graph.tree()}"
        for run in control.runs.values()
    )


def _render_worker_pool(control: WorkerPoolControl):
    from rich.panel import Panel

    return Panel(_worker_pool_text(control), title="Worker Pool")


def _print_worker_timeline(snapshots: list[tuple[str, list[WorkerRun]]]) -> None:
    print("\nWorker pool checkpoints:")
    for label, runs in snapshots:
        print(f"\n--- {label} ---")
        for run in runs:
            print(f"\n[{run.name}] {run.approach}")
            print(run.graph.tree())
            if run.graph.result():
                print("result:", run.graph.result())


def _drive_controller(
    controller: rflow.Flow,
    controller_graph: rflow.Graph,
    control: WorkerPoolControl,
    *,
    show_live: bool,
    delay: float,
) -> tuple[rflow.Graph, list[tuple[str, list[WorkerRun]]]]:
    snapshots: list[tuple[str, list[WorkerRun]]] = []
    _record_snapshot(snapshots, "workers start", control)
    step = 0
    if not show_live:
        while not controller_graph.finished:
            controller_graph = controller.step()
            step += 1
            _record_snapshot(snapshots, f"after controller step {step}", control)
            print(_worker_pool_text(control))
        return controller_graph, snapshots

    from rich.live import Live

    with Live(
        _render_worker_pool(control),
        vertical_overflow="visible",
        auto_refresh=False,
        redirect_stdout=False,
        redirect_stderr=False,
    ) as live:
        time.sleep(delay)
        while not controller_graph.finished:
            controller_graph = controller.step()
            step += 1
            _record_snapshot(snapshots, f"after controller step {step}", control)
            live.update(_render_worker_pool(control), refresh=True)
            time.sleep(delay)
    return controller_graph, snapshots


def _save_worker_graphs(control: WorkerPoolControl, out_dir: Path) -> None:
    for run in control.runs.values():
        path = run.graph.save(out_dir / "workers" / run.name)
        print(f"Graph saved to {path}")


def _best_finished_result(control: WorkerPoolControl) -> str:
    for run in control.runs.values():
        if run.graph.finished and run.graph.result():
            return run.graph.result()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Live graph-controller agent example")
    parser.add_argument("--worker-model", default="gpt-5-mini")
    parser.add_argument("--controller-model", default="gpt-5")
    parser.add_argument("--worker-max-depth", type=int, default=2)
    parser.add_argument("--worker-max-iters", type=int, default=12)
    parser.add_argument("--controller-max-iters", type=int, default=30)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--delay", type=float, default=LIVE_DELAY_SECONDS)
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "_runs" / "graph_controller_runs"),
        help="Save worker/controller graphs here.",
    )
    args = parser.parse_args()

    control = WorkerPoolControl(
        worker_factory=lambda: make_worker(
            build_llm(args.worker_model),
            max_depth=args.worker_max_depth,
            max_iters=args.worker_max_iters,
        )
    )
    controller = make_controller(
        control,
        build_llm(args.controller_model),
        max_iters=args.controller_max_iters,
    )
    controller_graph = controller.start(CONTROLLER_TASK)
    controller_graph, snapshots = _drive_controller(
        controller,
        controller_graph,
        control,
        show_live=not args.no_viz,
        delay=args.delay,
    )
    controller_answer = controller_graph.result()

    _print_worker_timeline(snapshots)
    print("controller answer:", controller_answer)
    print("best worker result:", _best_finished_result(control))

    out_dir = Path(args.out_dir)
    _save_worker_graphs(control, out_dir)
    path = controller_graph.save(out_dir / "controller")
    print(f"Graph saved to {path}")


if __name__ == "__main__":
    main()
