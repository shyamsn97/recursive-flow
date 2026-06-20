"""Rendering a Graph: text trees, transcripts, HTML viewer.

Several read-only renderers ship with recursive-flow:

- ``graph.tree()``                 — ASCII tree of agents + states
- ``graph.session(...)``           — full chat-style transcript across the run
- ``graph[aid].transcript()``      — one agent's transcript only
- ``graph.save_html(path)``        — interactive viewer page over the snapshots

This script writes ``examples/_runs/graph-render/viewer.html`` and prints the
text renderers to stdout.

Run:
    python examples/graph/07_render.py
    open examples/_runs/graph-render/viewer.html
"""

from __future__ import annotations

from pathlib import Path
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



def build_graph() -> rflow.Graph:
    root_q = rflow.UserQuery(agent_id="root", seq=0, content="write hello world")
    root_call = rflow.LLMAction(agent_id="root", seq=1, model="demo")
    root_reply = rflow.LLMOutput(
        agent_id="root",
        seq=2,
        reply="I'll delegate the file write.",
        code='await launch_subagents([{"name": "hello", "query": "write hello.py"}])',
        input_tokens=120,
        output_tokens=30,
    )
    root_exec = rflow.ExecAction(agent_id="root", seq=3, code=root_reply.code)
    root_sup = rflow.SupervisingOutput(
        agent_id="root",
        seq=4,
        waiting_on=["root.hello"],
    )
    root_done = rflow.DoneOutput(agent_id="root", seq=5, result="hello.py created")

    hello_q = rflow.UserQuery(agent_id="root.hello", seq=0, content="write hello.py")
    hello_reply = rflow.LLMOutput(
        agent_id="root.hello",
        seq=1,
        reply="writing the file",
        code='write_file("hello.py", "print(\\"hello\\")\\n")',
        input_tokens=80,
        output_tokens=20,
    )
    hello_exec = rflow.ExecAction(agent_id="root.hello", seq=2, code=hello_reply.code)
    hello_done = rflow.DoneOutput(agent_id="root.hello", seq=3, result="wrote hello.py")

    hello = rflow.Graph.from_meta_dict(
        {
            "agent_id": "root.hello",
            "depth": 1,
            "parent_agent_id": "root",
            "parent_node_id": root_reply.id,
            "query": "write hello.py",
        },
        nodes=[hello_q, hello_reply, hello_exec, hello_done],
    )
    return rflow.Graph.from_meta_dict(
        {"agent_id": "root", "depth": 0, "query": "write hello world"},
        nodes=[root_q, root_call, root_reply, root_exec, root_sup, root_done],
        children={"root.hello": hello},
    )


def banner(title: str) -> None:
    print("\n" + "─" * 60)
    print(title)
    print("─" * 60)


def main() -> None:
    g = build_graph()

    banner("graph.tree() — ASCII summary")
    print(g.tree())

    banner("graph['root.hello'].transcript() — single-agent transcript")
    print(g["root.hello"].transcript(include_system=False))

    banner("graph.session() — full chat-style transcript")
    print(g.session(include_system=False))

    banner("graph.save_html(...) — interactive viewer over the history")
    out_dir = _example_run_dir(__file__, "graph-render")
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = g.save_html(out_dir / "viewer.html")
    print(f"wrote {html_path} ({html_path.stat().st_size:,} bytes)")
    print(f"\nopen with: open {html_path}")
    _save_example_graph(g, __file__, "graph-render")


if __name__ == "__main__":
    main()
