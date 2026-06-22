from __future__ import annotations

import io
import inspect
import subprocess
import sys

from rich.console import Console

from rflow import (
    DoneOutput,
    ExecAction,
    ExecOutput,
    Graph,
    LLMOutput,
    SupervisingOutput,
    UserQuery,
)
from rflow.tui import (
    _context_inputs,
    agent_table,
    chat_bubbles,
    error_table,
    latest_table,
    node_counts_table,
    render_full_tree_panel,
    run_stats_table,
    waiting_table,
)


def _sample_graph() -> Graph:
    root = Graph(
        agent_id="root",
        query="research the thing",
        nodes=[
            UserQuery(agent_id="root", seq=0, global_step=0, content="research the thing"),
            LLMOutput(
                agent_id="root",
                seq=1,
                global_step=1,
                reply="I will inspect it.",
                code='print("checking")',
                model="test",
                input_tokens=10,
                output_tokens=6,
            ),
            ExecAction(
                agent_id="root",
                seq=2,
                global_step=2,
                code='print("checking")',
            ),
            SupervisingOutput(
                agent_id="root",
                seq=3,
                global_step=2,
                waiting_on=["root.lookup"],
            ),
        ],
    )
    child = Graph(
        agent_id="root.lookup",
        parent_agent_id="root",
        depth=1,
        query="lookup",
        nodes=[
            UserQuery(agent_id="root.lookup", seq=0, global_step=3, content="lookup"),
            ExecOutput(
                agent_id="root.lookup",
                seq=1,
                global_step=4,
                output="found evidence",
                content="found evidence",
            ),
            DoneOutput(
                agent_id="root.lookup",
                seq=2,
                global_step=5,
                result="child answer",
            ),
        ],
    )
    root.children[child.agent_id] = child
    return root


def _text(renderable) -> str:
    out = io.StringIO()
    Console(file=out, width=120, force_terminal=False).print(renderable)
    return out.getvalue()


def test_tui_chat_bubbles_render_every_node_once():
    graph = _sample_graph()
    bubbles = chat_bubbles(graph)

    assert len(bubbles) == len(list(graph.all_nodes))
    rendered = "\n".join(_text(bubble) for _, bubble in bubbles)
    assert "root / query" in rendered
    assert "root / exec" in rendered
    assert 'print("checking")' in rendered
    assert "shown in execute bubble" in rendered
    assert "root.lookup / done" in rendered

    seen = {node_id for node_id, _ in bubbles}
    assert chat_bubbles(graph, seen=seen) == []


def test_tui_tables_include_core_run_state():
    graph = _sample_graph()
    rendered = "\n".join(
        _text(table)
        for table in [
            run_stats_table(graph, busy=True),
            agent_table(graph),
            node_counts_table(graph),
            waiting_table(graph),
            error_table(graph),
            latest_table(graph),
        ]
    )

    assert "root.lookup" in rendered
    assert "running" in rendered
    assert "supervising" in rendered
    assert "llm" in rendered
    assert "none" in rendered or "0" in rendered


def test_tui_full_tree_panel_shows_nested_execution_tree():
    rendered = _text(render_full_tree_panel(_sample_graph()))

    assert "execution tree" in rendered
    assert "root.lookup" in rendered
    assert "waiting on root.lookup" in rendered
    assert "done - child answer" in rendered
    assert "├" in rendered or "└" in rendered


def test_tui_context_box_maps_to_context_input():
    assert _context_inputs("") is None
    assert _context_inputs("  \n  ") is None
    assert _context_inputs(" supporting material \n") == {
        "context": "supporting material"
    }


def test_rflow_import_exports_tui_without_textual_import():
    code = (
        "import sys, rflow\n"
        "assert callable(rflow.tui)\n"
        "assert hasattr(rflow.Flow, 'tui')\n"
        "assert 'textual' not in sys.modules, 'textual leaked into import path'\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_tui_api_waits_for_chat_input_instead_of_taking_query():
    import rflow

    assert "query" not in inspect.signature(rflow.tui).parameters
    assert "query" not in inspect.signature(rflow.Flow.tui).parameters
