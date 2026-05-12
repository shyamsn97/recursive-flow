"""End-to-end tests for the core RLMFlow engine + Graph data model."""

from __future__ import annotations

from importlib.util import find_spec

from rlmflow import (
    ErrorNode,
    Graph,
    RLMConfig,
    RLMFlow,
    ResultNode,
    Workspace,
)
from rlmflow.llm import LLMClient
from rlmflow.runtime.local import LocalRuntime
from rlmflow.utils.trace import load_trace, save_trace


class StaticLLM(LLMClient):
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def chat(self, messages, *args, **kwargs) -> str:
        return self.reply


def _agent(reply: str = '```repl\ndone("ok")\n```', **config_kwargs) -> RLMFlow:
    config_kwargs.setdefault("max_iterations", 3)
    return RLMFlow(
        StaticLLM(reply),
        runtime=LocalRuntime(),
        config=RLMConfig(**config_kwargs),
    )


def _run_full(agent: RLMFlow, graph: Graph) -> Graph:
    while not graph.finished:
        graph = agent.step(graph)
    return graph


# ── basic lifecycle ──────────────────────────────────────────────────


def test_start_returns_graph_with_root_agent():
    agent = _agent()
    graph = agent.start("say ok")

    assert isinstance(graph, Graph)
    assert graph.root_agent_id == "root"
    assert graph.query == "say ok"
    assert [state.type for state in graph.states] == ["query"]


def test_step_drives_observation_to_result_in_one_call():
    agent = _agent()
    graph = agent.step(agent.start("say ok"))

    assert isinstance(graph.current(), ResultNode)
    assert graph.result() == "ok"
    assert [s.type for s in graph.states] == ["query", "action", "result"]


def test_run_returns_result_string():
    agent = _agent()
    assert agent.run("say ok") == "ok"


def test_tree_contains_per_agent_timeline():
    agent = _agent()
    graph = _run_full(agent, agent.start("say ok"))
    tree = graph.tree()

    assert "root" in tree
    assert "query" in tree
    assert "result -> ok" in tree


# ── delegation ───────────────────────────────────────────────────────


def test_delegation_resumes_parent_with_child_result():
    class ScriptedLLM(LLMClient):
        def chat(self, messages, *args, **kwargs):
            prompt = messages[-1]["content"].lower()
            if "child task" in prompt:
                return '```repl\ndone("child-result")\n```'
            return (
                "```repl\n"
                'h = delegate("child", "child task", "")\n'
                "results = yield wait(h)\n"
                'done("parent:" + results[0])\n'
                "```"
            )

    agent = RLMFlow(
        ScriptedLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=1, max_iterations=5),
    )
    graph = _run_full(agent, agent.start("parent task"))

    assert graph.result() == "parent:child-result"
    assert "root.child" in graph
    assert graph["root.child"].depth == 1
    assert graph["root.child"].result() == "child-result"
    assert list(graph.children) == ["root.child"]


def test_resumed_parent_can_delegate_again_after_first_child_completes():
    class ScriptedLLM(LLMClient):
        def chat(self, messages, *args, **kwargs):
            prompt = messages[-1]["content"].lower()
            if "child task" in prompt:
                return '```repl\ndone("child-result")\n```'
            if "verify task" in prompt:
                return '```repl\ndone("verified")\n```'
            return (
                "```repl\n"
                'h = delegate("child", "child task", "")\n'
                "child_results = yield wait(h)\n"
                'v = delegate("verify", "verify task", "")\n'
                "verdict = yield wait(v)\n"
                'done("parent:" + verdict[0])\n'
                "```"
            )

    agent = RLMFlow(
        ScriptedLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=1, max_iterations=5),
    )
    graph = _run_full(agent, agent.start("parent task"))

    assert graph.result() == "parent:verified"
    assert set(graph.children) == {"root.child", "root.verify"}


# ── persistence ──────────────────────────────────────────────────────


def test_workspace_session_persists_states_and_context_payload(tmp_path):
    ws = Workspace.create(tmp_path / "workspace")
    agent = RLMFlow(
        StaticLLM('```repl\ndone("ok")\n```'),
        workspace=ws,
        config=RLMConfig(max_iterations=2),
    )

    graph = _run_full(agent, agent.start("say ok", context="hello"))

    assert isinstance(graph.current(), ResultNode)
    assert ws.context.read("context") == "hello"
    reload = ws.session.load_graph()
    assert [s.type for s in reload.states] == ["query", "action", "result"]


# ── tree, transcript, sessions ───────────────────────────────────────


def test_agent_transcript_renders_one_agent_chain():
    agent = _agent()
    graph = _run_full(agent, agent.start("say ok"))

    transcript = graph.transcript(include_system=False)

    assert "--- query ---\nsay ok" in transcript
    assert '--- assistant ---\n```repl\ndone("ok")\n```' in transcript
    assert "--- result ---\nok" in transcript


def test_graph_session_flattens_every_agent():
    class ScriptedLLM(LLMClient):
        def chat(self, messages, *args, **kwargs):
            prompt = messages[-1]["content"].lower()
            if "child" in prompt:
                return '```repl\ndone("c")\n```'
            return (
                "```repl\n"
                'h = delegate("c", "child", "")\n'
                "r = yield wait(h)\n"
                "done(r[0])\n"
                "```"
            )

    agent = RLMFlow(
        ScriptedLLM(), runtime=LocalRuntime(), config=RLMConfig(max_depth=1)
    )
    graph = _run_full(agent, agent.start("parent"))
    session = graph.session(include_system=False)

    assert "[root] query" in session
    assert "[root.c] query" in session
    assert "[root.c] result" in session


# ── plot kinds ───────────────────────────────────────────────────────


def test_graph_plot_returns_plotly_figure():
    if find_spec("plotly") is None:
        return
    agent = _agent()
    graph = _run_full(agent, agent.start("say ok"))

    fig = graph.plot(title="sample")
    assert fig.layout.title.text.startswith("<b>sample</b>")
    assert len(fig.data) >= 2


def test_graph_plot_supports_static_formats():
    agent = _agent()
    graph = _run_full(agent, agent.start("say ok"))

    assert graph.plot("tree").startswith("● root")
    assert graph.plot("mermaid").startswith("stateDiagram-v2")
    assert graph.plot("flowchart").startswith("flowchart TD")
    assert graph.plot("dot").startswith("digraph rlmflow")
    assert "root" in graph.plot("d2")


def test_graph_plot_supports_gantt_html():
    agent = _agent()
    graph = _run_full(agent, agent.start("say ok"))
    html = graph.plot("gantt", title="sample gantt")

    assert "<html>" in html
    assert "sample gantt" in html


# ── trace persistence ────────────────────────────────────────────────


def test_trace_persists_graph_snapshots(tmp_path):
    agent = _agent()
    graphs = [agent.start("say ok")]
    while not graphs[-1].finished:
        graphs.append(agent.step(graphs[-1]))

    path = save_trace(graphs, tmp_path / "trace")
    trace = load_trace(path)

    assert len(trace.graphs) == len(graphs)
    assert trace.graphs[-1].result() == "ok"
    assert isinstance(trace.graphs[0], Graph)


# ── filters over states ──────────────────────────────────────────────


def _delegating_graph() -> Graph:
    """A finished run with one ErrorNode (no_code_block) and one ResultNode."""

    class StumblingLLM(LLMClient):
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                # No code block → engine records an ErrorNode.
                return "I forgot the code block."
            return '```repl\ndone("ok")\n```'

    agent = RLMFlow(
        StumblingLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_iterations=5, max_depth=0),
    )
    return _run_full(agent, agent.start("kick"))


def test_graph_errors_finds_only_error_states():
    graph = _delegating_graph()
    errors = graph.nodes.errors()
    assert errors
    assert all(isinstance(e, ErrorNode) for e in errors)


def test_graph_results_finds_only_result_states():
    graph = _delegating_graph()
    results = graph.nodes.results()
    assert results
    assert all(isinstance(e, ResultNode) for e in results)


def test_graph_where_filters_by_predicate_and_kwargs():
    graph = _delegating_graph()
    actions = graph.nodes.where(type="action")
    assert actions and all(e.type == "action" for e in actions)

    on_root = graph.nodes.where(lambda e: e.agent_id == "root")
    assert on_root and all(e.agent_id == "root" for e in on_root)


def test_graph_indexing_returns_subgraph_rooted_at_agent():
    graph = _delegating_graph()
    sub = graph["root"]
    assert sub.agent_id == "root"
    assert isinstance(sub, type(graph))


# ── model labels ─────────────────────────────────────────────────────


def test_graph_model_label_combines_key_and_actual():
    g = Graph(agent_id="root.fast", config={"model": "fast"}, model="gpt-5-mini")
    assert g.model_label == "fast:gpt-5-mini"


def test_graph_model_label_falls_back_to_key():
    g = Graph(agent_id="root", config={"model": "default"})
    assert g.model_label == "default"


# ── child runtime + scope ────────────────────────────────────────────


def test_child_agent_has_its_own_runtime_session():
    class ScriptedLLM(LLMClient):
        def chat(self, messages, *args, **kwargs):
            prompt = messages[-1]["content"].lower()
            if "child task" in prompt:
                return '```repl\nprint(AGENT_ID, DEPTH)\ndone("child")\n```'
            return (
                "```repl\n"
                'h = delegate("child", "child task", "")\n'
                "results = yield wait(h)\n"
                "done(results[0])\n"
                "```"
            )

    agent = RLMFlow(
        ScriptedLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=1, max_iterations=5),
    )
    graph = _run_full(agent, agent.start("parent"))

    child = graph["root.child"]
    assert child.depth == 1
    assert child.runtime is not None
    assert child.runtime.id != "root"
    assert child.result() == "child"


def test_delegate_passes_child_context_payload(tmp_path):
    class ScriptedLLM(LLMClient):
        def chat(self, messages, *args, **kwargs):
            prompt = messages[-1]["content"].lower()
            if "child task" in prompt:
                return '```repl\ndone(CONTEXT.read())\n```'
            return (
                "```repl\n"
                'h = delegate("child", "child task", "child payload")\n'
                "results = yield wait(h)\n"
                "done(results[0])\n"
                "```"
            )

    workspace = Workspace.create(tmp_path / "workspace")
    agent = RLMFlow(
        ScriptedLLM(),
        workspace=workspace,
        config=RLMConfig(max_depth=1, max_iterations=5),
    )
    graph = _run_full(agent, agent.start("parent", context="root payload"))

    assert graph.result() == "child payload"
    assert (
        workspace.context.read("context", agent_id="root.child") == "child payload"
    )
