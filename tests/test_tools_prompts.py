"""Phase 3 — tools, builtins, HISTORY, and the ported prompt builder."""

from __future__ import annotations

import pytest

from rflow import (
    DEFAULT_BUILDER,
    FILE_TOOLS,
    Flow,
    PromptBuilder,
    SYSTEM_PROMPT,
    get_tool_metadata,
    tool,
)
from rflow.graph import ChildHandle, WaitRequest
from rflow.integrations.structured import StructuredOutputError
from rflow.repl import DoneSignal
from rflow.tools import format_tool_line
from rflow.tools.builtins import (
    ENV_AGENT_ID,
    ENV_DONE_RESULT,
    ENV_OUTPUT_SCHEMA,
    History,
    make_delegate,
    make_done,
    make_history,
    make_launch_subagents,
    make_show_vars,
    make_wait,
)
from rflow.tools.filesystem import (
    append_file,
    edit_file,
    grep,
    line_count,
    ls,
    read_file,
    read_lines,
    write_file,
)
from rflow.tools.registry import HIDDEN_REPL_TOOL_NAMES, partition_repl_namespace

from .helpers import StubLLM, make_flow, run_to_completion

DONE_OK = '```repl\ndone("ok")\n```'


# ── @tool / metadata ────────────────────────────────────────────────────


def test_tool_decorator_and_metadata():
    @tool("Add two ints.", name="adder")
    def add(a: int, b: int) -> int:
        return a + b

    meta = get_tool_metadata(add)
    assert meta is not None
    assert meta.name == "adder"
    assert meta.description == "Add two ints."
    assert get_tool_metadata(lambda: None) is None


def test_tool_default_name_strips_prefix():
    @tool("x")
    def tool_search() -> None: ...

    assert get_tool_metadata(tool_search).name == "search"


def test_format_tool_line_renders_signature():
    @tool("Read a file.")
    def read_file_(path: str) -> str:
        return path

    line = format_tool_line(read_file_)
    assert line.startswith("- `read_file_(")
    assert line.endswith("`: Read a file.")
    assert "path" in line
    assert format_tool_line(lambda: None) == ""


def test_get_tool_metadata_unwraps_bound_method():
    flow = make_flow()
    # llm_query_batched is a @tool-decorated method
    meta = get_tool_metadata(flow.llm_query_batched)
    assert meta is not None and meta.name == "llm_query_batched"


# ── filesystem tools ────────────────────────────────────────────────────


def test_filesystem_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert write_file("sub/a.txt", "hello\nworld\n").startswith("Wrote")
    assert read_file("sub/a.txt") == "hello\nworld\n"
    assert line_count("sub/a.txt") == 2
    assert read_lines("sub/a.txt", 0, 1) == "hello"
    edit_file("sub/a.txt", ("world", "there"))
    assert "there" in read_file("sub/a.txt")
    assert "sub/a.txt" in ls("sub")
    hits = grep("hello", "sub")
    assert "hello" in hits


def test_file_tools_collection_decorated():
    names = {get_tool_metadata(fn).name for fn in FILE_TOOLS}
    assert {"read_file", "write_file", "grep", "ls"} <= names


# ── registry partition ──────────────────────────────────────────────────


def test_partition_repl_namespace_hides_control_tools():
    flow = make_flow()
    flow.start("q")
    namespace = flow.build_tools({})
    visible, hidden = partition_repl_namespace(namespace)
    assert set(hidden) == set(HIDDEN_REPL_TOOL_NAMES)
    assert "done" in visible and "launch_subagents" in visible
    assert "llm_query_batched" in visible
    assert "flow_delegate" not in visible


def test_partition_skips_private_and_noncallable():
    visible, hidden = partition_repl_namespace(
        {"_x": lambda: None, "data": "str", "SHOW_VARS": lambda: None, "tool_a": lambda: None}
    )
    assert set(visible) == {"tool_a"}
    assert not hidden


# ── builtins factories ──────────────────────────────────────────────────


def test_make_wait_errors_and_success():
    flow_wait = make_wait()
    with pytest.raises(ValueError):
        flow_wait()
    with pytest.raises(TypeError, match="non-handle"):
        flow_wait("refusal string")
    req = flow_wait(ChildHandle("a"), ChildHandle("b"))
    assert isinstance(req, WaitRequest) and req.agent_ids == ["a", "b"]


def test_make_launch_subagents_spec_validation():
    import asyncio

    def fake_delegate(**spec):
        return ChildHandle(spec["name"])

    async def fake_wait(*handles):
        return [h.agent_id for h in handles]

    launch = make_launch_subagents(fake_delegate, fake_wait)
    with pytest.raises(TypeError, match="list of dict"):
        asyncio.run(launch("notalist"))
    with pytest.raises(TypeError, match="every spec to be a dict"):
        asyncio.run(launch([123]))
    with pytest.raises(KeyError, match="query"):
        asyncio.run(launch([{"name": "x"}]))
    out = asyncio.run(launch([{"name": "a", "query": "q"}, {"name": "b", "query": "q"}]))
    assert out == ["a", "b"]


def test_make_done_plain_and_structured():
    flow = make_flow()
    # plain: no schema in env
    env: dict = {}
    done = make_done(flow, env)
    with pytest.raises(DoneSignal):
        done("  hi  ")
    assert env[ENV_DONE_RESULT] == "hi"

    # structured: schema stashed in env (as repl_for does per agent)
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    env2: dict = {ENV_OUTPUT_SCHEMA: schema}
    done2 = make_done(flow, env2)
    with pytest.raises(DoneSignal):
        done2({"x": 5})
    assert env2[ENV_DONE_RESULT] == '{"x": 5}'
    with pytest.raises(StructuredOutputError):
        done2({"x": "nope"})


# ── HISTORY ─────────────────────────────────────────────────────────────


def test_history_over_finished_run():
    flow = make_flow(DONE_OK)
    graph = run_to_completion(flow, "summarize this")
    hist = History(lambda: graph)
    msgs = hist.messages()
    assert len(hist) == len(msgs) >= 2
    assert msgs[0]["role"] == "user"
    assert any(m["role"] == "assistant" for m in msgs)
    # no system message in HISTORY
    assert all(m["role"] != "system" for m in msgs)
    assert hist.last(1) == [msgs[-1]]
    assert hist.read(0) == msgs[0]
    assert "summarize" in hist.text()
    assert hist.grep("summarize")


def test_history_injected_into_repl_namespace():
    flow = make_flow(DONE_OK)
    agent = flow.start("q")
    repl = flow.repl_for(agent)
    assert isinstance(repl.namespace["HISTORY"], History)
    assert "SHOW_VARS" not in repl.namespace  # off by default


def test_history_reflects_live_trajectory():
    flow = make_flow(DONE_OK)
    flow.start("q")
    # Resolver reads the live run, so it tracks the graph across steps/copies.
    hist = History(lambda: flow.graph)
    before = len(hist)
    flow.step()  # appends llm_action + llm_output
    assert len(hist) > before


def test_make_history_tracks_live_graph_across_functional_adopt():
    # make_history resolves the agent by id, so the functional step API
    # (graph = flow.step(graph), which deep-copies + adopts) is reflected with
    # no rebinding of the REPL's HISTORY.
    flow = make_flow(DONE_OK)
    graph = flow.start("q")
    hist = make_history(flow, {ENV_AGENT_ID: graph.agent_id})
    before = len(hist)
    graph = flow.step(graph)
    assert len(hist) > before


# ── prompt builder + build_system_prompt ────────────────────────────────


def test_static_system_prompt_content():
    assert "CONTEXT" not in SYSTEM_PROMPT
    assert "llm_query_batched" in SYSTEM_PROMPT
    assert "HISTORY" in SYSTEM_PROMPT
    assert "launch_subagents" in SYSTEM_PROMPT


def test_build_system_prompt_has_dynamic_tool_docs():
    flow = make_flow()
    agent = flow.start("q")
    prompt = flow.build_system_prompt(agent)
    assert "## Tools" in prompt
    assert "`done(" in prompt and "`launch_subagents(" in prompt
    assert "`llm_query_batched(" in prompt
    # hidden control tools are not advertised
    assert "`flow_delegate(" not in prompt
    # recursion status
    assert "recursion depth" in prompt


def test_build_system_prompt_structured_section():
    flow = make_flow()
    agent = flow.start("q", output_schema={"type": "object", "properties": {"x": {"type": "integer"}}})
    prompt = flow.build_system_prompt(agent)
    assert "JSON Schema" in prompt
    assert "Structured Output" in prompt


def test_system_prompt_is_rendered_once_and_stored_on_graph():
    flow = make_flow()
    agent = flow.start("q")
    # builder path stores the full rendered prompt on the graph (not "")
    assert agent.system_prompt
    assert "## Tools" in agent.system_prompt
    # survives serialization round-trip
    from rflow import Graph

    assert Graph.from_dict(agent.to_dict()).system_prompt == agent.system_prompt


def test_explicit_system_prompt_is_escape_hatch():
    flow = make_flow(system_prompt="STATIC PROMPT")
    agent = flow.start("q")
    assert flow.build_system_prompt(agent) == "STATIC PROMPT"


def test_explicit_system_prompt_appends_schema():
    flow = make_flow(system_prompt="STATIC")
    agent = flow.start("q", output_schema={"type": "object", "properties": {"x": {"type": "integer"}}})
    prompt = flow.build_system_prompt(agent)
    assert prompt.startswith("STATIC")
    assert "JSON Schema" in prompt


def test_prompt_builder_update_is_immutable():
    custom = DEFAULT_BUILDER.update("role", "You are a security auditor.")
    assert "security auditor" in custom.build()
    assert "security auditor" not in DEFAULT_BUILDER.build()
    assert isinstance(custom, PromptBuilder)


def test_tool_section_lists_models_when_multimodel():
    flow = Flow(StubLLM(DONE_OK), llm_clients={"fast": StubLLM(DONE_OK)})
    agent = flow.start("q")
    prompt = flow.build_system_prompt(agent)
    assert "Available models" in prompt
    assert "`fast`" in prompt


# ── first_prompt / depth notes / no-code error ──────────────────────────


def test_first_prompt_has_manifest_and_depth_note():
    flow = make_flow()
    msg = flow.first_prompt("do a thing", {"doc": "abc"}, depth=0)
    assert "do a thing" in msg
    assert 'INPUTS["doc"]  (str, 3 chars)' in msg
    assert "recursion depth 0" in msg
    assert "abc" not in msg  # values are never dumped


def test_first_prompt_depth_limit_note():
    flow = make_flow(max_depth=2)
    at_limit = flow.first_prompt("q", {}, depth=2)
    assert "cannot spawn sub-agents" in at_limit


def test_no_code_block_message_used():
    flow = make_flow("I will not write a code block.")
    flow.start("q")
    flow.step()  # CallLLM -> LLMOutput with empty code
    flow.step()  # Exec -> no_code_block error
    errors = flow.graph.all_nodes.errors()
    assert errors and errors[0].error == "no_code_block"
    assert "repl" in errors[0].content


# ── reserved names + show_vars ──────────────────────────────────────────


def test_reserved_names_include_history_and_tools():
    for name in ("HISTORY", "SHOW_VARS", "llm_query_batched", "done"):
        assert name in Flow._RESERVED
    flow = make_flow()
    with pytest.raises(ValueError, match="reserved"):
        flow.start("q", {"HISTORY": "x"})


def test_show_vars_opt_in():
    flow = make_flow(DONE_OK, show_vars=True)
    agent = flow.start("q", {"doc": "hello"})
    repl = flow.repl_for(agent)
    assert "SHOW_VARS" in repl.namespace
    # Inputs live under INPUTS now, not as a top-level `doc` variable.
    assert repl.namespace["INPUTS"] == {"doc": "hello"}
    assert "doc" not in repl.namespace
    show = repl.namespace["SHOW_VARS"]
    out = show()
    assert out.get("INPUTS") == "dict"
    assert "done" not in out  # tools excluded
    # role section advertises it when enabled
    assert "SHOW_VARS()" in flow.build_system_prompt(agent)


def test_make_show_vars_filters_tools_and_hidden():
    @tool("a tool")
    def done():
        return None

    ns = {"x": 5, "_p": 1, "done": done, "flow_delegate": (lambda: None)}
    get = make_show_vars(ns)
    out = get()
    assert out == {"x": "int"}  # tool (metadata) + hidden + private all filtered


# ── FileFlow integration ────────────────────────────────────────────────


class FileFlow(Flow):
    def build_tools(self, env):
        file_tools = {get_tool_metadata(fn).name: fn for fn in FILE_TOOLS}
        return super().build_tools(env) | file_tools


def test_file_flow_writes_and_reads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reply = (
        "```repl\n"
        'write_file("out.txt", "generated")\n'
        'done(read_file("out.txt"))\n'
        "```"
    )
    flow = FileFlow(StubLLM(reply), max_iters=3)
    result = flow.run("write a file")
    assert result == "generated"
    assert (tmp_path / "out.txt").read_text() == "generated"
    # file tools advertised in the prompt
    agent = flow.graph
    assert "`write_file(" in flow.build_system_prompt(agent)


# ── filesystem tools: edge cases (ported from legacy test_filesystem_tools) ─


def test_ls_returns_workspace_relative_paths_for_relative_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "item.txt").write_text("x")
    assert ls(".") == ["nested"]
    assert ls("nested") == ["nested/item.txt"]
    assert ls("nested/item.txt") == ["nested/item.txt"]


def test_ls_preserves_absolute_paths_for_absolute_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "item.txt").write_text("x")
    assert ls(str(tmp_path / "nested")) == [str(tmp_path / "nested" / "item.txt")]


def test_append_file_creates_dirs_and_appends(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    msg = append_file("logs/run.txt", "first\n")
    assert msg == "Appended 6 bytes to logs/run.txt"
    append_file("logs/run.txt", "second\n")
    assert read_file("logs/run.txt") == "first\nsecond\n"


def test_edit_file_reports_partial_and_no_match(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_file("f.txt", "alpha beta alpha")
    # only the first occurrence of a matched edit is replaced; misses are counted.
    msg = edit_file("f.txt", ("alpha", "X"), ("missing", "Y"))
    assert msg == "Applied 1/2 edits to f.txt"
    assert read_file("f.txt") == "X beta alpha"


def test_read_lines_and_line_count(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_file("f.txt", "l0\nl1\nl2\nl3")
    assert line_count("f.txt") == 4
    assert read_lines("f.txt", 0, 1) == "l0"  # 0-indexed, end-exclusive
    assert read_lines("f.txt", 1, 3) == "l1\nl2"


def test_line_count_and_read_lines_on_empty_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_file("empty.txt", "")
    assert line_count("empty.txt") == 0
    assert read_lines("empty.txt", 0, 5) == ""


def test_grep_searches_directory_and_caps_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_file("a.txt", "needle here\nno match\nneedle again")
    write_file("b.txt", "nothing")
    out = grep("needle", ".")
    lines = out.splitlines()
    assert len(lines) == 2
    assert all(line.startswith("a.txt:") and "needle" in line for line in lines)
    # max_results truncates
    assert len(grep("needle", ".", max_results=1).splitlines()) == 1
    assert grep("absent-pattern", ".") == ""


def test_file_tools_collection_is_complete():
    names = {get_tool_metadata(fn).name for fn in FILE_TOOLS}
    assert names == {
        "read_file",
        "write_file",
        "append_file",
        "edit_file",
        "ls",
        "read_lines",
        "line_count",
        "grep",
    }


# ── make_delegate unit behavior (ported from legacy test_builtins) ──────


def test_flow_delegate_is_keyword_only_and_wires_spawn():
    import inspect

    flow = make_flow()
    agent = flow.start("q")
    captured = {}

    def fake_spawn(parent_agent_id, name, query, inputs, model, output_schema):
        captured.update(
            parent_agent_id=parent_agent_id,
            name=name,
            query=query,
            inputs=inputs,
            model=model,
            output_schema=output_schema,
        )
        return ChildHandle(f"{parent_agent_id}.{name}")

    flow.spawn_child = fake_spawn  # type: ignore[assignment]
    env = {ENV_AGENT_ID: agent.agent_id}
    delegate = make_delegate(flow, env)

    params = inspect.signature(delegate).parameters
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values())

    handle = delegate(name="kid", query="do it", inputs={"k": "v"}, model="default")
    assert isinstance(handle, ChildHandle) and handle.agent_id == "root.kid"
    assert captured == {
        "parent_agent_id": "root",
        "name": "kid",
        "query": "do it",
        "inputs": {"k": "v"},
        "model": "default",
        "output_schema": None,
    }


def test_launch_subagents_mixes_refusal_strings_and_handles():
    import asyncio

    calls = iter([ChildHandle("root.ok"), "refused: max depth"])

    def fake_delegate(**spec):
        return next(calls)

    async def fake_wait(*handles):
        return [f"result:{h.agent_id}" for h in handles]

    launch = make_launch_subagents(fake_delegate, fake_wait)
    out = asyncio.run(
        launch([{"query": "a"}, {"query": "b"}])
    )
    # handle slot gets the awaited result; refusal slot keeps the string.
    assert out == ["result:root.ok", "refused: max depth"]


# ── PromptBuilder structure (ported from legacy test_prompt_rendering) ──


def test_default_builder_section_order():
    assert DEFAULT_BUILDER.names == [
        "role",
        "strategy",
        "format",
        "examples",
        "final",
        "structured-output",
        "tools",
        "status",
    ]


def test_prompt_builder_anchors_and_immutability():
    base = PromptBuilder().section("a", "A", title="A").section("c", "C", title="C")
    inserted = base.section("b", "B", title="B", before="c")
    assert inserted.names == ["a", "b", "c"]
    after = base.section("z", "Z", title="Z", after="a")
    assert after.names == ["a", "z", "c"]
    # original never mutated
    assert base.names == ["a", "c"]


def test_prompt_builder_callable_sections_and_overrides():
    seen = {}

    def body(engine, graph):
        seen["args"] = (engine, graph)
        return "callable body"

    builder = PromptBuilder().section("dyn", body, title="Dyn")
    out = builder.build("ENG", "GRAPH")
    assert "callable body" in out and seen["args"] == ("ENG", "GRAPH")
    # keyword override wins over the callable for one render
    assert "overridden" in builder.build("ENG", "GRAPH", dyn="overridden")


def test_default_prompt_skips_structured_section_without_schema():
    flow = make_flow()
    agent = flow.start("q")
    assert agent.output_schema is None
    assert "## Structured Output" not in flow.build_system_prompt(agent)
