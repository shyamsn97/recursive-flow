"""Core engine behavior on the minimal stack (``rflow.Flow`` + ``rflow.Graph``).

Covers the loop the whole library is built on: start/step/run lifecycle, the
strict observation/action alternation, recursive delegation (tight + verify +
``launch_subagents``), resume semantics, and the recoverable-error paths
(no code block, forced final answer, invalid ``await``). This is the
new-stack replacement for the core legacy engine tests.
"""

from __future__ import annotations

import re

from rflow import Flow, Graph, is_done, is_errored, is_supervising
from tests.helpers import ScriptedLLM, first_user_text, make_flow, run_to_completion, types


# ── lifecycle ─────────────────────────────────────────────────────────


def test_start_records_query_node_at_seq_zero():
    flow = make_flow()
    graph = flow.start("say ok")
    assert isinstance(graph, Graph)
    assert graph.agent_id == "root"
    assert graph.query == "say ok"
    assert types(graph) == ["user_query"]
    assert graph.nodes[0].seq == 0


def test_one_shot_run_reaches_done():
    flow = make_flow('```repl\ndone("ok")\n```')
    assert flow.run("say ok") == "ok"
    assert types(flow.graph) == [
        "user_query",
        "llm_action",
        "llm_output",
        "exec_action",
        "done_output",
    ]
    assert is_done(flow.graph.current())


def test_step_splits_llm_half_then_exec_half():
    flow = make_flow('```repl\ndone("ok")\n```')
    graph = flow.start("say ok")
    flow.step()  # LLM half: LLMAction + LLMOutput
    assert graph.current().type == "llm_output"
    flow.step()  # exec half: ExecAction + DoneOutput
    assert is_done(graph.current())
    assert graph.result() == "ok"


def test_run_returns_result_string_and_marks_finished():
    flow = make_flow()
    flow.run("say ok")
    assert flow.graph.finished
    assert flow.graph.result() == "ok"


def test_global_step_is_stamped_and_monotonic():
    flow = make_flow('```repl\ndone("ok")\n```')
    g = run_to_completion(flow, "say ok")
    steps = [n.global_step for n in g.nodes]
    assert steps[0] == 0  # bootstrap query stamped at step 0
    assert steps == sorted(steps)
    assert g.max_global_step() == max(steps)


# ── single-agent state machine ────────────────────────────────────────


def test_two_turn_stateful_repl_persists_variables():
    def reply_for(messages):
        joined = "\n".join(m["content"] for m in messages)
        if "STASH" in joined:
            return '```repl\ndone("got:" + STASH)\n```'
        return "```repl\nSTASH = 'value'\nprint('hello')\n```"

    flow = Flow(ScriptedLLM(reply_for), max_depth=0, max_iters=5)
    g = run_to_completion(flow, "hi")
    assert types(g) == [
        "user_query",
        "llm_action",
        "llm_output",
        "exec_action",
        "exec_output",
        "llm_action",
        "llm_output",
        "exec_action",
        "done_output",
    ]
    assert g.result() == "got:value"


def test_done_captures_stdout_from_same_block():
    flow = make_flow('```repl\nprint("hello world")\ndone("ok")\n```', max_depth=0)
    g = run_to_completion(flow, "say ok")
    terminal = g.current()
    assert is_done(terminal)
    assert terminal.result == "ok"
    assert "hello world" in terminal.output
    assert "hello world" in terminal.content


def test_inputs_are_exposed_via_INPUTS_dict():
    flow = make_flow('```repl\ndone(INPUTS["payload"].upper())\n```', max_depth=0)
    g = run_to_completion(flow, "echo", inputs={"payload": "hi there"})
    assert g.result() == "HI THERE"
    # First prompt advertises inputs as a manifest, not their values.
    assert "payload" in g.nodes[0].content
    assert "hi there" not in g.nodes[0].content


# ── recoverable error paths ───────────────────────────────────────────


def test_no_code_block_records_error_then_recovers():
    calls = {"n": 0}

    def reply_for(_messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return "I forgot the code block."
        return '```repl\ndone("ok")\n```'

    flow = Flow(ScriptedLLM(reply_for), max_depth=0, max_iters=5)
    g = run_to_completion(flow, "p")
    assert types(g) == [
        "user_query",
        "llm_action",
        "llm_output",
        "exec_action",
        "error_output",
        "llm_action",
        "llm_output",
        "exec_action",
        "done_output",
    ]
    err = g.nodes[4]
    assert is_errored(err) and err.error == "no_code_block"
    assert g.result() == "ok"


def test_max_iters_forces_final_answer_turn():
    def reply_for(messages):
        if flow.FINAL in "\n".join(m["content"] for m in messages):
            return '```repl\ndone("final answer")\n```'
        return "```repl\nx = 1\n```"

    flow = Flow(ScriptedLLM(reply_for), max_depth=0, max_iters=1)
    g = run_to_completion(flow, "answer")
    assert g.result() == "final answer"


def test_invalid_await_is_rejected_before_running():
    calls = {"n": 0}

    def reply_for(_messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return '```repl\nlaunch_subagents([{"query": "x"}])\n```'  # not awaited
        return '```repl\ndone("recovered")\n```'

    flow = Flow(ScriptedLLM(reply_for), max_depth=1, max_iters=5)
    g = run_to_completion(flow, "p")
    err = next(n for n in g.nodes if is_errored(n))
    assert err.error == "invalid_wait"
    assert g.result() == "recovered"


# ── delegation / recursion ────────────────────────────────────────────


def _tight_parent_child(messages):
    """Parent delegates one child (tight: delegate→wait→done in one block)."""
    if "child task" in first_user_text(messages):
        return '```repl\ndone("c")\n```'
    return (
        "```repl\n"
        'h = flow_delegate(name="child", query="child task")\n'
        "results = await flow_wait(h)\n"
        'done("p:" + results[0])\n'
        "```"
    )


def test_tight_delegation_records_supervising_then_resume():
    flow = Flow(ScriptedLLM(_tight_parent_child), max_depth=1, max_iters=5)
    g = run_to_completion(flow, "parent")

    assert types(g) == [
        "user_query",
        "llm_action",
        "llm_output",
        "exec_action",
        "supervising_output",
        "resume_action",
        "done_output",
    ]
    assert types(g["root.child"]) == [
        "user_query",
        "llm_action",
        "llm_output",
        "exec_action",
        "done_output",
    ]
    sup = next(n for n in g.nodes if is_supervising(n))
    assert set(sup.waiting_on) == {"root.child"}
    assert g.result() == "p:c"
    # spawn link points back at the action node that was running.
    assert g["root.child"].parent_agent_id == "root"
    assert g["root.child"].parent_node_id in {n.id for n in g.nodes}


def test_launch_subagents_fans_out_in_order():
    def reply_for(messages):
        task = first_user_text(messages)
        if "task a" in task:
            return '```repl\ndone("A")\n```'
        if "task b" in task:
            return '```repl\ndone("B")\n```'
        return (
            "```repl\n"
            'rs = await launch_subagents([{"query": "task a"}, {"query": "task b"}])\n'
            'done("|".join(rs))\n'
            "```"
        )

    flow = Flow(ScriptedLLM(reply_for), max_depth=2, max_iters=6, max_concurrency=2)
    g = run_to_completion(flow, "p")
    assert g.result() == "A|B"
    assert len(g.children) == 2


def test_structured_child_result_is_parsed_not_a_json_string():
    """A child with ``output_schema`` returns a parsed value to its parent.

    Regression for the word-search baseline: the parent did
    ``results[0]["word"]`` and hit ``TypeError: string indices must be
    integers`` because the child's structured result came back as the raw JSON
    string instead of the validated list/dict the prompt promises.
    """
    item_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"word": {"type": "string"}},
            "required": ["word"],
            "additionalProperties": False,
        },
    }

    def reply_for(messages):
        task = first_user_text(messages)
        if "child task" in task:
            return '```repl\ndone([{"word": "AGENT"}])\n```'
        return (
            "```repl\n"
            "[hits] = await launch_subagents([\n"
            '    {"query": "child task", "output_schema": '
            + repr(item_schema)
            + "}\n"
            "])\n"
            'done("first=" + hits[0]["word"])\n'
            "```"
        )

    flow = Flow(ScriptedLLM(reply_for), max_depth=1, max_iters=6)
    g = run_to_completion(flow, "parent")
    assert g.result() == "first=AGENT"


def test_resume_does_not_leak_child_result_into_prompt():
    secret = "SECRET_CHILD_RESULT"
    captured = {}

    def reply_for(messages):
        task = first_user_text(messages)
        if "child task" in task:
            return f'```repl\ndone("{secret}")\n```'
        prior_assistant = "\n".join(
            m["content"] for m in messages if m["role"] == "assistant"
        )
        if "flow_wait" in prior_assistant:
            captured["resume_messages"] = messages
            return '```repl\ndone("parent:" + results[0])\n```'
        return (
            "```repl\n"
            'h = flow_delegate(name="child", query="child task")\n'
            "results = await flow_wait(h)\n"
            'print("MARKER")\n'
            "```"
        )

    flow = Flow(ScriptedLLM(reply_for), max_depth=1, max_iters=6)
    g = run_to_completion(flow, "parent")

    assert g.result() == f"parent:{secret}"
    resume_prompt = "\n".join(m["content"] for m in captured["resume_messages"])
    assert secret not in resume_prompt  # child result is never injected upstream


def test_recursive_depth_chain():
    """Each level delegates one child until the configured depth, then a leaf."""
    max_depth = 3

    def reply_for(messages):
        task = first_user_text(messages)
        m = re.search(r"level:(\d+)", task)
        level = int(m.group(1)) if m else 0
        if level >= max_depth:
            return f'```repl\ndone("leaf@{level}")\n```'
        return (
            "```repl\n"
            f'h = flow_delegate(name="child", query="level:{level + 1}")\n'
            "results = await flow_wait(h)\n"
            f'done("d{level}->" + results[0])\n'
            "```"
        )

    flow = Flow(ScriptedLLM(reply_for), max_depth=max_depth, max_iters=6)
    g = run_to_completion(flow, "level:0")

    assert g.result() == "d0->d1->d2->leaf@3"
    chain = ["root", "root.child", "root.child.child", "root.child.child.child"]
    assert all(aid in g.agents for aid in chain)
    assert g["root.child.child.child"].depth == 3


def test_max_depth_refusal_keeps_run_alive():
    def reply_for(messages):
        task = first_user_text(messages)
        if "refused" in task:  # never happens; refusal comes back as a result string
            return '```repl\ndone("unreachable")\n```'
        return (
            "```repl\n"
            'r = flow_delegate(name="c", query="go")\n'
            'done(r if isinstance(r, str) else "delegated")\n'
            "```"
        )

    flow = Flow(ScriptedLLM(reply_for), max_depth=0, max_iters=3)
    g = run_to_completion(flow, "p")
    assert "refused" in g.result()
    assert not g.children


# ── ported engine invariants (legacy test_engine.py / test_timeline.py) ─


def test_done_signal_not_swallowed_by_broad_except():
    # done() raises DoneSignal (a BaseException), so a broad `except Exception`
    # in agent code can't swallow a successful finish.
    reply = '```repl\ntry:\n    done("real")\nexcept Exception:\n    done("swallowed")\n```'
    flow = make_flow(reply, max_depth=0)
    g = run_to_completion(flow, "q")
    assert g.result() == "real"


def test_build_messages_never_emits_consecutive_user_turns():
    # Two LLM turns: a no-code reply (recoverable error → user nudge) then done.
    # The rebuilt prompt must coalesce same-role turns (no two adjacent users).
    replies = iter(["no code here, sorry", '```repl\ndone("ok")\n```'])
    flow = Flow(ScriptedLLM(lambda _m: next(replies)), max_depth=0, max_iters=5)
    g = run_to_completion(flow, "q")
    msgs = flow.build_messages(g, force_final=False)
    roles = [m["role"] for m in msgs]
    assert all(a != b for a, b in zip(roles, roles[1:])), roles


def test_action_and_observation_share_global_step():
    flow = make_flow('```repl\ndone("ok")\n```', max_depth=0)
    g = run_to_completion(flow, "q")
    # the LLM action and its llm_output observation are one engine step.
    by_type = {n.type: n.global_step for n in g.nodes}
    assert by_type["llm_action"] == by_type["llm_output"]
    assert by_type["exec_action"] == by_type["done_output"]


def test_supervisor_not_runnable_until_child_finishes():
    flow = Flow(ScriptedLLM(_tight_parent_child), max_depth=1, max_iters=5)
    g = flow.start("parent")
    saw_child_runnable_alone = False
    while not g.finished:
        runnable = g.get_runnable_nodes()
        sup = next((n for n in g.nodes if is_supervising(n)), None)
        if sup is not None and not g["root.child"].finished:
            # While the child is unfinished, the paused supervisor is NOT
            # runnable — only the child (or its descendants) is.
            assert "root" not in runnable
            if runnable == ["root.child"]:
                saw_child_runnable_alone = True
        flow.step()
    assert saw_child_runnable_alone
    assert g.result() == "p:c"
