"""Tests for how graph state is projected back into LLM chat messages."""

from __future__ import annotations

from rlmflow import Graph, LLMClient, RLMConfig, RLMFlow, is_errored
from rlmflow.prompts.messages import NO_CODE_BLOCK
from rlmflow.runtime.local import LocalRuntime


class _CapturingLLM(LLMClient):
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages, *args, **kwargs) -> str:
        self.calls.append([dict(m) for m in messages])
        idx = len(self.calls) - 1
        return self.replies[min(idx, len(self.replies) - 1)]


def _run(agent: RLMFlow, graph: Graph) -> Graph:
    while not graph.finished:
        graph = agent.step(graph)
    return graph


def _agent(client: _CapturingLLM, **config_kwargs) -> RLMFlow:
    config_kwargs.setdefault("max_depth", 0)
    config_kwargs.setdefault("max_iterations", 8)
    return RLMFlow(client, runtime=LocalRuntime(), config=RLMConfig(**config_kwargs))


def _roles(messages: list[dict[str, str]]) -> list[str]:
    return [m["role"] for m in messages]


def test_first_turn_projects_system_and_user_messages():
    client = _CapturingLLM(['```repl\ndone("ok")\n```'])
    agent = _agent(client)

    graph = agent.start("say ok")
    graph = agent.step(graph)

    assert graph.current().type == "llm_output"
    assert len(client.calls) == 1
    messages = client.calls[0]
    assert _roles(messages) == ["system", "user"]
    first_user = messages[1]["content"]
    assert "Query: say ok" in first_user
    assert "First inspect/decompose" in first_user
    assert "many independent units/scopes" in first_user
    assert "Your next action:" in first_user
    assert "assigned worker task" not in first_user
    assert "files/checks/components/trials" not in first_user


def test_root_and_child_first_turn_use_same_generic_action_shape():
    client = _CapturingLLM(
        [
            (
                "```repl\n"
                'h = rlm_delegate(name="child", query="child task", context="")\n'
                "results = yield rlm_wait(h)\n"
                'done("parent:" + results[0])\n'
                "```"
            ),
            '```repl\ndone("child-result")\n```',
            '```repl\ndone("parent:child-result")\n```',
        ]
    )
    agent = _agent(client, max_depth=1)

    graph = _run(agent, agent.start("coordinate"))

    assert graph.result() == "parent:child-result"
    root_first = client.calls[0][1]["content"]
    child_first = client.calls[1][1]["content"]
    for message, query in ((root_first, "coordinate"), (child_first, "child task")):
        assert f"Query: {query}" in message
        assert "First inspect/decompose" in message
        assert "Your next action:" in message
        assert "assigned worker task" not in message
        assert "semantic checks/extractions" not in message


def test_no_code_block_retry_projects_malformed_reply_error_and_continue_nudge():
    bad = "repl\nprint('this is missing markdown fences')"
    client = _CapturingLLM([bad, '```repl\ndone("ok")\n```'])
    agent = _agent(client)

    graph = _run(agent, agent.start("say ok"))

    assert graph.result() == "ok"
    assert any(is_errored(s) and s.error == "no_code_block" for s in graph.states)
    assert len(client.calls) == 2

    retry = client.calls[1]
    assert _roles(retry) == ["system", "user", "assistant", "user", "user"]
    assert retry[2]["content"] == bad
    assert retry[3]["content"] == NO_CODE_BLOCK
    assert "```repl\n# Python code here\n```" in retry[3]["content"]


def test_normal_exec_output_retry_projects_stdout_and_continue_nudge():
    client = _CapturingLLM(
        [
            '```repl\nprint("HELLO_FROM_REPL")\n```',
            '```repl\ndone("ok")\n```',
        ]
    )
    agent = _agent(client)

    graph = _run(agent, agent.start("say ok"))

    assert graph.result() == "ok"
    assert len(client.calls) == 2
    retry = client.calls[1]
    assert _roles(retry) == ["system", "user", "assistant", "user", "user"]
    assert retry[2]["content"].startswith("```repl")
    assert "REPL output:" in retry[3]["content"]
    assert "HELLO_FROM_REPL" in retry[3]["content"]
    assert "The history before is your previous interactions" in retry[4]["content"]
    assert 'original query: "say ok"' in retry[4]["content"]
    assert "not interacted with the REPL environment" not in retry[4]["content"]


def test_resume_turn_gets_verify_nudge_without_child_result_leak():
    client = _CapturingLLM(
        [
            (
                "```repl\n"
                'h = rlm_delegate(name="child", query="child task", context="")\n'
                "results = yield rlm_wait(h)\n"
                'print("parent resumed")\n'
                "```"
            ),
            '```repl\ndone("SECRET_CHILD_RESULT")\n```',
            '```repl\ndone("parent done")\n```',
        ]
    )
    agent = _agent(client, max_depth=1, max_iterations=8)

    graph = _run(agent, agent.start("coordinate"))

    assert graph.result() == "parent done"
    assert len(client.calls) == 3
    resumed = client.calls[2]
    assert _roles(resumed) == ["system", "user", "assistant", "user", "user"]
    assert "parent resumed" in resumed[3]["content"]
    assert "Children just finished: root.child" in resumed[4]["content"]
    assert "Before any new `rlm_delegate`" in resumed[4]["content"]
    assert "get_runs()" not in resumed[4]["content"]
    assert "SECRET_CHILD_RESULT" not in "\n".join(m["content"] for m in resumed)


def test_exec_exception_retry_projects_traceback_and_continue_nudge():
    client = _CapturingLLM(
        [
            '```repl\nraise ValueError("boom")\n```',
            '```repl\ndone("recovered")\n```',
        ]
    )
    agent = _agent(client)

    graph = _run(agent, agent.start("recover"))

    assert graph.result() == "recovered"
    assert any(is_errored(s) and s.error == "exec_exception" for s in graph.states)
    retry = client.calls[1]
    assert _roles(retry) == ["system", "user", "assistant", "user", "user"]
    assert "REPL output:" in retry[3]["content"]
    assert "ValueError: boom" in retry[3]["content"]


def test_repeated_no_code_blocks_accumulate_each_bad_reply_and_error_feedback():
    first_bad = "repl\nx = 1"
    second_bad = "repl\ny = 2"
    client = _CapturingLLM(
        [
            first_bad,
            second_bad,
            '```repl\ndone("ok")\n```',
        ]
    )
    agent = _agent(client)

    graph = _run(agent, agent.start("say ok"))

    assert graph.result() == "ok"
    final_retry = client.calls[2]
    assert _roles(final_retry) == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "user",
    ]
    assistant_messages = [m["content"] for m in final_retry if m["role"] == "assistant"]
    user_messages = [m["content"] for m in final_retry if m["role"] == "user"]
    assert assistant_messages == [first_bad, second_bad]
    assert user_messages.count(NO_CODE_BLOCK) == 2
