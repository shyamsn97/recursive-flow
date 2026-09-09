import asyncio

from helpers import StubLLM

from rlmflow import (
    ExecAction,
    ExecOutput,
    Flow,
    LLMChunk,
    LLMOutput,
    LLMUsage,
    ReplRun,
    ReplStatus,
    Runtime,
    WrappedRuntime,
    start,
)
from rlmflow.engine.steps import complete, run_repl, to_action


def test_step_advances_exactly_one_state_transition():
    flow = Flow(StubLLM(lambda _messages: '```repl\nfinish("ok")\n```'))
    root = start("query")

    produced = asyncio.run(flow.step(root))

    assert produced.type == "plan_query"
    assert root.frontier is produced
    assert root.transcript() == [root, produced]


def test_a_run_can_be_driven_by_hand_one_step_at_a_time():
    flow = Flow(StubLLM(lambda _messages: '```repl\nfinish("ok")\n```'))
    root = start("query")

    produced = [asyncio.run(flow.step(root.frontier)) for _ in range(3)]

    assert [node.type for node in produced] == [
        "plan_query",
        "llm_output",
        "exec_action",
    ]
    final = asyncio.run(flow.step(root.frontier))
    assert final.type == "done_output"
    assert root.terminal and root.result() == "ok"


def test_complete_runs_with_a_flow_and_fake_llm():
    seen = []

    class FakeLLM:
        async def stream(self, messages):
            seen.append(messages)
            yield LLMChunk(
                text="reply",
                usage=LLMUsage(1, 2),
            )

    flow = Flow(FakeLLM())
    root = start("query")
    plan = asyncio.run(flow.step(root))
    landed = asyncio.run(complete(flow, plan))

    assert isinstance(landed, LLMOutput)
    assert landed.usage == LLMUsage(1, 2)
    assert seen[0][-1]["content"] == flow.build_messages(plan)[-1]["content"]
    assert plan.instruction() in seen[0][-1]["content"]


def test_run_repl_uses_the_wrapped_runtime():
    class UnusedLLM:
        async def stream(self, _messages):
            raise AssertionError("stream is not used")
            yield LLMChunk()

    class FakeRuntime(Runtime):
        def open(self, agent):
            raise AssertionError("open is not used")

        async def execute(self, node, code):
            return ReplRun(output="observed", status=ReplStatus.OK)

    class FakeRepl:
        def seed(self, tools, inputs):
            self.seeded = (tools, inputs)

    root = start("query")
    action = ExecAction(code="print('observed')")
    root.append(action)
    runtime = FakeRuntime()
    repl = FakeRepl()
    runtime.repls[root.id] = repl
    flow = Flow(UnusedLLM(), runtime=runtime)
    flow.wrapped_runtime = WrappedRuntime(runtime, lambda _node: {"tool": "value"})

    landed = asyncio.run(run_repl(flow, action))

    assert isinstance(landed, ExecOutput)
    assert landed.content == "observed"
    assert asyncio.run(to_action(flow, LLMOutput(content="x", code="print(1)"))).code == "print(1)"
    assert repl.seeded == ({"tool": "value"}, {})
