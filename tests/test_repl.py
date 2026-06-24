"""Phase 0 — the in-process REPL (``rflow.repl``).

Ported from the legacy REPL-yield coverage to the new ``rflow.repl.REPL``
(no namespace ctor arg; ``repl.namespace`` is public). Covers top-level-await
detection, suspend/resume on ``WaitRequest``, the unsupported-await guard in the
driver, and BaseException capture (errors and ``SystemExit`` become recorded
output rather than crashing the host).
"""

from __future__ import annotations

import ast

from rflow.graph import WaitRequest
from rflow.repl import REPL, _has_top_level_await

# ── top-level await detection ─────────────────────────────────────────


def _await(code: str) -> bool:
    return _has_top_level_await(ast.parse(code))


def test_top_level_await_detected():
    assert _await("x = await launch_subagents(specs)") is True


def test_await_inside_nested_function_is_not_top_level():
    assert _await("async def f():\n    return await g()") is False


def test_await_inside_comprehension_is_not_top_level():
    assert _await("[await f() for x in xs]") is False


def test_no_await_anywhere():
    assert _await("x = 1\nprint(x)") is False


# ── plain (non-suspending) execution ──────────────────────────────────


def test_block_with_no_await_runs_to_completion():
    r = REPL()
    suspended, out = r.start("print('hi')\nx = 41 + 1")
    assert suspended is False and out == "hi"
    assert r.namespace["x"] == 42 and r.errored is False


def test_generator_defined_and_consumed_does_not_suspend():
    r = REPL()
    suspended, out = r.start("def g():\n    yield 1\n    yield 2\nprint(sum(g()))")
    assert suspended is False and out == "3"


def test_generator_expression_does_not_suspend():
    r = REPL()
    suspended, out = r.start("print(sum(i for i in range(4)))")
    assert suspended is False and out == "6"


def test_namespace_persists_across_blocks():
    r = REPL()
    r.start("acc = []")
    r.start("acc.append(1)")
    suspended, out = r.start("acc.append(2)\nprint(acc)")
    assert suspended is False and out == "[1, 2]"


# ── suspend / resume on WaitRequest ───────────────────────────────────


def _wait_repl() -> REPL:
    r = REPL()
    async def wait_agents(*agent_ids):
        return await WaitRequest(list(agent_ids))

    r.namespace["wait_agents"] = wait_agents
    return r


def test_await_wait_suspends_with_agent_ids():
    r = _wait_repl()
    suspended, payload = r.start("res = await wait_agents('root.a')")
    assert suspended is True
    request, pre = payload
    assert isinstance(request, WaitRequest) and request.agent_ids == ["root.a"]
    assert pre == ""


def test_multiple_handles_in_one_wait():
    r = _wait_repl()
    suspended, payload = r.start("res = await wait_agents('root.a', 'root.b')")
    assert suspended is True
    request, _ = payload
    assert request.agent_ids == ["root.a", "root.b"]


def test_resume_returns_send_value_to_block():
    r = _wait_repl()
    r.start("res = await wait_agents('root.a', 'root.b')")
    suspended, out = r.resume(["A", "B"])
    assert suspended is False
    assert r.namespace["res"] == ["A", "B"]


def test_nested_async_helper_can_suspend_and_resume():
    r = _wait_repl()
    code = (
        "async def helper():\n"
        "    vals = await wait_agents('root.a', 'root.b')\n"
        "    return '-'.join(vals)\n"
        "res = await helper()\n"
        "print(res)\n"
    )
    suspended, payload = r.start(code)
    assert suspended is True
    request, _pre = payload
    assert request.agent_ids == ["root.a", "root.b"]
    suspended, out = r.resume(["A", "B"])
    assert suspended is False
    assert out == "A-B"
    assert r.namespace["res"] == "A-B"


def test_nested_async_helper_can_suspend_twice():
    r = _wait_repl()
    code = (
        "async def helper():\n"
        "    first = await wait_agents('root.a')\n"
        "    second = await wait_agents('root.b')\n"
        "    return first + second\n"
        "res = await helper()\n"
    )
    suspended, payload = r.start(code)
    assert suspended is True
    request, _pre = payload
    assert request.agent_ids == ["root.a"]
    suspended, payload = r.resume(["A"])
    assert suspended is True
    request, _pre = payload
    assert request.agent_ids == ["root.b"]
    suspended, _out = r.resume(["B"])
    assert suspended is False
    assert r.namespace["res"] == ["A", "B"]


def test_pre_output_captured_before_suspension():
    r = _wait_repl()
    suspended, payload = r.start(
        "print('before wait')\nres = await wait_agents('root.a')"
    )
    assert suspended is True
    _request, pre = payload
    assert pre == "before wait"


# ── error / control-flow capture ──────────────────────────────────────


def test_runtime_error_is_captured_as_output():
    r = REPL()
    suspended, out = r.start("print(missing_name)")
    assert suspended is False
    assert r.errored is True and "NameError" in out


def test_syntax_error_is_captured():
    r = REPL()
    suspended, out = r.start("def (:")
    assert suspended is False
    assert r.errored is True and "SyntaxError" in out


def test_system_exit_in_block_is_captured_not_propagated():
    r = REPL()
    suspended, out = r.start("raise SystemExit('stop')")
    assert suspended is False
    assert r.errored is True and "SystemExit" in out


def test_sys_exit_call_in_block_is_captured():
    r = REPL()
    suspended, out = r.start("import sys\nsys.exit(2)")
    assert suspended is False
    assert r.errored is True


def test_unsupported_await_in_driver_is_rejected():
    r = REPL()

    class _Boom:
        def __await__(self):
            yield "not a wait request"

    r.namespace["Boom"] = _Boom
    suspended, out = r.start("await Boom()")
    assert suspended is False
    assert r.errored is True
    assert "only graph-aware awaits" in out


def test_resume_without_suspension_errors():
    r = REPL()
    suspended, out = r.resume(None)
    assert suspended is False
    assert r.errored is True and "no suspended coroutine" in out


def test_close_is_noop():
    r = REPL()
    assert r.close() is None
