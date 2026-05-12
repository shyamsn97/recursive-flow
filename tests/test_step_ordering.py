"""Engine stepping order: per-agent node-type sequences across all shapes.

These tests pin down the exact ordered sequence of node types each agent goes
through, in concert with delegation / supervising / resume. They complement
``test_nested_delegation`` (which checks delegation depth + final results)
and ``test_rlmflow_core`` (which checks single-step lifecycles).

Two delegation patterns are exercised because both are valid and have
different recorded sequences:

* **Tight** — ``delegate(...); yield wait(h); done(result)`` in a single
  block. The generator yields, the engine resumes it with results, ``done``
  is called immediately, the engine records ``ResultNode`` straight off
  the supervising state. Per-agent shape: ``query → action → supervising
  → result``.

* **Verify** — block ends right after ``yield wait(h)``; the agent reads
  results back and calls ``done`` on the next turn. The engine records
  ``ResumeNode`` + a fresh observation/action/result triple. Per-agent
  shape: ``query → action → supervising → resume → action → result``.

Invariants asserted across every test:

* Per-agent ``[s.type for s in graph.states]`` matches the expected pattern.
* Per-agent seqs are ``0, 1, 2, ...`` with no gaps or duplicates.
* Every ``SupervisingNode.waiting_on`` matches the actually-spawned children
  (set-equal — order doesn't matter).
* ``parent_node_id`` on every child equals the action it spawned from.
"""

from __future__ import annotations

from rlmflow import (
    Graph,
    LLMClient,
    RLMConfig,
    RLMFlow,
    SupervisingNode,
)
from rlmflow.runtime.local import LocalRuntime


def _run(agent: RLMFlow, graph: Graph) -> Graph:
    while not graph.finished:
        graph = agent.step(graph)
    return graph


def _types(g: Graph) -> list[str]:
    return [s.type for s in g.states]


def _assert_seqs_monotonic(g: Graph) -> None:
    """Every agent in the subtree has seq 0..n-1 with no gaps or dupes."""
    for sub in g.walk():
        seqs = [s.seq for s in sub.states]
        assert seqs == list(range(len(seqs))), (
            f"{sub.agent_id}: seqs={seqs} (expected 0..{len(seqs) - 1})"
        )


def _assert_spawn_links(g: Graph, spawner_seq: int = 1) -> None:
    """Every direct child's parent_node_id equals the agent's seq=spawner_seq state."""
    spawner = g.states[spawner_seq]
    for child in g.children.values():
        assert child.parent_node_id == spawner.id, (
            f"{child.agent_id}.parent_node_id={child.parent_node_id} "
            f"expected {g.agent_id}.seq={spawner_seq}.id={spawner.id}"
        )
        assert child.parent_agent_id == g.agent_id


# ── single agent ─────────────────────────────────────────────────────


class _OneShotLLM(LLMClient):
    def chat(self, messages, *args, **kwargs):
        return '```repl\ndone("ok")\n```'


def test_single_agent_one_shot_is_query_action_result():
    agent = RLMFlow(
        _OneShotLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=0, max_iterations=3),
    )
    g = _run(agent, agent.start("hi"))

    assert _types(g) == ["query", "action", "result"]
    _assert_seqs_monotonic(g)
    assert g.result() == "ok"


class _TwoTurnLLM(LLMClient):
    """Turn 1: print + stash a value. Turn 2: read stash and ``done``."""

    def chat(self, messages, *args, **kwargs):
        joined = "\n".join(m["content"] for m in messages)
        if "STASH" in joined:
            return '```repl\ndone("got:" + STASH)\n```'
        return "```repl\nSTASH = 'value'\nprint('hello')\n```"


def test_single_agent_loop_is_query_action_observation_action_result():
    agent = RLMFlow(
        _TwoTurnLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=0, max_iterations=5),
    )
    g = _run(agent, agent.start("hi"))

    assert _types(g) == ["query", "action", "observation", "action", "result"]
    _assert_seqs_monotonic(g)
    assert g.result() == "got:value"


# ── tight pattern: done in the same block as wait ────────────────────


class _TightChildLLM(LLMClient):
    """Tight pattern with one child."""

    def chat(self, messages, *args, **kwargs):
        prompt = messages[-1]["content"].lower()
        if "child task" in prompt:
            return '```repl\ndone("c")\n```'
        return (
            "```repl\n"
            'h = delegate("child", "child task", "")\n'
            "results = yield wait(h)\n"
            'done("p:" + results[0])\n'
            "```"
        )


def test_tight_pattern_one_child_pins_both_chains():
    agent = RLMFlow(
        _TightChildLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=1, max_iterations=5),
    )
    g = _run(agent, agent.start("parent"))

    # Tight pattern: root agent doesn't get a resume turn.
    assert _types(g) == ["query", "action", "supervising", "result"]
    assert _types(g["root.child"]) == ["query", "action", "result"]
    _assert_seqs_monotonic(g)
    _assert_spawn_links(g, spawner_seq=1)

    sup = next(s for s in g.states if isinstance(s, SupervisingNode))
    assert set(sup.waiting_on) == {"root.child"}
    assert g.result() == "p:c"


class _TightManySiblingLLM(LLMClient):
    """Tight pattern — N siblings spawned from one action."""

    def __init__(self, n_children: int) -> None:
        self.n_children = n_children

    def chat(self, messages, *args, **kwargs):
        prompt = messages[-1]["content"].lower()
        if "leaf task" in prompt:
            return '```repl\ndone("leaf:" + AGENT_ID)\n```'
        names = [f"c{i}" for i in range(self.n_children)]
        delegations = "\n".join(
            f'h{i} = delegate("{n}", "leaf task", "")' for i, n in enumerate(names)
        )
        handles = ", ".join(f"h{i}" for i in range(self.n_children))
        return (
            "```repl\n"
            f"{delegations}\n"
            f"results = yield wait({handles})\n"
            'done(",".join(results))\n'
            "```"
        )


def test_tight_pattern_many_siblings_share_one_supervising():
    n = 6
    agent = RLMFlow(
        _TightManySiblingLLM(n_children=n),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=1, max_iterations=5),
    )
    g = _run(agent, agent.start("fan out"))

    assert _types(g) == ["query", "action", "supervising", "result"]
    child_ids = {f"root.c{i}" for i in range(n)}
    assert set(g.children) == child_ids
    for cid in child_ids:
        assert _types(g[cid]) == ["query", "action", "result"]

    sup = next(s for s in g.states if isinstance(s, SupervisingNode))
    assert set(sup.waiting_on) == child_ids

    _assert_seqs_monotonic(g)
    _assert_spawn_links(g, spawner_seq=1)

    assert set(g.result().split(",")) == {f"leaf:{cid}" for cid in child_ids}


# ── verify pattern: block ends after wait, done on resume turn ───────


class _VerifyChildLLM(LLMClient):
    """Verify pattern — agent resumes, then ``done`` on its own next turn.

    Block 1: ``delegate; yield wait(h)``  (no done)
    Block 2: runtime resumed → ``done(...)``

    Distinguishes the resume turn by detecting prior assistant traffic
    containing ``yield wait`` in the message history.
    """

    def chat(self, messages, *args, **kwargs):
        prompt = messages[-1]["content"]
        if "child task" in prompt.lower():
            return '```repl\ndone("c")\n```'
        # Resume turn: there's a prior assistant message containing wait().
        prior_assistant = "\n".join(
            m["content"] for m in messages if m.get("role") == "assistant"
        )
        if "yield wait" in prior_assistant:
            return '```repl\ndone("p:c-verified")\n```'
        return (
            "```repl\n"
            'h = delegate("child", "child task", "")\n'
            "yield wait(h)\n"
            "```"
        )


def test_verify_pattern_one_child_records_resume_action_result():
    agent = RLMFlow(
        _VerifyChildLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=1, max_iterations=8),
    )
    g = _run(agent, agent.start("parent"))

    # Verify pattern: full q → a → s → r → a → r.
    assert _types(g) == [
        "query",
        "action",
        "supervising",
        "resume",
        "action",
        "result",
    ]
    assert _types(g["root.child"]) == ["query", "action", "result"]
    _assert_seqs_monotonic(g)
    _assert_spawn_links(g, spawner_seq=1)

    sup = next(s for s in g.states if isinstance(s, SupervisingNode))
    assert set(sup.waiting_on) == {"root.child"}
    assert g.result() == "p:c-verified"


class _VerifyManySiblingLLM(LLMClient):
    """Verify pattern — N siblings, then verify-and-done on resume turn."""

    def __init__(self, n_children: int) -> None:
        self.n_children = n_children

    def chat(self, messages, *args, **kwargs):
        prompt = messages[-1]["content"]
        if "leaf task" in prompt.lower():
            return '```repl\ndone("leaf:" + AGENT_ID)\n```'
        prior_assistant = "\n".join(
            m["content"] for m in messages if m.get("role") == "assistant"
        )
        if "yield wait" in prior_assistant:
            # Resume turn: ``results`` is rebound to the child outputs.
            return '```repl\ndone(",".join(results))\n```'
        names = [f"c{i}" for i in range(self.n_children)]
        delegations = "\n".join(
            f'h{i} = delegate("{n}", "leaf task", "")' for i, n in enumerate(names)
        )
        handles = ", ".join(f"h{i}" for i in range(self.n_children))
        return "```repl\n" + delegations + f"\nresults = yield wait({handles})\n```"


def test_verify_pattern_many_siblings_matches_boids_shape():
    """The boids run: 6 siblings under one supervising, verified on resume."""
    n = 6
    agent = RLMFlow(
        _VerifyManySiblingLLM(n_children=n),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=1, max_iterations=8),
    )
    g = _run(agent, agent.start("fan out"))

    assert _types(g) == [
        "query",
        "action",
        "supervising",
        "resume",
        "action",
        "result",
    ]
    child_ids = {f"root.c{i}" for i in range(n)}
    assert set(g.children) == child_ids
    for cid in child_ids:
        assert _types(g[cid]) == ["query", "action", "result"]

    sup = next(s for s in g.states if isinstance(s, SupervisingNode))
    assert set(sup.waiting_on) == child_ids

    _assert_seqs_monotonic(g)
    _assert_spawn_links(g, spawner_seq=1)

    assert set(g.result().split(",")) == {f"leaf:{cid}" for cid in child_ids}


# ── deep recursion ───────────────────────────────────────────────────


class _DeepChainLLM(LLMClient):
    """Each level delegates to one child until depth==max_child_depth.

    Uses the tight pattern (``done`` in the same block as ``wait``).
    """

    def __init__(self, max_child_depth: int) -> None:
        self.max_child_depth = max_child_depth

    def chat(self, messages, *args, **kwargs):
        depth, max_depth = self._depth(messages)
        if depth < max_depth and depth < self.max_child_depth:
            return (
                "```repl\n"
                'h = delegate("child", "go deeper", "")\n'
                "results = yield wait(h)\n"
                'done(AGENT_ID + "->" + results[0])\n'
                "```"
            )
        return '```repl\ndone("leaf:" + AGENT_ID)\n```'

    @staticmethod
    def _depth(messages):
        system = (
            messages[0]["content"]
            if messages and messages[0].get("role") == "system"
            else ""
        )
        marker = "You are at recursion depth **"
        if marker not in system:
            return 0, 0
        rest = system.split(marker, 1)[1]
        depth_text, rest = rest.split("**", 1)
        max_text = rest.split("max **", 1)[1].split("**", 1)[0]
        return int(depth_text), int(max_text)


def test_deep_chain_each_level_has_supervising_sequence():
    agent = RLMFlow(
        _DeepChainLLM(max_child_depth=3),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=3, max_iterations=5),
    )
    g = _run(agent, agent.start("kick"))

    # All non-leaf agents use the tight pattern: q → a → s → r.
    expected_non_leaf = ["query", "action", "supervising", "result"]
    expected_leaf = ["query", "action", "result"]

    chain = ["root", "root.child", "root.child.child", "root.child.child.child"]
    for aid in chain[:-1]:
        sub = g[aid]
        assert _types(sub) == expected_non_leaf, f"{aid} types"
        sup = next(s for s in sub.states if isinstance(s, SupervisingNode))
        assert set(sup.waiting_on) == {aid + ".child"}
        _assert_spawn_links(sub, spawner_seq=1)
    assert _types(g[chain[-1]]) == expected_leaf

    _assert_seqs_monotonic(g)

    expected_result = (
        "root->root.child->root.child.child->leaf:root.child.child.child"
    )
    assert g.result() == expected_result


# ── large / mixed trees: depth 5+ ────────────────────────────────────


class _BranchingLLM(LLMClient):
    """At depth d an agent spawns ``fanouts[d]`` children (tight pattern).

    Once depth exceeds ``len(fanouts)`` — or ``fanouts[d] == 0`` — the
    agent is a leaf and returns ``leaf:<AGENT_ID>``. Non-leaves wrap their
    children's outputs as ``<AGENT_ID>(child1,child2,...)`` so the final
    result string is a faithful pre-order traversal of the tree.
    """

    def __init__(self, fanouts: list[int]) -> None:
        self.fanouts = fanouts

    def chat(self, messages, *args, **kwargs):
        depth = self._depth(messages)
        if depth >= len(self.fanouts) or self.fanouts[depth] == 0:
            return '```repl\ndone("leaf:" + AGENT_ID)\n```'
        n = self.fanouts[depth]
        delegations = "\n".join(
            f'h{i} = delegate("c{i}", "work at d{depth + 1}", "")'
            for i in range(n)
        )
        handles = ", ".join(f"h{i}" for i in range(n))
        return (
            "```repl\n"
            f"{delegations}\n"
            f"results = yield wait({handles})\n"
            'done(AGENT_ID + "(" + ",".join(results) + ")")\n'
            "```"
        )

    @staticmethod
    def _depth(messages):
        system = (
            messages[0]["content"]
            if messages and messages[0].get("role") == "system"
            else ""
        )
        marker = "You are at recursion depth **"
        if marker not in system:
            return 0
        rest = system.split(marker, 1)[1]
        return int(rest.split("**", 1)[0])


def _expected_agent_ids(prefix: str, fanouts: list[int]) -> list[str]:
    """Pre-order enumeration of the agent_ids produced by ``fanouts``."""
    ids = [prefix]
    if not fanouts:
        return ids
    n = fanouts[0]
    for i in range(n):
        ids.extend(_expected_agent_ids(f"{prefix}.c{i}", fanouts[1:]))
    return ids


def _expected_result(prefix: str, fanouts: list[int]) -> str:
    """Pre-order rendering of the leaf strings combined by their parents."""
    if not fanouts or fanouts[0] == 0:
        return f"leaf:{prefix}"
    n = fanouts[0]
    children = ",".join(
        _expected_result(f"{prefix}.c{i}", fanouts[1:]) for i in range(n)
    )
    return f"{prefix}({children})"


def test_depth_5_chain_each_level_has_supervising_sequence():
    """6 agents (depth 0..5) chained one-deep at every step."""
    fanouts = [1, 1, 1, 1, 1]
    agent = RLMFlow(
        _BranchingLLM(fanouts),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=5, max_iterations=5),
    )
    g = _run(agent, agent.start("kick"))

    expected_ids = _expected_agent_ids("root", fanouts)
    assert len(expected_ids) == 6
    actual_ids = [sub.agent_id for sub in g.walk()]
    assert sorted(actual_ids) == sorted(expected_ids)

    leaf_id = expected_ids[-1]
    for aid in expected_ids:
        sub = g[aid]
        if aid == leaf_id:
            assert _types(sub) == ["query", "action", "result"]
        else:
            assert _types(sub) == ["query", "action", "supervising", "result"]
            sup = next(s for s in sub.states if isinstance(s, SupervisingNode))
            assert set(sup.waiting_on) == set(sub.children)
            _assert_spawn_links(sub, spawner_seq=1)

    _assert_seqs_monotonic(g)
    assert g.result() == _expected_result("root", fanouts)


def test_depth_5_mixed_branching_tree_records_correct_sequences():
    """Depth-5 tree with a 3-way fan-out at depth 2.

    Shape (12 agents)::

        root                                              [d0]
        └── root.c0                                       [d1]
            └── root.c0.c0                                [d2]
                ├── root.c0.c0.c0 ─ ... ─ leaf            [d3..d5]
                ├── root.c0.c0.c1 ─ ... ─ leaf            [d3..d5]
                └── root.c0.c0.c2 ─ ... ─ leaf            [d3..d5]
    """
    fanouts = [1, 1, 3, 1, 1]
    agent = RLMFlow(
        _BranchingLLM(fanouts),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=5, max_iterations=5),
    )
    g = _run(agent, agent.start("kick"))

    expected_ids = _expected_agent_ids("root", fanouts)
    assert len(expected_ids) == 12  # 1 + 1 + 1 + 3 + 3 + 3
    actual_ids = sorted(sub.agent_id for sub in g.walk())
    assert actual_ids == sorted(expected_ids)

    # Leaves live at depth 5; everything else is non-leaf.
    leaves = {aid for aid in expected_ids if aid.count(".") == 5}
    assert len(leaves) == 3

    for aid in expected_ids:
        sub = g[aid]
        if aid in leaves:
            assert _types(sub) == ["query", "action", "result"], aid
            assert not sub.children
        else:
            assert _types(sub) == ["query", "action", "supervising", "result"], aid
            sup = next(s for s in sub.states if isinstance(s, SupervisingNode))
            assert set(sup.waiting_on) == set(sub.children), aid
            _assert_spawn_links(sub, spawner_seq=1)

    # Depth invariant — supplied by the engine, not asserted elsewhere.
    for aid in expected_ids:
        assert g[aid].depth == aid.count(".")

    _assert_seqs_monotonic(g)
    assert g.result() == _expected_result("root", fanouts)


# ── delegation + intra-agent loop ────────────────────────────────────


class _LoopyDelegatorLLM(LLMClient):
    """Root: action → observation → delegate → done.

    Verifies that observations don't get confused with the supervising/resume
    chain when delegation happens *after* an inline turn.
    """

    def chat(self, messages, *args, **kwargs):
        joined = "\n".join(m["content"] for m in messages)
        if "child task" in joined.lower() and 'delegate("child"' not in joined:
            return '```repl\ndone("c")\n```'
        if "READY" not in joined:
            return "```repl\nprint('READY')\n```"
        if 'delegate("child"' not in joined:
            return (
                "```repl\n"
                'h = delegate("child", "child task", "")\n'
                "results = yield wait(h)\n"
                'done("p:" + results[0])\n'
                "```"
            )
        return '```repl\ndone("p:c")\n```'


def test_intra_agent_loop_then_delegation_pins_full_root_sequence():
    agent = RLMFlow(
        _LoopyDelegatorLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=1, max_iterations=8),
    )
    g = _run(agent, agent.start("parent"))

    # Root: q → a → obs → a → s → r  (tight pattern after the observation).
    assert _types(g) == [
        "query",
        "action",
        "observation",
        "action",
        "supervising",
        "result",
    ]
    assert _types(g["root.child"]) == ["query", "action", "result"]
    _assert_seqs_monotonic(g)

    # Child was spawned by root's seq=3 action (the one after the observation).
    spawn_action = g.states[3]
    assert spawn_action.type == "action"
    assert g["root.child"].parent_node_id == spawn_action.id

    assert g.result() == "p:c"
