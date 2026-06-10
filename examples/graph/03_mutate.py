"""Editing a Graph in place.

A Graph is mutable. The engine writes through ``add_node`` /
``add_child`` during a run, but you can use the same helpers offline to
edit a persisted graph: rewrite a result, drop an agent, swap a node.

Demonstrates:

- ``graph.add_node(node)``                    — append a node
- ``graph.update_node(node_id, **changes)``   — copy-with-changes by id
- ``graph.set_node(node_id, new_node)``       — full swap by id
- ``graph.remove_node(node_id)``              — drop a node
- ``graph.add_child(child)`` / ``graph.remove_child(aid)``
- ``graph.update(**fields)``                  — bulk top-level edit
- ``graph.all_nodes.replace / update / remove``   — by id, anywhere in subtree
- ``graph.copy(deep=True)``                   — clone before mutating

Run:
    python examples/graph/03_mutate.py
"""

from __future__ import annotations

from rlmflow.graph import DoneOutput, Graph, UserQuery


def base_graph() -> Graph:
    root_q = UserQuery(agent_id="root", seq=0, content="hello")
    root_done = DoneOutput(agent_id="root", seq=1, result="ok")
    child_q = UserQuery(agent_id="root.child", seq=0, content="sub")
    child_done = DoneOutput(agent_id="root.child", seq=1, result="sub ok")
    child = Graph.from_meta_dict(
        {"agent_id": "root.child", "depth": 1, "parent_agent_id": "root"},
        nodes=[child_q, child_done],
    )
    return Graph.from_meta_dict(
        {"agent_id": "root", "depth": 0, "query": "hello"},
        nodes=[root_q, root_done],
        children={"root.child": child},
    )


def banner(title: str) -> None:
    print("\n" + "─" * 60)
    print(title)
    print("─" * 60)


def summary(g: Graph) -> str:
    return (
        f"agents={list(g.agents)} nodes={len(g.all_nodes)} "
        f"result={g.result()!r} model={g.model_label}"
    )


def main() -> None:
    g = base_graph()
    banner("baseline")
    print(summary(g))

    banner("graph.copy(deep=True) — clone before mutating")
    twin = g.copy(deep=True)
    twin.update(model="gpt-5", config={"temperature": 0.0})
    print(f"original: {summary(g)}")
    print(f"twin    : {summary(twin)}")

    banner("update_node — copy-with-changes by id")
    result_id = g.all_nodes.results()[0].id
    g.update_node(result_id, result="ok (rewritten)")
    print(summary(g))

    banner("set_node — swap a node object")
    g.set_node(result_id, DoneOutput(
        agent_id="root", seq=1, result="ok (full swap)", id=result_id,
    ))
    print(summary(g))

    banner("nodes.update — same edit, but addressed via the flat view")
    child_result_id = g["root.child"].all_nodes.results()[0].id
    g.all_nodes.update(child_result_id, result="sub ok (via subtree view)")
    print(f"root.child result -> {g['root.child'].result()!r}")

    banner("add_node — append onto a sub-Graph")
    g["root.child"].add_node(UserQuery(agent_id="root.child", seq=2, content="follow-up"))
    print(f"root.child nodes: {[n.type for n in g['root.child'].nodes]}")

    banner("add_child / remove_child — attach + detach sub-agents")
    sibling = Graph.from_meta_dict(
        {"agent_id": "root.sibling", "depth": 1, "parent_agent_id": "root"},
        nodes=[UserQuery(agent_id="root.sibling", seq=0, content="hi")],
    )
    g.add_child(sibling)
    print(f"after add_child : {list(g.agents)}")
    g.remove_child("root.child")
    print(f"after remove    : {list(g.agents)}")

    banner("graph.update — bulk top-level field edit")
    g.update(query="hello (updated)", config={"max_depth": 2})
    print(f"query={g.query!r} config={g.config}")


if __name__ == "__main__":
    main()
