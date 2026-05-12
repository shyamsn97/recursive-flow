"""Static topology exports (Mermaid, DOT, D2) for :class:`Graph` snapshots."""

from __future__ import annotations

from rlmflow.graph import Graph, Node, ResultNode


def _sanitize(node_id: str) -> str:
    return node_id.replace(".", "_").replace("-", "_") or "root"


def _truncate(text: str, n: int = 60) -> str:
    text = text.replace("\n", " ").strip()
    return text[: n - 1] + "..." if len(text) > n else text


def _escape_mermaid(text: str) -> str:
    return text.replace('"', "'").replace("\n", " ")


def _escape_dot(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


_MERMAID_FLOW_CLASS = {
    "query": "query",
    "observation": "obs",
    "action": "action",
    "supervising": "sup",
    "resume": "resume",
    "error": "err",
    "result": "result",
}

_NODE_COLOR = {
    "query": "#58a6ff",
    "observation": "#58a6ff",
    "action": "#d29922",
    "supervising": "#bc8cff",
    "resume": "#7ee787",
    "error": "#f85149",
    "result": "#3fb950",
}


def _state_label(state: Node) -> str:
    """Short human-readable state label for diagram nodes."""
    return f"{state.agent_id} ({state.type})"


def _state_result_text(state: Node) -> str | None:
    if isinstance(state, ResultNode) and state.result:
        return state.result
    return None


# ── Mermaid state diagram ────────────────────────────────────────────


def to_mermaid(graph: Graph, *, include_results: bool = True) -> str:
    """Render ``graph`` as a Mermaid ``stateDiagram-v2``."""
    declarations: list[str] = []
    transitions: list[str] = []
    roots = _root_nodes(graph)

    for state in graph.nodes:
        nid = _sanitize(state.id)
        declarations.append(
            f'    state "{_escape_mermaid(_state_label(state))}" as {nid}'
        )
    for root in roots:
        transitions.append(f"    [*] --> {_sanitize(root.id)}")
    for edge in graph.edges:
        transitions.append(f"    {_sanitize(edge.from_)} --> {_sanitize(edge.to)}")
    if include_results:
        for state in graph.nodes:
            res = _state_result_text(state)
            if res:
                transitions.append(
                    f"    {_sanitize(state.id)} --> [*] : {_escape_mermaid(_truncate(res))}"
                )

    return "\n".join(["stateDiagram-v2", *declarations, *transitions])


# ── DOT ──────────────────────────────────────────────────────────────


def to_dot(graph: Graph, *, include_results: bool = True) -> str:
    lines = [
        "digraph rlmflow {",
        "    rankdir=TB;",
        '    node [shape=box, style="rounded,filled", fontname="Helvetica"];',
        '    edge [fontname="Helvetica", fontsize=10];',
    ]
    for state in graph.nodes:
        nid = _sanitize(state.id)
        color = _NODE_COLOR.get(state.type, "#8b949e")
        parts = [state.agent_id or "root", state.type]
        if include_results:
            res = _state_result_text(state)
            if res:
                parts.append(_truncate(res, 40))
        label = "\\n".join(_escape_dot(part) for part in parts)
        lines.append(
            f'    {nid} [label="{label}", fillcolor="{color}22", color="{color}"];'
        )
    for edge in graph.edges:
        style = "solid" if edge.kind == "flows_to" else "dashed"
        lines.append(
            f"    {_sanitize(edge.from_)} -> {_sanitize(edge.to)} "
            f'[label="{edge.kind}", style={style}];'
        )
    lines.append("}")
    return "\n".join(lines)


# ── Mermaid flowchart ────────────────────────────────────────────────


def to_mermaid_flowchart(graph: Graph, *, include_results: bool = True) -> str:
    lines = ["flowchart TD"]
    for state in graph.nodes:
        nid = _sanitize(state.id)
        agent = state.agent_id or "root"
        body = f"{agent}<br/><i>{state.type}</i>"
        if include_results:
            res = _state_result_text(state)
            if res:
                body += f"<br/>{_escape_mermaid(_truncate(res, 40))}"
        lines.append(
            f'    {nid}["{body}"]:::{_MERMAID_FLOW_CLASS.get(state.type, "obs")}'
        )
    for edge in graph.edges:
        lines.append(
            f"    {_sanitize(edge.from_)} -->|{edge.kind}| {_sanitize(edge.to)}"
        )
    lines.extend(
        [
            "    classDef query    fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;",
            "    classDef obs      fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;",
            "    classDef action   fill:#d2992222,stroke:#d29922,color:#c9d1d9;",
            "    classDef sup      fill:#bc8cff22,stroke:#bc8cff,color:#c9d1d9;",
            "    classDef resume   fill:#7ee78722,stroke:#7ee787,color:#c9d1d9;",
            "    classDef err      fill:#f8514922,stroke:#f85149,color:#c9d1d9;",
            "    classDef result   fill:#3fb95022,stroke:#3fb950,color:#c9d1d9;",
        ]
    )
    return "\n".join(lines)


# ── Mermaid sequence diagram ─────────────────────────────────────────


def to_mermaid_sequence(graph: Graph) -> str:
    """Delegate / wait / done flow between agents."""
    lines = ["sequenceDiagram"]
    for aid in graph.agents:
        lines.append(f"    participant {_sanitize(aid)} as {aid}")

    spawns = graph.edges.spawns()
    by_id = {e.id: e for e in graph.nodes}
    for edge in spawns:
        parent = by_id.get(edge.from_)
        child = by_id.get(edge.to)
        if parent is None or child is None:
            continue
        parent_id = _sanitize(parent.agent_id)
        child_id = _sanitize(child.agent_id)
        lines.append(f"    {parent_id}->>+{child_id}: delegate")
        child_sub = graph.agents[child.agent_id]
        cur = child_sub.current()
        if cur is not None and cur.terminal:
            kind = "done" if cur.type == "result" else cur.type
            res = getattr(cur, "result", None)
            summary = _truncate(res, 30) if res else kind
            lines.append(f"    {child_id}-->>-{parent_id}: {_escape_mermaid(summary)}")
    return "\n".join(lines)


# ── D2 ───────────────────────────────────────────────────────────────


_D2_STYLES = {
    "query": '{ style: { fill: "#1f6feb22"; stroke: "#58a6ff" } }',
    "observation": '{ style: { fill: "#1f6feb22"; stroke: "#58a6ff" } }',
    "action": '{ style: { fill: "#d2992222"; stroke: "#d29922" } }',
    "supervising": '{ style: { fill: "#bc8cff22"; stroke: "#bc8cff" } }',
    "resume": '{ style: { fill: "#7ee78722"; stroke: "#7ee787" } }',
    "error": '{ style: { fill: "#f8514922"; stroke: "#f85149" } }',
    "result": '{ style: { fill: "#3fb95022"; stroke: "#3fb950" } }',
}


def to_d2(graph: Graph, *, include_results: bool = True) -> str:
    lines: list[str] = []
    for state in graph.nodes:
        nid = _sanitize(state.id)
        agent = state.agent_id or "root"
        label = f"{agent}\\n{state.type}"
        if include_results:
            res = _state_result_text(state)
            if res:
                label += f"\\n{_truncate(res, 40)}"
        style = _D2_STYLES.get(state.type, "")
        lines.append(f'{nid}: "{label}" {style}'.rstrip())
    for edge in graph.edges:
        lines.append(f"{_sanitize(edge.from_)} -> {_sanitize(edge.to)}: {edge.kind}")
    return "\n".join(lines)


# ── helpers ──────────────────────────────────────────────────────────


def _root_nodes(graph: Graph) -> list[Node]:
    """Nodes with no incoming edge — used as ``[*] --> X`` Mermaid roots."""
    targets = {edge.to for edge in graph.edges}
    return [n for n in graph.nodes if n.id not in targets]


__all__ = [
    "to_d2",
    "to_dot",
    "to_mermaid",
    "to_mermaid_flowchart",
    "to_mermaid_sequence",
]
