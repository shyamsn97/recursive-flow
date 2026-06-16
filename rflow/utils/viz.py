"""Terminal/HTML visualizations: live Rich tree, Gantt swimlanes, reports.

These accept the same sources as the viewer: an in-memory :class:`~rflow.graph.Graph`,
a graph list, a ``trace.json`` path, or a directory. Trace-wide views use every
snapshot; single-snapshot views use the latest graph.

Token/cost views from the legacy module are intentionally omitted until real token
accounting lands (Phase 2).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from rflow.graph import Graph, is_done, is_errored, is_supervising
from rflow.utils.export import _kind
from rflow.utils.viewer import ViewSource, _model_label, graph_tree, resolve_graphs


def _resolve_graphs(source: ViewSource) -> list[Graph]:
    return resolve_graphs(source)


def _resolve_latest_graph(source: ViewSource) -> Graph:
    graphs = _resolve_graphs(source)
    if not graphs:
        raise ValueError("expected at least one Graph")
    return graphs[-1]


# ── live tree ────────────────────────────────────────────────────────


class LiveView:
    """Live-updating Rich tree of a running flow."""

    def __init__(self, *, console: Any = None) -> None:
        from rich.console import Console

        self._console = console or Console()
        self._live: Any = None

    def __enter__(self) -> "LiveView":
        from rich.live import Live

        self._live = Live(
            console=self._console,
            vertical_overflow="visible",
            auto_refresh=False,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._live is not None:
            self._live.__exit__(*exc)
            self._live = None

    def __call__(self, source: ViewSource) -> None:
        if self._live is None:
            raise RuntimeError("LiveView used outside of its context manager")
        latest = _resolve_latest_graph(source)
        self._live.update(_render_rich_tree(latest), refresh=True)


def _render_rich_tree(graph: Graph):
    from rich.text import Text
    from rich.tree import Tree

    def is_settled(aid: str) -> bool:
        sub = graph.agents.get(aid)
        cur = sub.current() if sub is not None else None
        return bool(cur and (is_done(cur) or is_errored(cur)))

    def running_children(sub: Graph) -> tuple[int, int] | None:
        cur = sub.current()
        if not is_supervising(cur):
            return None
        waiting_on = list(cur.waiting_on or [])
        active = sum(1 for child_id in waiting_on if not is_settled(child_id))
        return active, len(waiting_on)

    def label_for(aid: str) -> Text:
        sub = graph.agents[aid]
        cur = sub.current()
        info = Text()
        info.append(aid, style="bold")
        info.append(f" [{_model_label(sub)}]", style="cyan")
        if cur is not None:
            info.append(
                f" [{cur.type}]", style="magenta" if not cur.terminal else "green"
            )
        counts = running_children(sub)
        if counts is not None:
            active, total = counts
            info.append(f" | children running {active}/{total}", style="cyan")
        return info

    def populate(tree: Tree, aid: str) -> None:
        for child_aid in graph.agents[aid].children:
            child = tree.add(label_for(child_aid), guide_style="dim")
            populate(child, child_aid)

    def build(aid: str) -> Tree:
        tree = Tree(label_for(aid), guide_style="dim")
        populate(tree, aid)
        return tree

    return build(graph.agent_id)


def live_view(**kwargs: Any) -> LiveView:
    return LiveView(**kwargs)


def live(flow: Any, source: ViewSource | None = None) -> list[Graph]:
    """Drive ``flow``'s step loop to completion while streaming a live tree.

    Uses the functional step API: ``flow.start(query)`` (or ``step(query=...)``)
    must have seeded the run, then each ``graph = flow.step(graph)`` returns a
    fresh, frozen snapshot. Returns the list of per-tick snapshots.
    """
    graph = flow.graph if source is None else _resolve_latest_graph(source)
    if graph is None:
        raise ValueError("live() needs flow.start(query) to have been called first")
    graphs = [graph]
    with LiveView() as lv:
        lv(graph)
        while not graph.finished:
            graph = flow.step(graph)
            graphs.append(graph)
            lv(graph)
    return graphs


# ── gantt swimlanes ──────────────────────────────────────────────────


def _gantt_per_step(graphs: list[Graph]) -> tuple[list[str], list[list[str | None]]]:
    order: list[str] = []
    seen: set[str] = set()
    per_step: list[dict[str, str]] = []
    for graph in graphs:
        step_map: dict[str, str] = {}
        for aid, sub in graph.agents.items():
            cur = sub.current()
            if cur is None:
                continue
            step_map[aid] = f"{_kind(cur)} ({_model_label(sub)})"
            if aid not in seen:
                seen.add(aid)
                order.append(aid)
        per_step.append(step_map)
    return order, [[step.get(aid) for step in per_step] for aid in order]


_TYPE_CELL = {
    "query": ("Q", "blue"),
    "llm_call": ("a", "yellow"),
    "llm": ("A", "yellow"),
    "exec_call": ("o", "blue"),
    "exec": ("O", "blue"),
    "supervising": ("S", "magenta"),
    "resume_call": ("r", "green"),
    "resume": ("R", "green"),
    "errored": ("E", "red"),
    "done": ("F", "green"),
}

_TYPE_HTML = {
    "query": "#58a6ff",
    "llm_call": "#a98a2a",
    "llm": "#d29922",
    "exec_call": "#b87650",
    "exec": "#ff9e64",
    "supervising": "#bc8cff",
    "resume_call": "#5fa067",
    "resume": "#7ee787",
    "errored": "#f85149",
    "done": "#3fb950",
}


def gantt(source: ViewSource) -> None:
    """Print a per-step swimlane table (one row per agent) to the terminal."""
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    graphs = _resolve_graphs(source)
    agents, rows = _gantt_per_step(graphs)
    table = Table(
        show_header=True,
        header_style="dim",
        show_lines=False,
        pad_edge=False,
        padding=(0, 0),
    )
    table.add_column("agent", style="bold", no_wrap=True)
    for i in range(len(graphs)):
        table.add_column(str(i), justify="center", no_wrap=True)
    for aid, row in zip(agents, rows):
        cells = [Text(aid)]
        for kind in row:
            if kind is None:
                cells.append(Text(" ", style="dim"))
            else:
                node_type = kind.split(" ", 1)[0]
                glyph, color = _TYPE_CELL.get(node_type, ("?", "dim"))
                cells.append(Text(glyph, style=color))
        table.add_row(*cells)
    Console().print(table)


def gantt_html(source: ViewSource, *, title: str = "recursive-flow gantt") -> str:
    """Render the per-step swimlane as a self-contained HTML page."""
    graphs = _resolve_graphs(source)
    agents, rows = _gantt_per_step(graphs)
    n_steps = len(graphs)
    cells_html: list[str] = []
    for aid, row in zip(agents, rows):
        cells_html.append(
            f'<div class="row"><div class="name">{aid}</div>'
            f'<div class="bars" style="grid-template-columns: repeat({n_steps}, 1fr)">'
        )
        for kind in row:
            if kind is None:
                cells_html.append('<div class="cell empty"></div>')
            else:
                node_type = kind.split(" ", 1)[0]
                color = _TYPE_HTML.get(node_type, "#8b949e")
                cells_html.append(
                    f'<div class="cell" style="background:{color}" title="{kind}"></div>'
                )
        cells_html.append("</div></div>")
    legend = "".join(
        f'<span><i style="background:{c}"></i>{s}</span>' for s, c in _TYPE_HTML.items()
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; background: #0d1117; color: #e6edf3; margin: 24px; }}
h1 {{ font-size: 14px; color: #8b949e; font-weight: 500; margin: 0 0 12px; }}
.row {{ display: grid; grid-template-columns: 220px 1fr; gap: 8px; align-items: center; margin: 2px 0; }}
.name {{ font-family: monospace; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.bars {{ display: grid; gap: 1px; height: 18px; background: #161b22; border: 1px solid #30363d; border-radius: 3px; overflow: hidden; }}
.cell {{ height: 100%; }} .cell.empty {{ background: transparent; }}
.legend {{ margin-top: 16px; font-size: 12px; color: #8b949e; }} .legend span {{ margin-right: 16px; }}
.legend i {{ display: inline-block; width: 10px; height: 10px; margin-right: 4px; border-radius: 2px; vertical-align: middle; }}
</style></head><body>
<h1>{title} - {n_steps} steps, {len(agents)} agents</h1>
{"".join(cells_html)}
<div class="legend">{legend}</div>
</body></html>"""


# ── error / code log / report ────────────────────────────────────────


def error_summary(source: ViewSource) -> str:
    """Group every :class:`ErrorOutput` in ``source`` by ``error`` kind."""
    graph = _resolve_latest_graph(source)
    errors = [e for e in graph.all_nodes if is_errored(e)]
    if not errors:
        return "(no errors)"
    by_kind: Counter[str] = Counter()
    samples: dict[str, str] = {}
    for err in errors:
        kind = err.error or "(unknown)"
        by_kind[kind] += 1
        if kind not in samples:
            head = (err.content or "").strip().splitlines()
            samples[kind] = head[0] if head else ""
    lines = [f"{len(errors)} error(s) across {len(by_kind)} kind(s):"]
    for kind, count in by_kind.most_common():
        lines.append(f"  {kind}: {count}")
        if samples.get(kind):
            lines.append(f"    └─ {samples[kind][:120]}")
    return "\n".join(lines)


def code_log(source: ViewSource, agent_id: str | None = None) -> str:
    """Render every executed code block in the run, paired with its output."""
    graphs = _resolve_graphs(source)
    if not graphs:
        return "(no code blocks)"
    graph = graphs[-1]

    nodes = list(graph.all_nodes)
    if agent_id:
        nodes = [n for n in nodes if n.agent_id == agent_id]

    by_agent: dict[str, list] = {}
    for n in nodes:
        by_agent.setdefault(n.agent_id, []).append(n)

    out: list[str] = []
    for aid, states in by_agent.items():
        for i, node in enumerate(states):
            if node.type not in ("exec_action", "resume_action"):
                continue
            code = getattr(node, "code", "") or ""
            if not code:
                continue
            out.append(f"# [{aid}] {node.type}")
            out.append(code.strip())
            obs = states[i + 1] if i + 1 < len(states) else None
            output = ""
            if obs is not None:
                output = (
                    getattr(obs, "content", "")
                    or getattr(obs, "output", "")
                    or getattr(obs, "result", "")
                    or ""
                )
            if output:
                out.append("→ " + output.strip()[:240])
            out.append("")
    return "\n".join(out).rstrip() or "(no code blocks)"


def report_md(source: ViewSource, *, title: str = "recursive-flow run") -> str:
    """Render a Markdown summary of a run: stats + tree + errors + result."""
    graphs = _resolve_graphs(source)
    if not graphs:
        return f"# {title}\n\n(empty trace)\n"

    final = graphs[-1]
    parts: list[str] = [f"# {title}", ""]
    parts.append(f"**Steps:** {len(graphs)}")
    parts.append(f"**Agents:** {len(final.agents)}")
    current = final.current()
    parts.append(f"**Outcome:** {current.type if current else 'empty'}")
    errors = [e for e in final.all_nodes if is_errored(e)]
    if errors:
        parts.append(f"**Errors:** {len(errors)}")

    parts.extend(["", "## Tree", "", "```", graph_tree(final), "```"])
    if errors:
        parts.extend(["", "## Errors", "", "```", error_summary(final), "```"])

    result = final.result()
    if result:
        parts.extend(["", "## Result", "", "```", str(result), "```"])

    # Token/cost accounting is omitted until real usage lands (Phase 2).
    return "\n".join(parts) + "\n"


__all__ = [
    "LiveView",
    "code_log",
    "error_summary",
    "gantt",
    "gantt_html",
    "live",
    "live_view",
    "report_md",
]
