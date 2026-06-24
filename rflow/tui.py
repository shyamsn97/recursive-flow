"""Full-screen Textual TUI for driving a :class:`rflow.Flow`.

The module is import-light on purpose. Rich/Textual are imported lazily so
``import rflow`` remains a small engine import unless the user explicitly opens
the TUI.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from rflow.graph import (
    ActionNode,
    CodeObservation,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    Graph,
    LLMOutput,
    Node,
    ResumeAction,
    SupervisingOutput,
    UserQuery,
)

# pyright: reportMissingImports=false


if TYPE_CHECKING:
    from rflow.flow import Flow


def tui(
    flow: "Flow",
    *,
    max_steps_per_turn: int | None = None,
) -> Graph | None:
    """Open an interactive terminal chat for ``flow`` and return the latest graph.

    The app starts idle and waits for the first prompt in the bottom input.
    Each submitted prompt starts or appends a root turn and drives it until the
    run settles.
    """

    return run_tui(
        flow,
        max_steps_per_turn=max_steps_per_turn,
    )


def run_tui(
    flow: "Flow",
    *,
    max_steps_per_turn: int | None = None,
) -> Graph | None:
    """Run the Textual app, importing optional dependencies only here."""

    try:
        from rich.console import Group
        from rich.panel import Panel
        from rich.text import Text
        from textual import work
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Horizontal, Vertical, VerticalScroll
        from textual.widgets import (
            Footer,
            Header,
            RichLog,
            Static,
            TabbedContent,
            TabPane,
            TextArea,
        )
    except ImportError as exc:  # pragma: no cover - exercised by environments
        raise ImportError(
            "flow.tui() requires the TUI extra: " "`pip install rlmflow[tui]`."
        ) from exc

    class FlowTUI(App[None]):
        """Live chat + graph dashboard around one Flow instance."""

        TITLE = "rlmflow TUI"
        CSS = """
        Screen {
            layout: vertical;
        }
        #main {
            height: 1fr;
        }
        #chat-column {
            width: 3fr;
            min-width: 50;
        }
        #side-column {
            width: 2fr;
            min-width: 42;
        }
        #chat {
            height: 1fr;
            border: round $primary;
            padding: 0 1;
        }
        #tabs {
            height: 1fr;
        }
        #prompt {
            margin-top: 1;
            height: 5;
            border: round $primary;
        }
        #context {
            margin-top: 1;
            height: 5;
            border: round $primary;
        }
        TabPane {
            padding: 0 1;
        }
        #overview-scroll, #tree-scroll, #agents-scroll,
        #counts-scroll, #waiting-scroll, #errors-scroll, #latest-scroll {
            height: 1fr;
            overflow-y: auto;
        }
        #overview, #tree, #agents, #counts, #waiting, #errors, #latest {
            height: auto;
            width: 100%;
        }
        """
        BINDINGS = [
            ("ctrl+c", "quit", "Quit"),
            Binding("ctrl+enter", "submit_prompt", "Send", priority=True),
            ("ctrl+r", "run_until_done", "Run"),
            ("ctrl+t", "step_once", "Step"),
        ]

        def __init__(self) -> None:
            super().__init__()
            self.flow = flow
            self.graph: Graph | None = flow.graph
            self.max_steps_per_turn = max_steps_per_turn
            self._seen_nodes: set[str] = set()
            self._pending: list[tuple[str | None, dict[str, str] | None, Any]] = []
            self._busy = False

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="main"):
                with Vertical(id="chat-column"):
                    yield RichLog(id="chat", wrap=True, markup=False, highlight=True)
                    yield TextArea(
                        id="prompt",
                        soft_wrap=True,
                        placeholder=(
                            "Query... " "(what should the agent do? Ctrl+Enter to send)"
                        ),
                    )
                    yield TextArea(
                        id="context",
                        soft_wrap=True,
                        placeholder=(
                            "Context... "
                            "(optional supporting text, passed as INPUTS['context'])"
                        ),
                    )
                with Vertical(id="side-column"):
                    with TabbedContent(initial="overview-tab", id="tabs"):
                        with TabPane("Overview", id="overview-tab"):
                            with VerticalScroll(id="overview-scroll"):
                                yield Static(id="overview")
                        with TabPane("Tree", id="tree-tab"):
                            with VerticalScroll(id="tree-scroll"):
                                yield Static(id="tree")
                        with TabPane("Agents", id="agents-tab"):
                            with VerticalScroll(id="agents-scroll"):
                                yield Static(id="agents")
                        with TabPane("Counts", id="counts-tab"):
                            with VerticalScroll(id="counts-scroll"):
                                yield Static(id="counts")
                        with TabPane("Waiting", id="waiting-tab"):
                            with VerticalScroll(id="waiting-scroll"):
                                yield Static(id="waiting")
                        with TabPane("Errors", id="errors-tab"):
                            with VerticalScroll(id="errors-scroll"):
                                yield Static(id="errors")
                        with TabPane("Latest", id="latest-tab"):
                            with VerticalScroll(id="latest-scroll"):
                                yield Static(id="latest")
            yield Footer()

        def on_mount(self) -> None:
            self._refresh()
            self.query_one("#chat", RichLog).write(
                Panel(
                    "Type a query below, add optional context, then press Ctrl+Enter. "
                    "Ctrl+R continues a paused run.",
                    title="ready",
                    border_style="cyan",
                )
            )

        def action_submit_prompt(self) -> None:
            query_area = self.query_one("#prompt", TextArea)
            context_area = self.query_one("#context", TextArea)
            query = query_area.text.strip()
            if not query:
                return
            context = context_area.text.strip()
            query_area.text = ""
            turn_inputs = _context_inputs(context)
            if turn_inputs is None and self.graph is not None:
                turn_inputs = {"context": ""}
            self._queue(query, turn_inputs, None)

        def action_run_until_done(self) -> None:
            if self.graph is None:
                self._status("No graph yet. Type a prompt first.", style="yellow")
                return
            self._queue(None, None, None)

        def action_step_once(self) -> None:
            if self.graph is None:
                self._status("No graph yet. Type a prompt first.", style="yellow")
                return
            if self._busy:
                self._status("A turn is already running.", style="yellow")
                return
            self._busy = True
            self._step_once()

        def _queue(
            self,
            prompt: str | None,
            turn_inputs: dict[str, str] | None,
            schema: Any,
        ) -> None:
            self._pending.append((prompt, turn_inputs, schema))
            if prompt:
                self.query_one("#chat", RichLog).write(
                    Panel(prompt, title="you", border_style="blue")
                )
            if not self._busy:
                self._busy = True
                self._drain_queue()
            else:
                self._status("Queued prompt.", style="cyan")

        @work(thread=True, exclusive=True)
        def _drain_queue(self) -> None:
            try:
                while self._pending:
                    prompt, turn_inputs, schema = self._pending.pop(0)
                    self._advance(prompt, turn_inputs, schema, until_done=True)
            finally:
                self.call_from_thread(self._set_busy, False)

        @work(thread=True, exclusive=True)
        def _step_once(self) -> None:
            try:
                self._advance(None, None, None, until_done=False)
            finally:
                self.call_from_thread(self._set_busy, False)

        def _advance(
            self,
            prompt: str | None,
            turn_inputs: dict[str, str] | None,
            schema: Any,
            *,
            until_done: bool,
        ) -> None:
            try:
                if self.graph is None:
                    if prompt is None:
                        return
                    self.graph = self.flow.start(
                        prompt,
                        turn_inputs,
                        output_schema=schema,
                    )
                    self.call_from_thread(self._refresh)
                elif prompt is not None or not self.graph.finished:
                    self.graph = self.flow.step(
                        self.graph,
                        query=prompt,
                        inputs=turn_inputs,
                        output_schema=schema,
                    )
                    self.call_from_thread(self._refresh)

                steps = 0
                while until_done and self.graph is not None and not self.graph.finished:
                    if (
                        self.max_steps_per_turn is not None
                        and steps >= self.max_steps_per_turn
                    ):
                        self.call_from_thread(
                            self._status,
                            f"Stopped at step cap ({self.max_steps_per_turn}).",
                            "yellow",
                        )
                        return
                    self.graph = self.flow.step(self.graph)
                    steps += 1
                    self.call_from_thread(self._refresh)
            except Exception as exc:  # noqa: BLE001 - TUI should surface failures
                self.call_from_thread(
                    self._status,
                    f"{type(exc).__name__}: {exc}",
                    "red",
                )

        def _set_busy(self, value: bool) -> None:
            self._busy = value
            self._refresh()

        def _status(self, message: str, style: str = "cyan") -> None:
            self.query_one("#chat", RichLog).write(
                Panel(message, title="status", border_style=style)
            )

        def _refresh(self) -> None:
            graph = self.graph
            if graph is None:
                empty = Panel(
                    "No run yet. Type a prompt in the chat input.",
                    title="rlmflow",
                    border_style="dim",
                )
                for wid in (
                    "#overview",
                    "#tree",
                    "#agents",
                    "#counts",
                    "#waiting",
                    "#errors",
                    "#latest",
                ):
                    self.query_one(wid, Static).update(empty)
                return

            chat = self.query_one("#chat", RichLog)
            for node_id, bubble in chat_bubbles(graph, seen=self._seen_nodes):
                self._seen_nodes.add(node_id)
                chat.write(bubble)

            self.query_one("#overview", Static).update(
                Panel(
                    Group(
                        run_stats_table(graph, busy=self._busy),
                        Text(""),
                        waiting_table(graph),
                    ),
                    title="overview",
                )
            )
            self.query_one("#tree", Static).update(render_full_tree_panel(graph))
            self.query_one("#agents", Static).update(
                Panel(agent_table(graph), title="agents")
            )
            self.query_one("#counts", Static).update(
                Panel(node_counts_table(graph), title="node counts")
            )
            self.query_one("#waiting", Static).update(
                Panel(waiting_table(graph), title="waiting")
            )
            self.query_one("#errors", Static).update(
                Panel(error_table(graph), title="errors")
            )
            self.query_one("#latest", Static).update(
                Panel(latest_table(graph), title="latest nodes")
            )

    app = FlowTUI()
    app.run()
    return app.graph


def chat_bubbles(
    graph: Graph, *, seen: set[str] | None = None
) -> list[tuple[str, Any]]:
    """Return Rich panels for graph nodes not present in ``seen``."""

    from rich.panel import Panel

    seen = seen or set()
    out: list[tuple[str, Any]] = []
    for node in _ordered_nodes(graph):
        if node.id in seen:
            continue
        kind = _kind(node)
        title = _node_title(node, kind)
        body = _node_renderable(node)
        out.append(
            (
                node.id,
                Panel(
                    body,
                    title=title,
                    border_style=_node_style(node),
                    padding=(0, 1),
                ),
            )
        )
    return out


def _context_inputs(context: str) -> dict[str, str] | None:
    text = context.strip()
    return {"context": text} if text else None


def render_tree_panel(graph: Graph) -> Any:
    """Render the live Rich tree used by the side panel."""

    from rich.panel import Panel

    from rflow.utils.viz import _render_rich_tree

    return Panel(_render_rich_tree(graph), title="live tree", border_style="green")


def render_full_tree_panel(graph: Graph) -> Any:
    """Render the full graph as a scrollable Rich tree diagram."""

    from rich.panel import Panel

    return Panel(
        _render_execution_diagram(graph),
        title="execution tree",
        border_style="green",
    )


def _render_execution_diagram(graph: Graph) -> Any:
    from rich.text import Text
    from rich.tree import Tree

    from rflow.utils.viewer import _model_label

    def agent_label(sub: Graph) -> Text:
        label = Text()
        label.append(sub.agent_id or "root", style="bold")
        model = _model_label(sub)
        if model:
            label.append(f" ({model})", style="cyan")
        if sub.query:
            label.append(f" - {_clip_one_line(sub.query, 56)}", style="dim")
        return label

    def node_label(node: Node) -> Text:
        label = Text()
        step = "-" if node.global_step is None else str(node.global_step)
        kind = _kind(node)
        label.append(f"[{node.seq:>2} | step {step}] ", style="dim")
        label.append(kind, style=_node_style(node))
        summary = _diagram_summary(node)
        if summary:
            label.append(f" - {summary}", style="dim")
        return label

    def add_agent(sub: Graph, parent: Tree | None = None) -> Tree:
        branch = (
            Tree(agent_label(sub), guide_style="green")
            if parent is None
            else parent.add(agent_label(sub), guide_style="green")
        )
        state_ids = {node.id for node in sub.nodes}
        sup_for_agent: dict[str, str] = {}
        for node in sub.nodes:
            if isinstance(node, SupervisingOutput):
                for child_id in node.waiting_on:
                    sup_for_agent[child_id] = node.id

        attach_at: dict[str, list[Graph]] = {}
        unplaced: list[Graph] = []
        for child in sub.children.values():
            key = sup_for_agent.get(child.agent_id) or (
                child.parent_node_id if child.parent_node_id in state_ids else None
            )
            if key:
                attach_at.setdefault(key, []).append(child)
            else:
                unplaced.append(child)

        if not sub.nodes:
            branch.add(Text("(no nodes)", style="dim"), guide_style="dim")

        for node in sub.nodes:
            node_branch = branch.add(node_label(node), guide_style="dim")
            for child in attach_at.get(node.id, []):
                add_agent(child, node_branch)
        for child in unplaced:
            add_agent(child, branch)
        return branch

    return add_agent(graph)


def _diagram_summary(node: Node) -> str:
    if isinstance(node, SupervisingOutput):
        return "waiting on " + (", ".join(node.waiting_on) or "-")
    if isinstance(node, DoneOutput):
        return _clip_one_line(node.result or node.output, 60)
    if isinstance(node, ErrorOutput):
        return _clip_one_line(node.error or node.content or node.output, 60)
    if isinstance(node, UserQuery):
        return _clip_one_line(node.content, 60)
    if isinstance(node, LLMOutput):
        return _clip_one_line(_assistant_body(node), 60)
    if isinstance(node, ExecOutput):
        return _clip_one_line(node.content or node.output, 60)
    if isinstance(node, ResumeAction):
        return "resumed from " + (", ".join(node.resumed_from) or "-")
    return ""


def run_stats_table(graph: Graph, *, busy: bool = False) -> Any:
    """Small table of whole-run counters."""

    from rich.table import Table

    inp, out = graph.tokens()
    nodes = list(graph.all_nodes)
    table = Table.grid(expand=True)
    table.add_column(style="dim")
    table.add_column(justify="right")
    table.add_row(
        "status",
        "running" if busy and not graph.finished else _graph_status(graph),
    )
    table.add_row("agents", str(len(graph.agents)))
    table.add_row("nodes", str(len(nodes)))
    table.add_row("max depth", str(max((g.depth for g in graph.walk()), default=0)))
    table.add_row("runnable", ", ".join(graph.get_runnable_nodes()) or "-")
    table.add_row("tokens in", str(inp))
    table.add_row("tokens out", str(out))
    return table


def agent_table(graph: Graph) -> Any:
    """One row per agent with status, current node, and token counts."""

    from rich.table import Table

    table = Table(show_header=True, header_style="bold dim", expand=True)
    table.add_column("agent", overflow="fold")
    table.add_column("status", no_wrap=True)
    table.add_column("current", no_wrap=True)
    table.add_column("depth", justify="right")
    table.add_column("tokens", justify="right")
    for agent in graph.walk():
        cur = agent.current()
        table.add_row(
            agent.agent_id,
            _agent_status(agent),
            _kind(cur) if cur is not None else "-",
            str(agent.depth),
            str(agent.total_tokens(recursive=False)),
        )
    return table


def node_counts_table(graph: Graph) -> Any:
    """Counts by displayed node kind."""

    from rich.table import Table

    counts = Counter(_kind(node) for node in graph.all_nodes)
    table = Table(show_header=True, header_style="bold dim", expand=True)
    table.add_column("kind")
    table.add_column("count", justify="right")
    for kind, count in sorted(counts.items()):
        table.add_row(kind, str(count))
    if not counts:
        table.add_row("-", "0")
    return table


def waiting_table(graph: Graph) -> Any:
    """Supervisors currently waiting on child agents."""

    from rich.table import Table

    table = Table(show_header=True, header_style="bold dim", expand=True)
    table.add_column("agent")
    table.add_column("waiting on", overflow="fold")
    rows = 0
    for agent in graph.walk():
        cur = agent.current()
        if isinstance(cur, SupervisingOutput):
            table.add_row(agent.agent_id, ", ".join(cur.waiting_on) or "-")
            rows += 1
    if rows == 0:
        table.add_row("-", "none")
    return table


def error_table(graph: Graph) -> Any:
    """Errors grouped by kind."""

    from rich.table import Table

    errors = [node for node in graph.all_nodes if isinstance(node, ErrorOutput)]
    counts = Counter(err.error or "error" for err in errors)
    table = Table(show_header=True, header_style="bold dim", expand=True)
    table.add_column("kind")
    table.add_column("count", justify="right")
    for kind, count in counts.most_common():
        table.add_row(kind, str(count))
    if not counts:
        table.add_row("-", "0")
    return table


def latest_table(graph: Graph, *, limit: int = 8) -> Any:
    """Most recent nodes across all agents."""

    from rich.table import Table

    nodes = sorted(
        graph.all_nodes,
        key=lambda node: (
            -1 if node.global_step is None else node.global_step,
            node.agent_id,
            node.seq,
        ),
    )[-limit:]
    table = Table(show_header=True, header_style="bold dim", expand=True)
    table.add_column("step", justify="right", no_wrap=True)
    table.add_column("agent", overflow="fold")
    table.add_column("kind", no_wrap=True)
    for node in nodes:
        table.add_row(
            "-" if node.global_step is None else str(node.global_step),
            node.agent_id,
            _kind(node),
        )
    if not nodes:
        table.add_row("-", "-", "-")
    return table


def _ordered_nodes(graph: Graph) -> list[Node]:
    return sorted(
        list(graph.all_nodes),
        key=lambda node: (
            -1 if node.global_step is None else node.global_step,
            node.agent_id,
            node.seq,
        ),
    )


def _kind(node: Node | None) -> str:
    if node is None:
        return "-"
    from rflow.utils.export import _kind as display_kind

    return display_kind(node)


def _graph_status(graph: Graph) -> str:
    if graph.finished:
        return "finished"
    runnable = graph.get_runnable_nodes()
    return "ready" if runnable else "waiting"


def _agent_status(agent: Graph) -> str:
    cur = agent.current()
    if cur is None:
        return "empty"
    if isinstance(cur, DoneOutput):
        return "done"
    if isinstance(cur, ErrorOutput):
        return "error"
    if isinstance(cur, SupervisingOutput):
        return "supervising"
    if cur.terminal:
        return "terminal"
    return "ready"


def _node_style(node: Node) -> str:
    if isinstance(node, ErrorOutput):
        return "red"
    if isinstance(node, DoneOutput):
        return "green"
    if isinstance(node, SupervisingOutput):
        return "yellow"
    if isinstance(node, LLMOutput):
        return "magenta"
    if isinstance(node, (ExecAction, ExecOutput, ResumeAction)):
        return "cyan"
    if isinstance(node, UserQuery):
        return "blue"
    return "dim"


def _node_title(node: Node, kind: str) -> str:
    step = "-" if node.global_step is None else str(node.global_step)
    return f"{node.agent_id} / {kind} · step {step}"


def _node_renderable(node: Node) -> Any:
    from rich.console import Group
    from rich.syntax import Syntax
    from rich.text import Text

    if isinstance(node, ExecAction):
        code = _clip(node.code, limit=6_000) or "# empty code block"
        return Syntax(
            code,
            "python",
            theme="ansi_dark",
            word_wrap=True,
            line_numbers=False,
        )
    if isinstance(node, LLMOutput):
        body = _assistant_body(node)
        if node.code:
            return Group(
                Text(body or "Emitted a REPL block.", style=""),
                Text(
                    f"repl block: {len(node.code)} chars (shown in execute bubble)",
                    style="dim",
                ),
            )
        return Text(body or "(empty assistant reply)")
    body = _node_body(node)
    return Text(body or f"({node.type})")


def _assistant_body(node: LLMOutput) -> str:
    reply = (node.reply or "").strip()
    if not node.code:
        return _clip(reply)
    # Avoid showing the same generated code twice: the paired ExecAction gets a
    # dedicated syntax-highlighted bubble.
    if "```" in reply:
        before = reply.split("```", 1)[0].strip()
        after = reply.rsplit("```", 1)[-1].strip()
        reply = "\n\n".join(part for part in (before, after) if part)
    return _clip(reply)


def _node_body(node: Node) -> str:
    if isinstance(node, UserQuery):
        return _clip(node.content)
    if isinstance(node, ResumeAction):
        return "resumed from: " + (", ".join(node.resumed_from) or "-")
    if isinstance(node, SupervisingOutput):
        waiting = ", ".join(node.waiting_on) or "-"
        output = f"\n\n{node.output}" if node.output else ""
        return _clip(f"waiting on: {waiting}{output}")
    if isinstance(node, ErrorOutput):
        return _clip(node.content or node.output or node.error)
    if isinstance(node, DoneOutput):
        return _clip(node.result or node.output)
    if isinstance(node, CodeObservation):
        return _clip(node.output or node.content)
    if isinstance(node, ActionNode):
        return node.type
    return str(node)


def _clip(value: str, limit: int = 4000) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit].rstrip() + f"\n...[truncated {omitted} chars]"


def _clip_one_line(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


__all__ = [
    "agent_table",
    "chat_bubbles",
    "error_table",
    "latest_table",
    "node_counts_table",
    "render_tree_panel",
    "run_stats_table",
    "run_tui",
    "tui",
    "waiting_table",
]
