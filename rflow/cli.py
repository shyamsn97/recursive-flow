"""``rlmflow`` command-line entry point.

Three sub-commands, all operating on paths — no agent construction:

    rlmflow view     <path>              open the Gradio viewer
    rlmflow render   <path> --format F   write a static render
    rlmflow version                      print package + environment info

``<path>`` may be a ``trace.json`` (``{"steps": [...]}``), a standalone
``Graph`` JSON snapshot, a JSON list of snapshots, or a directory containing
``trace.json`` / ``graph.json``.

``--format`` accepts text formats (``mermaid`` / ``mermaid-flowchart`` /
``mermaid-sequence`` / ``dot`` / ``d2`` / ``tree`` / ``report-md`` /
``gantt-html`` / ``code-log`` / ``error-summary``) and figure formats
(``html`` stepper, ``image`` PNG/SVG, ``steps`` one image per snapshot under
``--out``). Figure formats need the ``viewer``/``image`` extras.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rflow.graph import Graph


def _load(path: Path) -> list[Graph]:
    """Return graph snapshots for a trace, graph dump, or directory."""
    from rflow.utils.viewer import resolve_graphs

    try:
        return resolve_graphs(path)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"rlmflow: {exc}") from None


# ── commands ─────────────────────────────────────────────────────────


def cmd_view(args: argparse.Namespace) -> int:
    from rflow.utils.viewer import open_viewer

    graphs = _load(Path(args.path))
    launch_kwargs: dict = {}
    if args.share:
        launch_kwargs["share"] = True
    if args.port is not None:
        launch_kwargs["server_port"] = args.port
    if args.host is not None:
        launch_kwargs["server_name"] = args.host
    open_viewer(graphs, **launch_kwargs)
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from rflow.utils.export import (
        to_d2,
        to_dot,
        to_mermaid,
        to_mermaid_flowchart,
        to_mermaid_sequence,
    )
    from rflow.utils.viewer import graph_tree
    from rflow.utils.viz import code_log, error_summary, gantt_html, report_md

    graphs = _load(Path(args.path))
    topo = graphs[-1]
    fmt = args.format

    if fmt in ("html", "image", "steps"):
        return _render_figure(args, graphs, topo, fmt)

    if fmt == "mermaid":
        out = to_mermaid(topo)
    elif fmt == "mermaid-flowchart":
        out = to_mermaid_flowchart(topo)
    elif fmt == "mermaid-sequence":
        out = to_mermaid_sequence(topo)
    elif fmt == "dot":
        out = to_dot(topo)
    elif fmt == "d2":
        out = to_d2(topo)
    elif fmt == "tree":
        out = graph_tree(topo)
    elif fmt == "gantt-html":
        out = gantt_html(graphs)
    elif fmt == "report-md":
        out = report_md(graphs)
    elif fmt == "code-log":
        out = code_log(topo)
    elif fmt == "error-summary":
        out = error_summary(topo)
    else:
        raise SystemExit(f"rlmflow: unknown format {fmt!r}")

    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(out)
        if not out.endswith("\n"):
            sys.stdout.write("\n")
    return 0


def _render_figure(
    args: argparse.Namespace, graphs: list[Graph], topo: Graph, fmt: str
) -> int:
    from rflow.utils.viewer import save_html, save_image, save_steps

    if fmt == "html":
        if not args.out:
            raise SystemExit("rlmflow: --format html requires --out PATH")
        path = save_html(graphs, args.out, title=args.title or "rlmflow run")
        print(f"wrote {path}", file=sys.stderr)
        return 0

    if fmt == "image":
        if not args.out:
            raise SystemExit(
                "rlmflow: --format image requires --out PATH (e.g. graph.png)"
            )
        path = save_image(
            topo, args.out, width=args.width, height=args.height, scale=args.scale
        )
        print(f"wrote {path}", file=sys.stderr)
        return 0

    if fmt == "steps":
        if not args.out:
            raise SystemExit("rlmflow: --format steps requires --out DIR")
        path = save_steps(
            graphs,
            args.out,
            fmt=args.image_format,
            width=args.width,
            height=args.height,
            scale=args.scale,
        )
        print(f"wrote images under {path}", file=sys.stderr)
        return 0

    raise SystemExit(f"rlmflow: unknown figure format {fmt!r}")


def cmd_version(_args: argparse.Namespace) -> int:
    import platform

    try:
        from importlib.metadata import version as _pkg_version

        pkg = _pkg_version("rlmflow")
    except Exception:
        pkg = "unknown"

    def _status(mod: str) -> str:
        import importlib.util

        return "available" if importlib.util.find_spec(mod) else "not installed"

    print(f"rlmflow  {pkg}")
    print(f"python  {platform.python_version()} ({sys.platform})")
    print(f"rich    {_status('rich')}")
    print(f"plotly  {_status('plotly')}")
    print(f"gradio  {_status('gradio')}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rlmflow",
        description="rlmflow command-line tools",
    )
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    v = sub.add_parser("view", help="open the Gradio viewer on a trace/graph")
    v.add_argument("path", help="trace.json, graph dump, or directory")
    v.add_argument("--share", action="store_true", help="create a public URL")
    v.add_argument("--port", type=int, default=None, help="server port")
    v.add_argument("--host", default=None, help="server host / bind address")
    v.set_defaults(func=cmd_view)

    r = sub.add_parser("render", help="render a trace/graph in one of several formats")
    r.add_argument("path", help="trace.json, graph dump, or directory")
    r.add_argument(
        "--format",
        "-f",
        required=True,
        choices=[
            "mermaid",
            "mermaid-flowchart",
            "mermaid-sequence",
            "dot",
            "d2",
            "tree",
            "gantt-html",
            "report-md",
            "code-log",
            "error-summary",
            "html",
            "image",
            "steps",
        ],
        help="output format",
    )
    r.add_argument(
        "--out",
        "-o",
        default=None,
        help="write to file (default: stdout). Required for html/image/steps.",
    )
    r.add_argument("--title", default=None, help="title for --format html")
    r.add_argument("--width", type=int, default=1800, help="image width in pixels")
    r.add_argument("--height", type=int, default=1350, help="image height in pixels")
    r.add_argument(
        "--scale", type=float, default=2.0, help="kaleido density multiplier"
    )
    r.add_argument(
        "--image-format",
        default="png",
        help="image suffix for --format steps (default: png)",
    )
    r.set_defaults(func=cmd_render)

    ver = sub.add_parser("version", help="print package and environment info")
    ver.set_defaults(func=cmd_version)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
