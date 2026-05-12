"""Tests for the public save_image / save_steps / save_html / render_html APIs."""

from __future__ import annotations

import importlib.util

import pytest

from rlmflow import Graph, LLMClient, LLMUsage, RLMConfig, RLMFlow
from rlmflow.runtime.local import LocalRuntime
from rlmflow.utils import (
    render_html,
    save_gif,
    save_html,
    save_image,
    save_steps,
)
from rlmflow.utils.viewer import _scale_figure_elements

KALEIDO_INSTALLED = importlib.util.find_spec("kaleido") is not None
PIL_INSTALLED = importlib.util.find_spec("PIL") is not None
PLOTLY_INSTALLED = importlib.util.find_spec("plotly") is not None


class _DelegatingLLM(LLMClient):
    """Tiny scripted LLM that produces a 1-child run."""

    ROOT = (
        "```repl\n"
        "h = delegate('child', 'do the thing', '')\n"
        "results = yield wait(h)\n"
        "done('root:' + results[0])\n"
        "```"
    )
    CHILD = "```repl\ndone('child-answer')\n```"

    def chat(self, messages, *args, **kwargs):
        self.last_usage = LLMUsage(input_tokens=10, output_tokens=5)
        for message in messages:
            if "do the thing" in (message.get("content") or ""):
                return self.CHILD
        return self.ROOT


def _run() -> list[Graph]:
    agent = RLMFlow(
        llm_client=_DelegatingLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )
    graph = agent.start("kick off")
    graphs = [graph]
    while not graph.finished:
        graph = agent.step(graph)
        graphs.append(graph)
        assert len(graphs) < 25
    return graphs


# ── render_html / save_html ──────────────────────────────────────────


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_render_html_contains_one_slide_per_state():
    graphs = _run()
    html = render_html(graphs, title="trace test")

    assert html.startswith("<!doctype html>")
    assert "<title>trace test</title>" in html
    section_steps = [
        '<section class="slide active" data-step="1"',
        *[
            f'<section class="slide" data-step="{i}"'
            for i in range(2, len(graphs) + 1)
        ],
    ]
    for marker in section_steps:
        assert marker in html, f"missing slide marker {marker!r}"
    assert 'id="prev"' in html and 'id="next"' in html
    assert html.count("https://cdn.plot.ly/plotly") <= 1


def test_render_html_rejects_empty_states():
    with pytest.raises(ValueError, match="at least one graph"):
        render_html([])


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_save_html_writes_file(tmp_path):
    out = save_html(_run(), tmp_path / "trace.html", title="t")

    assert out == tmp_path / "trace.html"
    assert out.exists()
    assert "<title>t</title>" in out.read_text(encoding="utf-8")


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_save_html_creates_parent_dirs(tmp_path):
    out = save_html(_run(), tmp_path / "nested" / "deep" / "trace.html")
    assert out.exists()
    assert out.parent.is_dir()


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_graph_save_html_shorthand(tmp_path):
    """Graph.save_html(path) renders a single-slide stepper of just that graph."""
    final = _run()[-1]
    out = final.save_html(tmp_path / "trace.html")

    assert out.exists()
    assert out.read_text().count('<section class="slide') == 1


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_render_html_normalize_labels_default_strips_top_positions():
    graphs = _run()
    html_default = render_html(graphs)
    html_alt = render_html(graphs, normalize_labels=False)

    assert "top center" not in html_default
    assert "top center" in html_alt


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_render_html_marker_mult_shows_up_in_embedded_json():
    graphs = _run()
    html_base = render_html(graphs, marker_mult=1.0)
    html_big = render_html(graphs, marker_mult=5.0)

    assert html_big != html_base
    assert html_base.count('"size":11') >= 1


# ── element_mult scaling ──────────────────────────────────────────────


def _marker_text_sizes(fig):
    for trace in fig.data:
        mode = getattr(trace, "mode", "") or ""
        if "markers" in mode and "text" in mode and trace.marker.size is not None:
            font = getattr(trace, "textfont", None)
            return trace.marker.size, getattr(font, "size", None)
    return None, None


def test_graph_plot_element_mult_scales_markers():
    pytest.importorskip("plotly.graph_objects")
    final = _run()[-1]
    fig_normal = final.plot(element_mult=1.0)
    fig_big = final.plot(element_mult=2.0)

    n_size, _ = _marker_text_sizes(fig_normal)
    b_size, _ = _marker_text_sizes(fig_big)
    assert n_size is not None and b_size is not None
    assert all(b == n * 2 for n, b in zip(n_size, b_size))


def test_graph_plot_uses_uniform_base_marker_sizes():
    """Marker size should not encode token counts."""
    pytest.importorskip("plotly.graph_objects")
    final = _run()[-1]
    fig = final.plot(element_mult=1.0)
    sizes, _ = _marker_text_sizes(fig)
    assert sizes is not None
    assert len(set(sizes)) == 1


def test_graph_plot_split_marker_text_mult():
    pytest.importorskip("plotly.graph_objects")
    final = _run()[-1]
    fig_base = final.plot()
    fig_split = final.plot(marker_mult=4.0, text_mult=2.0)

    base_markers, base_font = _marker_text_sizes(fig_base)
    split_markers, split_font = _marker_text_sizes(fig_split)
    assert base_markers and split_markers
    assert all(s == b * 4.0 for b, s in zip(base_markers, split_markers))
    if base_font is not None:
        assert split_font == base_font * 2.0


def test_graph_plot_normalize_labels():
    pytest.importorskip("plotly.graph_objects")
    final = _run()[-1]
    fig = final.plot(normalize_labels=True)

    seen_top = False
    seen_bottom = False
    for trace in fig.data:
        mode = getattr(trace, "mode", "") or ""
        if "text" not in mode:
            continue
        positions = getattr(trace, "textposition", None)
        if positions is None:
            continue
        if isinstance(positions, str):
            seen_top |= positions.startswith("top")
            seen_bottom |= positions.startswith("bottom")
        else:
            seen_top |= any((p or "").startswith("top") for p in positions)
            seen_bottom |= any((p or "").startswith("bottom") for p in positions)

    assert not seen_top
    assert seen_bottom


def test_scale_figure_elements_noop_when_mult_one():
    pytest.importorskip("plotly.graph_objects")
    final = _run()[-1]
    fig = final.plot()

    def snapshot(f):
        out = []
        for t in f.data:
            marker = getattr(t, "marker", None)
            size = getattr(marker, "size", None) if marker is not None else None
            if isinstance(size, (list, tuple)):
                out.append(("list", tuple(size)))
            elif isinstance(size, (int, float)):
                out.append(("scalar", float(size)))
            else:
                out.append(("none", None))
        return out

    before = snapshot(fig)
    _scale_figure_elements(fig, 1.0, 1.0)
    after = snapshot(fig)
    assert before == after


# ── save_image / save_steps (kaleido-gated) ───────────────────────────


@pytest.mark.skipif(not KALEIDO_INSTALLED, reason="kaleido not installed")
def test_save_image_writes_png(tmp_path):
    final = _run()[-1]
    out = save_image(
        final,
        tmp_path / "snap.png",
        width=400,
        height=300,
        scale=1.0,
        element_mult=1.5,
    )
    assert out.exists()
    assert out.stat().st_size > 0
    with open(out, "rb") as fh:
        assert fh.read(4) == b"\x89PNG"


@pytest.mark.skipif(not KALEIDO_INSTALLED, reason="kaleido not installed")
def test_save_steps_writes_one_per_state(tmp_path):
    graphs = _run()
    out_dir = save_steps(
        graphs,
        tmp_path / "frames",
        width=400,
        height=300,
        scale=1.0,
        element_mult=1.5,
    )
    assert out_dir == tmp_path / "frames"
    files = sorted(p.name for p in out_dir.glob("step_*.png"))
    assert len(files) == len(graphs)
    assert files[0] == "step_00.png"


@pytest.mark.skipif(not KALEIDO_INSTALLED, reason="kaleido not installed")
def test_graph_save_image_method(tmp_path):
    final = _run()[-1]
    out = final.save_image(
        tmp_path / "shorthand.png",
        width=400,
        height=300,
        scale=1.0,
    )
    assert out.exists()
    assert out.suffix == ".png"


def test_save_steps_empty_returns_dir(tmp_path):
    out = save_steps([], tmp_path / "empty")
    assert out == tmp_path / "empty"
    assert out.is_dir()
    assert list(out.iterdir()) == []


@pytest.mark.skipif(
    not (KALEIDO_INSTALLED and PIL_INSTALLED),
    reason="needs both kaleido and pillow",
)
def test_save_gif_writes_gif(tmp_path):
    out = save_gif(
        _run(),
        tmp_path / "trace.gif",
        duration=200,
        width=300,
        height=240,
        scale=1.0,
        element_mult=1.5,
    )
    assert out.exists()
    with open(out, "rb") as fh:
        magic = fh.read(6)
    assert magic in (b"GIF87a", b"GIF89a")


def test_save_gif_empty_states_raises(tmp_path):
    with pytest.raises(ValueError, match="at least one graph"):
        save_gif([], tmp_path / "empty.gif")


def test_save_image_raises_helpful_error_without_kaleido(tmp_path, monkeypatch):
    pio = pytest.importorskip("plotly.io")

    final = _run()[-1]

    def _boom(*_args, **_kwargs):
        raise ValueError("Image export requires the kaleido package.")

    monkeypatch.setattr(pio, "write_image", _boom)
    with pytest.raises(ImportError, match="kaleido"):
        save_image(
            final,
            tmp_path / "fail.png",
            width=200,
            height=150,
        )
