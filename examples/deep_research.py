"""Deep research example with real web search tools.

This example gives an RLMFlow live search/fetch tools and asks it to build a
grounded research report. The graph should show the useful RLM pattern:

1. split a broad question into research angles,
2. delegate each angle to a child,
3. verify weak/contradictory claims,
4. synthesize a cited final report.

Search provider:
- DuckDuckGo HTML search (free, no API key, best-effort scraping).

Usage:
    python examples/deep_research.py
    python examples/deep_research.py --query "Should we bet on geothermal by 2035?"
    python examples/deep_research.py --model gpt-4o --fast-model gpt-4o-mini
    python examples/deep_research.py --workspace examples/runs/deep_research/smr
"""

from __future__ import annotations

import argparse
import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from rlmflow import RLMConfig, RLMFlow, Workspace
from rlmflow.llm import AnthropicClient, OpenAIClient
from rlmflow.prompts import DEFAULT_BUILDER
from rlmflow.prompts.default import ROLE_TEXT
from rlmflow.runtime.local import LocalRuntime
from rlmflow.tools import FILE_TOOLS, tool
from rlmflow.utils.trace import save_trace


DEFAULT_QUERY = """\
Research whether small modular reactors are likely to be commercially viable
in the US by 2035.

Build a deep research report with:
- market drivers
- regulatory blockers
- technical risks
- cost comparisons
- strongest bull case
- strongest bear case
- open questions

For every major claim, include citations or source snippets. Separate evidence
from speculation, and call out contradictions between sources.
"""


DEEP_RESEARCH_RULES = """\
When doing deep research:
- Start broad, then split into independent research angles.
- Use `web_search(query)` to find sources and `fetch_url(url)` to inspect them.
- Ask child calls for structured JSON: claims, evidence, sources, contradictions,
  uncertainty, and suggested followups.
- On the resume turn, parse child outputs, identify weak claims or contradictions,
  and delegate targeted verification before `done(...)`.
- Every major final claim needs a source URL or quoted snippet.
- Do not present search snippets as certainty; fetch source pages when possible.
"""


def _request_text(url: str, *, headers: dict[str, str] | None = None) -> str:
    request_headers = {"User-Agent": "Mozilla/5.0 rlmflow-deep-research/0.1"}
    if headers:
        request_headers.update(headers)
    req = urllib.request.Request(url, headers=request_headers)
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read(2_000_000)
    return raw.decode("utf-8", errors="replace")


def _clean_html(text: str, *, max_chars: int) -> str:
    text = re.sub(r"(?is)<script.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?</style>", " ", text)
    text = re.sub(r"(?is)<(br|p|li|h[1-6]|tr|div)\b[^>]*>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()[:max_chars]


def _duckduckgo_search(query: str, max_results: int) -> list[dict[str, str]]:
    params = urllib.parse.urlencode({"q": query})
    text = _request_text(f"https://duckduckgo.com/html/?{params}")
    results: list[dict[str, str]] = []
    blocks = re.findall(r'(?is)<div class="result[^"]*">(.*?)</div>\s*</div>', text)
    for block in blocks:
        title_match = re.search(r'(?is)<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block)
        if not title_match:
            continue
        href = html.unescape(title_match.group(1))
        parsed = urllib.parse.urlparse(href)
        query_params = urllib.parse.parse_qs(parsed.query)
        url = query_params.get("uddg", [href])[0]
        title = _clean_html(title_match.group(2), max_chars=300)
        snippet_match = re.search(r'(?is)<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', block)
        snippet = _clean_html(snippet_match.group(1), max_chars=500) if snippet_match else ""
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results


@tool("Search the web. Returns JSON list of {title, url, snippet}.")
def web_search(query: str, max_results: int = 5) -> str:
    max_results = max(1, min(int(max_results), 10))
    try:
        results = _duckduckgo_search(query, max_results)
    except (urllib.error.URLError, TimeoutError) as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}", "results": []})
    return json.dumps(
        {"provider": "duckduckgo_html", "query": query, "results": results},
        indent=2,
    )


@tool("Fetch a URL and return cleaned page text with the source URL.")
def fetch_url(url: str, max_chars: int = 12000) -> str:
    max_chars = max(1000, min(int(max_chars), 30000))
    try:
        text = _request_text(url)
    except (urllib.error.URLError, TimeoutError) as exc:
        return json.dumps({"url": url, "error": f"{type(exc).__name__}: {exc}"})
    return json.dumps(
        {
            "url": url,
            "text": _clean_html(text, max_chars=max_chars),
        },
        indent=2,
    )


def build_prompt_builder():
    return (
        DEFAULT_BUILDER.section(
            "role",
            "You are a recursive deep research analyst. You search, read sources, "
            "split independent research angles into child calls, verify weak claims, "
            f"and produce grounded reports.\n\n{ROLE_TEXT}",
            title="Role",
        )
        .section(
            "deep_research",
            DEEP_RESEARCH_RULES,
            title="Deep Research Rules",
            after="strategy",
        )
    )


def make_llm(model: str):
    return AnthropicClient(model) if model.startswith("claude") else OpenAIClient(model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a recursive deep research example.")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--fast-model", default=None)
    parser.add_argument("--workspace", type=Path, default=Path("examples/runs/deep_research"))
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=18)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    workspace = Workspace.create(args.workspace)
    runtime = LocalRuntime(workspace=workspace)
    runtime.register_tools([*FILE_TOOLS, web_search, fetch_url])

    llm_clients = None
    if args.fast_model:
        llm_clients = {
            "fast": {
                "model": make_llm(args.fast_model),
                "description": "Cheaper/faster model for scoped research subtasks.",
            }
        }

    agent = RLMFlow(
        llm_client=make_llm(args.model),
        runtime=runtime,
        workspace=workspace,
        llm_clients=llm_clients,
        config=RLMConfig(max_depth=args.max_depth, max_iterations=args.max_iterations),
        prompt_builder=build_prompt_builder(),
    )

    graph = agent.start(args.query)
    graphs = [graph]
    while not graph.finished:
        graph = agent.step(graph)
        graphs.append(graph)
        print(graph.tree())

    result = graph.result()
    print("\n" + "=" * 80)
    print(result or "(no result)")

    trace_dir = args.workspace / "trace"
    save_trace(graphs, trace_dir, metadata={"query": args.query})
    print(f"\nTrace saved to {trace_dir}")

    if not args.no_viewer:
        try:
            from rlmflow.utils.viewer import save_html

            save_html(graphs, args.workspace / "viewer.html")
            print(f"Viewer saved to {args.workspace / 'viewer.html'}")
        except ImportError as exc:
            print(f"Viewer not saved: {exc}")


if __name__ == "__main__":
    main()
