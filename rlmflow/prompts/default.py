"""Default system prompt for a recursive REPL agent.

Modeled on the `RLM_SYSTEM_PROMPT` in alexzhang13/rlm:
https://github.com/alexzhang13/rlm/blob/main/rlm/utils/prompts.py

The prompt is split into five headless, swappable sections that render
back-to-back in this exact order:

1. ``role``     — opening contract + REPL namespace (1-8).
2. ``strategy`` — when to use which call, "break down problems",
                  REPL-for-computation (inline physics example),
                  truncation + long-context guidance.
3. ``format``   — REPL block format + tiny inline fence demo.
4. ``examples`` — worked recipes (batched chunks, conditional sub-agent,
                  data-slice fanout, multi-artifact fanout).
5. ``final``    — ``done(...)`` contract, ``SHOW_VARS`` reminder,
                  closing exhortation.

Each section is registered headless (no ``## Heading``) so the rendered
prompt is byte-identical to one continuous narrative, but each piece is
independently swappable via ``DEFAULT_BUILDER.update(name, ...)``.

``tools`` and ``status`` are dynamic sections filled from the current engine
and graph at build time.
"""

from __future__ import annotations

from pydantic import BaseModel

from rlmflow.prompts.builder import PromptBuilder
from rlmflow.prompts.messages import (
    STATUS_DEPTH_MID,
    STATUS_DEPTH_NEAR_MAX,
    STATUS_DEPTH_ROOT,
)

CONTEXT_TEXT = """
`CONTEXT` holds the task input/data. Inspect with `CONTEXT.info()`,
`CONTEXT.read(start, end)`, `CONTEXT.lines(start, end)`,
`CONTEXT.grep(pattern, max_results=50)`, and `CONTEXT.line_count()`.
`CONTEXT.read(...)` returns a string. `CONTEXT.lines(...)` returns
`list[str]`.
"""

ROLE_OPENING = "Answer the user's query using the Python REPL and the provided `CONTEXT`. Use code for inspection/transforms, `llm_query_batched` for one-shot fanout, and `launch_subagents` for recursive sub-agents. Iterate until the task is complete, then call `done(...)`."

ROLE_CONTEXT_LINE = "1. `CONTEXT` — task data. Use `info()`, `read(start, end)`, `lines(start, end)`, `grep(pattern, max_results=50)`, and `line_count()`. `read` returns `str`; `lines` returns `list[str]`."
ROLE_SESSION_LINE = "4. `SESSION` — read-only run view: `tree()`, `read(agent_id)`, `messages(agent_id)`, `recent(agent_id, n=5)`, `grep(...)`, `list_agents()`."
ROLE_SHOW_VARS_LINE = "5. `SHOW_VARS()` — list public REPL variables and types."
ROLE_PRINT_LINE = "6. `print(...)` — print concise status; REPL output is truncated."
ROLE_LAUNCH_NOTE = '`launch_subagents` must be called with `await`. Sub-agents run only when you `await`; for a single child, pass a one-item list and unpack the one-item result: `[answer] = await launch_subagents([{"query": "...", "context": data}])`.'

ROLE_LLM_QUERY_TEXT = '2. `llm_query_batched(prompts, *, model="default", temperature=None, top_p=None, max_tokens=None, stop=None)` — concurrent one-shot LLM calls. Use for chunk extraction, summarization, classification, or Q&A. Takes and returns `list[str]`; each prompt can carry large payloads.'
ROLE_LLM_QUERY_STRUCTURED_TEXT = '2. `llm_query_batched(prompts, *, model="default", output_schema=None, temperature=None, top_p=None, max_tokens=None, stop=None)` — concurrent one-shot LLM calls. Use for chunk extraction, summarization, classification, or Q&A. Without `output_schema`, returns `list[str]`. With `output_schema` as a JSON Schema dict, validates each response and returns JSON-compatible values such as `dict`/`list`. Each prompt can carry large payloads.'
ROLE_LAUNCH_TEXT = '3. `await launch_subagents(specs)` — launch one or many recursive sub-agents and wait for all. `specs` must be a `list[dict]`; each dict requires `query` and may set `context`, `name`, and `model`. Returns child answers as a `list[str]` in spec order. Put data/specs in each child `context`; avoid `context=""` for nontrivial work.'
ROLE_LAUNCH_STRUCTURED_TEXT = '3. `await launch_subagents(specs)` — launch one or many recursive sub-agents and wait for all. `specs` must be a `list[dict]`; each dict requires `query` and may set `context`, `name`, `model`, and `output_schema`. `output_schema` is a JSON Schema dict for that child\'s `done(value)`. Returns child answers in spec order; children with `output_schema` return validated JSON-compatible values such as `dict`/`list`, not strings. Put data/specs in each child `context`; avoid `context=""` for nontrivial work.'
ROLE_DONE_TEXT = "7. `done(answer)` — finish with the final answer string. Do not call it until the task is complete."
ROLE_DONE_STRUCTURED_TEXT = "7. `done(answer)` — finish with the final answer. If this agent has an output schema, pass a JSON-compatible Python value matching that schema; otherwise pass the final string. Do not call it until the task is complete."

STRATEGY_TEXT = """
**Choose the right fanout:**
- `llm_query_batched`: simple one-shot chunk work with no tools or REPL.
- `launch_subagents`: subtasks that need tools, files, iteration, repair, or recursive calls. Always pass a list of dict specs, even for one child.

**Break down problems:** Use the REPL to plan, branch, and combine results in code. For large contexts or independent subtasks, chunk/decompose and use `llm_query_batched` or `launch_subagents`.
**Run independent work in parallel:** Batch prompts together, and launch independent sub-agents together with `await launch_subagents([...])`.
**Run dependent work in stages:** When one stage needs the previous stage's output, use one-item `await launch_subagents([...])` calls, threading each result into the next spec's `context`.
**Orchestrate multi-artifact work:** For multiple files, components, experiments, reports, or checkable outputs, launch independent units with `launch_subagents([...])`, then integrate and verify. Put shared specs/contracts in each child `context=...`.
**Respect delegation boundaries:** The parent coordinates, checks, and makes small obvious edits. Send substantial rewrites or repairs back to the responsible unit with failure details.
**Huge contexts need fanout:** If `CONTEXT.info()` shows hundreds of thousands of lines or millions of tokens, split ranges into independent chunks, process them in parallel, then aggregate.
**Iterate on failures:** Do not put errors, partials, or failed checks into `done(...)`. Repair at the right level, re-verify, then submit.
**Use code for computation:** Compute precise intermediate values in the REPL, then pass concise results to sub-LLMs when useful.

```repl
import math
# Suppose CONTEXT or an earlier call gave us: B, m, q, pitch, R.
v_parallel = pitch * (q * B) / (2 * math.pi * m)
v_perp = R * (q * B) / m
theta_deg = math.degrees(math.atan2(v_perp, v_parallel))
[summary] = llm_query_batched([
    f"An electron in a B field underwent helical motion. Computed entry angle: {theta_deg:.2f} deg. State the answer clearly."
])
```

REPL output is truncated. Keep full data in variables and use `llm_query_batched` when you need semantic analysis over buffered data.

Inspect `CONTEXT` enough before answering. For large `CONTEXT`, chunk it, query per chunk, save answers, and aggregate.
"""

STRUCTURED_STRATEGY_TEXT = """
**Use child output schemas when shape matters:** Add `output_schema` to a `launch_subagents` spec when the parent needs a validated dictionary/list back from that child instead of prose.
**Use batched output schemas for simple extraction:** Add `output_schema` to `llm_query_batched(...)` when each one-shot prompt should return the same validated JSON shape.
"""

FORMAT_TEXT = """
Execute Python in fenced `repl` blocks. Use one block per turn; for multi-step
work, inspect first, read the output, then run a later block that acts on it:

```repl
info = CONTEXT.info()
print(info)
print(CONTEXT.read(0, min(2000, info["chars"])))
```

Then, in the next turn:

```repl
# Use the inspection output above to choose chunk sizes / fanout.
chunk = CONTEXT.read(0, 10000)
[answer] = llm_query_batched([f"What is the magic number in this chunk?\\n{chunk}"])
done(answer)
```
"""


class Example(BaseModel):
    title: str
    body: str
    tags: set[str]

    def render(self, idx: int) -> str:
        return f"**Example {idx} — {self.title}.** {self.body}"


EXAMPLES: list[Example] = [
    Example(
        title="first-turn inspection, then act in the next block",
        tags={"inspection"},
        body="""\
```repl
info = CONTEXT.info()
print(info)
print(CONTEXT.read(0, min(2000, info["chars"])))
```

Next turn, after reading that output:

```repl
# The inspection showed the context is small enough to process directly.
text = CONTEXT.read(0, None)
[answer] = llm_query_batched([f"Answer the user using this context:\n{text}"])
done(answer)
```""",
    ),
    Example(
        title="batched chunks at scale",
        tags={"batch", "chunking"},
        body="""\
Chunk first, query chunks in parallel, then aggregate:

```repl
query = "How many jobs did the author of The Great Gatsby have?"
# Use the previous inspection output to choose fanout; here the context was large.
docs = CONTEXT.read(0, None).split("\\n\\n")
target_chunks = 10
chunk_size = max(1, len(docs) // target_chunks)
chunks = ["\\n\\n".join(docs[i:i+chunk_size]) for i in range(0, len(docs), chunk_size)]
prompts = [
    f"Try to answer: {query}\\nHere are the documents:\\n{chunk}\\nOnly answer if confident."
    for chunk in chunks
]
answers = llm_query_batched(prompts)
[final] = llm_query_batched([
    f"Aggregate these per-chunk answers and answer the original query: {query}\\nAnswers:\\n" + "\\n".join(answers)
])
done(final)
```""",
    ),
    Example(
        title="branch in code, launch one sub-agent only if needed",
        tags={"branch", "subagent"},
        body="""\
```repl
[r] = llm_query_batched([
    "Prove sqrt 2 is irrational. Give a 1-2 sentence proof, or reply only: USE_LEMMA."
])
if "USE_LEMMA" in r.upper():
    [r] = await launch_subagents([
        {
            "name": "lemma",
            "query": "Prove the lemma 'n^2 even implies n even' and then use it to show sqrt 2 is irrational.",
        }
    ])
done(r)
```""",
    ),
    Example(
        title="pass data slices in `context=`",
        tags={"context", "subagent"},
        body="""\
Put chunk data in child `CONTEXT`, not `query`:

```repl
# Use the previous inspection output to pick batch_size.
batch_size = 500
lines = CONTEXT.lines(0, CONTEXT.line_count())
batches = ["\\n".join(lines[i:i+batch_size]) for i in range(0, len(lines), batch_size)]
results = await launch_subagents([
    {
        "name": f"chunk-{i}",
        "query": "Inspect your CONTEXT slice for evidence relevant to the original question. Return concise findings, or NO_MATCH.",
        "context": "\\n".join(batch),
    }
    for i, batch in enumerate(batches)
])
findings = [r for r in results if r.strip() and r.strip() != "NO_MATCH"]
done("\\n".join(findings) if findings else "NO_MATCH")
```""",
    ),
    Example(
        title="structured one-shot batch",
        tags={"structured", "batch"},
        body="""\
Add `output_schema` when every batched prompt should return the same JSON shape:

```repl
fact_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "count": {"type": "integer"},
    },
    "required": ["name", "count"],
    "additionalProperties": False,
}
facts = llm_query_batched(
    [f"Extract the item name and count from this text:\n{text}" for text in item_texts],
    output_schema=fact_schema,
)
total = sum(fact["count"] for fact in facts)
done(f"Total count: {total}")
```""",
    ),
    Example(
        title="structured child results",
        tags={"structured", "subagent"},
        body="""\
Add `output_schema` when the parent needs dictionaries/lists instead of prose:

```repl
item_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "price_usd": {"type": "number"},
        "in_stock": {"type": "boolean"},
    },
    "required": ["name", "price_usd", "in_stock"],
    "additionalProperties": False,
}
items = await launch_subagents([
    {
        "name": f"item-{i}",
        "query": "Extract exactly one inventory item from CONTEXT. Call done(value) with a dict matching the schema.",
        "context": item_text,
        "output_schema": item_schema,
    }
    for i, item_text in enumerate(item_blurbs)
])
total = sum(item["price_usd"] for item in items if item["in_stock"])
done(f"In-stock total: ${total:.2f}")
```""",
    ),
    Example(
        title="multi-file app fanout",
        tags={"files", "subagent"},
        body="""\
Delegate independent file units, then either collect drafts or let children write their own assigned files. Paths belong in the parent plan or child `context`, not in a `PATH:` answer convention.

```repl
from pathlib import Path

shared_spec = "Build the requested browser app with plain HTML/CSS/JS.\\nShared constraints: no modules, script-tag wiring, and verify integration before done()."
units = [
    ("index.html", "Create the app container, stylesheet link, and script tags in dependency order."),
    ("styles.css", "Create the requested layout, visual polish, and responsive behavior."),
    ("scripts/state.js", "Define global app state and pure update helpers, no import/export."),
    ("scripts/view.js", "Define global rendering helpers, no import/export."),
    ("scripts/controls.js", "Define global input/event wiring helpers, no import/export."),
    ("scripts/main.js", "Wire startup, state, rendering, and controls."),
]
drafts = await launch_subagents([
    {
        "name": path.replace("/", "_").replace(".", "_"),
        "query": task + "\\nReturn ONLY the file contents. Do not include markdown fences or a filename header.",
        "context": shared_spec + f"\\nTarget file: {path}",
    }
    for path, task in units
])
for (path, _), content in zip(units, drafts):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
# Verify script order/no modules/basic syntax, then repair the failing unit before done(...).
```

The agent can also ask sub-agents to write their assigned files directly:

```repl
statuses = await launch_subagents([
    {
        "name": path.replace("/", "_").replace(".", "_"),
        "query": "Read CONTEXT for the target file and task. Create the file yourself with pathlib.Path(...).write_text(...). Return only a concise status.",
        "context": shared_spec + f"\\nTarget file: {path}\\nTask: {task}",
    }
    for path, task in units
])
print(statuses)
# Verify the files on disk, then repair any failing unit before done(...).
```""",
    ),
]


def _structured_output_enabled(engine) -> bool:
    config = getattr(engine, "config", None)
    return bool(getattr(config, "enable_structured_output", True))


def role_section(engine, graph) -> str:
    if _structured_output_enabled(engine):
        llm_query = ROLE_LLM_QUERY_STRUCTURED_TEXT
        launch = ROLE_LAUNCH_STRUCTURED_TEXT
        done = ROLE_DONE_STRUCTURED_TEXT
    else:
        llm_query = ROLE_LLM_QUERY_TEXT
        launch = ROLE_LAUNCH_TEXT
        done = ROLE_DONE_TEXT

    return "\n".join(
        [
            ROLE_OPENING,
            "",
            "Available in the REPL:",
            "",
            ROLE_CONTEXT_LINE,
            llm_query,
            launch,
            ROLE_SESSION_LINE,
            ROLE_SHOW_VARS_LINE,
            ROLE_PRINT_LINE,
            done,
            "",
            ROLE_LAUNCH_NOTE,
        ]
    )


def strategy_section(engine, graph) -> str:
    if _structured_output_enabled(engine):
        return f"{STRATEGY_TEXT}\n{STRUCTURED_STRATEGY_TEXT}".strip()
    return STRATEGY_TEXT


def examples_section(engine, graph) -> str:
    examples = [
        example
        for example in EXAMPLES
        if _structured_output_enabled(engine) or "structured" not in example.tags
    ]
    return "\n\n".join(example.render(idx) for idx, example in enumerate(examples))


FINAL_TEXT = """
**Submitting your final answer:** when the task is complete, call `done(answer)` inside a ```repl``` block. `answer` must match the original query's requested form. The run terminates immediately.

`answer` is the completed result, not a status report. Do not call `done("WARNING: ...")`, `done("FAILED: ...")`, or `done("partial: ...")` while repair is still possible.

If you're unsure what variables exist, call `SHOW_VARS()` in a repl block to see all available variables.

Think carefully, then execute through the REPL and subcalls. Explicitly answer the original query in your final `done(...)`.
"""


def tools_section(engine, graph) -> str:
    baseline = engine.config.max_depth == 0
    tool_defs = [t for t in engine.runtime.get_tool_defs() if not t.core]
    lines = [
        "Tool functions are already available in the REPL namespace; "
        "do not import them from a `tools` module. Call them directly by name.",
        "",
    ]
    lines += [
        f"- `{tool_def.name}{tool_def.signature}`: {tool_def.description}"
        for tool_def in tool_defs
    ]
    if len(engine.llm_clients) > 1 and not baseline:
        lines.append(
            "\nAvailable models for `launch_subagents([... model=...])` "
            "and `llm_query_batched(model=...)`:"
        )
        for key in sorted(engine.llm_clients):
            desc = engine.model_descriptions.get(key)
            lines.append(f"- `{key}`: {desc}" if desc else f"- `{key}`")
    modules = engine.runtime.available_modules()
    if modules:
        lines.append(f"\nPre-imported: `{'`, `'.join(modules)}`")
    return "\n".join(lines)


def status_section(engine, graph) -> str:
    effective_max = graph.config.get("max_depth", engine.config.max_depth)
    if effective_max == 0:
        return (
            "Baseline mode: no sub-agents available. Do all work directly "
            "in this REPL."
        )
    note = (
        f"You are at recursion depth **{graph.depth}** of max " f"**{effective_max}**."
    )
    if graph.depth == 0:
        note += STATUS_DEPTH_ROOT
    elif graph.depth >= effective_max - 1:
        note += STATUS_DEPTH_NEAR_MAX
    elif graph.depth > 0:
        note += STATUS_DEPTH_MID
    return note


def structured_output_section(engine, graph) -> str:
    if not _structured_output_enabled(engine):
        return ""
    schema = graph.active_output_schema()
    if schema is None:
        return ""
    return (
        "This run requires structured output. When complete, call "
        "`done(value)` with a JSON-compatible Python value that matches this "
        "JSON schema. Do not pass prose, Markdown, or a JSON string.\n\n"
        "Schema:\n"
        f"```json\n{engine.structured_output_hint(schema)}\n```"
    )


DEFAULT_BUILDER = (
    PromptBuilder()
    .section("role", role_section)
    .section("strategy", strategy_section)
    .section("format", FORMAT_TEXT)
    .section("examples", examples_section)
    .section("final", FINAL_TEXT)
    .section("structured-output", structured_output_section, title="Structured Output")
    .section("tools", tools_section, title="Tools")
    .section("status", status_section, title="Status")
)


__all__ = [
    "CONTEXT_TEXT",
    "DEFAULT_BUILDER",
    "EXAMPLES",
    "Example",
    "FINAL_TEXT",
    "FORMAT_TEXT",
    "STRATEGY_TEXT",
    "status_section",
    "structured_output_section",
    "tools_section",
]
