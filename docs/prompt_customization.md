# Prompt Customization

`Flow` builds a system prompt from named sections. Most customization should
derive from the default builder instead of replacing the whole prompt, because
the default sections carry the REPL protocol, the
`launch_subagents` delegation rules,
`INPUTS`, `HISTORY`, and the worked examples that keep recursive execution
well-formed.

Use full replacement only when you want to own that entire protocol yourself.

## Inspect The Prompt

Before changing the prompt, render the one your agent already sees:

```python
graph = agent.start("Summarize this document.", context=document)
print(agent.build_system_prompt(graph))
```

You can also render without starting a run by constructing the graph shape you
want to inspect:

```python
import rflow

graph = rflow.Graph(
    query="Summarize this document.",
    agent_id="root",
    depth=0,
    config=agent.node_config(),
)
print(agent.build_system_prompt(graph))
```

Each `Graph` stores the prompt snapshot that was used for that agent's
first call:

```python
print(graph.system_prompt)
```

## Default Builder Shape

The default builder has seven sections, in order:

| Section | Purpose |
| --- | --- |
| `role` | Opening contract + REPL namespace (`INPUTS`, `HISTORY`, `llm_query_batched`, `launch_subagents`, `SHOW_VARS`, `print`, `done`). |
| `strategy` | When to use `llm_query_batched` vs `launch_subagents`, "break down problems", REPL-for-computation with an inline physics example, truncation + long-context guidance. |
| `format` | REPL block fence rules + tiny inline demo. |
| `examples` | Five worked recipes (chunked scan, batched chunks, branch on delegate, program-style fanout, parallel fanout). |
| `final` | `done(...)` contract, `SHOW_VARS` reminder, closing exhortation. |
| `tools` | Runtime-generated tool list (custom user tools registered with the engine). |
| `status` | Runtime-generated agent id, depth, and config status. |

The first five render headless and back-to-back, so the rendered prompt reads
as one continuous narrative; the split exists so each piece is independently
swappable via `DEFAULT_BUILDER.update(name, ...)`. `tools` and `status` are
callable sections filled from the current engine and graph at build time.

## Recommended: Derive From `DEFAULT_BUILDER`

The default prompt is a `PromptBuilder`: an ordered list of named sections.
`.section(...)` returns a new builder, so the module-level default is never
mutated.

### Add Project Rules

Add a new section anywhere relative to the existing ones:

```python
import rflow
from rflow.prompts.default import DEFAULT_BUILDER

project_rules = """
- Preserve API compatibility unless the task explicitly asks for a breaking change.
- Prefer small patches with focused tests.
- When changing public behavior, update docs in the same pass.
"""

prompt = DEFAULT_BUILDER.section(
    "project_rules",
    project_rules,
    title="Project Rules",
    after="final",
)

agent = rflow.Flow(llm, max_depth=2)
agent.prompt_builder = prompt
```

### Swap A Single Section

Replace just the piece you want to customize. The rest of the prompt is unchanged:

```python
from rflow.prompts.default import DEFAULT_BUILDER

domain_strategy = """
**When to delegate:** spawn one child per independent file/module. Keep the root
agent's job to planning, dispatch, and integration. Verify children mechanically
before `done()`.
"""

prompt = DEFAULT_BUILDER.update("strategy", domain_strategy)
```

### Prepend A Persona

Slip a small role section before `role` rather than overwriting the protocol:

```python
prompt = DEFAULT_BUILDER.section(
    "persona",
    "You are a recursive security auditor. Reproduce concrete risks and "
    "propose minimal fixes.",
    title="Persona",
    before="role",
)
```

### Remove A Section

You can remove sections, but only the ones you added — removing `system`
removes the entire delegation protocol.

```python
prompt = DEFAULT_BUILDER.remove("project_rules")
```

### Build A Prompt From Scratch

Use this when you want complete control while still using the section renderer.
If you want the standard runtime-generated tools and status blocks, include the
built-in callable sections.

```python
from rflow.prompts import PromptBuilder, status_section, tools_section

prompt = (
    PromptBuilder()
    .section("role", "You are a minimal REPL agent.", title="Role")
    .section(
        "protocol",
        """
- Use exactly one ```repl``` block per assistant message.
- Call `done(answer)` exactly once when finished.
- Use tools to inspect or modify files.
""",
        title="Protocol",
    )
    .section("tools", tools_section, title="Tools")
    .section("status", status_section, title="Status")
)
```

## Full System Prompt Replacement

`Flow(system_prompt=...)` bypasses the builder entirely:

```python
import rflow

agent = rflow.Flow(
    llm,
    system_prompt="""
You are a Python REPL agent.

- Use exactly one ```repl``` block per assistant message.
- Use available tools to make progress.
- Call `done(answer)` exactly once when finished.
""",
)
```

This is the most fragile option. If the prompt omits `launch_subagents`, `INPUTS`, `HISTORY`, or the
`done(...)` rule, the model will not reliably use those features.

## Dynamic Prompts

Subclass `Flow` when the prompt should depend on the current agent,
depth, query, available tools, or project state. The hook receives the
agent's `Graph` — all run-invariants are flat fields on it
(`agent_id`, `depth`, `query`, `config`, `model`, …).

```python
import rflow
from rflow.prompts.default import DEFAULT_BUILDER


class AuditFlow(rflow.Flow):
    def build_system_prompt(self, graph: rflow.Graph) -> str:
        extra = (
            "At root depth, produce an executive summary after verification."
            if graph.depth == 0
            else "As a child call, return only structured findings."
        )

        builder = DEFAULT_BUILDER.section(
            "audit_depth_rules",
            extra,
            title="Depth Rules",
            after="strategy",
        )
        return builder.build(self, graph)
```

You can also replace narrower callable sections directly:

```python
from rflow.prompts import tools_section


def careful_tools(engine, graph):
    return tools_section(engine, graph) + "\n- Prefer read-only tools before write tools."


prompt = DEFAULT_BUILDER.section("tools", careful_tools, title="Tools")
```

## Callable Sections

The dynamic prompt hook above works, but it is heavier than it needs to be for
small additions like skills, memory, or project rules. A prompt section can be
either static text or a function:

```python
def section(flow: rflow.Flow, graph: rflow.Graph) -> str:
    ...
```

The signature is intentionally just `flow, graph`. There is no context dict
and no separate prompt context object. If a section needs file-backed skills,
runtime tools, model registrations, config, or the current agent id, those are
already reachable from `flow` and `graph`.

For example, a small tool note can wrap the built-in tools section:

```python
from rflow.prompts import tools_section


def careful_tools(flow, graph):
    return tools_section(flow, graph) + "
- Prefer read-only tools before write tools."


prompt = DEFAULT_BUILDER.section("tools", careful_tools, title="Tools")
```

And a dynamic skills section is just a file reader:

```python
from pathlib import Path

skill_paths = [
    Path("skills/careful-research/SKILL.md"),
    Path("skills/coding-style/SKILL.md"),
]


def skills_section(flow, graph):
    blocks = []
    for path in skill_paths:
        if not path.exists():
            continue
        body = path.read_text(encoding="utf-8").strip()
        blocks.append(f"### `{path}`

{body}")
    return "

".join(blocks)
```

Then the builder setup stays simple:

```python
flow = rflow.Flow(llm)
flow.prompt_builder = (
    DEFAULT_BUILDER
    .section("skills", skills_section, title="Skills", before="tools")
    .section("tools", tools_section, title="Tools")
    .section("status", status_section, title="Status")
)
```

With callable sections, skills are not an engine config knob. They are ordinary
files plus a prompt section that decides what to include for the current
`flow, graph`.

See [`examples/skills.py`](../examples/skills.py) for a runnable version with a
concrete NumPy linear-algebra `SKILL.md`.

## Child-Specific Prompts

The easiest way to steer a child is the query you pass in a
`launch_subagents([...])` spec. Use the global prompt for stable behavior and
use child queries for local contracts.

```python
results = await launch_subagents([
    {
        "name": "api",
        "query": "Implement src/api.py. Return ONLY JSON {\"files\": [str], \"checks\": [str]}.",
        "inputs": {"spec": api_spec},
    },
    {
        "name": "tests",
        "query": "Implement tests for src/api.py. Return ONLY JSON {\"files\": [str], \"checks\": [str]}.",
        "inputs": {"spec": test_spec},
    },
])
```

If every child of a flow needs a different system prompt, use a subclass and
branch on `graph.depth`, `graph.agent_id`, or `graph.query`.
