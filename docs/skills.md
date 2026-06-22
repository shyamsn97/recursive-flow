# Customizable Skills

Skills are ordinary repo files that become part of an agent's prompt when they
matter. Use them for stable guidance you want to reuse across runs: project
style guides, domain playbooks, child-agent contracts, benchmark heuristics, or
lessons distilled from previous traces.

rlmflow keeps skills as files. Add a callable prompt section that decides which
files belong in the current agent's context. See
[`examples/skills.py`](../examples/skills.py) for a small runnable version.

## Suggested Layout

```text
skills/
+-- project-style/
|   `-- SKILL.md
+-- numpy-linear-algebra/
|   `-- SKILL.md
+-- child-agent-contract/
|   `-- SKILL.md
`-- run-memory/
    +-- debugging.md
    `-- eval-lessons.md
```

Each `SKILL.md` should be short, concrete, and action-oriented. Prefer rules the
agent can follow during a run over long background explanations.

## Always-On Project Skills

Load project conventions into every agent:

```python
from pathlib import Path

import rflow
from rflow.prompts import DEFAULT_BUILDER


def project_skill(flow: rflow.Flow, graph: rflow.Graph) -> str:
    return Path("skills/project-style/SKILL.md").read_text(encoding="utf-8")


flow = rflow.Flow(rflow.OpenAIClient(model="gpt-4o-mini"))
flow.prompt_builder = DEFAULT_BUILDER.section(
    "project_skill",
    project_skill,
    title="Project Skill",
    before="tools",
)
```

## Query-Selected Skills

Choose domain skills from the current task:

```python
from pathlib import Path

import rflow
from rflow.prompts import DEFAULT_BUILDER

SKILL_DIR = Path("skills")


def _read_skill(name: str) -> str:
    path = SKILL_DIR / name / "SKILL.md"
    if not path.exists():
        return ""
    body = path.read_text(encoding="utf-8").strip()
    return f"### {name}\n{body}"


def workspace_skills(flow: rflow.Flow, graph: rflow.Graph) -> str:
    query = graph.query.lower()
    skills = [_read_skill("project-style")]

    if "numpy" in query or "linear algebra" in query:
        skills.append(_read_skill("numpy-linear-algebra"))
    if graph.depth > 0:
        skills.append(_read_skill("child-agent-contract"))

    return "\n\n".join(skill for skill in skills if skill)


flow = rflow.Flow(rflow.OpenAIClient(model="gpt-4o-mini"))
flow.prompt_builder = DEFAULT_BUILDER.section(
    "workspace_skills",
    workspace_skills,
    title="Workspace Skills",
    before="tools",
)
```

## Child-Only Skills

Give spawned agents a tighter contract than the root planner:

```python
from pathlib import Path

import rflow
from rflow.prompts import DEFAULT_BUILDER


def child_contract(flow: rflow.Flow, graph: rflow.Graph) -> str:
    if graph.depth == 0:
        return ""
    return Path("skills/child-agent-contract/SKILL.md").read_text(encoding="utf-8")


flow = rflow.Flow(rflow.OpenAIClient(model="gpt-4o-mini"))
flow.prompt_builder = DEFAULT_BUILDER.section(
    "child_contract",
    child_contract,
    title="Child Agent Contract",
    after="strategy",
)
```

## Run-Memory Skills

Turn lessons from previous runs into reusable guidance:

```python
from pathlib import Path

import rflow
from rflow.prompts import DEFAULT_BUILDER

MEMORY_DIR = Path("skills/run-memory")


def run_memory(flow: rflow.Flow, graph: rflow.Graph) -> str:
    blocks = []
    for path in sorted(MEMORY_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            blocks.append(f"### {path.stem}\n{text}")
    return "\n\n".join(blocks)


flow = rflow.Flow(rflow.OpenAIClient(model="gpt-4o-mini"))
flow.prompt_builder = DEFAULT_BUILDER.section(
    "run_memory",
    run_memory,
    title="Run Memory",
    before="examples",
)
```

## Combining Skills With Other Prompt Changes

Skills are prompt sections, so they compose with the rest of the prompt builder:

```python
flow.prompt_builder = (
    DEFAULT_BUILDER
    .section(
        "workspace_skills",
        workspace_skills,
        title="Workspace Skills",
        before="tools",
    )
    .section("run_memory", run_memory, title="Run Memory", before="examples")
)
```

For lower-level prompt mechanics, see
[`prompt_customization.md`](prompt_customization.md).
