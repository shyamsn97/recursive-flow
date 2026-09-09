# Customizable Skills

Skills are ordinary repo files that become part of an agent's prompt when they matter. Use them for stable guidance you want to reuse across runs: project style guides, domain playbooks, child-agent contracts, benchmark heuristics, or lessons distilled from previous traces.

rlmflow keeps skills as files. Subclass `PromptBuilder` and concatenate the files that belong in the current agent's context. See [`examples/skills.py`](https://github.com/shyamsn97/rlmflow/blob/main/examples/skills.py) for a small runnable version.

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

Each `SKILL.md` should be short, concrete, and action-oriented. Prefer rules the agent can follow during a run over long background explanations.

## Always-On Project Skills

Load project conventions into every agent:

```python
from pathlib import Path

import rlmflow
from rlmflow import PromptBuilder
from rlmflow.llm import OpenAIClient


class ProjectSkillPrompt(PromptBuilder):
    def __call__(self, flow=None, node=None) -> str:
        skill = Path("skills/project-style/SKILL.md").read_text(encoding="utf-8")
        return super().__call__(flow, node) + "\n\n" + skill


flow = rlmflow.Flow(OpenAIClient(model="gpt-4o-mini"), system_prompt=ProjectSkillPrompt())
```

## Query-Selected Skills

Choose domain skills from the current task:

```python
from pathlib import Path

import rlmflow
from rlmflow import PromptBuilder
from rlmflow.llm import OpenAIClient

SKILL_DIR = Path("skills")


def _read_skill(name: str) -> str:
    path = SKILL_DIR / name / "SKILL.md"
    if not path.exists():
        return ""
    body = path.read_text(encoding="utf-8").strip()
    return f"### {name}\n{body}"


class WorkspaceSkillsPrompt(PromptBuilder):
    def __call__(self, flow=None, node=None) -> str:
        agent = None if node is None else node.parent_agent
        query = (agent.content if agent is not None else "").lower()
        skills = [_read_skill("project-style")]
        if "numpy" in query or "linear algebra" in query:
            skills.append(_read_skill("numpy-linear-algebra"))
        if agent is not None and agent.config.depth > 0:
            skills.append(_read_skill("child-agent-contract"))
        extra = "\n\n".join(skill for skill in skills if skill)
        text = super().__call__(flow, node)
        return f"{text}\n\n{extra}" if extra else text


flow = rlmflow.Flow(
    OpenAIClient(model="gpt-4o-mini"),
    system_prompt=WorkspaceSkillsPrompt(),
)
```

## Child-Only Skills

Give spawned agents a tighter contract than the root planner:

```python
from pathlib import Path

import rlmflow
from rlmflow import PromptBuilder
from rlmflow.llm import OpenAIClient


class ChildContractPrompt(PromptBuilder):
    def __call__(self, flow=None, node=None) -> str:
        agent = None if node is None else node.parent_agent
        text = super().__call__(flow, node)
        if agent is None or agent.config.depth == 0:
            return text
        contract = Path("skills/child-agent-contract/SKILL.md").read_text(
            encoding="utf-8"
        )
        return f"{text}\n\n{contract}"


flow = rlmflow.Flow(
    OpenAIClient(model="gpt-4o-mini"),
    system_prompt=ChildContractPrompt(),
)
```

## Run-Memory Skills

Turn lessons from previous runs into reusable guidance:

```python
from pathlib import Path

import rlmflow
from rlmflow import PromptBuilder
from rlmflow.llm import OpenAIClient

MEMORY_DIR = Path("skills/run-memory")


class RunMemoryPrompt(PromptBuilder):
    def __call__(self, flow=None, node=None) -> str:
        blocks = []
        for path in sorted(MEMORY_DIR.glob("*.md")):
            text = path.read_text(encoding="utf-8").strip()
            if text:
                blocks.append(f"### {path.stem}\n{text}")
        extra = "\n\n".join(blocks)
        text = super().__call__(flow, node)
        return f"{text}\n\n{extra}" if extra else text


flow = rlmflow.Flow(
    OpenAIClient(model="gpt-4o-mini"),
    system_prompt=RunMemoryPrompt(),
)
```

## Combining Skills With Other Prompt Changes

Skills are just more text after `super().__call__`, so they compose with other extras in the same subclass:

```python
class CombinedPrompt(PromptBuilder):
    def __call__(self, flow=None, node=None) -> str:
        parts = [super().__call__(flow, node), workspace_skills(flow, node), run_memory()]
        return "\n\n".join(part for part in parts if part)
```

For lower-level prompt mechanics, see [`prompt_customization.md`](prompt_customization.md).
