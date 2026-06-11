"""Skills as workspace artifacts plus a dynamic prompt section.

This example demonstrates the intended shape for user-authored skills:

1. A skill is just a user-chosen workspace artifact path, e.g.
   ``skills/numpy-linear-algebra/SKILL.md``. There is no hardcoded artifacts
   directory.
2. A prompt-builder callable section reads selected skill artifacts at prompt
   render time.
3. The agent runs with a real LLM client; pass ``--print-prompt`` to inspect
   the rendered prompt before the LLM call.

    export OPENAI_API_KEY=...
    python examples/basics/skills.py --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
from pathlib import Path

import rflow
from rflow.prompts.default import DEFAULT_BUILDER
from rflow.runtime.local import LocalRuntime

DEFAULT_WORKSPACE = (
    Path(__file__).resolve().parents[1] / "_runs" / "example-workspaces" / "skills-demo"
)


NUMPY_LINEAR_ALGEBRA_SKILL = """\
# NumPy Linear Algebra

Use this skill when the user asks for a concrete matrix/vector computation,
least-squares fit, eigenvalue calculation, or numerical verification.

## Instructions

- Use NumPy in the REPL for the actual arithmetic. Do not do matrix algebra by
  hand when code can compute it exactly enough.
- Print the key intermediate arrays or scalars so the trace is auditable.
- Use `np.linalg.solve` for square nonsingular systems and `np.linalg.lstsq`
  for overdetermined systems.
- Verify the answer by computing a residual, reconstruction, or direct
  substitution check.
- In the final answer, include the numeric result and the verification residual.

## Example

For a least-squares line `y = m*x + b` through points:

```python
import numpy as np

x = np.array([0, 1, 2, 3], dtype=float)
y = np.array([1, 2, 2, 4], dtype=float)
A = np.column_stack([x, np.ones_like(x)])

coeffs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
m, b = coeffs
pred = A @ coeffs
residual = y - pred
residual_norm = np.linalg.norm(residual)

print("A =", A)
print("coeffs =", coeffs)
print("pred =", pred)
print("residual =", residual)
print("residual_norm =", residual_norm)
```

Then report `m`, `b`, `pred`, and `residual_norm`.
"""


def install_example_skill(workspace: rflow.Workspace) -> str:
    """Write a sample SKILL.md as a normal workspace artifact."""

    path = "skills/numpy-linear-algebra/SKILL.md"
    if not workspace.artifacts.exists(path):
        workspace.artifacts.write_text(path, NUMPY_LINEAR_ALGEBRA_SKILL)
    return path


def skills_section(skill_paths: list[str]):
    """Return a callable prompt section that reads skill artifacts dynamically."""

    def render(engine, graph) -> str:
        if engine.workspace is None:
            return ""

        sections: list[str] = []
        for path in skill_paths:
            if not engine.workspace.artifacts.exists(path):
                continue
            body = engine.workspace.artifacts.read_text(path).strip()
            sections.append(f"### `{path}`\n\n{body}")
        if not sections:
            return ""
        return (
            "Use these skills when they match the user's task. Each skill is a "
            "workspace artifact loaded into the prompt for this run.\n\n"
            + "\n\n".join(sections)
        )

    return render


def build_agent(workspace: rflow.Workspace, *, model: str) -> rflow.RecursiveFlow:
    """Create an agent configured to include the installed skill."""

    skill_path = install_example_skill(workspace)
    prompt_builder = DEFAULT_BUILDER.section(
        "skills",
        skills_section([skill_path]),
        title="Skills",
        before="tools",
    )
    return rflow.RecursiveFlow(
        llm_client=rflow.OpenAIClient(model=model),
        runtime=LocalRuntime(workspace=workspace),
        workspace=workspace,
        prompt_builder=prompt_builder,
        config=rflow.FlowConfig(max_iterations=5),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help=f"Workspace directory where the sample SKILL.md artifact is written. Defaults to {DEFAULT_WORKSPACE}.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use.",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the rendered prompt before making the LLM call.",
    )
    args = parser.parse_args()

    workspace = rflow.Workspace.create(args.workspace)
    agent = build_agent(workspace, model=args.model)

    query = (
        "Fit the least-squares line y = m*x + b through points "
        "(0, 1), (1, 2), (2, 2), (3, 4), then report m, b, the "
        "predicted values, and the L2 residual norm."
    )
    graph = agent.start(query)

    print(f"Wrote skill artifact under: {workspace.root}")
    print("- skills/numpy-linear-algebra/SKILL.md")
    if args.print_prompt:
        print("\n--- rendered system prompt ---\n")
        print(graph.system_prompt)
        print("\n--- live run ---\n")

    while not graph.finished:
        graph = agent.step(graph)
    print(graph.current().result)


if __name__ == "__main__":
    main()
