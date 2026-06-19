# Prompt Reliability Action Plan

Internal plan for replacing the bloated default prompt with a small,
high-signal recursive coding agent prompt.

## Decision

Stop growing the prompt. The current direction should be closer to
`avilum/minrlm`: compact rules, direct tool descriptions, and a few hard
behavioral constraints instead of long strategy prose and many examples.

This is documentation/planning only. Do not make runtime prompt or `Flow`
implementation changes from this plan until explicitly approved.

The target is not to port minrlm literally. It uses a different API
(`input_0`, `sub_llm`, `FINAL`). We should copy the shape:

```text
identity -> code-block contract -> preloaded globals -> mandatory probe ->
tool semantics -> small approach rules -> final-answer contract
```

For `rflow`, that maps to:

- `INPUTS` for string inputs.
- The original query must also be present in `INPUTS`, as `INPUTS["query"]`.
- `llm_query_batched(...)` for simple one-shot fanout.
- `await launch_subagents([...])` for recursive subtasks.
- `done(answer)` for final answer.
- Extra registered tools in a filtered `## Tools` section.

## Target Prompt Shape

Replace the strategy-heavy prompt with a small base prompt like:

```text
You are a recursive coding agent. You only act through Python REPL blocks.
Write exactly one ```repl block per assistant turn. No prose outside the block.

Preloaded globals:
- INPUTS: dict[str, str]. Includes the original query under INPUTS["query"] plus
  caller-provided inputs. `query` is the only special key; every other key is
  user-defined, so inspect `list(INPUTS)` instead of assuming fixed names.
- llm_query_batched(prompts, ...): one-shot LLM fanout. Use for simple
  extraction, summarization, classification, and Q&A.
- await launch_subagents(specs): recursive child agents. Use for independent
  units that need tools, files, iteration, or repair. Always pass a list of
  dict specs; each child sees only its query and inputs.
- done(answer): final answer only. Never call it with a status, warning,
  partial result, or failed check.

Mandatory behavior:
1. First block probes the query/input state from INPUTS. Print list(INPUTS),
   len(INPUTS["query"]), tiny bounded samples such as
   `INPUTS["query"][:500]`, input sizes, and any obvious independent units. Do
   not print full large input contents.
2. If work splits into independently checkable units, launch subagents for
   those units and keep the current agent for sequencing, integration, and
   verification.
3. Do not print large payloads. Store data in variables and pass slices or
   focused inputs to subcalls.
4. Verify before done(...). A failed check is work to repair, not a final answer.
5. Use exactly one ```repl block each turn.
```

Keep dynamic sections:

- `## Structured Output` only when the current graph has an output schema and
  structured output is enabled.
- `## Tools` only for extra registered tools. Do not repeat prompt-documented
  builtins (`done`, `launch_subagents`, `llm_query_batched`) there.
- `## Status` for recursion depth / model availability if needed.

## Examples Policy

Default prompt should have at most one or two tiny examples. Prefer no generic
examples over a bloated bank.

Good default examples:

```repl
print(list(INPUTS))
print("query chars", len(INPUTS["query"]))
for name, value in INPUTS.items():
    if name != "query":
        print(name, len(value))
```

```repl
shared_inputs = {name: value for name, value in INPUTS.items() if name != "query"}
units = [
    {"name": "part-a", "query": "Own one independent part of INPUTS['query'].", "inputs": shared_inputs},
    {"name": "part-b", "query": "Own another independent part of INPUTS['query'].", "inputs": shared_inputs},
]
results = await launch_subagents(units)
print(results)
```

Task-specific few-shot examples can be injected by users or examples, but should
not live in the global default prompt. Those examples may choose task-specific
input names, but no name besides `query` should be treated as framework-level.

## Character Budget

The raw static prompt must stay under `MAX_STATIC_PROMPT_CHARS = 10_000`.

This should be enforced by tests, not runtime truncation:

```python
raw = DEFAULT_BUILDER.build()
assert SYSTEM_PROMPT == raw
assert len(raw) < MAX_STATIC_PROMPT_CHARS
```

No runtime clamp. If the prompt grows too large, the test should fail loudly.

## Task Injection

Do not keep the user query only as chat text / graph metadata. That departs from
the RLM shape: upstream passes the prompt payload into the environment as
context, so the model can inspect the query from code.

This is the key missing invariant. A recursive agent should not have to recover
its own assignment from chat history; the assignment is part of the working
state it can inspect, slice, pass to children, and verify against.

`rflow` should do the same:

```python
repl.namespace["INPUTS"] = {"query": agent.query, **agent.inputs}
```

Rules:

- `Graph.query` remains metadata and chat history.
- `INPUTS["query"]` is always present and is the REPL-readable query string.
- User-provided inputs are still under their own keys. They are freeform and
  chosen by the caller or agent for the specific task.
- The prompt must not define a fixed schema beyond `query`. Treat every other
  input name as an ordinary user-defined key.
- If a user already passes `inputs={"query": ...}`, either reject it as reserved
  or store the original query under a reserved key such as `__query__`. Prefer
  rejecting `query` as reserved so the prompt can stay simple.
- Children should receive their query as `INPUTS["query"]` too, plus whatever
  explicit `inputs` the parent passes.

This matters because prompt examples and first-turn probing should not rely on
`HISTORY.read(0)` to recover the query. The REPL-facing invariant should be:

```text
Every agent can inspect its own query with INPUTS["query"].
```

## Query Metadata

The first user turn should include metadata about REPL-visible inputs, not dump
the full query and inputs into chat.

Use this shape:

```text
Your REPL INPUTS contain:
- query: str, N chars
- <user key>: str, N chars
Total input chars: M.
Print only small bounded samples, e.g. `print(INPUTS["query"][:500])`; never
print a full large input.
```

If the caller or agent chooses structured keys, list the actual keys explicitly:

```text
Your REPL INPUTS contain:
- query: str, N chars
- <chosen task key>: str, N chars
- <another chosen key>: str, N chars
Total input chars: M.
Print only small bounded samples, e.g. `print(INPUTS["query"][:500])`; never
print a full large input.
```

This mirrors upstream RLM's metadata idea (`context_type`,
`context_total_length`, context count), adapted to `INPUTS`.

## History Policy

Do not inject `HISTORY` by default and do not describe it in the small base
prompt. Once `INPUTS["query"]` is always present, history is no longer needed to
recover the task.

Keep the `HISTORY` code path available behind an opt-in flag or custom prompt
for debugging/long-horizon recovery, but it should not be part of the default
REPL namespace or default prompt.

## Current Failure: Coding Notebook

The boids notebook is still the clearest regression case. The user query is
short, explicit, and naturally decomposes into independently checkable work:

```text
Create a runnable browser-based boids simulation...
- Write separate files:
    - index.html
    - style.css
    - boids.js
- Verify that all files exist, script tags are ordered correctly, and the
  JavaScript has no obvious syntax/runtime wiring errors before returning.
```

Observed bad trajectory:

- The graph contains only `root`; there are no child agents.
- The first turn probes correctly, but the second turn writes `index_html`,
  `style_css`, and `boids_js` inline in the current REPL.
- The assistant emitted multiple fenced `repl` blocks in one response in at
  least one run, causing a syntax error before doing useful work.
- After the syntax error, it retried by writing all files inline again instead
  of delegating.
- Verification was weak: file existence and line counts / small snippets, not a
  real integration check. It did not check script order robustly, no-module
  constraints, static JS syntax, canvas wiring, or likely runtime issues.

Why this matters:

- This is not a large-context problem. It is a small task whose shape should be
  enough to trigger recursive orchestration.
- The prompt is still letting the model choose the locally easy path: one big
  root REPL block that solves everything inline.
- Adding more abstract strategy prose has not reliably fixed the behavior.

Prompt target for this case:

- Keep rules compact, but include one concrete good-behavior example where the
  parent:
  1. probes `INPUTS`;
  2. identifies independent tool-using units;
  3. launches children with `await launch_subagents([...])`;
  4. passes only non-`query` supporting inputs in child `inputs`;
  5. integrates child outputs;
  6. runs deterministic verification;
  7. delegates repair for failing units;
  8. calls `done(...)` only after verification passes.
- The example must be one valid `repl` block. Do not show multiple fenced blocks
  in a single assistant response anywhere in the default prompt.
- The example must not hardcode boids, browser apps, or filenames. It should
  show generic units/artifacts and generic verification/repair flow.

Validation target:

- A fresh coding-notebook run should create at least one child agent for the
  independent units.
- For multi-artifact coding tasks, line counts and existence checks alone are
  not enough verification. The trajectory should show meaningful checks against
  the task contract, and repair should be attempted before `done(...)` if checks
  fail.
- If no delegation happens, the trace must explicitly explain why the work is
  tiny/tightly coupled enough to do inline. For the boids notebook, that
  explanation should not pass.

## Implementation Plan

1. Collapse `STRATEGY_TEXT`, `FORMAT_TEXT`, and the large `EXAMPLES` bank into
   the compact prompt above.
2. Remove examples that hardcode `doc`, `corpus`, browser apps, boids, or file
   names from the global default prompt.
3. Keep `tools_section(...)` filtering builtins out of `## Tools`.
4. Keep structured-output text dynamic and outside the static prompt unless
   needed.
5. Inject every agent's query into its REPL `INPUTS`.
6. Add `query` to reserved input names or otherwise prevent user input collision.
7. Add query/input metadata to the first user turn.
8. Stop injecting `HISTORY` by default; keep the implementation available for
   opt-in use.
9. Keep first-turn user text short:

```text
You have not interacted with the REPL environment or seen your inputs yet.
Inspect them first; do not provide a final answer yet.

<input metadata manifest>

Answer the original prompt: ...
```

10. Run only prompt-focused verification:

```text
python - <<'PY'
from rflow.prompts.default import DEFAULT_BUILDER, SYSTEM_PROMPT, MAX_STATIC_PROMPT_CHARS
raw = DEFAULT_BUILDER.build()
print(len(raw), MAX_STATIC_PROMPT_CHARS, raw == SYSTEM_PROMPT)
PY
pytest tests/test_tools_prompts.py -q
```

## Validation Criteria

Pass:

- Static prompt under `MAX_STATIC_PROMPT_CHARS`.
- No runtime truncation helper.
- `## Tools` does not list `done`, `launch_subagents`, or
  `llm_query_batched`.
- Root and child REPLs expose their own query at `INPUTS["query"]`.
- First user turn includes the input metadata manifest.
- `HISTORY` is not present in the default REPL namespace/prompt.
- First turn probes before final answer.
- Coding notebook trace shows either subagents for independent file units or a
  clear explanation in the trajectory why direct execution is better.

Fail:

- Prompt grows via more global examples.
- Prompt repeats core builtins in `## Tools`.
- Prompt hardcodes boids/browser-file names into base behavior.
- Agent must use `HISTORY.read(0)` to discover its own query.
- `HISTORY` remains injected by default.
- Root/current agent writes every independently checkable artifact inline
  without planning subcalls.
