# Node Model

`recursive-flow` records every agent run as a typed trajectory. The trajectory is a
strict alternation of **observations** and **actions**:

- **Observations** are inputs the system received or observed: a user query, an
  LLM reply, REPL output, a suspension, an error, or a terminal result.
- **Actions** are work the system did: call the LLM, execute code, or resume a
  suspended runtime.

Every action is followed by exactly one observation. This makes each transition
auditable: the graph says what the engine decided to do and what happened next.

## Hierarchy

```text
Node
├── ObservationNode
│   ├── UserQuery
│   ├── LLMOutput
│   └── CodeObservation
│       ├── ExecOutput
│       ├── SupervisingOutput
│       ├── ErrorOutput
│       └── DoneOutput
└── ActionNode
    ├── LLMAction
    ├── ExecAction
    └── ResumeAction
```

There are nine concrete leaf types under four base classes. Use
`isinstance(node, CodeObservation)` or `is_code_observation(node)` for "any
result from running or resuming code."

## Node Fields

All nodes share:

- `type`: stable serialized discriminator, such as `"llm_output"`;
- `id`: generated node ID;
- `agent_id`: owning agent ID, such as `"root"` or `"root.search"`;
- `seq`: per-agent sequence number.

The concrete payloads are:

| Class | `type` | Base | Key payload |
| --- | --- | --- | --- |
| `UserQuery` | `user_query` | `ObservationNode` | `content` |
| `LLMAction` | `llm_action` | `ActionNode` | `model` |
| `LLMOutput` | `llm_output` | `ObservationNode` | `reply`, `code`, `model`, `input_tokens`, `output_tokens` |
| `ExecAction` | `exec_action` | `ActionNode` | `code` |
| `ExecOutput` | `exec_output` | `CodeObservation` | `output`, `content`, `resumed_from` |
| `SupervisingOutput` | `supervising_output` | `CodeObservation` | `output`, `waiting_on`, `resumed_from` |
| `ErrorOutput` | `error_output` | `CodeObservation` | `error`, `content`, `output`, `resumed_from` |
| `DoneOutput` | `done_output` | `CodeObservation` | `result`, `content`, `output`, `resumed_from` |
| `ResumeAction` | `resume_action` | `ActionNode` | `code`, `resumed_from` |

`resumed_from` is empty for fresh code execution and populated when the
observation came from resuming a suspended parent after children completed.

`LLMOutput.code` is the source of truth for executed code. `ExecAction.code` and
`ResumeAction.code` are debug/UI echoes of what was run or resumed.

## Normal Flow

A one-turn successful run looks like this:

```text
UserQuery
  -> LLMAction
  -> LLMOutput(code="done('answer')")
  -> ExecAction
  -> DoneOutput(result="answer")
```

A multi-turn run loops through LLM and exec halves:

```text
UserQuery
  -> LLMAction
  -> LLMOutput(code="x = compute()")
  -> ExecAction
  -> ExecOutput(output="...")
  -> LLMAction
  -> LLMOutput(code="done(x)")
  -> ExecAction
  -> DoneOutput(result="...")
```

Errors are observations too. The next LLM turn sees the error message and can
recover:

```text
LLMOutput(code="1 / 0")
  -> ExecAction
  -> ErrorOutput(error="exec_exception", output="ZeroDivisionError: ...")
  -> LLMAction
  -> LLMOutput(code="done(...)")
```

If the LLM reply contains no executable code block, the engine records a normal
exec half with `ErrorOutput(error="no_code_block")`.

## Delegation And Resume Flow

Agents delegate with:

```python
results = await launch_subagents([
    {"name": "search", "query": "Find the evidence", "context": chunk},
])
```

When code awaits `launch_subagents(...)`, the parent runtime suspends and the
engine writes:

```text
ExecAction
  -> SupervisingOutput(waiting_on=["root.search"])
```

The scheduler then runs the child agent. When all children listed in
`waiting_on` are terminal, the parent becomes runnable again:

```text
SupervisingOutput(waiting_on=["root.search"])
  -> ResumeAction(resumed_from=["root.search"])
  -> ExecOutput(resumed_from=["root.search"], output="...")
```

After the resume observation, the parent returns to normal LLM/exec flow.

## Step Semantics

`step(graph)` advances each runnable agent by one observation-to-observation
transition. That means one logical reasoning turn usually takes two `step`
rounds:

1. LLM half: `ObservationNode -> LLMAction -> LLMOutput`.
2. Exec half: `LLMOutput -> ExecAction -> CodeObservation`.

Resume is also an observation-to-observation transition:

```text
SupervisingOutput -> ResumeAction -> CodeObservation
```

The pure scheduling logic decides which agents are runnable:

- finished agents do nothing;
- an agent at `LLMOutput` runs code next;
- an agent at `UserQuery`, `ExecOutput`, or `ErrorOutput` calls the LLM next;
- an agent at `SupervisingOutput` resumes only after all children in
  `waiting_on` are terminal;
- otherwise, the scheduler descends into unfinished children.

## Persistence

Node sequence numbers are assigned by the session append path. Callers populate
payload fields; the engine assigns `agent_id`, `seq`, and `id`.

Workspaces persist the per-agent trajectory in `session/<agent-id>/`, while the
recursive graph manifest links agents through `Graph.children`. Cross-agent
edges are derived from the recursive graph structure and `SupervisingOutput`
wait sets; there is no separate edge object to maintain by hand.

Use the predicate helpers from `rflow.graph` when inspecting traces:

```python
import rflow

for node in graph.all_nodes:
    if rflow.is_supervising(node):
        print(node.agent_id, "waiting on", node.waiting_on)
    elif rflow.is_code_observation(node):
        print(node.agent_id, node.type)
    elif rflow.is_done(node):
        print("result:", node.result)
```
