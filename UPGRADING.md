# Upgrading to 0.5

Version 0.5 is a deliberate breaking beta release. It does not include compatibility aliases for the pre-0.5 graph, completion, persistence, or TUI APIs.

## Agent completion

Agent code must call `finish(value)`. The old `done(value)` name is not present in the REPL.

Without `output_schema`, `finish(value)` keeps JSON-compatible Python values (dicts, lists, numbers, bools, `None`, and strings). Values that cannot round-trip through JSON are still `str(value)`. With an explicit `output_schema`, `Flow.run()`, `agent.result()`, and `wait_for_result()` return the validated typed value.

## Structured output guidance

`Flow(enable_structured_output=True)` is now the default. This flag controls whether spawn-capable agents see guidance for optional child `output_schema` values; it does not require every child or root to use a schema. Pass `enable_structured_output=False` to hide that guidance.

When a schema is supplied, pass a JSON Schema object with explicit `properties`, `required`, and `additionalProperties`. Without a schema, child and root results retain JSON-compatible Python values as described above.

## Node.append

`node.append(child)` mutates the tree and returns `None`. Capture the child before appending:

```python
created = LLMOutput(content=reply, code=code_block(reply))
node.append(created)
return created
```

Do not write `return node.append(...)` or `next = node.append(...)`. `append_child` and `AppendChild.attach` still return the attached subtree.

## Graph traversal and identity

- Use `node.walk()` for iterative parent-before-child traversal.
- Use `node.iter_backwards()` for same-agent history.
- Use `node.to_record()` for one shallow record.
- Use `root.usage` and `root.stats` for indexed whole-run aggregates.
- Use `node.subtree_usage()` when an explicit subtree scan is intended.
- `node.fork()` always creates fresh graph and agent identities.

The removed forms are `walk(reverse=True)`, `to_dict()`, `tokens()`, and `fork(new_ids=False)`.

## Agent limits and truncation

Library runs are unbounded by default: `AgentConfig.max_iters` and `max_budget` are `None`. The CLI still supplies its own finite iteration default. Set limits explicitly for unattended or user-facing runs.

`max_output_length` and `max_query_chars` now default to 20,000 characters. `max_output_length` bounds one REPL observation; `max_query_chars` bounds a delegated child's goal. Set either field on `AgentConfig`, through `flow.start(...)`, or in `Flow(root_config=...)`.

## Persistence

Persistence v3 stores one flat, parent-linked graph document. Use `persistence.to_document()` and `persistence.from_document()`. Loading rejects older versions and unknown node types instead of guessing. Register custom Node types with `persistence.register_node_type`.

There is no in-process v2 migration path. Export or regenerate old development runs with the version that created them before upgrading.

## Checkpointing and TUI

`GraphCheckpointer` uses `interval_s` and `interval_nodes`; call `flush()` when an immediate checkpoint is required.

Initialize `FlowTUI` with `ui.init(flow)` and then call `ui.run()`. The old `ui.run(drive)` callback entry point is removed.

## Agent and model helpers

- Call `AgentInfo.result()`; `get_result()` is removed.
- Import `client_for` from `rlmflow.llm`; `examples.common.build_client` is removed.

## Step table

`await flow.step(node)` returns the created `Node`. Register automatic producers with `@Flow.transitions.on(...)`. Declare model-selectable routes with `.choices(current, *targets)`; `transition("name")` then lands the selected `UserQuery` directly from the current `ExecAction`. A blank table is `Flow(llm, transitions=Transitions())`. Extra automatic behavior is `@Review.transitions.on(...)`.

Removed: `StepFunction`, `LLMRequestStep`, `LLMOutputStep`, `ExecActionStep`, `update_step_fn`, `DEFAULT_STEPS`, `DEFAULT_TRANSITIONS`, `Flow.always`, `Flow.when`, `MessageBuilder`, and `FlowMessages`. The queue still uses an internal `Transition`; do not take `.created` off `flow.step`.

## Planning control query

`PlanQuery` now precedes every ordinary model action. Its user turn is `Turn N/M:` (plus a first-turn INPUTS safeguard). `InspectQuery`, `INSPECTION_ACTION`, and the automatic `InspectQuery -> PlanQuery` transition are removed. Custom step registrations must use `PlanQuery`; persisted development runs containing `inspect_query` nodes must be regenerated before loading.

## System prompts

`SystemPromptBuilder`, `.sections`, `DEFAULT_BUILDER`, `Section`, and `Sections` are removed. Subclass `PromptBuilder` and concatenate after `super().__call__(flow, node)`, or pass a string / `(flow, node) -> str` to `Flow(system_prompt=...)`.

`Flow.build_system_prompt(node)` is the inherited protocol. `UserQuery.build_system_prompt(flow)` is what a turn sends; `PlanQuery` wraps with the orchestrator addendum when the agent can spawn, and `FinalQuery` wraps with last-turn submit guidance. `Node.build_system_prompt` is gone — actions and outputs have no prompt hook. Inspect a prompt with `flow.build_system_prompt(root)` or `flow.build_messages(node)`, not `flow.system_prompt.render(...)`.

## Tools and toolsets

Use `@tool` for one callable and `@toolset("Title")` for a class that owns several related tools and their prompt documentation. Pass either form as an item in `Flow(tools=[...])`; toolset methods are flattened into the REPL namespace, so agent code calls the method name directly rather than through the class.

A toolset may define `description(flow, node)` and `example(flow, node)` to render context-sensitive prompt text. Reserved builtins use `inject=False` signatures because `Flow` installs their real per-node closures separately.

Normal tools are serialized into local workers. Use `@tool(..., proxy=True)` when a tool must execute against host-owned mutable state or another host-only resource.

## REPL preimports

Every runtime now preimports the modules in `rlmflow.runtime.DEFAULT_PREIMPORTS`: `json`, `re`, `os`, `sys`, `pathlib`, `csv`, `math`, `asyncio`, `collections`, `datetime`, `itertools`, `statistics`, `textwrap`, and `time`.

Pass `preimports=` to `LocalRuntime`, `SubprocessRuntime`, `DockerRuntime`, or another runtime to replace that list. `preimports=[]` binds no modules; names may include installed third-party packages.

## Runtime trust boundary

Worker-to-host tool arguments, `ENV`, and `Runtime.get_var()` results must be JSON-compatible data. Arbitrary Python objects may still be copied from the trusted host into a worker, but executable worker-controlled payloads are never deserialized on the host.

Publish the plain fields the host needs through `ENV` instead of retrieving a custom worker object.

## Execution approval

Pass `execution_guard=` to `Flow`. The guard receives each `ExecAction` before runtime execution and returns `None` to allow it or an error message to reject it.
