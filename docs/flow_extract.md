# `Flow` is the facade

Implemented. `flow.py` is the constructor, the stream loop, and wrappers. Restore, builtin factories, child launch, and prompt assembly are `(flow, ...)` functions.

`flow.py` was 900 lines because the REPL builtins, restore, child launch, prompt assembly, and tool registry still lived as methods. The step table already left: producers are `(flow, node)` functions in `engine/steps.py`. Do the same for the rest. `Flow` keeps the public names. The bodies move.

Do not invent a second object. No `MessageBuilder`, no mixin pile, no services bag.

## What is still in the file

Approximate, by block:

| Block                | Lines | What it is                                                       |
| -------------------- | ----- | ---------------------------------------------------------------- |
| constructor / wiring | ~80   | llm, runtime, pool, flags, `BuiltIns()`, `WrappedRuntime`        |
| prompts              | ~160  | `profile`, `build_messages`, `render_tools`, `transition_footer` |
| llm pool             | ~60   | `llm_for_step`, `call_stream`, `call_chat`                       |
| step / `_drive`      | ~50   | already the thin driver                                          |
| tool registry        | ~70   | `add_tool`, `inject`, `_bind_toolset`, `build_tools`             |
| builtin closures     | ~100  | `finish_tool`, `transition_tool`, `launch_tool`, wait / observe  |
| child launch         | ~110  | `resolve_child`, `new_child`, `submit_child`                     |
| restore              | ~55   | `replay`, `ensure_replayed`                                      |
| stream loop          | ~120  | `run_streaming`, `arun`, `run`, `aclose`                         |

`BuiltIns` already owns the prompt text (`inject=False` stubs). The real closures are still hand-written on `Flow`. `llm_query` already lives as `llm_query(flow)` in `tools/llm_query.py`. Finish and launch should look like that.

`view/replay.py` is animation. Restore of REPL namespaces is a different word.

## Target

```text
rlmflow/flow.py                 constructor, public wrappers, llm pool, stream loop
rlmflow/engine/steps.py         producers (done)
rlmflow/engine/restore.py       replay(flow, root), ensure_replayed(flow, agent)
rlmflow/engine/delegation.py    resolve_child, new_child, submit_child, launch_tool
rlmflow/tools/namespace.py      as_tool_items, bind, build_namespace, finish / transition / wait / observe
rlmflow/prompts/messages.py     build_messages(flow, node), transition_footer(flow, node)
rlmflow/tools/builtins.py       prompt source only (already)
```

Functions take `flow`. They read `flow.queue`, `flow.runtime`, `flow.transitions`. They do not grow a parallel type.

```python
# tools/namespace.py
def finish_tool(flow, node):
    schema = node.parent_agent.config.output_schema

    @tool("Submit this agent's final answer and end its run.", proxy=True)
    def finish(answer: object) -> None:
        value = (
            parse_structured_answer(answer, schema)
            if schema is not None
            else as_finish_value(answer)
        )
        raise DoneSignal(value)

    return finish


def build_namespace(flow, node) -> dict[str, Any]:
    running = (
        tuple(current for current, _task in flow.queue.running.values())
        if flow.queue is not None
        else ()
    )
    agents = build_agent_directory(node.parent_agent, running_nodes=running)
    namespace = {
        **flow.tools,
        "finish": finish_tool(flow, node),
        "transition": transition_tool(flow, node),
        "launch_subagent": launch_tool(flow, node),
        "asyncio": asyncio,
        "INPUTS": node.parent_agent.config.inputs,
        AGENTS_BINDING: agents,
        AGENT_OBSERVE_TOOL: observe_agent_tool(flow, node),
        AGENT_WAIT_TOOL: wait_agent_tool(flow, node),
    }
    if flow.use_agent_tree:
        namespace["AGENTS"] = agents
    return namespace
```

```python
# engine/restore.py
async def replay(flow, root: AgentStart) -> None:
    ...


async def ensure_replayed(flow, agent: AgentStart) -> None:
    ...
```

`Flow` keeps the names callers already use:

```python
class Flow:
    def build_tools(self, node):
        return build_namespace(self, node)

    def build_messages(self, node):
        return build_messages(self, node)

    async def replay(self, root):
        await replay(self, root)
```

`RAOFlow` and tests that override `_drive` / `llm_for_step` / `build_tools` keep working. They override the facade.

## What stays on `Flow`

- Constructor and flags.
- `transitions` and `__init_subclass__`.
- `step` / `_drive`. The queue submits `_drive`.
- `llm_for_step`, `call_stream`, `call_chat`. The pooled client is the flow's.
- `run_streaming` / `arun` / `run` / `start` / `aclose`. The stream loop is the product.

`run_streaming` can move later as `engine/driver.py:stream(flow, ...)`. Not first. It is one function and it is the public driver.

## What we do not do

- **Do not resurrect `MessageBuilder` / `FlowMessages`.** `build_messages(flow, node)` is a function in `prompts/messages.py`.
- **Do not put implementations on `BuiltIns`.** That class is the prompt. `inject=False` stays. The factories live in `tools/namespace.py`.
- **Do not name restore `replay` on disk.** `rlmflow.view.replay` already means frames. The module is `engine/restore.py`. The method on `Flow` can stay `replay` — that is the public word for "rebuild namespaces from recorded code."
- **Do not extract `__init__`.** Construction stays on the class.
- **Do not add a `ToolBag` or `RestoreState` object** unless a function would otherwise close over four private fields. `_restored_agents` and `_restore_lock` can move with `ensure_replayed` as a small helper, or stay as `Flow` attributes the function reads. Prefer attributes on `Flow` for now.
- **Do not move `budget_exceeded` / `timed` until `_drive` wants them.** They are ten lines next to the driver.

## Order

1. **Restore.** `replay` / `ensure_replayed` have one caller (`run_streaming`) and no decorator table. `engine/restore.py`, wrappers on `Flow`.
1. **Builtin factories + `build_namespace`.** `finish_tool` through `observe_agent_tool`, `as_tool_items`, `add_tool` / `_bind_toolset` if they come along for free. `launch_tool` stays a factory here; `resolve_child` / `new_child` / `submit_child` can move in the same pass or the next.
1. **Delegation.** `resolve_child`, `new_child`, `submit_child` next to `launch_tool` in `engine/delegation.py`. They mutate the tree and the queue. They are not prompt code.
1. **Prompt assembly.** `build_messages`, `transition_footer`, `render_tools` into `prompts/messages.py`. `profile` and `build_system_prompt` can stay one-liners on `Flow` or move with them. No new type.
1. Stop. `run_streaming` only if the file is still fat after 1–4.

Each step is a move. Public method names do not change.

Extract-by-kind (producers vs guards vs graph accessors) is a different cut, and it is the wrong one for a default row. At the time of this extract, named transitions still hopped through staged request state; the historical follow-up is [`colocate.md`](colocate.md). The current path resolves `.choices(...)` directly in `run_repl`.

## Who imports whom

```text
flow.py
  -> engine.steps          (bottom import; registers @transitions.on)
  -> engine.restore
  -> engine.delegation
  -> tools.namespace
  -> prompts.messages

engine.steps / restore / delegation / tools.namespace
  -> Flow only as a type / argument
```

Same cycle as steps: `flow.py` finishes, then the modules that decorate or that `Flow` wraps. `tools.namespace` must not import `engine.steps`. `BuiltIns` stays import-light.

## Acceptance

- `flow.py` is the constructor, the stream loop, and one-line wrappers. Restore, builtin factories, and child launch are not defined in it.
- `await flow.replay(root)` and `flow.build_tools(node)` still work.
- `BuiltIns.description` is unchanged. The stubs still raise if someone calls them.
- `from rlmflow.engine.restore import replay` is the function. `from rlmflow.view.replay import ...` is still frames.
- No new public type. Callers keep importing `Flow`.
