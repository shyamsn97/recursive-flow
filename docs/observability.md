# Observability

Everything you need to debug a run lives in the `Graph` snapshot returned by every `start` / `step` call.

## Data model

A `Graph` is a recursive structure: it represents **one agent**, and
`graph[other_aid]` returns the `Graph` rooted at any descendant agent.
Per-agent invariants live as flat fields on `Graph` itself; sub-agents
live in `graph.children`; trajectory lives in `graph.nodes`.

```python
graph.agent_id     # str — this agent's id
graph.depth        # int — recursion depth
graph.query        # str — original task
graph.system_prompt
graph.config       # dict — engine knobs at spawn
graph.runtime      # RuntimeRef | None
graph.model        # str | None — concrete model name (if set)
graph.parent_agent_id / graph.parent_node_id

graph.nodes       # list[Node] — this agent's trajectory (seq order)
graph.children     # dict[str, Graph] — direct sub-agents

# subtree views (every agent / node / edge in the recursion)
graph.agents       # Mapping[agent_id, Graph]
graph.all_nodes        # every Node in the subtree (iterable, queryable)
graph.edges        # derived flows_to / spawns edges

graph.tree()       # ASCII per-agent timeline
```

`Node` carries only what changes per turn — the node payload
(`content`, `code`, `output`, `reply`, `result`, `error`, token
deltas) — and is tagged by its `type`. Trajectories strictly
alternate **observation** and **action** nodes; every action is
followed by exactly one observation. Nine concrete leaf classes
live under four base classes. See [`node_model.md`](node_model.md) for
the full flow and wait/resume semantics:

| `type`                | Class                | Base                   | Carries                                                           |
|-----------------------|----------------------|------------------------|-------------------------------------------------------------------|
| `user_query`          | `UserQuery`          | `ObservationNode`      | initial task content (root user query / spawn prompt for a child) |
| `llm_action`          | `LLMAction`          | `ActionNode`           | "called the LLM" — model name + call metadata                     |
| `llm_output`          | `LLMOutput`          | `ObservationNode`      | reply, extracted REPL code, token deltas                          |
| `exec_action`         | `ExecAction`         | `ActionNode`           | "ran fresh code" — optional code echo                             |
| `exec_output`         | `ExecOutput`         | `CodeObservation` (obs)| runtime stdout/stderr                                             |
| `supervising_output`  | `SupervisingOutput`  | `CodeObservation` (obs)| code suspended at an awaited launcher; `waiting_on` lists pending children |
| `error_output`        | `ErrorOutput`        | `CodeObservation` (obs)| failure observation                                               |
| `done_output`         | `DoneOutput`         | `CodeObservation` (obs)| terminal answer from `done(...)`                                  |
| `resume_action`       | `ResumeAction`       | `ActionNode`           | "supervisor resumed paused code" — produces a `CodeObservation`   |

Use `isinstance(n, CodeObservation)` for "any code result"
(`ExecOutput`, `SupervisingOutput`, `ErrorOutput`, `DoneOutput`).

## Querying the graph

```python
graph.tree()                                   # ASCII render
graph.current()                                # latest node on the root agent
graph.result()                                 # terminal answer
graph.finished                                 # root agent's current node and descendants are terminal
graph.tokens()                                 # (in, out) — recursive by default
graph.tokens(recursive=False)                  # (in, out) — just this agent

graph["root.scanner_api"]                      # sub-Graph rooted at that agent
graph.agents["root.scanner_api"]               # same, but explicit
graph.children                                 # dict[str, Graph] of spawned children
graph.parent_id                                # str | None — id of the spawning agent

graph.agents[aid].nodes                       # ordered list[Node] for one agent
graph.agents[aid].result()                     # the latest DoneOutput payload
graph.agents[aid].tokens()                     # (in, out) for that subtree

graph.all_nodes                                    # iterate every node (agent then seq)
graph.all_nodes.find("n_abc...")                   # bare Node lookup by id
graph.all_nodes.errors()                           # list[ErrorOutput]
graph.all_nodes.results()                          # list[DoneOutput]
graph.all_nodes.supervising()                      # list[SupervisingOutput]
graph.all_nodes.where(type="llm_output", agent_id="root")  # kwargs match attrs
graph.all_nodes.where(lambda n: n.type == "error_output")  # or pass a predicate

graph.edges.spawns()                           # list[Edge] — cross-agent delegation
graph.edges.flows_to()                         # list[Edge] — same-agent continuity
```

## Workspace persistence

A workspace is the durable run. It separates per-agent node logs,
the graph manifest, task payloads, and user-controlled artifact files:

```text
workspace/
  graph.json                  # workspace manifest: root + agent list
  session/
    root/
      agent.json              # per-agent invariants written once
      session.jsonl           # one Node per line, in seq order
      latest.json             # cached summary of the latest node
      transcript.json         # exact LLM conversation + per-message metadata
    root.child/
      agent.json
      session.jsonl
      latest.json
      transcript.json
  context/
    root/context.txt          # CONTEXT payload + metadata
    root.child/context.txt
  skills/
    numpy-linear-algebra/SKILL.md  # user artifact, via workspace.artifacts
  reports/
    summary.md                      # user artifact, via workspace.artifacts
```

`transcript.json` is the ground-truth record of what each agent's LLM
actually saw: `messages` is the flat `[{role, content}, ...]`
conversation across every turn, and `metadata` is a parallel list with
one dict per message (per-assistant entries carry `model`, token counts,
`elapsed_s`, and the node/seq the turn was appended after). Useful for
debugging prompt issues, replaying a turn under a different model, or
auditing context growth. Read it via the `Session` API
(`session.read_transcript(agent_id)`).

`Workspace.open_path(...).load_graph()` rehydrates the persisted node log
as the same `Graph` shape the engine emits — `flows_to` edges are
derived from node order, `spawns` edges come straight from
`graph.json`. See [`internals.md`](internals.md#persistence) for the
full session/transcript/context layout.

`workspace.artifacts` is the safe API for ordinary user-controlled workspace
files. It uses the paths you choose, rejects absolute paths and `..`, and hides
engine-owned paths like `session/`, `context/`, and `graph.json`:

```python
workspace.artifacts.write_text("skills/review/SKILL.md", skill_text)
workspace.artifacts.read_text("reports/summary.md")
workspace.artifacts.list("skills")
```

## Live terminal

```python
from rlmflow.utils.viz import live

for graph in live(agent, agent.start(query)):
    pass
```

Or just `print(graph.tree())` in a step loop.

## Gantt swimlane

One row per agent, one column per step, colored by node type. Makes
parallelism and the critical path obvious at a glance.

```python
from rlmflow.utils.viz import gantt, gantt_html

gantt(graphs)                                  # print to terminal (Rich)
Path("run.html").write_text(gantt_html(graphs, title="run 1"))
```

## Topology exports

Static renders of the graph for READMEs, issues, and post-mortems.

```python
from rlmflow.utils.export import to_mermaid, to_dot, to_d2

print(to_mermaid(graphs[-1]))                  # stateDiagram-v2 — paste into GitHub
Path("run.dot").write_text(to_dot(graphs[-1]))
Path("run.d2").write_text(to_d2(graphs[-1]))
# $ dot -Tsvg run.dot -o run.svg
# $ d2  run.d2 run.svg
```

Per-agent transcripts and ASCII tree boxes are one call:

```python
from rlmflow.utils.viz import ascii_boxes, message_stream

print(message_stream("root.boid_js", graphs[-1]))
print(ascii_boxes(graphs[-1]))
```

## Image and HTML snapshots

For blog posts, PR comments, papers, or CI artifacts — render the
graph to a PNG (or SVG/PDF), or to a single self-contained HTML
stepper.

```python
from rlmflow.utils import save_image, save_steps, save_html

save_image("runs/deep_research", "final.png")       # latest workspace snapshot
save_html("runs/deep_research", "viewer.html")      # standalone viewer
save_steps(graphs, "frames/")                       # if you kept a history list
```

Or via the graph convenience methods:

```python
graph.save_image("final.png")
graph.save_html("viewer.html")
```

Markers, edges, and fonts share the same default element scale
(`element_mult=1.0`) across image, GIF, steps, and HTML export so
static artifacts stay close to the interactive viewer. Tune `width` /
`height` / `scale` / `element_mult` to taste.

```python
save_steps(
    graphs,
    "frames/",
    width=1800,
    height=1350,
    scale=2.0,           # kaleido density multiplier (hi-dpi crispness)
    element_mult=1.0,    # marker / edge / font size multiplier
)
```

Image export needs `kaleido`:

```
pip install rlmflow[image]
```

For an animated GIF instead of separate frames, add Pillow:

```python
from rlmflow.utils import save_gif

save_gif(graphs, "trace.gif", duration=400)    # ~2.5 fps
```

```
pip install rlmflow[image] pillow
```

The HTML stepper has no static-image dependency — it embeds Plotly
from CDN and runs in any browser.

## Viewer

```python
from rlmflow.utils.viewer import open_viewer

open_viewer("runs/deep_research")              # workspace path
open_viewer(graph)                             # single snapshot
open_viewer(graphs)                            # explicit in-memory history
```

Requires `rlmflow[viewer]`.

## CLI

The same helpers are reachable from a shell. `view` and `render` take workspace
directories.

```
rlmflow view   runs/deep_research/
rlmflow view   runs/deep_research/ --port 7861
rlmflow render runs/deep_research/ -f gantt-html -o run1.html
rlmflow render runs/deep_research/ -f mermaid          # stdout
rlmflow render runs/deep_research/ -f dot -o graph.dot
rlmflow render runs/deep_research/ -f tree
rlmflow version
```
