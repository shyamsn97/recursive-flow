# Examples

The examples are grouped by what you are trying to learn:

- [`basics/`](basics/) — first API examples: running agents, structured output,
  batched queries, skills, and the viewer.
- [`graph/`](graph/) — offline tours of graph querying, editing, timeline,
  forking, and rendering.
- [`control/`](control/) — steering execution: delegation, branching,
  injection, replay, and controller-authored graph edits.
- [`use_cases/`](use_cases/) — concrete workloads like summarization,
  needle-in-haystack search, autoresearch, and coding-agent demos.
- [`providers/`](providers/) — model/tool provider adapters such as DSPy, MCP,
  and Tinker.
- [`sandboxes/`](sandboxes/) — runtime isolation providers such as Modal, E2B,
  and Daytona.
- [`notebooks/`](notebooks/) — notebook walkthroughs.

Generated workspaces and bulky fixtures live under [`_runs/`](_runs/) and
[`_data/`](_data/) so source examples stay easy to scan.

Most compute examples (`use_cases/summarizer.py`,
`use_cases/needle_haystack.py`, `use_cases/needle_haystack_filesystem.py`,
`basics/showcase.py`, `use_cases/coding_agent/agent.py`) take the same flags. Defaults can
vary; run `--help` for the exact values.

| Flag | Default | Meaning |
|---|---|---|
| `--model MODEL` | varies | Main LLM. Prefix decides client (`claude*` → Anthropic, else OpenAI). |
| `--fast-model MODEL` | varies | Optional cheap secondary model registered as `fast` for delegates. |
| `--docker-image IMAGE` | unset | If set, run agent code inside this Docker image. Must have `recursive-flow` installed. Leaving this unset uses `LocalRuntime`. |
| `--max-depth N` | `3` | Max delegation depth. |
| `--max-iterations N` | `15` | Max LLM calls per agent. |
| `--no-viz` | off | Disable the live terminal visualization. |

## Running under Docker

The repo ships a `Dockerfile` at its root that builds an image with `recursive-flow`
preinstalled. Build it once:

```bash
docker build -t recursive-flow:local .
```

Then just pass `--docker-image recursive-flow:local` to any example — presence of
the flag is what enables the Docker runtime:

```bash
python examples/use_cases/summarizer.py                 --docker-image recursive-flow:local
python examples/use_cases/needle_haystack.py            --docker-image recursive-flow:local
python examples/use_cases/needle_haystack_filesystem.py --docker-image recursive-flow:local
python examples/basics/showcase.py                        --docker-image recursive-flow:local
python examples/use_cases/coding_agent/agent.py --workspace ./proj --docker-image recursive-flow:local
```

The host workspace is bind-mounted at `/workspace` inside the container, so
registered workspace tools work identically in both modes.

Each compute example writes its durable run state into its workspace. Reopen or
export it with:

```bash
recursive-flow view path/to/workspace
recursive-flow render path/to/workspace -f html -o viewer.html
```

The workspace is the saved run.

## Modal, E2B, and Daytona

Remote sandbox examples live under [`examples/sandboxes/`](sandboxes/). They run
a small platformer-building task, so set `OPENAI_API_KEY` plus the provider's
sandbox credentials:

```bash
python examples/sandboxes/modal_agent.py --model gpt-5 --no-live
python examples/sandboxes/e2b_agent.py --model gpt-5
python examples/sandboxes/daytona_agent.py --model gpt-5
```

Install the matching extra first: `recursive-flow[modal]`, `recursive-flow[e2b]`,
`recursive-flow[daytona]`, or `recursive-flow[sandbox]` for all three.

For fully locked-down runs, `DockerRuntime` takes the usual Docker knobs
directly when built by hand:

```python
from rflow.runtime.docker import DockerRuntime

runtime = DockerRuntime(
    image="recursive-flow:local",
    mounts={"./data": "/workspace"},
    env={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]},
    network="none",       # air-gap the container
    cpus=1.0,
    memory="512m",
)
```

## Smoke runner

`run_examples.py` is the manifest-driven smoke runner. By default it runs the
deterministic/offline examples; use `--include-optional`, `--include-live`,
`--include-sandbox`, `--include-manual`, or `--all --list` to expand or inspect
the suite. Notebooks are documented separately and are not executed by this
runner.

See each subdirectory README for the examples in that group.
