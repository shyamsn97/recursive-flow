# Examples

The examples are grouped by what they demonstrate:

- [`core-api/`](core-api/) — small examples for the core API surface.
- [`control/`](control/) — delegation, branching, forking, and graph edits.
- [`applications/`](applications/) — concrete workloads like summarization and
  needle-in-haystack search.
- [`integrations/`](integrations/) — DSPy and alternate inference backends.
- [`advanced/`](advanced/) — multi-step replay and graph surgery flows.
- [`graph-features/`](graph-features/) — direct graph querying, mutation, and export.
- [`sandbox/`](sandbox/) — Modal, E2B, and Daytona runtime examples.

Most compute examples (`applications/summarizer.py`,
`applications/needle_haystack.py`, `applications/needle_haystack_filesystem.py`,
`core-api/showcase.py`, `coding-agent/agent.py`) take the same flags. Defaults can
vary; run `--help` for the exact values.

| Flag | Default | Meaning |
|---|---|---|
| `--model MODEL` | varies | Main LLM. Prefix decides client (`claude*` → Anthropic, else OpenAI). |
| `--fast-model MODEL` | varies | Optional cheap secondary model registered as `fast` for delegates. |
| `--docker-image IMAGE` | unset | If set, run agent code inside this Docker image. Must have `rlmflow` installed. Leaving this unset uses `LocalRuntime`. |
| `--max-depth N` | `3` | Max delegation depth. |
| `--max-iterations N` | `15` | Max LLM calls per agent. |
| `--no-viz` | off | Disable the live terminal visualization. |

## Running under Docker

The repo ships a `Dockerfile` at its root that builds an image with `rlmflow`
preinstalled. Build it once:

```bash
docker build -t rlmflow:local .
```

Then just pass `--docker-image rlmflow:local` to any example — presence of
the flag is what enables the Docker runtime:

```bash
python examples/applications/summarizer.py                 --docker-image rlmflow:local
python examples/applications/needle_haystack.py            --docker-image rlmflow:local
python examples/applications/needle_haystack_filesystem.py --docker-image rlmflow:local
python examples/core-api/showcase.py                        --docker-image rlmflow:local
python examples/coding-agent/agent.py --workspace ./proj --docker-image rlmflow:local
```

The host workspace is bind-mounted at `/workspace` inside the container, so
registered workspace tools work identically in both modes.

Each compute example writes its durable run state into its workspace. Reopen or
export it with:

```bash
rlmflow view path/to/workspace
rlmflow render path/to/workspace -f html -o viewer.html
```

The workspace is the saved run.

## Modal, E2B, and Daytona

Remote sandbox examples live under [`examples/sandbox/`](sandbox/). They run
a small platformer-building task, so set `OPENAI_API_KEY` plus the provider's
sandbox credentials:

```bash
python examples/sandbox/modal_agent.py --model gpt-5 --no-live
python examples/sandbox/e2b_agent.py --model gpt-5
python examples/sandbox/daytona_agent.py --model gpt-5
```

Install the matching extra first: `rlmflow[modal]`, `rlmflow[e2b]`,
`rlmflow[daytona]`, or `rlmflow[sandbox]` for all three.

For fully locked-down runs, `DockerRuntime` takes the usual Docker knobs
directly when built by hand:

```python
from rlmflow.runtime.docker import DockerRuntime

runtime = DockerRuntime(
    image="rlmflow:local",
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
`--include-notebooks`, `--include-sandbox`, or `--all --list` to expand or inspect
the suite.

See each subdirectory README for the examples in that group.
