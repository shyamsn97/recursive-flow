# Examples

Single-file scripts live at the root. Multi-file tasks and API tours live in
named folders. Generated runs go under [`_runs/`](_runs/); fixtures under
[`_data/`](_data/).

## Scripts

| Script | What it shows |
|---|---|
| [`showcase.py`](showcase.py) | End-to-end `Flow` run + live terminal viz |
| [`drop_in_llm.py`](drop_in_llm.py) | `Flow` as a drop-in `LLMClient` |
| [`llm_query_batched.py`](llm_query_batched.py) | `llm_query_batched` in the REPL |
| [`skills.py`](skills.py) | On-disk skills + dynamic prompt section |
| [`structured_output.py`](structured_output.py) | Root + child `output_schema` validation |
| [`view_demo.py`](view_demo.py) | Gradio viewer on synthetic graphs |
| [`summarizer.py`](summarizer.py) | Recursive map-reduce summarization |

```bash
python examples/showcase.py --no-viz
python examples/skills.py --model gpt-4o-mini
python examples/summarizer.py --sections 10 --no-viz
```

## Tasks

| Folder | What it shows |
|---|---|
| [`needle/`](needle/) | Needle-in-haystack (in-memory + filesystem variants) |
| [`coding/`](coding/) | Interactive file-editing agent |
| [`autoresearch/`](autoresearch/) | Karpathy-style research loop + circle-packing benchmark |

## Tours & integrations

| Folder | What it shows |
|---|---|
| [`graph/`](graph/) | Offline Graph API (query, edit, save, fork, render) |
| [`control/`](control/) | Delegation, branching, injection |
| [`sandboxes/`](sandboxes/) | Modal, E2B, Daytona remote execution |
| [`providers/`](providers/) | DSPy, MCP, Tinker adapters |
| [`notebooks/`](notebooks/) | Jupyter walkthroughs |

---

Most compute examples (`summarizer.py`, `needle/haystack.py`, `showcase.py`,
`coding/agent.py`) share the same flags. Run `--help` on any script for defaults.

| Flag | Default | Meaning |
|---|---|---|
| `--model MODEL` | varies | Main LLM. Prefix decides client (`claude*` → Anthropic, else OpenAI). |
| `--fast-model MODEL` | varies | Optional cheap secondary model registered as `fast` for delegates. |
| `--docker-image IMAGE` | unset | If set, run agent code inside this Docker image via a `DockerRuntime`. Must have `recursive-flow` installed. Leaving this unset uses the in-process `LocalRuntime`. |
| `--max-depth N` | `3` | Max delegation depth. |
| `--max-iters N` | `15` | Max LLM turns per agent. |
| `--no-viz` | off | Disable the live terminal visualization. |
| `--out-dir PATH` | `_runs/<example-name>/` | Save the final run here. Defaults use flat example names under [`_runs/`](_runs/). |

## Running under Docker

Build the image once:

```bash
docker build -t recursive-flow:local .
```

Then pass `--docker-image recursive-flow:local` to any example that supports it:

```bash
python examples/summarizer.py                 --docker-image recursive-flow:local
python examples/needle/haystack.py            --docker-image recursive-flow:local
python examples/needle/filesystem.py          --docker-image recursive-flow:local
python examples/coding/agent.py --workdir ./proj --docker-image recursive-flow:local
```

Examples that use file tools register them on the runtime
(`runtime.register_tools(FILE_TOOLS)`) and set `working_directory`, so relative
paths resolve into that directory the same way in local and Docker modes.

A finished run is saved automatically under `_runs/`; reopen it with:

```bash
python examples/summarizer.py        # saves to examples/_runs/summarizer/
recursive-flow view examples/_runs/summarizer
recursive-flow render examples/_runs/summarizer --format html -o viewer.html
```

The saved directory holds `graph.json` (and optionally `trace.json` when you
capture a step sequence with `save_trace`).

## Modal, E2B, and Daytona

Remote sandbox examples live under [`sandboxes/`](sandboxes/). They run a small
platformer-building task, so set `OPENAI_API_KEY` plus the provider's sandbox
credentials:

```bash
python examples/sandboxes/modal_agent.py --model gpt-5 --no-live
python examples/sandboxes/e2b_agent.py --model gpt-5
python examples/sandboxes/daytona_agent.py --model gpt-5
```

Install the matching extra first: `recursive-flow[modal]`, `recursive-flow[e2b]`,
`recursive-flow[daytona]`, or `recursive-flow[sandbox]` for all three.

For fully locked-down local runs, pass a `DockerRuntime`:

```python
from rflow import Flow, DockerRuntime
from rflow.clients import OpenAIClient

runtime = DockerRuntime("recursive-flow:local", working_directory="./proj")
flow = Flow(OpenAIClient(model="gpt-4o"), runtime=runtime)
```

## Smoke runner

`run_examples.py` is the manifest-driven smoke runner. By default it runs the
deterministic/offline examples; use `--include-optional`, `--include-live`,
`--include-sandbox`, `--include-manual`, or `--all --list` to expand or inspect
the suite. Notebooks are documented separately and are not executed by this
runner.

See each subdirectory README for the examples in that group.
