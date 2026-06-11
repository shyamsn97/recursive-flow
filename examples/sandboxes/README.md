# Sandbox Examples

These examples run a single RecursiveFlow task whose Python code executes inside a
remote sandbox: build a simple 2D side-scrolling platformer in plain HTML,
CSS, and JavaScript. They use `OpenAIClient`, so set `OPENAI_API_KEY` before
running them.

Each example writes durable run state under `examples/_runs/example-workspaces/`.

Good first-turn behavior is to inspect in a small standalone REPL block before
delegating or writing files. For example:

```repl
info = CONTEXT.info()
print(info)
print(CONTEXT.read(0, min(2000, info["chars"])))
```

Then use that output in the next turn to decide whether to delegate,
batch LLM calls, or write files.

## Modal

```bash
pip install -e ".[openai,modal]"
export OPENAI_API_KEY=...
modal setup
python examples/sandboxes/modal_agent.py --model gpt-5
```

The Modal example builds its image from this local checkout by copying
the repo into `/opt/recursive-flow` and running `pip install -e /opt/recursive-flow`
inside the image.

Useful Modal sandbox args:

```bash
python examples/sandboxes/modal_agent.py \
  --app-name recursive-flow-dev \
  --sandbox-timeout 600 \
  --remote-workdir /workspace
```

## E2B

```bash
pip install -e ".[openai,e2b]"
export OPENAI_API_KEY=...
export E2B_API_KEY=...
python examples/sandboxes/e2b_agent.py --model gpt-5
```

By default, `E2BRuntime` starts from E2B's base template and runs
`python -m pip install -q recursive-flow` inside the sandbox. Pass
`template=...` and `setup_commands=[]` if you maintain a prebuilt
template with `recursive-flow` already installed.

Useful E2B sandbox args:

```bash
python examples/sandboxes/e2b_agent.py \
  --template recursive-flow-dev \
  --skip-setup \
  --sandbox-timeout 600 \
  --remote-workdir /workspace
```

## Daytona

```bash
pip install -e ".[openai,daytona]"
export OPENAI_API_KEY=...
export DAYTONA_API_KEY=...
python examples/sandboxes/daytona_agent.py --model gpt-5
```

By default, `DaytonaRuntime` creates a default Python sandbox and runs
`python -m pip install -q recursive-flow` inside it. Pass provider-specific
`create_params` and `setup_commands=[]` for a prebuilt snapshot.

Useful Daytona sandbox args:

```bash
python examples/sandboxes/daytona_agent.py \
  --snapshot recursive-flow-dev \
  --skip-setup \
  --create-timeout 120 \
  --remote-workdir /workspace
```
