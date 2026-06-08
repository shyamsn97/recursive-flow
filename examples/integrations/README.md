# Integrations

Examples that connect RLMFlow to another library or inference backend.

Run commands below from the repository root after installing the matching extra.

## DSPy

Use RLMFlow as the LM behind a DSPy program.

```bash
export OPENAI_API_KEY=...
pip install -e ".[openai,dspy]"
python examples/integrations/dspy_drop_in.py
```

## MCP Weather

Starts a local FastMCP weather server backed by the real Open-Meteo API, registers
its tools with RLMFlow, delegates Seattle/Austin forecasts to child agents, and
combines the packing advice.

```bash
export OPENAI_API_KEY=...
pip install -e ".[openai,mcp]"
python examples/integrations/mcp_weather.py --no-viz
```

`mcp_weather.py` starts `mcp_weather_server.py` for you over stdio. To run the
FastMCP server directly for debugging:

```bash
pip install -e ".[mcp]"
python examples/integrations/mcp_weather_server.py
```

That command starts the server on stdio, which is what MCP clients expect. If you
want to inspect it with your own MCP client, point the client at:

```bash
python examples/integrations/mcp_weather_server.py
```

Useful flags:

```bash
python examples/integrations/mcp_weather.py --model gpt-5-mini
python examples/integrations/mcp_weather.py --workspace /tmp/mcp-weather-run --no-viz
```

The run calls Open-Meteo over the network and saves a workspace at
`examples/example-workspaces/mcp-weather` by default.

## Tinker

Run RLMFlow with Tinker inference and the live terminal graph view.

```bash
export TINKER_API_KEY=...
pip install -e ".[tinker]"
python examples/integrations/tinker_agent.py
```

Optional model flags:

```bash
python examples/integrations/tinker_agent.py --base-model Qwen/Qwen3-8B
python examples/integrations/tinker_agent.py --model-path tinker://run/weights/checkpoint
```

## Sandbox Providers

Remote runtime examples live under [`sandbox/`](sandbox/). They run the same
small platformer-building task against Modal, E2B, or Daytona.

```bash
export OPENAI_API_KEY=...
pip install -e ".[sandbox]"
python examples/integrations/sandbox/modal_agent.py --model gpt-5 --no-live
python examples/integrations/sandbox/e2b_agent.py --model gpt-5
python examples/integrations/sandbox/daytona_agent.py --model gpt-5
```
