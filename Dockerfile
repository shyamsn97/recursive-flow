# Sandbox image for recursive-flow's DockerRuntime.
#
# Build:
#   docker build -t recursive-flow:local .
#
# Use:
#   from rflow.runtime.docker import DockerRuntime
#   runtime = DockerRuntime("recursive-flow:local")
#
# Or via any of the bundled examples:
#   python examples/use_cases/summarizer.py --runtime docker --docker-image recursive-flow:local

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/recursive-flow
COPY pyproject.toml README.md ./
COPY rflow ./rflow
RUN pip install ".[openai,anthropic]"

# DockerRuntime bind-mounts the host workspace at /workspace.
WORKDIR /workspace

# DockerRuntime spawns: `docker run -i --rm <image> python -m rflow.runtime.repl`.
# Setting it as CMD also makes `docker run -i recursive-flow:local` work standalone.
CMD ["python", "-m", "rflow.runtime.repl"]
