.PHONY: clean clean-build clean-pyc clean-test coverage dist docs help install lint lint/flake8 format-md lint-md test test-all examples examples-list examples-optional examples-live examples-sandbox examples-all eval-help eval-smoke eval-test eval-run eval-wandb eval-benchmark eval-clean animation animation-preview animation-mp4 animation-gif animation-gif-small animation-clean bump-version
	{%- if cookiecutter.use_black == 'y' %} lint/black{% endif %}
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z0-9_/-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

define BUMP_VERSION_PYSCRIPT
from pathlib import Path
import re
import sys

part = sys.argv[1]
if part not in {"major", "minor", "patch"}:
    raise SystemExit("BUMP must be one of: major, minor, patch")

path = Path("pyproject.toml")
text = path.read_text(encoding="utf-8")
match = re.search(r'(?m)^version = "(\d+)\.(\d+)\.(\d+)"$$', text)
if not match:
    raise SystemExit("Could not find [project] version in pyproject.toml")

major, minor, patch = map(int, match.groups())
if part == "major":
    major, minor, patch = major + 1, 0, 0
elif part == "minor":
    minor, patch = minor + 1, 0
else:
    patch += 1

old_version = match.group(0).split('"')[1]
new_version = f"{major}.{minor}.{patch}"
text = text[: match.start(1)] + new_version + text[match.end(3) :]
path.write_text(text, encoding="utf-8")
print(f"{path}: {old_version} -> {new_version}")
endef
export BUMP_VERSION_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"
BUMP ?= patch


help: ## Show this help message.
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr coverage/
	rm -fr .pytest_cache

lint: ## check style with flake8
	isort --profile black rflow
	black rflow
	flake8 rflow
	python -m ruff check .

# ── Markdown ─────────────────────────────────────────────────────────
# Scoped to public-facing docs only.
# Install once:  pip install mdformat mdformat-gfm
MD_FILES := README.md CHANGELOG.md $(wildcard docs/*.md)

format-md: ## Auto-format public markdown (one paragraph per line, no hard wrap).
	python -m mdformat --wrap no $(MD_FILES)

lint-md: ## Check public markdown formatting without writing changes.
	python -m mdformat --wrap no --check $(MD_FILES)

install: clean lint
	python -m pip install . --upgrade

doc:
	rm -r docs/reference/
	pdocs as_markdown rflow -o docs/reference
	rm docs/reference/rflow/index.md
	cp examples/*.ipynb docs/examples/
	cp README.md docs/index.md

serve-docs:
	mkdocs serve

commit: install test doc
	git add .
	git commit -a

test: ## Run the default test suite.
	python -m pytest

test-all: build-docker-image ## Run all tests, including Docker-gated integration tests.
	RECURSIVE_FLOW_DOCKER_TEST=1 python -m pytest

test-html: test
	$(BROWSER) tests/cov-report/index.html

EXAMPLES_ARGS ?=

examples: ## Run deterministic/offline examples.
	python examples/run_examples.py $(EXAMPLES_ARGS)

examples-list: ## List all examples and skip reasons.
	python examples/run_examples.py --all --list $(EXAMPLES_ARGS)

examples-optional: ## Run offline + optional-dependency examples.
	python examples/run_examples.py --include-optional $(EXAMPLES_ARGS)

examples-live: ## Run offline + live LLM examples.
	python examples/run_examples.py --include-live $(EXAMPLES_ARGS)

examples-sandbox: ## Run offline + sandbox runtime examples.
	python examples/run_examples.py --include-sandbox $(EXAMPLES_ARGS)

examples-all: ## Run every example category.
	python examples/run_examples.py --all $(EXAMPLES_ARGS)

build-docker-image:
	docker build -t recursive-flow:local .

# ── Eval harness ─────────────────────────────────────────────────────
EVAL_PROVIDER ?= openai
EVAL_MODEL ?= gpt-5-mini
EVAL_TASKS ?= official_sniah
EVAL_RUNNERS ?= vanilla rflow official
EVAL_SEEDS ?= 0:10
EVAL_BENCHMARK_TASKS ?= official
EVAL_BENCHMARK_RUNNERS ?= rflow vanilla official
EVAL_BENCHMARK_SEEDS ?= 0:20
EVAL_MAX_ITERS ?= 20
EVAL_MAX_DEPTH ?= 2
EVAL_OUT_DIR ?= benchmarks/eval/runs
EVAL_REPORT_DIR ?= eval-runs
EVAL_TASK_PARAMS ?=
EVAL_ARGS ?=
EVAL_OFFICIAL_DATA_DIR ?= evals/data
EVAL_OFFICIAL_SPLIT ?=
EVAL_OFFICIAL_MAX_SAMPLES ?=
EVAL_OFFICIAL_MAX_CONTEXT_CHARS ?=
EVAL_OFFICIAL_MAX_CONTEXT_TOKENS ?=
EVAL_BROWSECOMP_MAX_DOCS ?=
EVAL_OFFICIAL_ARGS := --official-data-dir $(EVAL_OFFICIAL_DATA_DIR) \
	$(if $(EVAL_OFFICIAL_SPLIT),--official-split $(EVAL_OFFICIAL_SPLIT),) \
	$(if $(EVAL_OFFICIAL_MAX_SAMPLES),--official-max-samples $(EVAL_OFFICIAL_MAX_SAMPLES),) \
	$(if $(EVAL_OFFICIAL_MAX_CONTEXT_CHARS),--official-max-context-chars $(EVAL_OFFICIAL_MAX_CONTEXT_CHARS),) \
	$(if $(EVAL_OFFICIAL_MAX_CONTEXT_TOKENS),--official-max-context-tokens $(EVAL_OFFICIAL_MAX_CONTEXT_TOKENS),) \
	$(if $(EVAL_BROWSECOMP_MAX_DOCS),--browsecomp-max-docs $(EVAL_BROWSECOMP_MAX_DOCS),)
EVAL_WANDB_PROJECT ?= rflow-eval
EVAL_WANDB_ENTITY ?=
EVAL_WANDB_ENTITY_ARG := $(if $(EVAL_WANDB_ENTITY),--wandb-entity $(EVAL_WANDB_ENTITY),)

eval-help: ## Show eval harness CLI help.
	python -m benchmarks.eval --help

eval-smoke: ## Run local fake eval smoke across fake/vanilla/rflow.
	python -m benchmarks.eval \
		--provider fake \
		--model fake \
		--tasks sniah \
		--runners fake vanilla rflow \
		--seeds 0:3 \
		--task-param records=8 \
		--task-param filler_words=2 \
		--out-dir $(EVAL_OUT_DIR) \
		--report-dir $(EVAL_REPORT_DIR) \
		$(EVAL_ARGS)

eval-test: ## Run focused eval harness tests.
	python -m pytest tests/test_eval_benchmarks.py

eval-run: ## Run real eval harness. Override EVAL_PROVIDER/MODEL/TASKS/RUNNERS/SEEDS/ARGS.
	python -m benchmarks.eval \
		--provider $(EVAL_PROVIDER) \
		--model $(EVAL_MODEL) \
		--tasks $(EVAL_TASKS) \
		--runners $(EVAL_RUNNERS) \
		--seeds $(EVAL_SEEDS) \
		--max-iters $(EVAL_MAX_ITERS) \
		--max-depth $(EVAL_MAX_DEPTH) \
		--out-dir $(EVAL_OUT_DIR) \
		--report-dir $(EVAL_REPORT_DIR) \
		$(EVAL_OFFICIAL_ARGS) \
		$(EVAL_TASK_PARAMS) \
		$(EVAL_ARGS)

eval-wandb: ## Run real eval harness with W&B logging enabled.
	python -m benchmarks.eval \
		--provider $(EVAL_PROVIDER) \
		--model $(EVAL_MODEL) \
		--tasks $(EVAL_TASKS) \
		--runners $(EVAL_RUNNERS) \
		--seeds $(EVAL_SEEDS) \
		--max-iters $(EVAL_MAX_ITERS) \
		--max-depth $(EVAL_MAX_DEPTH) \
		--out-dir $(EVAL_OUT_DIR) \
		--report-dir $(EVAL_REPORT_DIR) \
		--wandb \
		--wandb-project $(EVAL_WANDB_PROJECT) \
		$(EVAL_WANDB_ENTITY_ARG) \
		$(EVAL_OFFICIAL_ARGS) \
		$(EVAL_TASK_PARAMS) \
		$(EVAL_ARGS)

eval-benchmark:
	python -m benchmarks.eval \
		--provider $(EVAL_PROVIDER) \
		--model $(EVAL_MODEL) \
		--tasks $(EVAL_BENCHMARK_TASKS) \
		--runners $(EVAL_BENCHMARK_RUNNERS) \
		--seeds $(EVAL_BENCHMARK_SEEDS) \
		--max-iters $(EVAL_MAX_ITERS) \
		--max-depth $(EVAL_MAX_DEPTH) \
		--out-dir $(EVAL_OUT_DIR) \
		--report-dir $(EVAL_REPORT_DIR) \
		--report-name "$(EVAL_BENCHMARK_TASKS)" \
		--wandb \
		--wandb-project $(EVAL_WANDB_PROJECT) \
		$(EVAL_WANDB_ENTITY_ARG) \
		$(EVAL_OFFICIAL_ARGS) \
		$(EVAL_TASK_PARAMS) \
		$(EVAL_ARGS)

eval-clean: ## Remove local eval run artifacts.
	rm -rf benchmarks/eval/runs/

# ── Animation (manim) ────────────────────────────────────────────────
ANIMATION_SRC := docs/rlm_animation.py
ANIMATION_SCENE := RecursiveFlowHero
ANIMATION_OUT_DIR := media/videos/rlm_animation/1080p60

animation: animation-mp4 animation-gif-small ## Render the recursive-flow animation: MP4 + share-friendly GIF.

animation-preview: ## Quick low-res manim preview of the recursive-flow animation.
	manim -pql $(ANIMATION_SRC) $(ANIMATION_SCENE)

animation-mp4: ## Render docs/rlm_animation.mp4 (1080p60).
	manim -qh $(ANIMATION_SRC) $(ANIMATION_SCENE)
	cp $(ANIMATION_OUT_DIR)/$(ANIMATION_SCENE).mp4 docs/rlm_animation.mp4

animation-gif: ## Raw 1080p60 GIF from manim — WARNING: ~1GB+ output.
	manim -qh --format=gif $(ANIMATION_SRC) $(ANIMATION_SCENE)
	cp "$$(ls -t $(ANIMATION_OUT_DIR)/$(ANIMATION_SCENE)*.gif | head -n 1)" docs/rlm_animation.gif

animation-gif-small: animation-mp4 ## Share-friendly GIF (ffmpeg, ~5MB) — replaces the README hero.
	ffmpeg -y -i docs/rlm_animation.mp4 \
		-vf "fps=20,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5" \
		docs/rlm_animation.gif

animation-clean: ## Remove manim render artifacts (media/).
	rm -rf media/

bump-version: ## Bump pyproject.toml version. Override with BUMP=minor or BUMP=major.
	python -c "$$BUMP_VERSION_PYSCRIPT" "$(BUMP)"
