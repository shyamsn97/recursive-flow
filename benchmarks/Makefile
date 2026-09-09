.PHONY: help eval-help eval-list-sets eval-smoke eval-test eval-run eval-reasoning eval-long-context eval-delegation eval-task-graph eval-research eval-code eval-compare eval-clean
.DEFAULT_GOAL := help

# Recipes run from the repo root (absolute imports: `from benchmarks.eval ...`).
REPO_ROOT := $(realpath $(dir $(realpath $(firstword $(MAKEFILE_LIST))))/..)

define PRINT_HELP_PYSCRIPT
import re, sys
for line in sys.stdin:
	match = re.match(r'^([a-zA-Z0-9_/-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-24s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

EVAL_PROVIDER ?= openai
EVAL_MODEL ?= gpt-5-mini
EVAL_SET ?= reasoning
EVAL_SEEDS ?= 0
EVAL_SAMPLE_ARGS ?= --limit 5
EVAL_MAX_ITERS ?= 20
EVAL_CHILD_MAX_ITERS ?= 20
EVAL_MAX_DEPTH ?= 2
EVAL_MAX_BUDGET ?= 100000
EVAL_INCLUDE_LLM_QUERY ?= true
EVAL_EXECUTOR ?= local
EVAL_PARALLEL ?= 1
EVAL_BEST_OF_N ?= 1
EVAL_ATTEMPT_TIMEOUT ?= 900
EVAL_MODAL_APP_NAME ?= rlmflow-benchmarks
EVAL_MODAL_CPU ?= 1
EVAL_MODAL_TIMEOUT ?= 3600
EVAL_OUT_DIR ?= benchmarks/eval/runs
EVAL_DATASET_PARAMS ?=
EVAL_RUNNER_PARAMS ?= --runner-param rlmflow-local.max_iters=$(EVAL_MAX_ITERS) --runner-param rlmflow-local.child_max_iters=$(EVAL_CHILD_MAX_ITERS) --runner-param rlmflow-local.max_depth=$(EVAL_MAX_DEPTH) --runner-param rlmflow-local.max_budget=$(EVAL_MAX_BUDGET) --runner-param rlmflow-local.use_llm_query=$(EVAL_INCLUDE_LLM_QUERY) --runner-param official-rlm.max_iters=$(EVAL_MAX_ITERS) --runner-param official-rlm.max_depth=$(EVAL_MAX_DEPTH) --runner-param official-rlm.max_budget=$(EVAL_MAX_BUDGET)
EVAL_ARGS ?=
EVAL_RLMFLOW_RUN ?=
EVAL_OFFICIAL_RUN ?=
EVAL_COMPARISON_OUT ?= benchmarks/eval/comparisons/latest
EVAL_RUN_ID ?=
EVAL_RESUME ?= false
EVAL_RUN_ID_ARG := $(if $(EVAL_RUN_ID),--run-id $(EVAL_RUN_ID),)
EVAL_RESUME_ARG := $(if $(filter true,$(EVAL_RESUME)),--resume,)

EVAL_LAUNCH = cd $(REPO_ROOT) && python -m benchmarks.eval --model $(EVAL_PROVIDER):$(EVAL_MODEL) --seeds $(EVAL_SEEDS) --out-dir $(EVAL_OUT_DIR) --executor $(EVAL_EXECUTOR) --parallel $(EVAL_PARALLEL) --best-of-n $(EVAL_BEST_OF_N) --attempt-timeout $(EVAL_ATTEMPT_TIMEOUT) --modal-app-name $(EVAL_MODAL_APP_NAME) --modal-cpu $(EVAL_MODAL_CPU) --modal-timeout $(EVAL_MODAL_TIMEOUT) $(EVAL_DATASET_PARAMS) $(EVAL_RUN_ID_ARG) $(EVAL_RESUME_ARG)

help: ## Show this help message.
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

eval-help: ## Show eval harness CLI help.
	cd $(REPO_ROOT) && python -m benchmarks.eval --help

eval-list-sets: ## Print named eval sets and exit.
	cd $(REPO_ROOT) && python -m benchmarks.eval --list-sets

eval-smoke: ## Local fake smoke (no API key).
	cd $(REPO_ROOT) && python -m benchmarks.eval --model fake --dataset smoke --seeds 0:3 --dataset-param synthetic_needle.records=8 --dataset-param synthetic_needle.filler_words=2 --runner-param rlmflow-local.max_iters=3 --runner-param rlmflow-local.max_depth=1 --runner-param rlmflow-local.live_save=false --out-dir $(EVAL_OUT_DIR) $(EVAL_ARGS)

eval-test: eval-smoke ## Alias for eval-smoke.

eval-run: ## Run EVAL_SET (default: reasoning). Runners come from the set unless --runner is in EVAL_ARGS.
	$(EVAL_LAUNCH) --dataset $(EVAL_SET) $(EVAL_SAMPLE_ARGS) $(EVAL_RUNNER_PARAMS) $(EVAL_ARGS)

eval-reasoning: ## AIME 2025 + Sudoku Extreme.
	$(MAKE) eval-run EVAL_SET=reasoning

eval-long-context: ## S-NIAH, OOLONG, OOLONG-Pairs, CodeQA.
	$(MAKE) eval-run EVAL_SET=long-context

eval-delegation: ## Routing set: local / batched query / spawn.
	$(MAKE) eval-run EVAL_SET=delegation EVAL_SAMPLE_ARGS=--full EVAL_PARALLEL=5 EVAL_RUNNER_PARAMS="$(EVAL_RUNNER_PARAMS) --runner-param rlmflow-local.prompt_condition=current_policy"

eval-task-graph: ## Frozen 20-problem telemetry pack.
	$(MAKE) eval-run EVAL_SET=task-graph EVAL_SAMPLE_ARGS=--full EVAL_RUNNER_PARAMS="$(EVAL_RUNNER_PARAMS) --runner-param rlmflow-local.prompt_condition=current_policy"

eval-research: ## BrowseComp-Plus. Download the corpus first.
	$(MAKE) eval-run EVAL_SET=research

eval-code: ## LiveCodeBench. Scoring requires Docker.
	$(MAKE) eval-run EVAL_SET=code

eval-compare: ## Pair two completed runs. Set EVAL_RLMFLOW_RUN and EVAL_OFFICIAL_RUN.
	@test -n "$(EVAL_RLMFLOW_RUN)" || (echo "EVAL_RLMFLOW_RUN is required" >&2; exit 1)
	@test -n "$(EVAL_OFFICIAL_RUN)" || (echo "EVAL_OFFICIAL_RUN is required" >&2; exit 1)
	cd $(REPO_ROOT) && python -m benchmarks.eval.compare --run rlmflow=$(EVAL_RLMFLOW_RUN) --run official=$(EVAL_OFFICIAL_RUN) --output-dir $(EVAL_COMPARISON_OUT)

eval-clean: ## Remove local eval run artifacts.
	rm -rf $(REPO_ROOT)/benchmarks/eval/runs/
