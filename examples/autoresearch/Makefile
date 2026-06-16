.PHONY: help sanity run smoke clean

TARGET ?= circle_packing
MODEL ?= gpt-5
BUDGET_S ?= 120
MAX_ITERATIONS ?= 20
MAX_SUBMISSIONS ?= 40
BRANCHES_PER_TURN ?= 8
MAX_DEPTH ?= 1
WORKDIR ?= ../_runs/autoresearch

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
    match = re.match(r'^([a-zA-Z0-9_/-]+):.*?## (.*)$$', line)
    if match:
        target, help = match.groups()
        print(f"{target:20s} {help}")
endef
export PRINT_HELP_PYSCRIPT

help: ## Show available targets
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

sanity: ## Run the circle_packing baseline evaluator
	python circle_packing/evaluate.py circle_packing/solution.py

run: ## Run autoresearch on $(TARGET)
	python autoresearch.py \
		--target $(TARGET) \
		--budget-s $(BUDGET_S) \
		--max-submissions $(MAX_SUBMISSIONS) \
		--model $(MODEL) \
		--branches-per-turn $(BRANCHES_PER_TURN) \
		--max-depth $(MAX_DEPTH) \
		--workdir $(WORKDIR)

smoke: ## Short no-UI run (baseline only, no child submissions)
	python autoresearch.py \
		--target $(TARGET) \
		--workdir $(WORKDIR)/smoke \
		--max-submissions 0 \
		--max-iterations 3 \
		--no-ui

clean: ## Remove local run artifacts
	rm -rf $(WORKDIR) $(WORKDIR)/smoke
