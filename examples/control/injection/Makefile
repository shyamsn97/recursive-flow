WIDTH ?= 1024
HEIGHT ?= 768
SCALE ?= 1.0

baseline_run := ../../_runs/word-search/baseline

render_baseline:
	recursive-flow render ./$(baseline_run) \
	--format steps \
	--out ./$(baseline_run)/frames/baseline \
	--width $(WIDTH) \
	--height $(HEIGHT) \
	--scale $(SCALE)

render_all: render_baseline
