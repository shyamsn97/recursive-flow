WIDTH ?= 1024
HEIGHT ?= 768
SCALE ?= 1.0

base_workspace := ../../_runs/word-search-workspace
baseline_workspace := $(base_workspace)/word-search-baseline
direct_scan_workspace := $(base_workspace)/word-search-direct-scan
cols_direct_workspace := $(base_workspace)/word-search-cols-direct
direct_scan_branch_id := direct-scan
cols_direct_branch_id := cols-direct

render_baseline:
	recursive-flow render ./$(baseline_workspace) \
	--format steps \
	--out ./$(baseline_workspace)/frames/baseline \
	--width $(WIDTH) \
	--height $(HEIGHT) \
	--scale $(SCALE)

render_direct_scan:
	recursive-flow render ./$(direct_scan_workspace) \
	--format steps \
	--branch-id $(direct_scan_branch_id) \
	--out ./$(direct_scan_workspace)/frames/direct-scan \
	--width $(WIDTH) \
	--height $(HEIGHT) \
	--scale $(SCALE)

render_cols_direct:
	recursive-flow render ./$(cols_direct_workspace) \
	--format steps \
	--branch-id $(cols_direct_branch_id) \
	--out ./$(cols_direct_workspace)/frames/cols-direct \
	--width $(WIDTH) \
	--height $(HEIGHT) \
	--scale $(SCALE)

render_all: render_baseline render_direct_scan render_cols_direct