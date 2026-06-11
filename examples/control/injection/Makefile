base_workspace := ../../_runs/word-search-workspace
baseline_workspace := $(base_workspace)/word-search-baseline
direct_scan_workspace := $(base_workspace)/word-search-direct-scan
cols_direct_workspace := $(base_workspace)/word-search-cols-direct

render_baseline:
	recursive-flow render ./$(baseline_workspace) \
	--format steps \
	--out ./$(baseline_workspace)/frames/baseline \
	--width 1600 \
	--height 1200 \
	--scale 2

render_direct_scan:
	recursive-flow render ./$(direct_scan_workspace) \
	--format steps \
	--out ./$(direct_scan_workspace)/frames/direct-scan \
	--width 1600 \
	--height 1200 \
	--scale 2

render_cols_direct:
	recursive-flow render ./$(cols_direct_workspace) \
	--format steps \
	--out ./$(cols_direct_workspace)/frames/cols-direct \
	--width 1600 \
	--height 1200 \
	--scale 2

render_all: render_baseline render_direct_scan render_cols_direct