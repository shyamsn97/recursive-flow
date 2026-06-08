base_workspace := sudoku-workspace/
naive_workspace := $(base_workspace)/sudoku-naive
backtracking_workspace := $(base_workspace)/sudoku-backtracking
cols_function_workspace := $(base_workspace)/sudoku-cols-function

render_naive:
	rlmflow render ./$(naive_workspace) \
	--format steps \
	--out ./$(naive_workspace)/frames/naive \
	--width 1600 \
	--height 1200 \
	--scale 2

render_backtracking:
	rlmflow render ./$(backtracking_workspace) \
	--format steps \
	--out ./$(backtracking_workspace)/frames/backtracking \
	--width 1600 \
	--height 1200 \
	--scale 2

render_cols_function:
	rlmflow render ./$(cols_function_workspace) \
	--format steps \
	--out ./$(cols_function_workspace)/frames/cols-function \
	--width 1600 \
	--height 1200 \
	--scale 2

render_all: render_naive render_backtracking render_cols_function