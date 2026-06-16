"""Static checks on agent-emitted code blocks.

The engine runs an agent's ```repl block as a coroutine only when it has a
*top-level* ``await launch_subagents(...)``. Anything else involving ``await``
(awaiting the wrong call, awaiting inside a function or comprehension, a bare
un-awaited ``launch_subagents(...)``) fails confusingly at runtime. This
pre-flight check turns those into a single clear, recoverable error string the
model can act on, before the block ever executes.
"""

from __future__ import annotations

import ast
import re

# Opener for *reading* code blocks: ```repl / ```python / bare ```.
# (Broader than the repl-only opener used for editing, below.)
_OPEN = re.compile(r"```(?:repl|python)?[ \t]*\n")
_CLOSE = re.compile(r"\n?```[ \t]*(?:\n|$)")
# Opener for *editing* code blocks — repl fences only, so injection/replay
# edits never rewrite an incidental ```python/```bash block.
_REPL_OPEN = re.compile(r"```repl[ \t]*\n")


def find_code_blocks(text: str) -> list[str]:
    """Extract fenced code blocks (```repl / ```python / bare ```).

    Greedy per block: the *last* closing fence after an opener wins, so markdown
    fences embedded inside a Python string (e.g. ``\"\"\"```bash ... ```\"\"\"``)
    don't prematurely close the block.
    """
    blocks: list[str] = []
    pos = 0
    while True:
        opening = _OPEN.search(text, pos)
        if not opening:
            break
        start = opening.end()
        last = None
        for m in _CLOSE.finditer(text, start):
            last = m
        if last is None:
            break
        blocks.append(text[start : last.start()].strip())
        pos = last.end()
    return blocks


def replace_code_block(text: str, new_code: str) -> str:
    """Keep text up to the first ```repl block, replacing its body with ``new_code``.

    Repl-specific on purpose: trajectory edits (inject/replace) target the
    agent's ``repl`` action block, not incidental fences.
    """
    opening = _REPL_OPEN.search(text)
    if not opening:
        return text
    start = opening.end()
    last = None
    for m in _CLOSE.finditer(text, start):
        last = m
    if last is None:
        return text
    return text[: opening.start()] + f"```repl\n{new_code}\n```"


# The only calls an agent may ``await`` at action-block top level.
# ``launch_subagents`` is the public surface; ``flow_wait`` is the internal
# primitive it composes over (kept awaitable for the engine's own machinery).
_AWAITABLE_CALLS = {"launch_subagents", "flow_wait"}


def check_wait_syntax(code: str) -> str | None:
    """Return an error string for unsupported ``await`` usage, else ``None``.

    A genuine ``SyntaxError`` returns ``None`` here — that's reported through
    the normal execution path with the interpreter's own message.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    checker = _WaitSyntaxChecker()
    checker.visit(tree)
    return "ERROR: " + "; ".join(checker.errors) if checker.errors else None


def _is_awaitable_call(node: ast.AST | None) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _AWAITABLE_CALLS
    )


class _WaitSyntaxChecker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.await_depth = 0
        self.errors: list[str] = []

    def _add(self, node: ast.AST, message: str) -> None:
        line = getattr(node, "lineno", None)
        prefix = f"Line {line}: " if line is not None else ""
        self.errors.append(prefix + message)

    def visit_Await(self, node: ast.Await) -> None:  # noqa: N802
        if not _is_awaitable_call(node.value):
            self._add(node, "only `await launch_subagents(...)` is supported")
        self.await_depth += 1
        self.generic_visit(node)
        self.await_depth -= 1

    def visit_Yield(self, node: ast.Yield) -> None:  # noqa: N802
        self._add(
            node,
            "use `await launch_subagents([...])`; top-level `yield` is not supported",
        )

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:  # noqa: N802
        self._add(node, "top-level `yield from` is not supported")

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if _is_awaitable_call(node) and self.await_depth == 0:
            name = node.func.id  # type: ignore[union-attr]
            self._add(node, f"`{name}(...)` must be awaited: `await {name}(...)`")
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:  # noqa: N802
        self._check_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:  # noqa: N802
        self._check_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:  # noqa: N802
        self._check_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:  # noqa: N802
        self._check_comprehension(node)

    def _check_comprehension(self, node: ast.AST) -> None:
        for child in ast.walk(node):
            if isinstance(child, ast.Await) or _is_awaitable_call(child):
                self._add(
                    node,
                    "`await launch_subagents(...)` is not supported in comprehensions",
                )
                return

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._check_nested(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._check_nested(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        self._check_nested(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._check_nested(node)

    def _check_nested(self, node: ast.AST) -> None:
        for child in ast.walk(node):
            if isinstance(child, ast.Await) or _is_awaitable_call(child):
                self._add(
                    node,
                    "`launch_subagents(...)` is only supported at action-block "
                    "top level",
                )
                return


__all__ = ["check_wait_syntax", "find_code_blocks", "replace_code_block"]
