"""Symbol-level code intelligence for the NLP tool loop (plan item B-#8).

Two pure-``ast`` tools that replace whole-file reads when the model needs
STRUCTURE instead of text:

- ``definitions(file)`` — every class/function with signature and line span,
  so the model can jump straight to the right ``read`` window.
- ``references(symbol)`` — file:line list of where a symbol is defined or
  used across the workspace (text-word match on AST name nodes plus string/
  comment hits, capped), so "where is X used" costs one call instead of a
  grep + N reads.

Both are read-only, stdlib-only, and workspace-scoped by the agent's path
resolution — same safety envelope as the existing ``search`` tool.
"""
from __future__ import annotations

import ast
import os
from typing import Iterator

#: Maximum number of reference hits returned (the result is model context).
MAX_REFERENCES = 60

#: Files bigger than this are skipped by the workspace scan (generated blobs).
_MAX_SCAN_BYTES = 1_500_000

_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache",
              ".ruff_cache", "backups", ".docs", "node_modules", ".venv"}

_DEF_KINDS = {
    ast.FunctionDef: "def",
    ast.AsyncFunctionDef: "async def",
    ast.ClassDef: "class",
}


def _first_line_of(node: ast.AST) -> int:
    """1-based start line of *node* (decorators included)."""
    return int(getattr(node, "lineno", 0))


def _last_line_of(node: ast.AST) -> int:
    """1-based end line of *node* (Python >= 3.8 provides end_lineno)."""
    return int(getattr(node, "end_lineno", 0) or getattr(node, "lineno", 0))


def _signature_of(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Compact one-line signature: ``name(args) -> ret``."""
    try:
        args = ast.unparse(node.args) if hasattr(node, "args") else ""
    except Exception:
        args = "..."
    ret = ""
    returns = getattr(node, "returns", None)
    if returns is not None:
        try:
            ret = f" -> {ast.unparse(returns)}"
        except Exception:
            ret = " -> ..."
    name = getattr(node, "name", "?")
    return f"{name}({args}){ret}"


def _bases_of(node: ast.ClassDef) -> str:
    if not node.bases:
        return ""
    try:
        return "(" + ", ".join(ast.unparse(b) for b in node.bases) + ")"
    except Exception:
        return "(...)"


def _walk_definitions(tree: ast.Module, prefix: str = "") -> list[str]:
    """Render definitions in document order, methods indented under classes."""
    out: list[str] = []
    for node in tree.body:
        kind = _DEF_KINDS.get(type(node))
        if isinstance(node, ast.ClassDef):
            out.append(
                f"  {prefix}class {node.name}{_bases_of(node)}  "
                f"[lines {_first_line_of(node)}-{_last_line_of(node)}]"
            )
            out.extend(_walk_class_body(node, prefix + "  "))
        elif kind in ("def", "async def") and isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            out.append(
                f"  {prefix}{kind} {_signature_of(node)}  "
                f"[lines {_first_line_of(node)}-{_last_line_of(node)}]"
            )
    return out


def _walk_class_body(node: ast.ClassDef, prefix: str) -> list[str]:
    out: list[str] = []
    for child in node.body:
        if isinstance(child, ast.FunctionDef):
            out.append(
                f"  {prefix}  def {_signature_of(child)}  "
                f"[lines {_first_line_of(child)}-{_last_line_of(child)}]"
            )
        elif isinstance(child, ast.AsyncFunctionDef):
            out.append(
                f"  {prefix}  async def {_signature_of(child)}  "
                f"[lines {_first_line_of(child)}-{_last_line_of(child)}]"
            )
        elif isinstance(child, ast.ClassDef):
            out.append(
                f"  {prefix}  class {child.name}  "
                f"[lines {_first_line_of(child)}-{_last_line_of(child)}]"
            )
            out.extend(_walk_class_body(child, prefix + "  "))
    return out


def collect_definitions(source: str, filename: str = "<src>") -> str:
    """Return the formatted definition index for *source*, or an error note."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return f"SyntaxError parsing {filename}: {exc}"
    lines = _walk_definitions(tree)
    if not lines:
        return f"No classes or functions found in {filename}."
    header = f"Definitions in {filename}:"
    return header + "\n" + "\n".join(lines)


def _iter_py_files(root: str) -> "Iterator[str]":
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def _word_is_symbol(word: str, symbol: str) -> bool:
    """Exact word match with attribute awareness.

    ``tool_loop`` matches ``tool_loop`` but not ``tool_loop_runner``;
    ``MUTATING_TOOLS`` inside ``agent_core.llm.tool_loop.MUTATING_TOOLS``
    matches too (dotted chains end with the bare name).
    """
    if word == symbol:
        return True
    parts = word.split(".")
    return bool(parts) and parts[-1] == symbol and all(parts)


def collect_references(
    symbol: str, root: str, max_results: int = MAX_REFERENCES,
) -> str:
    """Scan the workspace for *symbol* uses; return a capped file:line list.

    Lines are matched as whole words (attribute-aware), so ``run`` does not
    hit ``run_interactive``.  The symbol's own definition lines are included
    (marked ``def``), which is what "where is X defined/used" wants.
    """
    import re

    if not symbol or not symbol.replace("_", "").isalnum():
        return f"Invalid symbol name: {symbol!r}"

    pattern = re.compile(
        r"(?<![\w.])" + re.escape(symbol) + r"(?![\w])"
    )
    hits: list[tuple[str, int, str]] = []
    truncated_files = 0
    for path in _iter_py_files(root):
        rel = os.path.relpath(path, root).replace("\\", "/")
        try:
            if os.path.getsize(path) > _MAX_SCAN_BYTES:
                truncated_files += 1
                continue
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for lineno, line in enumerate(fh, 1):
                    if pattern.search(line):
                        hits.append((rel, lineno, line.strip()))
                        if len(hits) >= max_results:
                            break
        except OSError:
            continue
        if len(hits) >= max_results:
            break

    if not hits:
        return (
            f"No references to '{symbol}' found in .py files under "
            f"{os.path.basename(root) or root}."
        )

    out = [
        f"References to '{symbol}' ({len(hits)}"
        + ("+" if len(hits) >= max_results else "")
        + " hits, showing file:line):"
    ]
    current_file = None
    for rel, lineno, text in hits:
        if rel != current_file:
            out.append(f"  {rel}:")
            current_file = rel
        out.append(f"    {lineno}: {text[:160]}")
    if truncated_files:
        out.append(f"  ({truncated_files} oversized file(s) skipped)")
    return "\n".join(out)
