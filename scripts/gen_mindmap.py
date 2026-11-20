"""Regenerate FreeMind .mm maps with per-file class/function nodes.

Walks the tracked .py files referenced in .docs/freemind/*.mm and, for each
file node, appends child nodes for the top-level classes and functions found
via AST.  Idempotent: re-running replaces the generated children.

Usage:
    python scripts/gen_mindmap.py
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MM_DIR = ROOT / ".docs" / "freemind"


def top_level_defs(path: Path) -> list[str]:
    try:
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "class" if isinstance(node, ast.ClassDef) else "def"
            out.append(f"{node.name} ({kind})")
    return out


def process_mm(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    # Find file nodes: <node TEXT="...path.py"/> (possibly with intent parent already)
    # We operate on nodes whose TEXT ends with .py
    # Use regex to replace a leaf file node with parent(intent?)... simpler:
    # For each occurrence of a node whose text is exactly a .py path, wrap?
    # Instead, append class/def children after the file node if not present.
    added = 0
    lines = text.splitlines()
    out_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.search(r'<node TEXT="([^"]+\.py)"/>\s*$', line)
        if m:
            fpath = m.group(1)
            # File node text is just the path (child of an intent node).
            defs = top_level_defs(ROOT / fpath)
            out_lines.append(line)
            indent = line[: len(line) - len(line.lstrip())]
            child_indent = indent + "  "
            j = i + 1
            while (
                j < len(lines)
                and lines[j].startswith(child_indent)
                and re.search(r"\((class|def)\)", lines[j])
            ):
                j += 1
            if defs:
                for d in defs:
                    out_lines.append(f'{child_indent}<node TEXT="{d}"/>')
                added += len(defs)
            i = j
            continue
        out_lines.append(line)
        i += 1
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return added


def main() -> None:
    total = 0
    for mm in sorted(MM_DIR.glob("*.mm")):
        if mm.name == "agent1.mm":
            continue
        n = process_mm(mm)
        total += n
        print(f"{mm.name}: +{n} def nodes")
    print(f"total added: {total}")


if __name__ == "__main__":
    main()
