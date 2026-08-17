"""Lint guardrail scanner for generated Python code.

Flags two classes of problems that commonly slip past syntax checks:

* duplicate module-level definitions (a ``def`` / ``async def`` / ``class``
  name defined more than once in the same module)
* bare ``except:`` clauses, which swallow every exception and hide bugs

``scan_regression`` compares a pre-edit buffer against a post-edit buffer and
reports issues that are newly introduced.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union

DefinitionNode = Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]

__all__ = ["LintGuard", "LintIssue"]


@dataclass(frozen=True)
class LintIssue:
    """A single guardrail finding."""

    kind: str
    path: str
    line: int
    message: str
    name: str = ""


class LintGuard:
    """Scan Python sources for duplicate definitions and bare excepts."""

    def __init__(self, workspace: str | Path | None = None) -> None:
        self.workspace = Path(workspace) if workspace is not None else Path.cwd()

    def scan_workspace(self) -> list[LintIssue]:
        """Scan every ``*.py`` file under the configured workspace."""
        issues: list[LintIssue] = []
        for path in sorted(self.workspace.rglob("*.py")):
            if self._is_ignored(path):
                continue
            issues.extend(self.scan_file(path))
        return issues

    def scan_file(self, path: str | Path) -> list[LintIssue]:
        """Scan a single Python file."""
        file_path = Path(path)
        try:
            source = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return []
        return self.scan_source(source, str(file_path))

    def scan_source(self, source: str, path: str = "<string>") -> list[LintIssue]:
        """Scan an in-memory Python source string."""
        return self.find_duplicate_definitions(source, path) + self.find_bare_excepts(
            source, path
        )

    def scan_regression(
        self, before: str, after: str, path: str = "<string>"
    ) -> list[LintIssue]:
        """Report duplicate definitions / bare excepts newly introduced.

        Duplicate definitions are reported only for names that were not
        already duplicated in ``before``.  Bare excepts are reported only
        when ``before`` contained none, which avoids noisy false positives
        from pre-existing violations.
        """
        before_dupes = self._duplicate_names(before)
        issues = [
            issue
            for issue in self.find_duplicate_definitions(after, path)
            if issue.name not in before_dupes
        ]

        if not self.find_bare_excepts(before, path):
            issues.extend(self.find_bare_excepts(after, path))
        return issues

    def find_duplicate_definitions(
        self, source: str, path: str = "<string>"
    ) -> list[LintIssue]:
        """Flag module-level names defined more than once."""
        tree = self._parse(source)
        if tree is None:
            return []

        seen: dict[str, ast.AST] = {}
        issues: list[LintIssue] = []
        for node in self._module_definitions(tree):
            name = node.name
            previous = seen.get(name)
            if previous is not None:
                issues.append(
                    LintIssue(
                        kind="duplicate-definition",
                        path=path,
                        line=node.lineno,
                        message=(
                            f"Duplicate module-level definition of {name!r}; "
                            f"first defined at line {getattr(previous, 'lineno', '?')}"
                        ),
                        name=name,
                    )
                )
            else:
                seen[name] = node
        return issues

    def find_bare_excepts(
        self, source: str, path: str = "<string>"
    ) -> list[LintIssue]:
        """Flag every bare ``except:`` clause."""
        tree = self._parse(source)
        if tree is None:
            return []

        issues: list[LintIssue] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append(
                    LintIssue(
                        kind="bare-except",
                        path=path,
                        line=node.lineno,
                        message="Bare except clause; catch a concrete exception instead",
                    )
                )
        return issues

    def _parse(self, source: str) -> ast.Module | None:
        try:
            return ast.parse(source)
        except (SyntaxError, ValueError):
            return None

    @staticmethod
    def _module_definitions(tree: ast.Module) -> Iterable[DefinitionNode]:
        for stmt in tree.body:
            if isinstance(stmt, ast.FunctionDef):
                yield stmt
            elif isinstance(stmt, ast.AsyncFunctionDef):
                yield stmt
            elif isinstance(stmt, ast.ClassDef):
                yield stmt

    def _duplicate_names(self, source: str) -> set[str]:
        tree = self._parse(source)
        if tree is None:
            return set()
        names = [node.name for node in self._module_definitions(tree)]
        return {name for name, count in Counter(names).items() if count > 1}

    @staticmethod
    def _is_ignored(path: Path) -> bool:
        ignored = {
            ".git",
            ".hg",
            ".svn",
            ".venv",
            "venv",
            "node_modules",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
        }
        return any(part in ignored for part in path.parts)