"""Analyze a Python source file and report structural signatures via AST."""

import ast
import logging
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

_SIGNATURE_PATTERN: Final[str] = (
    r"^\s*(?:async\s+)?def\s+(\w+)\s*\((.*?)\)\s*(?:->\s*([^:]+))?\s*:"
)


class AnalyzeCommand:
    """Analyze a Python source file and report structural signatures."""

    @staticmethod
    def register(cls: type | None = None) -> None:
        """Registry compatibility placeholder (registry not used here)."""

    @property
    def name(self) -> str:
        return "analyze"

    async def handle(self, args: list[str]) -> int:
        if not args:
            logger.error("analyze requires a target file path")
            print("Usage: analyze <file_path>")
            return 1

        raw_target = args[0]

        try:
            resolved_path = Path(raw_target).resolve()
        except (OSError, ValueError) as exc:
            logger.error("Target file could not be resolved", extra={"error": str(exc)})
            print(f"Error resolving path: {exc}")
            return 1

        if not resolved_path.exists():
            logger.error(
                "Target file does not exist", extra={"file": str(resolved_path)}
            )
            print(f"File not found: {resolved_path}")
            return 1

        try:
            source_text = resolved_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.exception("Failed to read target file")
            print(f"Error reading file: {exc}")
            return 1

        tree: ast.AST
        try:
            tree = ast.parse(source_text, filename=str(resolved_path))
        except SyntaxError as exc:
            logger.warning(
                "Syntax error encountered during analysis",
                extra={"file": str(resolved_path)},
            )
            print(f"SyntaxError in {resolved_path}: {exc}")
            return 2

        signatures = self._extract_signatures(tree)
        complexity_score = len(signatures) + sum(
            1 for _ in ast.walk(tree) if isinstance(_, (ast.If, ast.For))
        )

        logger.info(
            "Analysis complete",
            extra={"file": str(resolved_path), "functions": len(signatures)},
        )

        print(f"File: {resolved_path}")
        print(f"Functions detected: {len(signatures)}")
        for func_name, params, return_annotation in signatures:
            param_str = ", ".join(params) if params else ""
            annotation_str = (
                f" -> {return_annotation}" if return_annotation else ""
            )
            print(f"  def {func_name}({param_str}){annotation_str}")

        print(f"Complexity estimate: {complexity_score}")
        return 0

    @staticmethod
    def _extract_signatures(
        tree: ast.AST,
    ) -> list[tuple[str, list[str], str | None]]:
        """Extract function signatures by walking the AST.

        The previous regex fallback was broken (searched ``tree.body.__class__.__name__``
        which is literally the string ``"Module"``).  AST traversal alone covers all valid
        Python source, so the buggy fallback has been removed entirely.
        """
        signatures: list[tuple[str, list[str], str | None]] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                param_names: list[str] = [arg.arg for arg in node.args.args]
                return_annotation: str | None = None
                if node.returns is not None and isinstance(
                    node.returns, ast.Name
                ):
                    return_annotation = node.returns.id

                signatures.append((node.name, param_names, return_annotation))

        return signatures
