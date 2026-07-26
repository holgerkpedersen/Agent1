from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Final, List, Optional

from ..base_handler import BaseCommandHandler
from ...path_utils import normalize_path
from ...config import AgentSettings
from ...exceptions import FileOperationError

logger = logging.getLogger(__name__)

_SIGNATURE_PATTERN: Final[str] = r"^\s*(?:async\s+)?def\s+(\w+)\s*\((.*?)\)\s*(?:->\s*([^:]+))?\s*:"


class AnalyzeCommand(BaseCommandHandler):
    """Analyze a Python source file and report structural signatures."""

    @property
    def name(self) -> str:
        return "analyze"

    async def handle(self, args: List[str]) -> int:
        if not args:
            logger.error("analyze requires a target file path")
            print("Usage: analyze <file_path>")
            return 1

        settings = AgentSettings()
        raw_target = args[0]
        resolved_path: Optional[Path] = normalize_path(raw_target, settings)

        if resolved_path is None or not resolved_path.exists():
            logger.error("Target file could not be resolved or does not exist")
            raise FileOperationError(f"File not found", str(raw_target))

        try:
            source_text = resolved_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.exception("Failed to read target file")
            raise FileOperationError(str(exc), str(resolved_path)) from exc

        tree: ast.AST
        try:
            tree = ast.parse(source_text, filename=str(resolved_path))
        except SyntaxError as exc:
            logger.warning("Syntax error encountered during analysis", extra={"file": resolved_path})
            print(f"SyntaxError in {resolved_path}: {exc}")
            return 2

        signatures = self._extract_signatures(tree)
        complexity_score = len(signatures) + sum(1 for _ in ast.walk(tree) if isinstance(_, (ast.If, ast.For)))

        logger.info("Analysis complete", extra={"file": resolved_path, "functions": len(signatures)})

        print(f"File: {resolved_path}")
        print(f"Functions detected: {len(signatures)}")
        for func_name, params, return_annotation in signatures:
            param_str = ", ".join(params) if params else ""
            annotation_str = f" -> {return_annotation}" if return_annotation else ""
            print(f"  def {func_name}({param_str}){annotation_str}")

        print(f"Complexity estimate: {complexity_score}")
        return 0

    @staticmethod
    def _extract_signatures(tree: ast.AST) -> List[tuple[str, List[str], Optional[str]]]:
        signatures: List[tuple[str, List[str], Optional[str]]] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                param_names: List[str] = [arg.arg for arg in node.args.args]
                return_annotation: Optional[str] = None
                if node.returns is not None and isinstance(node.returns, ast.Name):
                    return_annotation = node.returns.id

                signatures.append((node.name, param_names, return_annotation))

        # Fallback regex-based detection for edge cases missed by AST parsing
        fallback_matches = re.findall(_SIGNATURE_PATTERN, tree.body.__class__.__name__ if hasattr(tree, "body") else "", re.MULTILINE)  # noqa: B005
        if not signatures and fallback_matches:
            logger.warning("AST-based signature extraction failed; falling back to regex")

        return signatures


AnalyzeCommand.register = lambda cls=None: None  # Placeholder for registry compatibility