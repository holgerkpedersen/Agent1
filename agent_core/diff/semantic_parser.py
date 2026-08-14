from dataclasses import dataclass, field
from typing import List, Tuple

import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class ASTNode:
    """Represents a node in the semantic Abstract Syntax Tree."""
    node_type: str
    content: str
    children: List["ASTNode"] = field(default_factory=list)
    signature: str = ""


@dataclass
class DiffResult:
    """Result of a semantic comparison between two sources."""
    source_a: str
    source_b: str
    similarity_score: float
    differences: List[str]
    common_elements: List[str]


class SemanticDiffEngine:
    """Core logic for comparing AST structures and identifying similarities."""

    def compute_similarity(self, nodes_a: List[ASTNode], nodes_b: List[ASTNode]) -> float:
        try:
            sigs_a = {n.signature for n in nodes_a if n.signature}
            sigs_b = {n.signature for n in nodes_b if n.signature}

            if not sigs_a and not sigs_b:
                return 1.0

            intersection = len(sigs_a & sigs_b)
            union = len(sigs_a | sigs_b)
            return intersection / union if union > 0 else 0.0
        except Exception as exc:
            logger.warning("compute_similarity failed, returning 0.0: %s", exc)
            return 0.0

    def get_differences(self, nodes_a: List[ASTNode], nodes_b: List[ASTNode]) -> List[str]:
        try:
            sigs_a = {n.signature for n in nodes_a if n.signature}
            sigs_b = {n.signature for n in nodes_b if n.signature}

            diffs: List[str] = []
            for s in (sigs_a - sigs_b):
                diffs.append(f"Missing signature: {s}")
            for s in (sigs_b - sigs_a):
                diffs.append(f"Extra signature: {s}")
            return diffs
        except Exception as exc:
            logger.warning("get_differences failed, returning empty list: %s", exc)
            return []


class SemanticParser:
    """High-level interface for semantic code analysis and duplication detection."""

    def __init__(self, threshold: float = 0.75) -> None:
        self.threshold = threshold
        self._engine = SemanticDiffEngine()

    def parse(self, source: str) -> List[ASTNode]:
        """Parses raw source into a list of structural AST nodes."""
        nodes: List[ASTNode] = []
        try:
            lines = [line.strip() for line in source.split("\n") if line.strip()]

            for line in lines:
                try:
                    if "def " in line:
                        parts = line.split("def ")[1].split("(")[0].strip()
                        sig = hashlib.md5(parts.encode()).hexdigest()[:8]
                        nodes.append(ASTNode(node_type="function", content=line, signature=sig))
                    elif "if " in line:
                        nodes.append(ASTNode(node_type="conditional", content=line))
                    elif "class " in line:
                        parts = line.split("class ")[1].split("(")[0].strip()
                        sig = hashlib.md5(parts.encode()).hexdigest()[:8]
                        nodes.append(ASTNode(node_type="class", content=line, signature=sig))
                    else:
                        nodes.append(ASTNode(node_type="statement", content=line))
                except (IndexError, AttributeError) as exc:
                    logger.debug("Skipping malformed line during parse: %s — %s", line[:80], exc)
                    nodes.append(ASTNode(node_type="statement", content=line))
        except Exception as exc:
            logger.warning("parse failed for source, returning empty list: %s", exc)
        return nodes

    def extract_signatures(self, source: str) -> List[str]:
        """Extracts unique semantic signatures from a source."""
        try:
            nodes = self.parse(source)
            return [n.signature for n in nodes if n.signature]
        except Exception as exc:
            logger.warning("extract_signatures failed, returning empty list: %s", exc)
            return []

    def compare(self, source_a: str, source_b: str) -> DiffResult:
        """Compares two sources and returns structural similarity results."""
        try:
            nodes_a = self.parse(source_a)
            nodes_b = self.parse(source_b)

            score = self._engine.compute_similarity(nodes_a, nodes_b)
            diffs = self._engine.get_differences(nodes_a, nodes_b)

            return DiffResult(
                source_a=source_a,
                source_b=source_b,
                similarity_score=score,
                differences=diffs,
                common_elements=[]
            )
        except Exception as exc:
            logger.warning("compare failed, returning zero-similarity result: %s", exc)
            return DiffResult(
                source_a=source_a,
                source_b=source_b,
                similarity_score=0.0,
                differences=["Comparison failed"],
                common_elements=[]
            )

    def detect_duplication(self, sources: List[Tuple[str, str]]) -> List[DiffResult]:
        """Detects duplication across multiple pairs of source strings."""
        try:
            return [self.compare(a, b) for a, b in sources]
        except Exception as exc:
            logger.warning("detect_duplication failed, returning empty list: %s", exc)
            return []

    def is_duplicated(self, source_a: str, source_b: str) -> bool:
        """Checks if two sources exceed the similarity threshold."""
        try:
            result = self.compare(source_a, source_b)
            return result.similarity_score >= self.threshold
        except Exception as exc:
            logger.warning("is_duplicated failed, returning False: %s", exc)
            return False

    @property
    def signature(self) -> str:
        """Returns a unique identifier for this parser instance."""
        return "semantic_parser_v1"