from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import hashlib


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
        sigs_a = {n.signature for n in nodes_a if n.signature}
        sigs_b = {n.signature for n in nodes_b if n.signature}

        if not sigs_a and not sigs_b:
            return 1.0
        
        intersection = len(sigs_a & sigs_b)
        union = len(sigs_a | sigs_b)
        return intersection / union if union > 0 else 0.0

    def get_differences(self, nodes_a: List[ASTNode], nodes_b: List[ASTNode]) -> List[str]:
        sigs_a = {n.signature for n in nodes_a if n.signature}
        sigs_b = {n.signature for n in nodes_b if n.signature}
        
        diffs = []
        for s in (sigs_a - sigs_b):
            diffs.append(f"Missing signature: {s}")
        for s in (sigs_b - sigs_a):
            diffs.append(f"Extra signature: {s}")
        return diffs


class SemanticParser:
    """High-level interface for semantic code analysis and duplication detection."""

    def __init__(self, threshold: float = 0.75) -> None:
        self.threshold = threshold
        self._engine = SemanticDiffEngine()

    def parse(self, source: str) -> List[ASTNode]:
        """Parses raw source into a list of structural AST nodes."""
        nodes: List[ASTNode] = []
        lines = [line.strip() for line in source.split("\n") if line.strip()]

        for line in lines:
            if "def " in line:
                # Extract function name to create a unique signature
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
        return nodes

    def extract_signatures(self, source: str) -> List[str]:
        """Extracts unique semantic signatures from a source."""
        nodes = self.parse(source)
        return [n.signature for n in nodes if n.signature]

    def compare(self, source_a: str, source_b: str) -> DiffResult:
        """Compares two sources and returns structural similarity results."""
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

    def detect_duplication(self, sources: List[Tuple[str, str]]) -> List[DiffResult]:
        """Detects duplication across multiple pairs of source strings."""
        return [self.compare(a, b) for a, b in sources]

    def is_duplicated(self, source_a: str, source_b: str) -> bool:
        """Checks if two sources exceed the similarity threshold."""
        result = self.compare(source_a, source_b)
        return result.similarity_score >= self.threshold

    @property
    def signature(self) -> str:
        """Returns a unique identifier for this parser instance."""
        return "semantic_parser_v1"