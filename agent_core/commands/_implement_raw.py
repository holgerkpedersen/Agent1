import ast
from typing import Optional


class Command:
    @property
    def name(self) -> str:
        raise NotImplementedError

    def help_text(self) -> Optional[str]:
        return None

    def execute(self, source: str, filename: str, client: "LLMClientLike", radius: int = 400) -> Optional[str]:
        raise NotImplementedError


class LLMClientLike:
    def query(self, prompt: str) -> Optional[str]:
        raise NotImplementedError


def _is_stub_body(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> bool:
    """Return True if the node body consists only of a docstring or ellipsis."""
    body = node.body
    if not body:
        return False
    first = body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, (ast.Constant,)):
        value = first.value.value
        if value is None or isinstance(value, str):
            return len(body) == 1
    if isinstance(first, ast.Pass):
        return len(body) == 1
    return False


def _extract_file_context(source: str, filename: str, radius: int = 400) -> str:
    """Return a trimmed slice of the source around the file's stubs."""
    tree = ast.parse(source)
    stubs = collect_raw_stubs(source)
    if not stubs:
        return ""
    start = min(getattr(node, "lineno", 1) for node in stubs) - radius // 2
    end = max(getattr(node, "end_lineno", len(source.splitlines())) for node in stubs) + radius // 2
    lines = source.splitlines()
    start = max(0, start)
    end = min(len(lines), end)
    return "\n".join(lines[start:end])


def extract_signatures(source: str) -> dict[str, str]:
    """Map each top-level function/class name to its signature string."""
    tree = ast.parse(source)
    signatures: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            signatures[node.name] = ast.unparse(node).split(":")[0].strip()
    return signatures


def collect_raw_stubs(source: str) -> list[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]:
    """Collect all function/class nodes whose bodies are stubs."""
    tree = ast.parse(source)
    stubs: list[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and _is_stub_body(node):
            stubs.append(node)
    return stubs


def build_raw_prompt(source: str, filename: str, radius: int = 400) -> str:
    """Compose an LLM prompt describing the file context and stub bodies."""
    context = _extract_file_context(source, filename, radius=radius)
    signatures = extract_signatures(source)
    stubs = collect_raw_stubs(source)
    lines: list[str] = [
        f"Filename: {filename}",
        "--- Context ---",
        context,
        "--- Signatures ---",
    ]
    lines.extend(f"{name}: {signature}" for name, signature in signatures.items())
    if stubs:
        lines.append("--- Stub bodies to implement ---")
        lines.extend(ast.unparse(node) for node in stubs)
    return "\n".join(lines)


def generate_raw_implementation(
    source: str, filename: str, client: LLMClientLike, radius: int = 400
) -> Optional[str]:
    """Ask the LLM to produce a full implementation for a file needing generation."""
    if not file_needs_generation(filename):
        return None
    prompt = build_raw_prompt(source, filename, radius=radius)
    return client.query(prompt)


def file_needs_generation(filename: str) -> bool:
    """Return True if the filename indicates raw generation is needed."""
    return filename.endswith(".py") and "raw" in filename.lower()


class RawImplementCommand(Command):
    @property
    def name(self) -> str:
        return "implement-raw"

    def help_text(self) -> Optional[str]:
        return (
            "Generate raw implementations for files containing stub bodies. "
            "An LLM fills in function and class stubs using surrounding file context."
        )

    def execute(
        self, source: str, filename: str, client: LLMClientLike, radius: int = 400
    ) -> Optional[str]:
        return generate_raw_implementation(source, filename, client, radius=radius)


__all__ = [
    "RawImplementCommand",
    "LLMClientLike",
    "collect_raw_stubs",
    "build_raw_prompt",
    "generate_raw_implementation",
]