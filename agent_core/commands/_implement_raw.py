import ast
from typing import Optional, Protocol, runtime_checkable

from agent_core.commands.implement_cmd import (
    Command, _extract_file_context, _is_stub_body, extract_signatures, file_needs_generation,
)


@runtime_checkable
class LLMClientLike(Protocol):
    """Structural interface for anything capable of answering an LLM prompt."""

    def query(self, prompt: str) -> str: ...


def collect_raw_stubs(source: str) -> list[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]:
    """Return AST nodes whose bodies are stubs (pass-only or otherwise empty)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
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


class RawImplementCommand(Command):
    def name(self) -> str:
        return "implement-raw"

    def help_text(self) -> str:
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