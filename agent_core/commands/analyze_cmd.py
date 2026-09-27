"""Analyze command for agent interactive mode."""

from __future__ import annotations
import os
import re

from .base import Command, read_stdin
from agent_core import workspace_path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Agent


def _parse_imports(source: str) -> list[str]:
    """Extract local project module paths from *source* that could be resolved.

    Returns a list of relative file paths (e.g. ``agent_core/commands/fix_cmd.py``).
    """
    result = []
    for m in re.finditer(r"^(?:from|import)\s+(\S+)", source, re.MULTILINE):
        module = m.group(1)
        if module.startswith("."):
            continue
        top = module.split(".", 1)[0]
        if top in ("agent_core", "agent1", "tests", "src"):
            path = module.replace(".", "/")
            if path == top:
                path = f"{top}/__init__.py"
            else:
                path = f"{path}.py"
            result.append(path)
    return sorted(set(result))


def _parse_file_refs(text: str) -> list[str]:
    """Extract project file references from a text response.

    Matches patterns like ``agent_core/commands/fix_cmd.py``,
    backtick-wrapped filenames, and bare ``.py`` filenames.
    """
    refs: list[str] = []

    # Backtick-wrapped paths: `agent_core/commands/fix_cmd.py`
    for m in re.finditer(r"`([^`]+\.py)`", text):
        refs.append(m.group(1))

    # Explicit relative paths: agent_core/commands/fix_cmd.py or src/agent1/...
    for m in re.finditer(r"(?:agent_core|agent1|src|tests)/[\w/]+\.py", text):
        refs.append(m.group(0))

    return sorted(set(refs))


#: Directory analysis limits: a folder is summarized from a bounded sample so
#: one ``analyze agent_core/`` call cannot pull the whole tree into context.
_ANALYZE_MAX_FILES = 40
_ANALYZE_MAX_CHARS = 200_000
_ANALYZE_SKIP_DIRS = frozenset({
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "node_modules", ".docs", "backups", "reports",
    "generated", "dist", "build",
})
_ANALYZE_SKIP_SUFFIXES = (
    ".pyc", ".pyo", ".so", ".dll", ".exe", ".bin", ".png", ".jpg", ".jpeg",
    ".gif", ".ico", ".zip", ".whl", ".tar", ".gz", ".gguf", ".pdf",
)


def _collect_directory_files(root: str) -> list[str]:
    """Files under *root* worth analyzing (bounded, skip-list applied)."""
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in _ANALYZE_SKIP_DIRS and not d.startswith(".")
        )
        for name in sorted(filenames):
            if name.lower().endswith(_ANALYZE_SKIP_SUFFIXES):
                continue
            files.append(os.path.join(dirpath, name))
            if len(files) >= _ANALYZE_MAX_FILES:
                return files
    return files


class AnalyzeCommand(Command):
    """AI analysis of a file via LM Studio — follows imports and iterates with --desc."""

    @property
    def name(self) -> str:
        return "analyze"

    @property
    def help_text(self) -> str:
        return (
            'analyze <file|dir> [--desc "q"] [--stdin] [--deep] - AI analysis '
            "via LM Studio (a directory is summarized from a bounded sample; "
            "no path = whole workspace)"
        )

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        parts = list(args)

        desc_text = None
        deep_mode = False
        stdin_mode = "--stdin" in parts

        if "--desc" in parts:
            di = parts.index("--desc")
            if di + 1 < len(parts):
                desc_text = parts[di + 1].strip('"')
                deep_mode = True
            parts = [p for p in parts if p not in (parts[di], args[di + 1] if di + 1 < len(args) else "")]

        if "--deep" in parts:
            deep_mode = True
            parts = [p for p in parts if p != "--deep"]

        if stdin_mode:
            parts = [p for p in parts if p != "--stdin"]
            content = read_stdin("Paste text to analyze. Type --- on its own line when done, or Ctrl+Z to finish:")
            if not content.strip():
                self.error("No text provided.")
                return True
            question = desc_text or "Analyze the text above thoroughly."
            result = await agent.llm.chat([
                {"role": "system", "content": "You are an expert analyst. Answer the question concisely based on the provided text."},
                {"role": "user", "content": f"## Text:\n\n{content}\n\n## Question:\n{question}"},
            ])
            print(result)
            return True

        if len(parts) < 1:
            # Default (the schema's "whole workspace"): the directory analyzer
            # handles the workspace root.
            path = workspace_path(agent.workspace)
        else:
            path = parts[0]
        output_file = parts[1] if len(parts) > 1 else None

        ws = workspace_path(agent.workspace)
        resolved = (
            os.path.normpath(path)
            if os.path.isabs(path)
            else os.path.normpath(os.path.join(ws, path))
        )
        if os.path.isdir(resolved):
            # A FOLDER is a first-class target: summarize a bounded sample of
            # its files (opening it as a file used to raise PermissionError).
            result = await self._analyze_directory(path, desc_text, agent)
        elif deep_mode:
            result = await self._deep_analyze(path, desc_text, agent)
        else:
            content = await agent.read_file(path, track_read=False)
            if content.startswith("File not found:") or content.startswith("Error"):
                result = content
            else:
                result = await agent.llm.analyze_code(content)

        if output_file:
            invalid = set('<>:"/\\|?*')
            if any(c in output_file for c in invalid) or len(output_file) < 2:
                self.error(f"Invalid output filename: {output_file!r}")
                print(result)
                return True
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"# Analysis of {path}\n\n")
                f.write(result)
            print(f"Analysis written to {output_file}")
        else:
            print(result)

        return True

    async def _analyze_directory(
        self, path: str, question: str | None, agent: "Agent",
    ) -> str:
        """Analyze a DIRECTORY from a bounded sample of its files.

        ``analyze agent_core/`` used to fail with an opaque PermissionError
        (the folder was opened as a file); a directory is a first-class target
        now, and omitting the path analyzes the whole workspace.  The sample is
        capped by file count and total characters so one call cannot pull the
        entire tree into context.
        """
        ws = workspace_path(agent.workspace)
        root = (
            os.path.normpath(path)
            if os.path.isabs(path)
            else os.path.normpath(os.path.join(ws, path))
        )
        files = _collect_directory_files(root)
        if not files:
            return f"No analyzable files found under {path}."
        parts: list[str] = []
        total = 0
        for full in files:
            content = await agent.read_file(full, track_read=False)
            if content.startswith("File not found:") or content.startswith("Error"):
                continue
            rel = os.path.relpath(full, ws)
            chunk = f"\n\n# === {rel} ===\n{content[:20000]}"
            if total + len(chunk) > _ANALYZE_MAX_CHARS:
                break
            parts.append(chunk)
            total += len(chunk)
        try:
            shown = os.path.relpath(root, ws)
        except ValueError:  # different drive on Windows
            shown = root
        listing = "\n".join(f"- {os.path.relpath(f, ws)}" for f in files)
        system = (
            "You are an expert code reviewer. Analyze the DIRECTORY: its "
            "purpose, structure, key modules, risks and concrete improvements."
        )
        user = (
            f"## Directory: {shown}\n## Files ({len(files)}):\n{listing}\n"
            + "".join(parts)
            + (f"\n\n## Question:\n{question}" if question else "")
        )
        return await agent.llm.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])

    async def _deep_analyze(self, path: str, question: str | None, agent: "Agent") -> str:
        """Iteratively read files and deepen analysis, following import chains
        and file references in LLM responses."""
        ws = workspace_path(agent.workspace)
        content = await agent.read_file(path, track_read=False)
        if content.startswith("File not found:") or content.startswith("Error"):
            return content

        # Round 0: collect the target file + its imports
        combined = content
        read_paths: set[str] = set()
        read_paths.add(os.path.normpath(os.path.join(ws, path.replace("/", os.sep))) if not os.path.isabs(path) else os.path.normpath(path))

        read_cache: dict[str, str] = {}

        # Follow imports from the target file
        import_candidates: list[tuple[str, str]] = []
        for imp_path in _parse_imports(content):
            if len(import_candidates) >= 5:
                break
            full = os.path.normpath(os.path.join(ws, imp_path))
            if os.path.isfile(full) and full not in read_paths:
                import_candidates.append((imp_path, full))

        import_parts = []
        for imp_path, full in import_candidates:
            try:
                if full in read_cache:
                    imp_content = read_cache[full]
                else:
                    imp_content = await agent.read_file(full, track_read=False)
                    read_cache[full] = imp_content
                if not imp_content.startswith("File not found:") and not imp_content.startswith("Error"):
                    import_parts.append((imp_path, imp_content))
                    read_paths.add(full)
            except Exception as e:
                print(f"  Warning: failed to read {imp_path}: {e}")

        initial_followed = [imp for imp, _ in import_parts]
        combined += "".join(f"\n\n# === {p} ===\n{c}" for p, c in import_parts)

        if initial_followed:
            print(f"  Round 0: followed imports — {', '.join(initial_followed)}")

        # Round 1: first answer
        system = "You are an expert code reviewer. Answer the question concisely using the provided code as reference."
        user = f"## Code:\n\n{combined}\n\n## Question:\n{question}" if question else f"## Code:\n\n{combined}\n\nAnalyze thoroughly."
        answer = await agent.llm.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])

        # Iterate: follow file references mentioned in each answer
        for round_num in range(2, 5):  # rounds 2, 3, 4
            refs = _parse_file_refs(answer)
            ref_candidates: list[tuple[str, str]] = []
            for ref in refs:
                if len(ref_candidates) >= 4:
                    break
                full = os.path.normpath(os.path.join(ws, ref))
                if os.path.isfile(full) and full not in read_paths:
                    ref_candidates.append((ref, full))

            ref_parts = []
            for ref, full in ref_candidates:
                try:
                    if full in read_cache:
                        ref_content = read_cache[full]
                    else:
                        ref_content = await agent.read_file(full, track_read=False)
                        read_cache[full] = ref_content
                    if not ref_content.startswith("File not found:") and not ref_content.startswith("Error"):
                        ref_parts.append((ref, ref_content))
                        read_paths.add(full)
                except Exception as e:
                    print(f"  Warning: failed to read {ref}: {e}")

            new_files = [ref for ref, _ in ref_parts]
            combined += "".join(f"\n\n# === {p} (referenced in previous answer) ===\n{c}" for p, c in ref_parts)

            if not new_files:
                break  # Nothing new to follow

            print(f"  Round {round_num}: followed references — {', '.join(new_files)}")

            answer = await agent.llm.chat([
                {"role": "system", "content": system},
                {"role": "user", "content": f"## All code seen so far:\n\n{combined}\n\n## Question:\n{question}\n\nYour previous answer mentioned files listed above as 'referenced in previous answer'. Now that you have their full content, deepen your analysis with more detail about these referenced files."},
            ])

        return answer
