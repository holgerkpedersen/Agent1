"""Analyze command for agent interactive mode."""
import os
import re

from .base import Command
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
                path += "/__init__.py"
            else:
                path += ".py"
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


class AnalyzeCommand(Command):
    """AI analysis of a file via LM Studio — follows imports and iterates with --desc."""

    @property
    def name(self) -> str:
        return "analyze"

    @property
    def help_text(self) -> str:
        return 'analyze <file> [--desc "q"] [--stdin] [--deep] — AI analysis via LM Studio'

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
            print("Paste text to analyze, then press Enter on an empty line:")
            lines = []
            while True:
                try:
                    line = input()
                    if not line.strip():
                        break
                    lines.append(line)
                except EOFError:
                    break
            content = "\n".join(lines)
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
            self.error('Usage: analyze <path> [--desc "q"] [--stdin] [--deep]')
            return True

        path = parts[0]
        output_file = parts[1] if len(parts) > 1 else None

        if deep_mode:
            result = await self._deep_analyze(path, desc_text, agent)
        else:
            result = await agent.process_query(f"analyze {path}")

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"# Analysis of {path}\n\n")
                f.write(result)
            print(f"Analysis written to {output_file}")
        else:
            print(result)

        return True

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

        # Follow imports from the target file
        initial_followed: list[str] = []
        for imp_path in _parse_imports(content):
            if len(initial_followed) >= 5:
                break
            full = os.path.normpath(os.path.join(ws, imp_path))
            if os.path.isfile(full) and full not in read_paths:
                try:
                    imp_content = await agent.read_file(full, track_read=False)
                    if not imp_content.startswith("File not found:") and not imp_content.startswith("Error"):
                        combined += f"\n\n# === {imp_path} ===\n{imp_content}"
                        read_paths.add(full)
                        initial_followed.append(imp_path)
                except Exception:
                    pass

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
            new_files: list[str] = []

            for ref in refs:
                if len(new_files) >= 4:
                    break
                full = os.path.normpath(os.path.join(ws, ref))
                if os.path.isfile(full) and full not in read_paths:
                    try:
                        ref_content = await agent.read_file(full, track_read=False)
                        if not ref_content.startswith("File not found:") and not ref_content.startswith("Error"):
                            combined += f"\n\n# === {ref} (referenced in previous answer) ===\n{ref_content}"
                            read_paths.add(full)
                            new_files.append(ref)
                    except Exception:
                        pass

            if not new_files:
                break  # Nothing new to follow

            print(f"  Round {round_num}: followed references — {', '.join(new_files)}")

            answer = await agent.llm.chat([
                {"role": "system", "content": system},
                {"role": "user", "content": f"## All code seen so far:\n\n{combined}\n\n## Question:\n{question}\n\nYour previous answer mentioned files listed above as 'referenced in previous answer'. Now that you have their full content, deepen your analysis with more detail about these referenced files."},
            ])

        return answer
