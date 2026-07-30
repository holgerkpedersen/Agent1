"""Analyze command for agent interactive mode."""
import os
import re

from .base import Command

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
            # Top-level package import ('from agent_core import X') → __init__.py
            if path == top:
                path += "/__init__.py"
            else:
                path += ".py"
            result.append(path)
    return sorted(set(result))


class AnalyzeCommand(Command):
    """AI analysis of a file via LM Studio — follows imports with --desc."""

    @property
    def name(self) -> str:
        return "analyze"

    @property
    def help_text(self) -> str:
        return 'analyze <file> [analysis.md] [--desc "question"] [--deep] — AI analysis via LM Studio'

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        parts = list(args)

        desc_text = None
        deep_mode = False
        if "--desc" in parts:
            di = parts.index("--desc")
            if di + 1 < len(parts):
                desc_text = parts[di + 1].strip('"')
                deep_mode = True  # --desc automatically follows imports
            parts = [p for p in parts if p not in (parts[di], args[di + 1] if di + 1 < len(args) else "")]

        if "--deep" in parts:
            deep_mode = True
            parts = [p for p in parts if p != "--deep"]

        if len(parts) < 1:
            self.error('Usage: analyze <path> [analysis.md] [--desc "question"] [--deep]')
            return True

        path = parts[0]
        output_file = parts[1] if len(parts) > 1 else None

        if desc_text:
            content = await agent.read_file(path, track_read=False)
            if content.startswith("File not found:") or content.startswith("Error"):
                self.error(content)
                return True

            combined = content
            related: list[str] = []

            if deep_mode:
                imports = _parse_imports(content)
                base_dir = os.path.dirname(os.path.abspath(path))
                ws = agent.workspace.replace("/c/", "C:/").replace("\\", "/")

                for imp_path in imports:
                    if len(related) >= 5:
                        break
                    full = os.path.normpath(os.path.join(ws, imp_path))
                    if os.path.isfile(full) and full != os.path.abspath(path):
                        try:
                            imp_content = await agent.read_file(full, track_read=False)
                            if not imp_content.startswith("File not found:") and not imp_content.startswith("Error"):
                                combined += f"\n\n# === {imp_path} ===\n{imp_content}"
                                related.append(imp_path)
                        except Exception:
                            pass

                if related:
                    print(f"  Followed imports: {', '.join(related)}")

            result = await agent.llm.chat([
                {"role": "system", "content": "You are an expert code reviewer. Answer the question concisely using the provided code as reference."},
                {"role": "user", "content": f"## Code:\n\n{combined}\n\n## Question:\n{desc_text}"},
            ])
        elif deep_mode:
            content = await agent.read_file(path, track_read=False)
            if content.startswith("File not found:") or content.startswith("Error"):
                self.error(content)
                return True

            combined = content
            imports = _parse_imports(content)
            ws = agent.workspace.replace("/c/", "C:/").replace("\\", "/")
            follow_count = 0

            for imp_path in imports:
                if follow_count >= 5:
                    break
                full = os.path.normpath(os.path.join(ws, imp_path))
                if os.path.isfile(full) and full != os.path.abspath(path):
                    try:
                        imp_content = await agent.read_file(full, track_read=False)
                        if not imp_content.startswith("File not found:") and not imp_content.startswith("Error"):
                            combined += f"\n\n# === {imp_path} ===\n{imp_content}"
                            follow_count += 1
                    except Exception:
                        pass

            if follow_count:
                print(f"  Followed imports: {follow_count} files")

            result = await agent.llm.chat([
                {"role": "system", "content": "You are an expert code reviewer. Analyze this code thoroughly."},
                {"role": "user", "content": f"## Code:\n\n{combined}"},
            ])
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
