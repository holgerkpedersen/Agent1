"""Analyze command for agent interactive mode."""
from .base import Command

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Agent


class AnalyzeCommand(Command):
    """AI analysis of a file via LM Studio."""

    @property
    def name(self) -> str:
        return "analyze"

    @property
    def help_text(self) -> str:
        return "analyze <file> [analysis.md] [--desc \"question\"] — AI analysis via LM Studio"

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        parts = list(args)

        desc_text = None
        if "--desc" in parts:
            di = parts.index("--desc")
            if di + 1 < len(parts):
                desc_text = parts[di + 1].strip('"')
            parts = [p for p in parts if p not in (parts[di], args[di + 1] if di + 1 < len(args) else "")]

        if len(parts) < 1:
            self.error('Usage: analyze <path> [analysis.md] [--desc "question"]')
            return True

        path = parts[0]
        output_file = parts[1] if len(parts) > 1 else None

        if desc_text:
            content = await agent.read_file(path, track_read=False)
            if content.startswith("File not found:") or content.startswith("Error"):
                self.error(content)
                return True

            result = await agent.llm.chat([
                {"role": "system", "content": "You are an expert code reviewer. Answer the question concisely using only the provided code as reference."},
                {"role": "user", "content": f"## Code ({path}):\n\n{content}\n\n## Question:\n{desc_text}"},
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
