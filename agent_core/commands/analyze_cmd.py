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
        return "analyze <file> [analysis.md] - AI analysis via LM Studio"
    
    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        if len(args) < 1:
            self.error("Usage: analyze <path> [analysis.md]")
            return True
        
        path = args[0]
        output_file = args[1] if len(args) > 1 else None
        
        result = await agent.process_query(f"analyze {path}")
        
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"# Analysis of {path}\n\n")
                f.write(result)
            print(f"Analysis written to {output_file}")
        else:
            print(result)
        
        return True
