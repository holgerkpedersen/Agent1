"""Plan command for agent interactive mode."""
from .base import Command

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Agent


class PlanCommand(Command):
    """Generate coding plan from analysis file."""
    
    @property
    def name(self) -> str:
        return "plan"
    
    @property
    def help_text(self) -> str:
        return "plan <analysis.md> <plan.md> - Generate coding plan from analysis"
    
    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        if len(args) < 2:
            self.error("Usage: plan <analysis.md> <plan.md>")
            return True
        
        analysis_file = args[0]
        plan_file = args[1]
        
        try:
            with open(analysis_file, "r", encoding="utf-8") as f:
                analysis_content = f.read()
        except FileNotFoundError:
            self.error(f"File not found: {analysis_file}")
            return True
        
        messages = [
            {"role": "system", "content": "You are an expert software architect. Based on the code analysis provided, create a detailed coding plan with specific implementation steps, prioritized by impact and dependencies."},
            {"role": "user", "content": f"Create a coding plan based on this analysis:\n\n{analysis_content}"}
        ]
        plan = await agent.llm.chat(messages)
        
        with open(plan_file, "w", encoding="utf-8") as f:
            f.write("# Coding Plan\n\n")
            f.write(plan)
        print(f"Coding plan written to {plan_file}")
        
        return True