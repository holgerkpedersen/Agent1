"""Taskplan command for agent interactive mode."""
import os

from .base import Command

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Agent


class TaskplanCommand(Command):
    """Generate implementation task plan."""
    
    @property
    def name(self) -> str:
        return "taskplan"
    
    @property
    def help_text(self) -> str:
        return "taskplan <analysis.md> <plan.md> [tasks.md] - Generate implementation tasks"
    
    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        if len(args) < 2:
            self.error("Usage: taskplan <analysis.md> <plan.md> [tasks.md]")
            return True
        
        analysis_file = args[0]
        plan_file = args[1]
        tasks_file = args[2] if len(args) > 2 else "tasks.md"
        
        try:
            with open(analysis_file, "r", encoding="utf-8") as f:
                analysis_content = f.read()
            with open(plan_file, "r", encoding="utf-8") as f:
                plan_content = f.read()
        except FileNotFoundError as e:
            self.error(f"File not found: {e}")
            return True
        
        # Check for existing entities.py
        entities_content = ""
        entities_py = os.path.join(os.path.dirname(analysis_file), "entities.py")
        if os.path.exists(entities_py):
            with open(entities_py, "r", encoding="utf-8") as f:
                entities_content = f.read()
        
        messages = [
            {"role": "system", "content": "You are an expert project manager. Create a detailed task plan for implementing code changes. Break down work into concrete, actionable tasks with clear descriptions. Include task dependencies and priority.\n\nIMPORTANT: All new files MUST use sub-package paths (e.g. `agent_core/thing.py`, `agent1/module.py`). Never list bare filenames like `types.py` or `config.py` at workspace root — they shadow stdlib modules."},
            {"role": "user", "content": f"Create a task implementation plan from this analysis and plan:\n\n## Analysis:\n{analysis_content}\n\n## Plan:\n{plan_content}\n\n## Existing entities.py:\n{entities_content if entities_content else 'No entities.py found'}\n\nGenerate a tasks.md file with specific implementation tasks, organized by file, with clear steps for new and existing files. Ensure tasks respect the entity definitions in entities.py."}
        ]
        tasks = await agent.llm.chat(messages)
        
        with open(tasks_file, "w", encoding="utf-8") as f:
            f.write(f"# Implementation Tasks\n\n")
            f.write(tasks)
        print(f"Tasks written to {tasks_file}")
        
        return True
