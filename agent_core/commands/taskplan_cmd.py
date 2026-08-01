"""Taskplan command for agent interactive mode."""
import os
import re

from .base import Command

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Agent


def _collision_scan(workspace: str) -> str:
    """Brief collision warning for LLM prompts."""
    if not os.path.isdir(workspace):
        return ""
    taken: dict[str, list[str]] = {}
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache", "backups")]
        for f in files:
            if not f.endswith(".py"):
                continue
            try:
                with open(os.path.join(root, f), "r", encoding="utf-8") as fh:
                    names = {m.group(1) for m in re.finditer(r'^(?:class|def)\s+(\w+)', fh.read(), re.MULTILINE) if not m.group(1).startswith("__")}
                if names:
                    rel_dir = os.path.relpath(root, workspace).replace("\\", "/").rstrip(".")
                    taken.setdefault(rel_dir, []).extend(sorted(names)[:20])
            except Exception:
                pass
    if not taken:
        return ""
    lines = ["",
             "CRITICAL: These class and function names already exist in the project.",
             "DO NOT create new files that define these same names in the same directory:",
    ]
    for d, names in sorted(taken.items())[:12]:
        lines.append(f"  {d or 'root'}: {', '.join(names[:12])}")
    lines.append("If you need to add functionality to these, modify the existing file.")
    return "\n\n".join(lines)


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
        
        # Build collision warning from workspace
        ws_dir = os.path.dirname(os.path.abspath(analysis_file))
        collision_warning = _collision_scan(ws_dir)

        messages = [
            {"role": "system", "content": "You are an expert project manager. Create a detailed task plan for implementing code changes. Break down work into concrete, actionable tasks with clear descriptions. Include task dependencies and priority.\n\nCRITICAL RULES:\n- PATH: Every file path MUST use `agent_core/`, `agent1/`, or `src/agent1/` prefix. BAD: bare filenames, bare `src/`, or any other directory.\n- SIZE: New files max 150 lines (SRP — single responsibility). Split large concepts across multiple focused files. Modify existing files only with minimal changes — do not suggest rewriting entire large files."},
            {"role": "user", "content": f"Create a task implementation plan from this analysis and plan:\n\n## Analysis:\n{analysis_content}\n\n## Plan:\n{plan_content}\n\n## Existing entities.py:\n{entities_content if entities_content else 'No entities.py found'}\n\nGenerate a tasks.md file with specific implementation tasks, organized by file, with clear steps for new and existing files. Ensure tasks respect the entity definitions in entities.py.{collision_warning}"}
        ]
        tasks = await agent.llm.chat(messages)
        
        with open(tasks_file, "w", encoding="utf-8") as f:
            f.write(f"# Implementation Tasks\n\n")
            f.write(tasks)
        print(f"Tasks written to {tasks_file}")
        
        return True
