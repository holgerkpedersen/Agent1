"""Taskplan command for agent interactive mode."""
import os
import re

from .base import Command, auto_choice
from .doc_paths import find_input, resolve_output
from .plan_verifier import check_doc, apply_report, summarize

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Agent


def _detect_subpackages(workspace: str) -> list[str]:
    """Find existing subpackage directories (containing __init__.py)."""
    if not os.path.isdir(workspace):
        return []
    pkgs = set()
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', '.pytest_cache')]
        if '__init__.py' in files:
            rel = os.path.relpath(root, workspace).replace('\\', '/')
            pkgs.add(rel if rel != '.' else root.rsplit(os.sep, 1)[-1])
    return sorted(pkgs)


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
                print(f"Warning: Failed to scan {root}/{f}")
    if not taken:
        return ""
    lines = ["",
             "CRITICAL: These class and function names already exist in the project.",
             "DO NOT create new files that define these same names in the same directory:",
    ]
    lines += [f"  {d or 'root'}: {', '.join(names[:12])}" for d, names in sorted(taken.items())[:12]]
    lines.append("If you need to add functionality to these, modify the existing file.")
    return "\n\n".join(lines)


class TaskplanCommand(Command):
    """Generate implementation task plan."""
    
    @property
    def name(self) -> str:
        return "taskplan"
    
    @property
    def help_text(self) -> str:
        return ("taskplan <analysis.md> <plan.md> [tasks.md] - Generate implementation tasks\n"
                "  Output is regression-checked (paths, duplicate definitions);\n"
                "  flagged claims pause for confirmation unless --force.")

    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        if len(args) < 2:
            self.error("Usage: taskplan <analysis.md> <plan.md> [tasks.md]")
            return True

        force = "--force" in args
        clean_args = [a for a in args if a != "--force"]
        if not clean_args:
            self.error("Usage: taskplan <analysis.md> <plan.md> [tasks.md]")
            return True

        ws = agent.workspace
        analysis_file = find_input(ws, clean_args[0])
        plan_file = find_input(ws, clean_args[1])
        # Bare output filenames go to .docs/<timestamp>/ (the input's run
        # folder when it has one) — explicit paths are kept.
        tasks_file = resolve_output(ws, clean_args[2] if len(clean_args) > 2 else "tasks.md",
                                    sibling_of=analysis_file)
        
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
        
        # Build collision warning from workspace — the analysis file may live
        # in a .docs/<timestamp>/ run folder, which is not the workspace.
        analysis_parent = os.path.dirname(os.path.abspath(analysis_file))
        if os.path.basename(os.path.dirname(analysis_parent)) == ".docs":
            ws_dir = ws
        else:
            ws_dir = analysis_parent
        collision_warning = _collision_scan(ws_dir)

        # Dynamic path rules
        pkgs = _detect_subpackages(ws_dir)
        if not pkgs:
            pkgs = ["agent_core", "agent1", "src/agent1"]
        pkg_list = "`, `".join(pkgs[:5])
        path_rule = f"\n\nPATH RULES: New files MUST use a sub-package prefix (`{pkg_list}/`). BAD: bare filenames or `src/` without subdirectory.\nSIZE RULES: New files max 150 lines (SRP). Split large concepts. Modifying: minimal changes only."

        messages = [
            {"role": "system", "content": "You are an expert project manager. Create a detailed task plan for implementing code changes. Break down work into concrete, actionable tasks with clear descriptions. Include task dependencies and priority.\n\nFollow the PATH and SIZE rules in the prompt below."},
            {"role": "user", "content": f"Create a task implementation plan from this analysis and plan:\n\n## Analysis:\n{analysis_content}\n\n## Plan:\n{plan_content}\n\n## Existing entities.py:\n{entities_content if entities_content else 'No entities.py found'}\n\nGenerate a tasks.md file with specific implementation tasks, organized by file, with clear steps for new and existing files. Ensure tasks respect the entity definitions in entities.py.{collision_warning}{path_rule}"}
        ]
        tasks = await agent.llm.chat(messages)

        content = f"# Implementation Tasks\n\n{tasks}"
        result = check_doc("taskplan", content, ws)
        summarize(result, "taskplan")
        if not result.clean and not force:
            answer = auto_choice(
                "  Taskplan has unverifiable/colliding claims — write anyway? (y/N): ",
                default="n", auto_default="n",
            )
            if answer.strip().lower() not in ("y", "yes"):
                print("[taskplan] Halted — regenerate with corrected paths "
                      "(or rerun with --force).")
                return True
        content = apply_report(content, result)

        with open(tasks_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Tasks written to {tasks_file}")

        return True