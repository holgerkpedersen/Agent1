"""Entities command for agent interactive mode."""
from .base import Command, auto_choice
from .doc_paths import find_input, resolve_output
from .plan_verifier import check_doc, apply_report, summarize

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Agent


class EntitiesCommand(Command):
    """Generate shared entities from analysis and plan."""

    @property
    def name(self) -> str:
        return "entities"

    @property
    def help_text(self) -> str:
        return ("entities <analysis.md> <plan.md> [entities.md] - Generate shared entities\n"
                "  Output is regression-checked (python blocks parse, names unique);\n"
                "  flagged claims pause for confirmation unless --force.")

    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        if len(args) < 2:
            self.error("Usage: entities <analysis.md> <plan.md> [entities.md]")
            return True

        force = "--force" in args
        clean_args = [a for a in args if a != "--force"]
        if not clean_args:
            self.error("Usage: entities <analysis.md> <plan.md> [entities.md]")
            return True

        ws = agent.workspace
        analysis_file = find_input(ws, clean_args[0])
        plan_file = find_input(ws, clean_args[1])
        # Bare output filenames go to .docs/<timestamp>/ (the input's run
        # folder when it has one) — explicit paths are kept.
        entities_file = resolve_output(ws, clean_args[2] if len(clean_args) > 2 else "entities.md",
                                       sibling_of=analysis_file)

        try:
            with open(analysis_file, "r", encoding="utf-8") as f:
                analysis_content = f.read()
            with open(plan_file, "r", encoding="utf-8") as f:
                plan_content = f.read()
        except FileNotFoundError as e:
            self.error(f"File not found: {e}")
            return True

        messages = [
            {"role": "system", "content": "Create entities.md with ONLY Python code — no intro text. Start with ```python. All types must be valid — no unbound TypeVars, no forward-ref errors. Must pass mypy strict. Avoid circular imports."},
            {"role": "user", "content": f"Extract and define all shared entities from this analysis and plan:\n\n## Analysis:\n{analysis_content}\n\n## Plan:\n{plan_content}\n\nCreate an entities.md file with Python-ready entity definitions that can be centralized in an entities.py file for import across the project."}
        ]
        entities = await agent.llm.chat(messages)

        content = f"# Shared Entities\n\n{entities}"
        result = check_doc("entities", content, ws)
        summarize(result, "entities")
        if not result.clean and not force:
            answer = auto_choice(
                "  Entities have unparsable/duplicated definitions — write anyway? (y/N): ",
                default="n", auto_default="n",
            )
            if answer.strip().lower() not in ("y", "yes"):
                print("[entities] Halted — regenerate valid python blocks "
                      "(or rerun with --force).")
                return True
        content = apply_report(content, result)

        with open(entities_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Entities written to {entities_file}")

        return True
