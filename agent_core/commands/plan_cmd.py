"""Plan command for agent interactive mode."""
from .base import Command, auto_choice
from .doc_paths import find_input, resolve_output
from .plan_verifier import check_doc, apply_report, summarize

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
        return ("plan <analysis.md> <plan.md> - Generate coding plan from analysis\n"
                "  Output is regression-checked (paths exist or are marked new);\n"
                "  flagged claims pause for confirmation unless --force.")

    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        if len(args) < 2:
            self.error("Usage: plan <analysis.md> <plan.md>")
            return True

        force = "--force" in args
        clean_args = [a for a in args if a != "--force"]
        if not clean_args:
            self.error("Usage: plan <analysis.md> <plan.md>")
            return True

        # Bare input filenames fall back to the newest .docs run folder.
        analysis_file = find_input(agent.workspace, clean_args[0])
        # Bare output filenames go to .docs/<timestamp>/ (the input's run
        # folder when it has one) — explicit paths are kept.
        plan_file = resolve_output(agent.workspace, clean_args[1], sibling_of=analysis_file)

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

        content = f"# Coding Plan\n\n{plan}"
        result = check_doc("plan", content, agent.workspace)
        summarize(result, "plan")
        if not result.clean and not force:
            answer = auto_choice(
                "  Plan references unverifiable paths — write anyway? (y/N): ",
                default="n", auto_default="n",
            )
            if answer.strip().lower() not in ("y", "yes"):
                print("[plan] Halted — regenerate with corrected paths "
                      "(or rerun with --force).")
                return True
        content = apply_report(content, result)

        with open(plan_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Coding plan written to {plan_file}")

        return True
