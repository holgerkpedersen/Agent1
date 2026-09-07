"""Help command for agent interactive mode."""
from .base import Command

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Agent


class HelpCommand(Command):
    """Show all available commands with their help text."""

    @property
    def name(self) -> str:
        return "help"

    @property
    def help_text(self) -> str:
        return "help [command] - List all commands or show details for a specific command"

    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        from .registry import CommandRegistry
        # Build a fresh registry — same set of commands as run_interactive().
        registry = CommandRegistry()
        
        # Register all built-in commands (mirror _register_commands).
        from .read_cmd import ReadCommand
        from .write_cmd import WriteCommand
        from .search_cmd import SearchCommand
        from .clear_cmd import ClearCommand
        from .model_cmd import ModelCommand
        from .analyze_cmd import AnalyzeCommand
        from .plan_cmd import PlanCommand
        from .entities_cmd import EntitiesCommand
        from .taskplan_cmd import TaskplanCommand
        from .cleanup_cmd import CleanupCommand
        from .git_cmd import GitCommand
        from .implement_cmd import ImplementCommand
        from .fix_cmd import FixCommand
        from .workflow_cmd import WorkflowCommand
        from .optimize_cmd import OptimizeCommand
        from .perf_cmd import PerfCommand
        from .paste_cmd import PasteCommand
        from .paste_image_cmd import PasteImageCommand
        from .display_cmd import DisplayCommand
        from .decide_cmd import DecideCommand
        from .review_cmd import ReviewCommand
        from .run_cmd import RunCommand
        from .self_heal_cmd import SelfHealCommand
        from .reconstruct_cmd import ReconstructCommand
        from .multillm_cmd import MultiLlmCommand
        from .demo_data_cmd import DemoDataCommand
        from .mode_cmd import ModeCommand
        from .subagent_cmd import SubAgentCommand
        from .mcp_cmd import MCPCommand
        from .propose_cmd import ProposeCommand
        
        for cmd_cls in (ReadCommand, WriteCommand, SearchCommand, ClearCommand,
                        ModelCommand, AnalyzeCommand, PlanCommand, EntitiesCommand,
                        TaskplanCommand, CleanupCommand, GitCommand, ImplementCommand,
                        FixCommand, WorkflowCommand, OptimizeCommand, PerfCommand,
                        PasteCommand, PasteImageCommand, DisplayCommand, DecideCommand,
                        ReviewCommand, RunCommand, SelfHealCommand, ReconstructCommand,
                        MultiLlmCommand, DemoDataCommand, ModeCommand, SubAgentCommand,
                        MCPCommand, ProposeCommand, HelpCommand):
            registry.register(cmd_cls())

        if not args:
            # List all commands
            print("Available commands:")
            for name in sorted(registry.names()):
                cmd = registry.get(name)
                synopsis = cmd.help_text.splitlines()[0].strip() if cmd else name
                print(f"  {name:<25} {synopsis}")
        else:
            # Show help for a specific command
            cmd_name = args[0].lower()
            cmd = registry.get(cmd_name)
            if cmd:
                print(f"{cmd.name}:")
                for line in cmd.help_text.splitlines():
                    print(f"  {line}")
            else:
                # Suggest similar commands
                import difflib
                all_names = sorted(registry.names())
                matches = difflib.get_close_matches(cmd_name, all_names, n=3, cutoff=0.6)
                if matches:
                    print(f"Unknown command '{cmd_name}'. Did you mean:")
                    for m in matches:
                        cmd_m = registry.get(m)
                        synopsis = cmd_m.help_text.splitlines()[0].strip() if cmd_m else ""
                        print(f"  {m:<25} {synopsis}")
                else:
                    print(f"Unknown command '{cmd_name}'. Type 'help' for all commands.")
        return True
