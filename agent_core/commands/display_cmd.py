"""Display-mode command for agent interactive mode."""
from .base import Command

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Agent


class DisplayCommand(Command):
    """Show and set the REPL display mode (verbose|clean|quiet)."""

    @property
    def name(self) -> str:
        return "display"

    @property
    def help_text(self) -> str:
        return "display [verbose|clean|quiet] - Show/set NLP output verbosity"

    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        from agent_core.config import AgentDisplayMode, load_agent_settings
        env_path = ".env"

        modes = {m.value: m for m in AgentDisplayMode}

        try:
            settings = load_agent_settings()
            current = settings.display_mode
        except Exception:
            import os as _os
            raw = _os.environ.get("AGENT_DISPLAY_MODE", "").strip().lower() or "verbose"
            try:
                current = AgentDisplayMode(raw)
            except ValueError:
                current = AgentDisplayMode.VERBOSE

        if not args:
            print(f"Display mode is '{current.value}'.")
            print("Options: verbose (every call + result), clean (reason per call, summarized results), quiet (only final answer).")
            return True

        requested = str(args[0]).strip().lower()
        if requested not in modes:
            print(f"Unknown mode '{requested}'. Options: {', '.join(modes)}")
            return False

        # Persist the choice to .env so it survives REPL restarts.
        lines = []
        found = False
        import os as _os
        if _os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as ef:
                lines = ef.readlines()
        with open(env_path, "w", encoding="utf-8") as ef:
            for line in lines:
                if line.startswith("AGENT_DISPLAY_MODE="):
                    ef.write(f"AGENT_DISPLAY_MODE={requested}\n")
                    found = True
                else:
                    ef.write(line)
            if not found:
                ef.write(f"\n# NLP output verbosity (verbose|clean|quiet)\nAGENT_DISPLAY_MODE={requested}\n")

        print(f"Display mode set to '{requested}' (persisted to .env).")
        return True