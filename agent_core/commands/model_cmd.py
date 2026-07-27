"""Model command for agent interactive mode."""
import os
import difflib

from .base import Command
from agent_core.constants import KNOWN_MODELS

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Agent


class ModelCommand(Command):
    """Manage LLM models - list, reload, switch."""
    
    @property
    def name(self) -> str:
        return "model"
    
    @property
    def help_text(self) -> str:
        return "model [list|reload|name] - Manage LLM models"
    
    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        if len(args) < 1:
            self._list_models(agent)
            return True
        
        query = args[0].strip()
        
        if query == "list":
            self._list_models(agent)
            return True
        
        if query == "reload":
            await self._reload_model(agent)
            return True
        
        # Switch to matching model
        await self._switch_model(query, agent)
        return True
    
    def _list_models(self, agent: 'Agent'):
        """List all known models."""
        print(f"Current model: {agent.llm.model_name}")
        print(f"Known models ({len(KNOWN_MODELS)}):")
        for name, info in sorted(KNOWN_MODELS.items()):
            marker = " <- current" if name == agent.llm.model_name else ""
            print(f"  {name}{marker}\n    {info['desc']}")
    
    async def _reload_model(self, agent: 'Agent'):
        """Reload current model by cycling through alternate."""
        current = agent.llm.model_name
        alternates = [m for m in KNOWN_MODELS if m != current]
        if not alternates:
            print("No alternate model available for reload cycle.")
            return
        
        temp_model = alternates[0]
        print(f"Reloading {current} via {temp_model} cycle...")
        print(f"  1. Switching to {temp_model}...")
        agent.llm.model_name = temp_model
        try:
            await agent.llm.chat([{"role": "user", "content": "ping"}])
        except Exception:
            pass
        print(f"  2. Switching back to {current}...")
        agent.llm.model_name = current
        try:
            await agent.llm.chat([{"role": "user", "content": "ping"}])
        except Exception:
            pass
        print(f"Reload complete. Model: {current}")
    
    async def _switch_model(self, query: str, agent: 'Agent'):
        """Switch to a matching model."""
        # Find best match
        exact_match = next((m for m in KNOWN_MODELS if m == query), None)
        if exact_match:
            best_match = exact_match
        else:
            substring_matches = [m for m in KNOWN_MODELS if query.lower() in m.lower()]
            if substring_matches:
                best_match = substring_matches[0]
            else:
                close = difflib.get_close_matches(query, KNOWN_MODELS.keys(), n=1, cutoff=0.3)
                best_match = close[0] if close else None
        
        if not best_match:
            print(f"No match for '{query}'. Known models:")
            for name in KNOWN_MODELS:
                print(f"  {name}")
            return
        
        if best_match == agent.llm.model_name:
            print(f"Already using: {best_match}")
            return
        
        info = KNOWN_MODELS.get(best_match, {"desc": ""})
        print(f"Match: {best_match} - {info['desc']}")
        print(f"Switch from {agent.llm.model_name} to {best_match}? (y/n)")
        confirm = input().strip().lower()
        if confirm in ["y", "yes"]:
            agent.llm.model_name = best_match
            print(f"Model set to: {best_match}")
            self._save_to_env(best_match)
        else:
            print("Cancelled.")
    
    def _save_to_env(self, model_name: str):
        """Save model to .env file."""
        env_path = ".env"
        lines = []
        found = False
        if os.path.exists(env_path):
            with open(env_path, "r") as ef:
                lines = ef.readlines()
        with open(env_path, "w") as ef:
            for line in lines:
                if line.startswith("AGENT_MODEL="):
                    ef.write(f"AGENT_MODEL={model_name}\n")
                    found = True
                else:
                    ef.write(line)
            if not found:
                ef.write(f"\nAGENT_MODEL={model_name}\n")
        print(f"Saved to {env_path}")
