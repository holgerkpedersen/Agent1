import os
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.traceback import install

# Enable rich traceback for better debugging during development
install(show_locals=True)

# Internal module imports (Phase 2 & Phase 4 dependencies)
from src.types import (
    AgentConfig,
    Message,
    ToolDefinition,
    build_context_window,
    format_tool_result,
)
from src.mock_llm import MockLLM
from src.agent_v1_basic import BasicAgent
from src.agent_v2_tools import ToolAgent
from src.agent_v3_memory import MemoryAgent
from src.agent_v4_reasoning import ReasoningAgent

# Resolve project root relative to this file (src/agent_final.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class AgentOrchestrator:
    """
    Production orchestrator that composes capabilities from v1-v4 agents.
    Uses delegation instead of inheritance to maintain loose coupling and 
    testability across different agent versions.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.console = Console()
        load_dotenv(PROJECT_ROOT / ".env")

        # 1. Config-driven initialization
        self.config = self._load_config(config_path or DEFAULT_CONFIG_PATH)

        # 2. Initialize LLM Provider based on environment flags
        use_mock = os.getenv("USE_MOCK_LLM", "true").lower() == "true"
        if use_mock:
            self.llm = MockLLM(temperature=self.config.temperature)
            self.console.print("[dim]🔧 Using MockLLM provider (safe for development/testing)[/dim]")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                self.console.print("[yellow]⚠️ No API key found. Falling back to MockLLM.[/yellow]")
                self.llm = MockLLM(temperature=self.config.temperature)
            else:
                raise NotImplementedError("Live LLM provider integration is pending in this tutorial.")

        # 3. Compose features via delegation (v1 -> v4)
        self.basic_agent = BasicAgent(llm=self.llm, system_prompt=self.config.system_prompt)
        self.tool_agent = ToolAgent(llm=self.llm)
        self.memory_agent = MemoryAgent(max_turns=self.config.max_memory_turns)
        self.reasoning_agent = ReasoningAgent(
            llm=self.llm, 
            max_steps=self.config.max_reasoning_steps
        )

        # 4. Observability & State
        self.conversation_history: List[Message] = []
        self.console.rule("[bold cyan]🚀 Agent Orchestrator Initialized[/bold cyan]")

    def _load_config(self, path: Path) -> AgentConfig:
        """Load and validate configuration from YAML with safe fallbacks."""
        if not path.exists():
            self.console.print(f"[red]❌ Config file not found at {path}. Using defaults.[/red]")
            return AgentConfig()

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
                
            agent_cfg = raw.get("agent", {})
            self.console.print(f"[dim]📄 Loaded config from {path.name}[/dim]")
            return AgentConfig(**agent_cfg)
        except Exception as e:
            self.console.print(f"[red]⚠️ Failed to parse config: {e}. Using defaults.[/red]")
            return AgentConfig()

    def register_tool(self, name: str, func: Callable, description: str):
        """Expose tool registration to external users."""
        definition = ToolDefinition(name=name, description=description, function=func)
        self.tool_agent.register_tool(definition)
        self.console.print(f"[green]✅ Registered tool:[/green] {name}")

    def run(self, user_input: str) -> str:
        """Orchestrate a single interaction turn with error boundaries."""
        if not user_input.strip():
            return ""

        self.console.print(f"[bold blue]👤 User:[/bold blue] {user_input}")
        
        # Inject new message into memory buffer (v3)
        user_msg = Message(role="user", content=user_input)
        self.memory_agent.add_message(user_msg)

        try:
            # Retrieve context window respecting token limits (Phase 2 util)
            context_window = build_context_window(
                self.memory_agent.get_history(), 
                limit=self.config.max_tokens // 4
            )

            if self.config.enable_reasoning and len(self.tool_agent.tools) > 0:
                # ReAct Loop with Tools (v4 + v2)
                trace = self.reasoning_agent.execute(context_window, list(self.tool_agent.tools.values()))
                assistant_content = trace.final_answer or "Reasoning completed without explicit answer."
                
            elif self.config.enable_reasoning:
                # Pure Reasoning (v4)
                trace = self.reasoning_agent.execute(context_window, [])
                assistant_content = trace.final_answer or "Reasoning completed."

            elif len(self.tool_agent.tools) > 0:
                # Tool Execution without explicit ReAct loop (v2)
                result = self.tool_agent.execute(context_window)
                assistant_content = format_tool_result(result)

            else:
                # Fallback to Basic Chat (v1)
                response = self.basic_agent.chat(user_input)
                assistant_content = response

            # Persist assistant reply in memory (v3)
            assistant_msg = Message(role="assistant", content=assistant_content)
            self.memory_agent.add_message(assistant_msg)

            self.console.print(f"[bold green]🤖 Assistant:[/bold green] {assistant_content}")
            return assistant_content

        except Exception as e:
            error_msg = f"Orchestration failed: {str(e)}"
            self.console.print(f"[bold red]⚠️ Error:[/bold red] {error_msg}")
            
            # Graceful degradation to v1 basic agent on failure
            fallback = self.basic_agent.chat(user_input)
            return fallback

    def interactive_loop(self):
        """CLI driver for continuous conversation."""
        self.console.print("Type [bold]'exit'[/bold] or [bold]'quit'[/bold] to end the session.\n")
        
        while True:
            try:
                user_input = input("[dim]> [/dim]").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    self.console.print("[bold yellow]👋 Session ended.[/bold yellow]")
                    break
                self.run(user_input)
            except KeyboardInterrupt:
                self.console.print("\n[bold yellow]⚠️ Interrupted by user.[/bold yellow]")
                break


def main():
    """Entry point for CLI execution."""
    orchestrator = AgentOrchestrator()
    
    # Example: Register a built-in tool dynamically (if not in config)
    def get_weather(location: str) -> str:
        return f"The weather in {location} is sunny and 25°C."
        
    orchestrator.register_tool("get_weather", get_weather, "Get current weather for a location")
    
    orchestrator.interactive_loop()


if __name__ == "__main__":
    main()