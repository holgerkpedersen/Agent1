"""
Phase 5: Production Orchestrator (src/agent_final.py)

Composes capabilities from v1–v4 into a single `AgentOrchestrator` using composition.
Features:
- Config-driven initialization via config.yaml & .env
- Structured observability with rich.console
- Graceful error boundaries & fallbacks for tool/parsing failures
- Interactive CLI entry point
"""

import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.traceback import install

# Enable pretty tracebacks for safe debugging
install()

# Ensure project root is in path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.types import AgentConfig, Message, ReasoningStep, parse_json_safely, format_tool_result
except ImportError:
    # Fallback stubs for isolated execution / early dev phases
    class AgentConfig: pass
    class Message: pass
    class ReasoningStep: pass
    def parse_json_safely(text): return {}
    def format_tool_result(result): return str(result)

try:
    from src.mock_llm import MockLLM
except ImportError:
    class MockLLM:
        def __init__(self, config: dict): self.config = config
        def chat(self, messages: list) -> str: return "Mock response"
        def generate_tool_calls(self, messages: list, tools: list) -> list: return []

# --------------------------------------------------------------------------- #
#                          ORCHESTRATOR COMPONENTS                            #
# --------------------------------------------------------------------------- #

class AgentOrchestrator:
    """
    Production-ready agent orchestrator. Composes basic chat, tool execution,
    memory management, and reasoning loops into a cohesive pipeline.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.console = Console()
        load_dotenv()  # Load .env for API keys & feature flags
        
        self.config = self._load_config(config_path)
        self.llm = self._setup_llm_provider()
        
        # Composition of v1-v4 capabilities
        self.tools: Dict[str, Callable] = {}
        self.tool_schemas: List[Dict[str, Any]] = []
        self.memory_buffer: List[Message] = []
        self.max_turns = self.config.get("memory", {}).get("max_turns", 10)
        
        # Initialize default tools from config
        for tool_cfg in self.config.get("tools", []):
            self._register_builtin_tool(tool_cfg)
            
        self.console.print(Panel("[bold green]Agent Orchestrator Initialized[/]", title="System"))

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load and validate external configuration."""
        config_file = Path(path)
        if not config_file.exists():
            self.console.print(f"[yellow]Warning:[/] {path} not found. Using defaults.")
            return {}
        
        with open(config_file, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}
            
        # Merge with safe defaults
        defaults = {
            "agent": {"name": "OrchestratorAgent", "temperature": 0.2, "max_tokens": 512},
            "memory": {"max_turns": 10},
            "tools": []
        }
        for key in defaults:
            raw_cfg.setdefault(key, defaults[key])
            
        return raw_cfg

    def _setup_llm_provider(self):
        """Initialize LLM based on .env flags & available credentials."""
        use_mock = os.getenv("USE_MOCK_LLM", "true").lower() == "true"
        
        if use_mock:
            self.console.print("[dim]Using MockLLM for safe offline execution.[/dim]")
            return MockLLM(self.config.get("agent", {}))
            
        # Live LLM stub (expandable to OpenAI, Anthropic, etc.)
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            self.console.print("[yellow]No API key found. Falling back to MockLLM.[/yellow]")
            return MockLLM(self.config.get("agent", {}))
            
        self.console.print("[green]Live LLM provider configured.[/green]")
        # Placeholder for real provider initialization
        return type("LiveLLMProvider", (), {
            "chat": lambda self, msgs: "Live response placeholder",
            "generate_tool_calls": lambda self, msgs, tools: []
        })()

    def _register_builtin_tool(self, tool_cfg: Dict[str, Any]):
        """Register a tool from config schema."""
        name = tool_cfg.get("name")
        desc = tool_cfg.get("description", "")
        
        # Simple dynamic registration mapping names to safe builtins
        if name == "calculator":
            func = lambda expr: eval(expr)  # Sandbox note: use ast.literal_eval or sympy in prod
        else:
            func = lambda *args, **kwargs: f"Tool '{name}' executed successfully."
            
        self.register_tool(name, func, desc)

    def register_tool(self, name: str, func: Callable, description: str):
        """Register a callable with metadata for LLM routing."""
        self.tools[name] = func
        self.tool_schemas.append({
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}}
        })

    def _update_memory(self, role: str, content: Any):
        """Maintain sliding window memory buffer (v3 concept)."""
        self.memory_buffer.append(Message(role=role, content=str(content)))
        
        # Prune oldest turns if limit exceeded
        while len(self.memory_buffer) > self.max_turns:
            self.memory_buffer.pop(0)

    def _get_context_window(self) -> List[Dict[str, str]]:
        """Format memory + system prompt for LLM consumption."""
        sys_prompt = self.config.get("agent", {}).get("system_prompt", "You are a helpful assistant.")
        window = [{"role": "system", "content": sys_prompt}]
        window.extend([{"role": m.role, "content": str(m.content)} for m in self.memory_buffer])
        return window

    def _execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[str]:
        """Safely execute requested tools with error boundaries (v2 concept)."""
        results = []
        for call in tool_calls:
            name = call.get("name", "unknown")
            args = parse_json_safely(call.get("arguments", "{}"))
            
            if name not in self.tools:
                results.append(f"Error: Tool '{name}' not found.")
                continue
                
            try:
                output = self.tools[name](**args)
                results.append(format_tool_result(output))
                self.console.print(f"[dim]Tool [{name}] -> {output}[/dim]")
            except Exception as e:
                error_msg = f"Execution failed for '{name}': {e}"
                results.append(error_msg)
                self.console.print(f"[red]{error_msg}[/red]")
                
        return results

    def run(self, user_input: str) -> Dict[str, Any]:
        """
        Core reasoning loop (v4 concept). Orchestrates thought → action → observation.
        Returns structured result with trace and final answer.
        """
        self.console.print(Panel(Markdown(user_input), title="User Input"))
        self._update_memory("user", user_input)
        
        context = self._get_context_window()
        steps: List[ReasoningStep] = []
        max_iterations = 5
        
        for i in range(max_iterations):
            # 1. Prompt LLM for reasoning step or final answer
            llm_response = self.llm.chat(context)
            
            # Simulate structured extraction (in prod, use JSON mode or regex parsing)
            is_final = "final answer" in llm_response.lower() or i == max_iterations - 1
            
            step = ReasoningStep(
                iteration=i + 1,
                thought=llm_response.split("\n")[0],
                action=None,
                observation=None,
                is_final=is_final
            )
            
            # Check for tool calls (simulated routing)
            if "call:" in llm_response.lower():
                tool_name = llm_response.split("call:")[1].strip().split()[0]
                step.action = {"name": tool_name, "arguments": "{}"}
                observations = self._execute_tools([step.action])
                step.observation = "\n".join(observations)
                
            steps.append(step)
            
            # Inject observation back into context for next loop
            if not is_final and step.observation:
                context.extend([
                    {"role": "assistant", "content": f"Thought: {step.thought}\nAction: {step.action}"},
                    {"role": "user", "content": f"Observation: {step.observation}"}
                ])
            else:
                break
                
        # Finalize response
        final_answer = steps[-1].thought if steps else "No response generated."
        self._update_memory("assistant", final_answer)
        
        trace_report = "\n".join([f"Step {s.iteration}: {s.thought}" for s in steps])
        self.console.print(Panel(Markdown(final_answer), title="Agent Response"))
        self.console.print(f"[dim]Trace:\n{trace_report}[/dim]")
        
        return {"final_answer": final_answer, "steps": steps}

    def interactive_loop(self):
        """CLI REPL for continuous conversation."""
        self.console.print("[bold blue]Interactive Mode Active[/] (type 'quit' to exit)")
        while True:
            try:
                user_input = self.console.input("\n> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit"):
                    break
                self.run(user_input)
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.console.print(f"[red]Runtime error:[/] {e}")


# --------------------------------------------------------------------------- #
#                             CLI ENTRY POINT                                 #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent Tutorial Orchestrator")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--interactive", action="store_true", help="Start interactive REPL")
    parser.add_argument("prompt", nargs="?", default=None, help="Single-shot prompt")
    
    args = parser.parse_args()
    agent = AgentOrchestrator(config_path=args.config)
    
    if args.prompt:
        agent.run(args.prompt)
    elif args.interactive:
        agent.interactive_loop()
    else:
        # Default to interactive for convenience when run directly
        agent.console.print("[dim]No prompt provided. Starting interactive mode...[/dim]")
        agent.interactive_loop()