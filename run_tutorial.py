#!/usr/bin/env python3
"""
CLI Driver for AI Agent Tutorial
Executes progressive agent implementations with rich console output.
Usage:
    python run_tutorial.py --step all   # Run full tutorial
    python run_tutorial.py --step 2    # Jump to tools
    python run_tutorial.py --dry-run   # Show code without executing
    python run_tutorial.py --test      # Run pytest validation suite
"""

import argparse
import sys
import os
import subprocess
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.markdown import Markdown
import yaml

# Ensure the project root is in the Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

console = Console()


def load_config():
    """Load externalized configuration from config.yaml"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
    if not os.path.exists(config_path):
        console.print("[yellow]⚠️  config.yaml not found. Using defaults.[/yellow]")
        return {"agent": {"system_prompt": "You are a helpful assistant."}}
    
    with open(config_path) as f:
        return yaml.safe_load(f)


# Pedagogical step definitions mapping to source files
STEPS = {
    1: {
        "name": "Basic Agent Loop",
        "module": "src.agent_v1_basic",
        "cls": "BasicAgent",
        "prompt": "What is the result of 2 + 2?",
        "description": "Teaches anatomy: prompt → LLM call → output parsing."
    },
    2: {
        "name": "Tool-Using Agent",
        "module": "src.agent_v2_tools",
        "cls": "ToolAgent",
        "prompt": "Calculate the square root of 144.",
        "description": "Adds function calling, argument parsing, & execution."
    },
    3: {
        "name": "Memory & Context Agent",
        "module": "src.agent_v3_memory",
        "cls": "MemoryAgent",
        "prompt_1": "My name is Alice. Remember this.",
        "prompt_2": "What is my name?",
        "description": "Manages conversation history & sliding context windows."
    },
    4: {
        "name": "Reasoning (ReAct) Agent",
        "module": "src.agent_v4_reasoning",
        "cls": "ReasoningAgent",
        "prompt": "Find the capital of France and calculate its area in sq km.",
        "description": "Implements planning → tool use → reflection loops."
    },
    5: {
        "name": "Production Orchestrator",
        "module": "src.agent_final",
        "cls": "AgentOrchestrator",
        "prompt": "System status check. Calculate 10*5 and summarize.",
        "description": "Composes all features with logging & error handling."
    }
}


def run_step(step_num: int, config: dict, dry_run: bool = False) -> bool:
    """Execute a single tutorial step or simulate it."""
    step_info = STEPS[step_num]
    
    console.print(f"\n[bold cyan]📘 Step {step_num}: {step_info['name']}[/bold cyan]")
    console.print(f"[dim]{step_info['description']}[/dim]")
    
    if dry_run:
        console.print(Panel(
            f"• Import: `{step_info['module']}`\n"
            f"• Class:  `{step_info['cls']}`\n"
            f"• Prompt: \"{step_info.get('prompt', step_info.get('prompt_1'))}\"\n"
            f"\n[yellow]Skipping execution (Dry Run Mode)[/yellow]",
            title="Simulation", border_style="yellow"
        ))
        return True

    try:
        # Dynamic import to gracefully handle missing implementations during early phases
        module = __import__(step_info["module"], fromlist=[step_info["cls"]])
        agent_cls = getattr(module, step_info["cls"])
        
        console.print("[dim]⚙️  Initializing agent with MockLLM...[/dim]")
        
        # Import shared mock LLM
        from src.mock_llm import MockLLM
        llm_cfg = config.get("agent", {})
        llm = MockLLM(llm_cfg)
        
        if step_num == 1:
            agent = agent_cls(llm, system_prompt=llm_cfg.get("system_prompt"))
            result = agent.run(step_info["prompt"])
            
        elif step_num == 2:
            agent = agent_cls(llm)
            # Register a safe calculator tool for demonstration
            def calc(expr: str) -> str:
                try: 
                    return str(eval(expr)) if expr else "0"
                except Exception: 
                    return "Invalid expression"
            agent.register_tool("calculator", calc, "Evaluate math expressions")
            result = agent.run(step_info["prompt"])
            
        elif step_num == 3:
            agent = agent_cls(llm, max_turns=5)
            r1 = agent.run(step_info["prompt_1"])
            console.print(f"[green]✅ Turn 1:[/green] {r1}")
            result = agent.run(step_info.get("prompt_2", ""))
            
        elif step_num == 4:
            # ReAct agents often take tools list; we pass empty or mock for simplicity in driver
            agent = agent_cls(llm, tools=[], max_steps=5)
            result = agent.run(step_info["prompt"])
            
        else:  # Step 5 (Final/Orchestrator)
            agent = agent_cls(config=config)
            result = agent.run(step_info["prompt"])
            
        console.print(Panel(str(result), title="[bold green]🤖 Agent Output[/bold green]", border_style="green"))
        return True
        
    except ImportError as e:
        console.print(f"[red]❌ Module not found:[/red] {e}")
        console.print("[yellow]💡 Tip: Ensure you've completed previous phases or run `pip install -e .`[/yellow]")
        return False
    except Exception as e:
        console.print(f"[red]⚠️  Runtime Error in Step {step_num}:[/red] {type(e).__name__}: {e}")
        import traceback
        console.print(Panel(Markdown(traceback.format_exc()), title="[bold red]Traceback[/bold red]", border_style="red"))
        return False


def run_tests():
    """Integrate pytest for validation exercises."""
    console.print("\n[bold cyan]🧪 Running Validation Suite...[/bold cyan]")
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
    if not os.path.exists(test_dir):
        console.print("[yellow]⚠️  tests/ directory not found. Skipping.[/yellow]")
        return False
        
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_dir, "-v", "--tb=short"],
        capture_output=False
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="AI Agent Tutorial CLI Driver")
    parser.add_argument("--step", type=str, default="all", help="Run specific step (1-5) or 'all'")
    parser.add_argument("--dry-run", action="store_true", help="Show execution plan without running code")
    parser.add_argument("--test", action="store_true", help="Run pytest validation suite instead of tutorial steps")
    args = parser.parse_args()
    
    config = load_config()
    
    console.print(Panel.fit("🚀 Welcome to the AI Agent Tutorial!", subtitle="Progressive learning from basic loops to production agents"))
    
    if args.test:
        run_tests()
        return

    # Parse step argument
    if args.step == "all":
        steps_to_run = list(STEPS.keys())
    else:
        try:
            step_int = int(args.step)
            if step_int not in STEPS:
                console.print(f"[red]❌ Invalid step. Choose from {list(STEPS.keys())}[/red]")
                sys.exit(1)
            steps_to_run = [step_int]
        except ValueError:
            console.print("[red]❌ Step must be a number or 'all'[/red]")
            sys.exit(1)

    # Execute with progress tracking
    success_count = 0
    total = len(steps_to_run)
    
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), transient=True) as progress:
        task = progress.add_task("[cyan]Executing tutorial steps...", total=total)
        
        for s in steps_to_run:
            desc = f"Running Step {s}: {STEPS[s]['name']}..."
            progress.update(task, description=desc)
            if run_step(s, config, dry_run=args.dry_run):
                success_count += 1
            progress.advance(task, 1)

    console.print(f"\n[bold green]✅ Tutorial Complete![/bold green] ({success_count}/{total} steps passed)")
    console.print("💡 Next: Run `python run_tutorial.py --test` to validate your understanding.")


if __name__ == "__main__":
    main()