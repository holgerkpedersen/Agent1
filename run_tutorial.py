#!/usr/bin/env python3
"""CLI driver for the Agent Tutorial.

Supports step-by-step execution of agent versions (v1 → v4),
dry-run mode, and integrated test running via pytest.

Usage:
    python run_tutorial.py --step all      # Run all steps sequentially
    python run_tutorial.py --step 2        # Run only v2 (tools) step
    python run_tutorial.py --dry-run       # Preview without executing
    python run_tutorial.py --test          # Run the test suite
"""

import argparse
import importlib
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so that `src.*` imports work regardless
# of how this script is invoked.
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.markdown import Markdown
except ImportError:
    print("Error: 'rich' is required. Install with: pip install rich")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Lazy-import the tutorial modules so that run_tutorial.py can be executed
# even if some agent versions have not been implemented yet.  This makes it
# safe to run during incremental development (Phase-by-Phase).
# ---------------------------------------------------------------------------
console = Console()


def _import_module(module_name: str):
    """Import a module and return it, or None with a friendly message."""
    try:
        mod = importlib.import_module(module_name)
        return mod
    except ImportError as exc:
        console.print(f"[yellow]⚠  Module '{module_name}' not available yet ({exc}).[/yellow]")
        return None


# ── Step Definitions ───────────────────────────────────────────────────────

_STEP_DESCRIPTIONS = {
    1: ("Basic Chat", "src.agent_v1_basic", "BasicAgent"),
       2: ("Tool Use", "src.agent_v2_tools", "ToolAgent"),
    3: ("Memory / History", "src.agent_v3_memory", "MemoryAgent"),
    4: ("ReAct Reasoning", "src.agent_v4_reasoning", "ReasoningAgent"),
}


# ── Individual Step Runners ───────────────────────────────────────────────

def run_step_1_basic(dry_run: bool = False) -> None:
    """Execute Agent v1 — Basic Chat (no tools, no memory)."""
    console.print(Panel("Step 1: Basic Chat", subtitle="Input → LLM → Output"))

    if dry_run:
        console.print("[dim]Would create a BasicAgent with a system prompt and "
                      "call llm.chat() once.[/dim]")
        return

    agent_mod = _import_module("src.agent_v1_basic")
    mock_mod = _import_module("src.mock_llm")

    if agent_mod is None or mock_mod is None:
        console.print("[red]❌ Cannot run Step 1 — missing dependencies.[/red]")
        return

    llm = mock_mod.MockLLM()
    system_prompt = "You are a helpful assistant."
    agent_cls = getattr(agent_mod, "BasicAgent", None)
    if agent_cls is None:
        console.print("[red]❌ BasicAgent class not found in src.agent_v1_basic.[/red]")
        return

    agent = agent_cls(llm=llm, system_prompt=system_prompt)

    user_input = "What is the capital of France?"
    console.print(f"[blue]User:[/] {user_input}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Thinking...", total=None)
        response = agent.run(user_input)
        progress.update(task, description="Done")

    console.print(f"[green]Assistant:[/] {response}")


def run_step_2_tools(dry_run: bool = False) -> None:
    """Execute Agent v2 — Tool Use."""
    console.print(Panel("Step 2: Tool Use", subtitle="Input → LLM → Tool Call → Result → Output"))

    if dry_run:
        console.print("[dim]Would register tools (e.g., get_weather, search), "
                      "parse JSON tool calls from the LLM, execute them, and inject results.[/dim]")
        return

    agent_mod = _import_module("src.agent_v2_tools")
    mock_mod = _import_module("src.mock_llm")

    if agent_mod is None or mock_mod is None:
        console.print("[red]❌ Cannot run Step 2 — missing dependencies.[/red]")
        return

    llm = mock_mod.MockLLM()
    system_prompt = "You are a helpful assistant with access to tools."
    agent_cls = getattr(agent_mod, "ToolAgent", None)
    if agent_cls is None:
        console.print("[red]ToolAgent class not found in src.agent_v2_tools.[/red]")
        return

    agent = agent_cls(llm=llm, system_prompt=system_prompt)

    # Register a sample tool
    def get_weather(city: str) -> str:
        return f"The weather in {city} is 72°F and sunny."

    agent.register_tool("get_weather", get_weather, "Get the current weather for a city.")

    user_input = "What's the weather like in Paris?"
    console.print(f"[blue]User:[/] {user_input}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Thinking & executing tools...", total=None)
        response = agent.run(user_input)
        progress.update(task, description="Done")

    console.print(f"[green]Assistant:[/] {response}")


def run_step_3_memory(dry_run: bool = False) -> None:
    """Execute Agent v3 — Memory / Conversation History."""
    console.print(Panel("Step 3: Memory / History", subtitle="Multi-turn conversation with context window"))

    if dry_run:
        console.print("[dim]Would maintain a message history buffer, enforce max_turns "
                      "via build_context_window(), and alternate user/assistant roles.[/dim]")
        return

    agent_mod = _import_module("src.agent_v3_memory")
    mock_mod = _import_module("src.mock_llm")

    if agent_mod is None or mock_mod is None:
        console.print("[red]❌ Cannot run Step 3 — missing dependencies.[/red]")
        return

    llm = mock_mod.MockLLM()
    system_prompt = "You are a helpful assistant with memory."
    agent_cls = getattr(agent_mod, "MemoryAgent", None)
    if agent_cls is None:
        console.print("[red]❌ MemoryAgent class not found in src.agent_v3_memory.[/red]")
        return

    max_turns = 5
    agent = agent_cls(llm=llm, system_prompt=system_prompt, max_turns=max_turns)

    turns = [
        ("Hi, I'm Alex. Nice to meet you!", "Turn 1: Introduction"),
        ("What's my name?", "Turn 2: Recall name (tests memory)"),
        ("Tell me a fun fact about Python.", "Turn 3: Fun fact"),
    ]

    for user_msg, label in turns:
        console.print(f"[dim]--- {label} ---[/dim]")
        console.print(f"[blue]User:[/] {user_msg}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Thinking...", total=None)
            response = agent.run(user_msg)
            progress.update(task, description="Done")

        console.print(f"[green]Assistant:[/] {response}")
        console.print()


def run_step_4_reasoning(dry_run: bool = False) -> None:
    """Execute Agent v4 — ReAct Reasoning Loop."""
    console.print(Panel("Step 4: ReAct Reasoning", subtitle="Thought → Action → Observation loop"))

    if dry_run:
        console.print("[dim]Would run a ReAct loop: prompt LLM for {thought, action, input}, "
                      "execute tool, append observation, repeat until final answer or max steps.[/dim]")
        return

    agent_mod = _import_module("src.agent_v4_reasoning")
    mock_mod = _import_module("src.mock_llm")
    types_mod = _import_module("src.types")

    if agent_mod is None or mock_mod is None:
        console.print("[red]❌ Cannot run Step 4 — missing dependencies.[/red]")
        return

    llm = mock_mod.MockLLM()
    system_prompt = "You are a reasoning assistant. Use tools to answer questions."
    agent_cls = getattr(agent_mod, "ReasoningAgent", None)
    if agent_cls is None:
        console.print("[red]❌ ReasoningAgent class not found in src.agent_v4_reasoning.[/red]")
        return

    max_steps = 5
    
    def calculator(expression: str) -> str:
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {e}"
    
    calc_tool = {
        "name": "calculator",
        "description": "Evaluate math expressions",
        "func": calculator,
    }
    
    agent = agent_cls(llm=llm, tools=[calc_tool], max_steps=max_steps)


    user_input = "What is (42 * 3) + 15?"
    console.print(f"[blue]User:[/] {user_input}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Reasoning...", total=None)
        result = agent.run(user_input)
        progress.update(task, description="Done")

    # Result may be a ReasoningStep or a string depending on implementation
    if hasattr(result, "is_final"):
        console.print(f"[green]Final Answer:[/] {result.result}")
        if hasattr(result, "steps"):
            table = Table(title="Reasoning Trace", show_header=True)
            table.add_column("Step", style="dim")
            table.add_column("Thought")
            table.add_column("Action")
            table.add_column("Observation")
            for i, step in enumerate(result.steps, 1):
                table.add_row(
                    str(i),
                    getattr(step, "thought", ""),
                    getattr(step, "action", ""),
                    getattr(step, "observation", ""),
                )
            console.print(table)
    else:
        console.print(f"[green]Assistant:[/] {result}")


# ── Test Integration ───────────────────────────────────────────────────────

def run_tests(verbosity: str = "-v") -> None:
    """Run the pytest test suite programmatically."""
    console.print(Panel("Running Tests", subtitle="pytest"))

    tests_dir = _project_root / "tests"
    if not tests_dir.is_dir():
        console.print("[red]❌ 'tests/' directory not found.[/red]")
        return

    try:
        import pytest  # noqa: F401
    except ImportError:
        console.print("[red]❌ 'pytest' is required. Install with: pip install pytest[/red]")
        return

    args = [str(tests_dir), verbosity, "--tb=short"]
    console.print(f"[dim]$ pytest {' '.join(args)}[/dim]\n")

    exit_code = subprocess.call([sys.executable, "-m", "pytest"] + args)
    if exit_code == 0:
        console.print("\n[green]✅ All tests passed![/green]")
    else:
        console.print(f"\n[red]❌ Tests failed with exit code {exit_code}.[/red]")


# ── Summary Table ─────────────────────────────────────────────────────────

def print_summary() -> None:
    """Print a summary table of all tutorial steps."""
    table = Table(title="Agent Tutorial — Step Overview", show_header=True, header_style="bold magenta")
    table.add_column("#", justify="center", style="cyan")
    table.add_column("Name")
    table.add_column("Module")
    table.add_column("Class")

    for step_num, (name, module, cls) in _STEP_DESCRIPTIONS.items():
        available = "[green]✓[/green]" if _import_module(module) is not None else "[yellow]?[/yellow]"
        table.add_row(str(step_num), name, f"{available} {module}", cls)

    console.print(table)


# ── Main CLI Entrypoint ───────────────────────────────────────────────────

_STEP_RUNNERS = {
    1: run_step_1_basic,
    2: run_step_2_tools,
    3: run_step_3_memory,
    4: run_step_4_reasoning,
}


def parse_steps(step_arg: str) -> List[int]:
    """Parse the --step argument into a list of step numbers."""
    if step_arg.lower() == "all":
        return sorted(_STEP_RUNNERS.keys())

    parts = [p.strip() for p in step_arg.split(",")]
    steps: List[int] = []
    for part in parts:
        try:
            n = int(part)
        except ValueError:
            console.print(f"[red]❌ Invalid step number: '{part}'[/red]")
            sys.exit(1)
        if n not in _STEP_RUNNERS:
            console.print(f"[red]❌ Unknown step {n}. Valid steps: 1–4 or 'all'.[/red]")
            sys.exit(1)
        steps.append(n)

    return sorted(set(steps))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_tutorial",
        description="CLI driver for the Agent Tutorial — step-by-step agent development.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_tutorial.py --step all          Run all steps (v1 → v4)\n"
            "  python run_tutorial.py --step 2             Run only Step 2 (Tool Use)\n"
            "  python run_tutorial.py --step 1,3           Run Steps 1 and 3\n"
            "  python run_tutorial.py --dry-run            Preview steps without executing\n"
            "  python run_tutorial.py --test               Run the pytest suite\n"
        ),
    )

    parser.add_argument(
        "--step",
        type=str,
        default="all",
        help="Which step(s) to run. Use 'all' (default), a number (1-4), or comma-separated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without executing the agents.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the pytest test suite instead of tutorial steps.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a summary table and exit.",
    )

    args = parser.parse_args()

    # ── Mode: Summary ────────────────────────────────────────────────
    if args.summary:
        console.print()
        print_summary()
        return

    # ── Mode: Tests ──────────────────────────────────────────────────
    if args.test:
        run_tests()
        return

    # ── Mode: Tutorial Steps ─────────────────────────────────────────
    steps = parse_steps(args.step)
    dry_run = args.dry_run

    banner_text = "[bold]Agent Tutorial[/]"
    mode_text = " (Dry Run)" if dry_run else ""
    console.print(f"\n{banner_text}{mode_text}")
    console.print(f"Steps to execute: {', '.join(str(s) for s in steps)}\n")

    print_summary()
    console.print()

    # Separator between runs
    separator = "━" * 40

    for step_num in steps:
        runner = _STEP_RUNNERS[step_num]
        console.print(f"\n[{separator}]")
        try:
            runner(dry_run=dry_run)
        except Exception as exc:
            console.print(f"[red]❌ Step {step_num} failed:[/red] {exc}")

    console.print(f"\n[{separator}]\n[bold green]✅ Tutorial complete![/bold green]")


if __name__ == "__main__":
    main()