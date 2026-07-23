# 📘 Coding Plan: Beginner-Friendly `agent.py` Tutorial Project

This plan delivers a **progressive, self-contained tutorial** designed to teach beginners how to build, understand, and extend a Python-based AI agent. The architecture is intentionally framework-agnostic, uses lightweight dependencies, and emphasizes core agent concepts over boilerplate.

---

## 📁 1. Complete File Tree
```
agent-tutorial/
├── README.md                          # Main tutorial guide & cheat sheet
├── pyproject.toml                     # Dependencies, scripts, environment config
├── .env.example                       # Template for API keys / local settings
├── config.yaml                        # Agent configuration (model params, tools, memory)
├── src/
│   ├── __init__.py
│   ├── mock_llm.py                    # Zero-API-key LLM simulator for safe practice
│   ├── agent_v1_basic.py              # Step 1: Minimal agent loop & anatomy
│   ├── agent_v2_tools.py              # Step 2: Function calling & tool registration
│   ├── agent_v3_memory.py             # Step 3: Context window & conversation history
│   ├── agent_v4_reasoning.py          # Step 4: ReAct-style planning loop
│   └── agent_final.py                 # Polished, production-ready structure
├── tests/
│   ├── __init__.py
│   └── test_agent_tutorial.py         # Validation suite + fill-in-the-blank exercises
├── run_tutorial.py                    # CLI driver to execute steps sequentially
└── notebooks/
    └── interactive_walkthrough.ipynb  # Jupyter version for visual learners
```

---

## 📄 2. File-by-File Specification

### `README.md`
- **Purpose**: Central tutorial document. Progressive lessons, setup instructions, mental models, troubleshooting.
- **Key Sections**:
  - What is an AI Agent? (Input → Reasoning → Action → Output loop)
  - How to use this repo (`python run_tutorial.py --step 1`, etc.)
  - Step-by-step walkthroughs with explanations
  - Exercises & expected outputs
  - Common beginner mistakes & debugging tips
  - "Cheat Sheet" table mapping concepts to code patterns

### `pyproject.toml`
- **Purpose**: Modern Python packaging, dependency management, CLI scripts.
- **Contents**:
  ```toml
  [project]
  name = "agent-tutorial"
  version = "0.1.0"
  requires-python = ">=3.9"
  dependencies = [
      "rich>=13.0",
      "pyyaml>=6.0",
      "python-dotenv>=1.0",
      "pytest>=7.4"
  ]

  [project.scripts]
  run-tutorial = "run_tutorial:main"
  ```
- **Why**: Keeps dependencies minimal. `rich` for pretty CLI output, `pyyaml` for config parsing, `dotenv` for safe env vars.

### `.env.example`
```env
# Copy to .env and fill if using real LLMs later
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
USE_MOCK_LLM=true  # Set to false when ready for live APIs
```

### `config.yaml`
```yaml
agent:
  name: "BeginnerAgent"
  temperature: 0.2
  max_tokens: 512
  system_prompt: |
    You are a helpful assistant. Think step-by-step. 
    Use tools when needed. Keep responses concise.

tools:
  - name: calculator
    description: "Evaluate math expressions"
    args: ["expression"]
    
memory:
  max_turns: 10
  store_system_messages: false
```
- **Purpose**: Centralized, human-readable configuration. Teaches beginners how to externalize magic numbers and prompts.

### `src/mock_llm.py`
- **Purpose**: Safe, offline LLM simulator that mimics OpenAI-style responses without API keys.
- **Key Concepts**: Deterministic outputs, tool-call simulation, temperature mocking.
- **Structure**:
  ```python
  class MockLLM:
      def __init__(self, config: dict):
          self.config = config
      def chat(self, messages: list[dict]) -> str:
          # Returns predictable responses based on message content
          pass
      def generate_tool_calls(self, messages: list[dict], tools: list) -> list[dict]:
          # Simulates function calling format
          pass
  ```

### `src/agent_v1_basic.py`
- **Purpose**: Minimal agent loop. Teaches anatomy: prompt → LLM call → output parsing → response.
- **Key Concepts**: System/user messages, basic I/O, type hints, docstrings.
- **Structure**:
  ```python
  class BasicAgent:
      def __init__(self, llm, system_prompt: str): ...
      def run(self, user_input: str) -> str: ...
  # Includes clear comments mapping to the "Input → Process → Output" mental model
  ```

### `src/agent_v2_tools.py`
- **Purpose**: Add tool/function calling. Teaches registration, argument parsing, execution, and result injection.
- **Key Concepts**: Tool schema, JSON argument extraction, safe execution sandbox, fallback behavior.
- **Structure**:
  ```python
  class ToolAgent:
      def register_tool(self, name, func, description): ...
      def _parse_tool_calls(self, llm_output): ...
      def execute_tools(self, calls): ...
      def run(self, user_input: str) -> str: ...
  ```

### `src/agent_v3_memory.py`
- **Purpose**: Context management. Teaches conversation history, windowing, and system prompt persistence.
- **Key Concepts**: Message list immutability, sliding window, turn counting, memory serialization.
- **Structure**:
  ```python
  class MemoryAgent:
      def __init__(self, llm, max_turns=5): ...
      def add_to_history(self, role, content): ...
      def get_context_window(self) -> list[dict]: ...
      def run(self, user_input: str) -> str: ...
  ```

### `src/agent_v4_reasoning.py`
- **Purpose**: Multi-step reasoning loop (simplified ReAct). Teaches planning → tool use → reflection → final answer.
- **Key Concepts**: Loop control, max iterations, thought/action/observation pattern, early exit conditions.
- **Structure**:
  ```python
  class ReasoningAgent:
      def __init__(self, llm, tools, max_steps=5): ...
      def _format_react_prompt(self, history, observations): ...
      def run(self, user_input: str) -> dict: # Returns {thoughts, actions, final_answer}
  ```

### `src/agent_final.py`
- **Purpose**: Production-ready structure combining all concepts with logging, error handling, config loading, and clean CLI.
- **Key Concepts**: Composition over inheritance, graceful degradation, structured logging, environment-aware initialization.
- **Structure**: Modular imports from v1-v4, wrapped in `AgentOrchestrator` class with `__main__` block for direct execution.

### `tests/test_agent_tutorial.py`
- **Purpose**: Validation + interactive exercises. Beginners run tests to verify understanding.
- **Key Concepts**: pytest fixtures, assert patterns, fill-in-the-blank challenges (e.g., `_ = student_implement_tool_parsing()`).
- **Structure**: Parametrized tests for each step, clear failure messages pointing to tutorial sections.

### `run_tutorial.py`
- **Purpose**: CLI driver that executes steps sequentially with progress tracking and interactive prompts.
- **Key Concepts**: argparse, rich console output, step validation, auto-test integration.
- **Usage**: 
  ```bash
  python run_tutorial.py --step all   # Run full tutorial
  python run_tutorial.py --step 2    # Jump to tools
  python run_tutorial.py --dry-run   # Show code without executing
  ```

### `notebooks/interactive_walkthrough.ipynb`
- **Purpose**: Jupyter alternative for visual learners. Cell-by-cell execution with markdown explanations, plots of memory window, and tool-call traces.
- **Key Concepts**: `%run`, interactive widgets, output capture, pedagogical pacing.

---

## 🧭 3. Tutorial Execution Flow (Pedagogical Path)

| Step | File              | Learning Objective                          | Beginner Exercise                          |
|------|-------------------|---------------------------------------------|--------------------------------------------|
| 0    | `README.md`       | Setup environment, understand agent anatomy | Create `.env`, run `python -m pytest`      |
| 1    | `agent_v1_basic.py` | Message structure, basic LLM call         | Change system prompt, observe output shift |
| 2    | `agent_v2_tools.py` | Tool registration & JSON argument parsing | Add a `weather` tool, test with query      |
| 3    | `agent_v3_memory.py`| Context windowing & history management     | Set `max_turns=2`, verify message pruning  |
| 4    | `agent_v4_reasoning.py`| Planning loop & observation injection   | Trace step-by-step reasoning in console    |
| 5    | `agent_final.py`  | Composition, logging, production patterns  | Swap `MockLLM` → real API key, run live    |

---

## ⚙️ 4. Setup & Execution Instructions

1. **Clone & Install**
   ```bash
   git clone <repo> && cd agent-tutorial
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"
   cp .env.example .env  # Set USE_MOCK_LLM=true initially
   ```

2. **Run Tutorial**
   ```bash
   python run_tutorial.py --step 1  # Start basic agent
   python run_tutorial.py --step all  # Full progressive run
   ```

3. **Validate Learning**
   ```bash
   pytest tests/ -v --tb=short
   ```

4. **Go Live (Optional)**
   - Set `USE_MOCK_LLM=false` in `.env`
   - Add real API key
   - Update `src/mock_llm.py` import to use `openai.OpenAI()` or `langchain.llms`

---

## 🧠 5. Architectural & Pedagogical Design Notes

| Principle                  | Implementation                                                                 |
|---------------------------|--------------------------------------------------------------------------------|
| **Progressive Disclosure** | Each file adds exactly one new concept. No hidden complexity.                  |
| **Safe-by-Default**        | `MockLLM` prevents accidental API costs. Tools run in isolated scope.          |
| **Explicit Over Implicit** | All prompts, configs, and tool schemas are externalized & documented.          |
| **Test-as-Tutorial**       | Exercises live in `test_agent_tutorial.py`. Passing = mastery.                 |
| **Framework Agnostic**     | Zero LangChain/CrewAI dependency. Easy to port to any ecosystem later.         |
| **Debuggable by Design**   | Structured logging, step tracing, and `--dry-run` mode for safe exploration.   |

---

## 🔮 6. Extensibility Path (Post-Tutorial)

Once beginners complete the tutorial, they can extend:
- Add streaming responses (`asyncio` + token-by-token output)
- Implement tool validation with Pydantic schemas
- Swap `MockLLM` for OpenAI, Anthropic, or local Llama.cpp
- Add evaluation harness (accuracy, hallucination checks, cost tracking)
- Package as CLI tool or FastAPI backend

---

✅ **This plan delivers a complete, production-grade tutorial structure that teaches agent fundamentals intuitively, safely, and progressively. Every file serves a pedagogical purpose, and the architecture scales from beginner exercises to real-world deployment.** 

Ready for implementation. Let me know if you want any file fully coded or adapted to a specific framework (LangChain, AutoGen, CrewAI, etc.).