# 📋 Task Plan: Dependency-Ordered Implementation

This plan sequences files strictly by **code dependencies** first, then **pedagogical flow**. Each phase includes concrete implementation tasks, dependency prerequisites, and validation checkpoints to ensure safe progression.

---

## 🔹 Phase 1: Project Scaffolding & Configuration
**Dependencies:** None  
**Files:** `pyproject.toml`, `.env.example`, `config.yaml`, `src/__init__.py`, `tests/__init__.py`

| Task | Details |
|------|---------|
| ✅ Create package metadata | Add `[project]`, dependencies (`rich`, `pyyaml`, `python-dotenv`, `pytest`), and CLI entry points to `pyproject.toml` |
| ✅ Setup environment template | Copy `.env.example` with placeholder keys & `USE_MOCK_LLM=true` |
| ✅ Externalize config | Write `config.yaml` with agent params, tool schemas, and memory limits |
| ✅ Initialize modules | Create empty `__init__.py` in `src/` and `tests/` for Python package recognition |

🔍 **Validation:** `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"` → Verify no import errors.

---

## 🔹 Phase 2: Shared Types & Contracts (Root Dependency)
**Dependencies:** None  
**Files:** `src/types.py`

| Task | Details |
|------|---------|
| ✅ Implement type contracts | Paste provided `Message`, `ToolDefinition`, `LLMProvider`, `AgentBase`, `ReasoningStep`, `AgentConfig` |
| ✅ Add stateless utilities | Implement `parse_json_safely()`, `format_tool_result()`, `build_context_window()` |
| ✅ Enforce structural typing | Use `Protocol` + `TypedDict` to avoid inheritance chains & circular imports |

🔍 **Validation:** `python -c "from src.types import *; print('Types loaded successfully')"` → Run `mypy src/types.py --ignore-missing-imports` for type safety.

---

## 🔹 Phase 3: Safe Execution Layer
**Dependencies:** `src/types.py`  
**Files:** `src/mock_llm.py`

| Task | Details |
|------|---------|
| ✅ Implement `MockLLM` class | Satisfy `LLMProvider` protocol with deterministic `chat()` and `generate_tool_calls()` |
| ✅ Simulate tool routing | Parse message content to return predictable JSON tool calls matching registered schemas |
| ✅ Add temperature/seed mocking | Optional: support config-driven randomness for realistic testing |

🔍 **Validation:** Unit test mock responses against expected formats. Verify zero external API calls.

---

## 🔹 Phase 4: Progressive Agent Implementations
**Dependencies:** `src/types.py` (all), `src/mock_llm.py` (v1)  
*Note: Files are structurally independent but pedagogically sequential.*

### 📦 `src/agent_v1_basic.py`
| Task | Details |
|------|---------|
| ✅ Implement `BasicAgent` | Satisfy `AgentBase`. Accept `llm` + `system_prompt`, return string response |
| ✅ Message formatting | Build `[{"role":"system",...}, {"role":"user","content":input}]` → call `llm.chat()` |

### 📦 `src/agent_v2_tools.py`
| Task | Details |
|------|---------|
| ✅ Tool registry | `register_tool(name, func, desc)` storing `ToolDefinition` objects |
| ✅ Parse & execute | Use `parse_json_safely()` to extract calls → run in isolated scope → inject results via `format_tool_result()` |

### 📦 `src/agent_v3_memory.py`
| Task | Details |
|------|---------|
| ✅ History buffer | Maintain `list[Message]`, enforce `max_turns` via `build_context_window()` |
| ✅ Turn tracking | Alternate user/assistant roles, prune oldest pairs when limit exceeded |

### 📦 `src/agent_v4_reasoning.py`
| Task | Details |
|------|---------|
| ✅ ReAct loop | Max iterations → prompt LLM for `{thought, action, input}` → execute tool → append observation → repeat |
| ✅ Step tracking | Return `ReasoningStep` objects; early exit on `is_final=True` or max steps reached |

🔍 **Validation:** Run each agent independently with `MockLLM`. Verify outputs match expected mental models (Input→Process→Output, Tool→Result, Memory Pruning, Reasoning Trace).

---

## 🔹 Phase 5: Production Orchestrator
**Dependencies:** All v1–v4 agents, `src/types.py`, `config.yaml`  
**Files:** `src/agent_final.py`

| Task | Details |
|------|---------|
| ✅ Compose features | Import capabilities from v1–v4 into `AgentOrchestrator` using composition (not inheritance) |
| ✅ Config-driven init | Load `config.yaml`, apply to `AgentConfig`, initialize LLM provider based on `.env` flags |
| ✅ Add observability | Structured logging (`rich.console`), error boundaries, graceful fallbacks for missing tools/parsing failures |
| ✅ CLI entry point | `if __name__ == "__main__":` block accepting user input or streaming mode |

🔍 **Validation:** `python -m src.agent_final` → Verify config loading, mock/live toggle, and clean error handling.

---

## 🔹 Phase 6: Test Suite & CLI Driver
**Dependencies:** All agent files, `src/types.py`  
**Files:** `tests/test_agent_tutorial.py`, `run_tutorial.py`

| Task | Details |
|------|---------|
| ✅ Write parameterized tests | Cover each step (v1–v4) with clear assertions & pedagogical failure messages |
| ✅ Add fill-in exercises | Stub functions like `student_implement_tool_parsing()` that fail until completed |
| ✅ Build CLI driver | `argparse` + `rich` progress tracking. Support `--step N`, `--dry-run`, auto-test integration |

🔍 **Validation:** `pytest tests/ -v` → All pass. `python run_tutorial.py --step all` → Executes sequentially with clear console output.

---

## 🔹 Phase 7: Documentation & Interactive Materials
**Dependencies:** References all above (can be drafted in parallel during Phases 1–4)  
**Files:** `README.md`, `notebooks/interactive_walkthrough.ipynb`

| Task | Details |
|------|---------|
| ✅ Write tutorial guide | Setup instructions, mental models, step-by-step walkthroughs, cheat sheet table, troubleshooting |
| ✅ Build Jupyter notebook | Cell-by-cell execution matching v1→v4 flow, visual memory window plots, tool-call trace outputs |

🔍 **Validation:** Follow `README.md` from scratch on a clean machine. Run notebook cells sequentially → Verify all outputs match spec expectations.

---

## 🔗 Dependency Graph Summary
```
pyproject.toml + config.yaml
        ↓
   src/types.py  (ROOT)
     ↙    ↓    ↘
mock_llm.py  agent_v1_basic.py  agent_v2_tools.py  agent_v3_memory.py  agent_v4_reasoning.py
        ↖__________________________↙
                     ↓
              agent_final.py
                ↙         ↘
test_agent_tutorial.py   run_tutorial.py
        ↖_______________↙
            README.md + interactive_walkthrough.ipynb
```

## ✅ Execution Checklist
1. [ ] Phase 1: Scaffolding & Config
2. [ ] Phase 2: `src/types.py` (paste & validate)
3. [ ] Phase 3: `src/mock_llm.py`
4. [ ] Phase 4: `agent_v1` → `v2` → `v3` → `v4`
5. [ ] Phase 5: `agent_final.py`
6. [ ] Phase 6: Tests + CLI driver
7. [ ] Phase 7: Docs + Notebook
8. [ ] Final: `pytest`, `mypy`, clean-room walkthrough

Ready to implement phase-by-phase or generate full code for any specific file. Let me know your preferred starting point.