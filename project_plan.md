**[FIX]**
- [MUST] Patch `_execute_nlp_tool` command injection vulnerability by replacing `shell=True` with safe subprocess execution.
- [SHOULD] Replace silent `except Exception: pass` blocks with structured error logging and graceful degradation.
- [COULD] Standardize type hints and return annotations across core modules to improve static analysis reliability.

**[FEATURE]**
- [MUST] Implement comprehensive unit and integration test suites covering agent logic, tool routing, and edge cases.
- [SHOULD] Add LLM rate limiting, exponential backoff retries, and robust session/state persistence mechanisms.
- [COULD] Integrate lightweight RAG with vector embeddings to replace brittle regex-based code search fallbacks.

**[ARCH]**
- [MUST] Refactor monolithic `Agent` class by extracting CLI, NLP parsing, tool execution, and LLM wrapping into dedicated modules.
- [SHOULD] Consolidate duplicated path normalization and safety checks into a single shared utility module.
- [COULD] Integrate the standalone `tool_router.py` with Pydantic validation or remove it to eliminate architectural fragmentation.

**[OPS]**
- [MUST] Replace fragile manual `.env` parsing and hardcoded defaults with a validated configuration management library.
- [SHOULD] Standardize logging across all modules by replacing `print()` statements and error swallowing with structured loggers.
- [COULD] Implement centralized secret management and consistent workspace boundary enforcement for production deployments.
