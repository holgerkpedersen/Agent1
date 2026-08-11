1. `agent_core/config.py` — Replace fragile `.env` parsing with validated configuration management and type-safe defaults
2. `agent_core/llm/config.py` — Integrate robust config validation for LLM provider settings and retry parameters
3. `agent_core/security/path_utils.py` — Consolidate duplicated path normalization and workspace boundary enforcement logic
4. `agent_core/security/sanitizer.py` — Add centralized secret masking and consistent input sanitization utilities
5. `agent_core/logging_config.py` — Create structured logging configuration to replace scattered print statements and silent exceptions
6. `agent_core/llm/retry_adapter.py` — Implement exponential backoff retries and rate limiting for LLM API calls
7. `agent_core/tool_executor.py` — Extract tool execution logic from monolithic agent and replace unsafe shell=True subprocess calls
8. `agent_core/nlp_parser.py` — Extract NLP parsing and intent classification into a dedicated module with standardized type hints
9. `agent_core/llm/provider.py` — Refactor LLM wrapping to use structured logging, graceful degradation, and the new retry adapter
10. `agent_core/routing/bus.py` — Update routing bus to handle structured errors and integrate with centralized logging
11. `agent_core/file_context_retriever.py` — Replace inline path checks with consolidated security utilities and add context-aware retrieval fallbacks
12. `agent_core/commands/base.py` — Standardize command base class with type hints, structured error handling, and shared validation logic
13. `agent_core/commands/implement_cmd.py` — Patch _execute_nlp_tool to use safe subprocess execution without shell=True
14. `agent_core/handlers/base_handler.py` — Refactor handler base to enforce SRP, integrate structured logging, and remove silent exception swallowing
15. `agent_core/agent.py` — Decompose monolithic Agent class by delegating CLI, NLP, tool execution, and LLM tasks to extracted modules
16. `agent_core/cli/commands/clear.py` — Update clear command to use standardized logging and validated configuration paths
17. `agent1/providers/async_llm.py` — Integrate rate limiting and backoff retries into async provider implementation
18. `agent1/swarm/orchestrator.py` — Refactor orchestrator to leverage extracted agent modules and structured error handling
19. `agent_core/diff/semantic_parser.py` — Add type annotations and graceful fallbacks for semantic diff parsing failures
20. `agent_core/security/allowlist.py` — Enhance command allowlist with dynamic validation and structured audit logging
