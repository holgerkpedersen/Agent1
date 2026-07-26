Task 1: `agent_core/entities.py` — Merge duplicate exception definitions from top-level `entities.py`; replace generic TypeVar-based `Failure[E]` with concrete subclasses; enforce frozen dataclasses on config objects (AgentSettings).

Task 2: `agent_core/path_utils.py` — Rename private `_validate_path()` to public `normalize_path()`, add strict variant (`_strict_validate_path`) exported as `validate_path`.

Task 3: `agent_core/__init__.py` — Update exports to reflect renamed functions in path_utils; ensure proper aliasing of normalize_path/validate_path.

Task 4: `agent.py` — Fix `_normalize_path()` double-normalization mismatch in execute_tool("read_file"); define missing search_file method directly on Agent class wrapping _search_files().

Task 5: `agent_core/handlers/analyze_handler.py` — Remove dead `.register=lambda cls=None:None`; fix signature extraction to operate on source text instead of AST metadata; adopt ast.NodeVisitor pattern for cleaner traversal.

Task 6: `tool_router.py` — Register ShellCommandHandler for "run_command" tool definition; add schema validation enforcement before routing execution calls using pydantic models.

Task 7: `benchmark.py` — Convert string-based error returns ("File not found") to raise structured FileOperationError exceptions; make scoring thresholds configurable via CLI flags (--timeout-per-model, --retry-backoff-factor).

Task 8: Multiple files (`agent.py`, `entities.py`, etc.) — Standardize all modules to use `from __future__ import annotations` + PEP604 unions replacing Union[A,B] syntax.

Task 9: `benchmark.py` — Enhance haiku syllable estimation accuracy using vowel grouping heuristics accounting for silent trailing 'e'.

Task 10: New test files (`tests/test_agent_paths.py`, `tests/test_tool_router.py`) — Add unit tests covering path normalization edge cases (/c/, /d/) and tool dispatch routing correctness including invalid schema rejection.

Validation Step: Run `mypy --strict .` to confirm full type-checking compliance; verify no runtime circular imports introduced through static import graph analysis; grep audit confirms structured exception usage replaces string returns everywhere feasible.