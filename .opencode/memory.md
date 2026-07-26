# Agent1 Project Memory

## Prompt Rules (DO NOT VIOLATE)

### When editing LLM system prompts:

1. **Only remove fluff/narration** — text like "Let me analyze...", "Looking at this carefully..." that the LLM generates as intros. These waste tokens.

2. **NEVER remove requirements** — keep all instructions about:
   - Type safety (mypy strict, no unbound TypeVars, no forward-ref errors)
   - Validation (type-checking validation, py_compile, imports)
   - Anti-duplication (no _v1/_v2/_clean/_final variants)
   - Code quality (avoid circular imports)
   - Output format ([FILE: ...], ```python)

3. **CONFIRM** before bulk-editing multiple prompts simultaneously. Each prompt serves a specific purpose and requirements may differ.

### Anti-duplication safeguards added 2026-07-26:
- System prompts include: "NEVER create duplicate functions or classes. No _v1, _v2, _clean, _final variants. One implementation per concept."
- Post-generation: reject files with >10 near-duplicate functions
- Post-generation: reject files >50KB

### Model notes:
- `qwen3.6-27b-mtp` and `gemma-4-31b` — good results, no duplication
- `laguna-s-2.1` — generates duplicate function variants, avoid for code generation

## Architecture Notes

- agent.py is ~2200 lines monolithic CLI
- Phase 3 writes happen AFTER all batches generated — risk of data loss if interrupted
- All subprocess/file handling uses sync urllib (blocking in async context)
