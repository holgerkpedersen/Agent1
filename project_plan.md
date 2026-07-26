# Detailed Coding Plan: Comprehensive Bug Fixes and Code Quality Improvements

## Overview
This plan addresses all identified issues from the analysis report, focusing on fixing bugs while ensuring compliance with **mypy strict type checking**. We'll tackle each module systematically.

---

## Phase 1: Fix `agent.py` Issues  
**Estimated Effort:** Medium-High | **Files Affected:** `agent.py`, possibly related modules  

### Tasks:
#### ✅ Task A1 – Normalize Path Consistency (`read_file`) Deduplication Logic Fix  
- **Issue**: Double normalization mismatch causing deduplication failure.
- **Fix Strategy**:
  - Refactor `_normalize_path()` to accept both strict/non-strict modes uniformly.
  - Ensure `execute_tool("read_file")` uses consistent normalized path throughout flow:
    ```python
    def execute_tool(self, tool_name: str, args: dict[str, Any]) -> Result[Any]:
        ...
        if tool_name == "read_file":
            norm_path = self._normalize_path(args["path"], strict=True)
            if norm_path in self._files_read:
                return ToolResult(success=False, error=ToolExecutionError("Already read"))
            result = await self.read_file(norm_path)  # pass already-normalized path
    ```

#### ✅ Task A2 – Define Missing `search_file` Method Directly On Agent  
- **Issue**: Referenced in dispatch table but undefined directly.
- **Fix Strategy**:
  - Add explicit method definition on `Agent`:
    ```python
    async def search_file(self, pattern: str, directory: Optional[str] = None) -> List[FileInfo]:
        """Public-facing wrapper around _search_files"""
        return await self._search_files(pattern=pattern, root_dir=directory or ".")
    ```

#### ✅ Task A3 – Validate File Existence Before Returning `.py` Paths  
- **Issue**: `_parse_natural_language("read")` assumes existence without check.
- **Fix Strategy**:
  - Modify method to verify file exists before returning path:
    ```python
    def _parse_natural_language(self, intent: str) -> Optional[str]:
        if intent.startswith("read"):
            candidate = self._guess_py_file(intent[4:].strip())
            if os.path.exists(candidate):
                return candidate
            raise FileOperationError(f"Cannot find file matching {intent}")
    ```

---

## Phase 2: Consolidate Entities Definitions  
**Estimated Effort:** Low-Medium | **Files Affected:** `entities.py`, `agent_core/entities.py`  

### Tasks:
#### ✅ Task E1 – Merge Duplicate Exception Classes Into Single Source-of-truth (`agent_core/entities.py`)  
- **Issue**: Conflicting definitions between top-level and core package.
- **Fix Strategy**:
  - Move all shared exceptions/configs into `agent_core/entities.py`.
  - Update imports accordingly in dependent modules:
    ```python
    # In agent.py:
    from agent_core.entities import AgentError, FileOperationError, ToolExecutionError
    ```

#### ✅ Task E2 – Replace Generic TypeVar Usage With Concrete Exception Types  
- **Issue**: `Failure[E]` violates mypy strict rules due to unsupported generic over exceptions.
- **Fix Strategy**:
  - Rewrite using concrete subclass approach:
    ```python
    @dataclass(frozen=True)
    class Failure:
        reason: str
        error_type: ClassVar[type[Exception]] = ToolExecutionError

    # Or better yet, define subclasses explicitly for different failure types.
    ```

#### ✅ Task E3 – Enforce Immutability Where Appropriate  
- **Improvement**: Use frozen dataclasses for configuration objects.
- Example:
  ```python
  @dataclass(frozen=True)
  class AgentSettings:
      max_tokens: int = field(default=4096)
      temperature: float = field(default=0.7)
  ```

---

## Phase 3: Clean Up `analyze_handler.py` Dead Code & Improve Signature Extraction  
**Estimated Effort:** Medium | **Files Affected:** `agent_core/handlers/analyze_handler.py`, possibly `path_utils.py`  

### Tasks:
#### ✅ Task H1 – Fix `_SIGNATURE_PATTERN` Regex Input Source Mismatch  
- **Issue**: Operates on AST metadata instead of actual source text.
- **Fix Strategy**:
  - Pass raw function body string to regex matcher rather than class name placeholder:
    ```python
    def extract_signature(source_text: str) -> Optional[str]:
        match = _SIGNATURE_PATTERN.search(source_text)
        return match.group(1) if match else None
    ```

#### ✅ Task H2 – Remove `.register` Placeholder Assignment  
- **Issue**: Lambda assignment adds confusion.
- **Fix Strategy**:
  - Delete line entirely or replace with meaningful registration hook:
    ```python
    # Instead of useless lambda:
    AnalyzeCommand.register = functools.partial(AnalyzeCommand._default_register)
    ```

#### ✅ Task H3 – Adopt `ast.NodeVisitor` Pattern For Cleaner Traversal  
- **Improvement**: Replace manual AST inspection loops.
- Implementation sketch:
  ```python
  class FunctionExtractor(ast.NodeVisitor):
      def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
          signatures.append(node.name + "(" + ", ".join(arg.arg for arg in node.args.args) + ")")
          self.generic_visit(node)
  ```

---

## Phase 4: Address `path_utils.py` Export Mismatch  
**Estimated Effort:** Low | **Files Affected:** `agent_core/path_utils.py`, `agent_core/__init__.py`  

### Tasks:
#### ✅ Task P1 – Rename Private `_validate_path` To Public `normalize_path` Or Properly Alias It  
- **Issue**: Exported as public but defined privately.
- **Fix Strategy Options**:
  - Option A: Rename internal method to match export name directly.
  - Option B: Explicitly alias in `__init__.py`:
    ```python
    # In __init__.py:
    normalize_path = _validate_path
    validate_path = _strict_validate_path
    ```

---

## Phase 5: Complete Handler Wiring in `tool_router.py`  
**Estimated Effort:** Medium | **Files Affected:** `tool_router.py`, possibly new handler files  

### Tasks:
#### ✅ Task R1 – Register Shell Command Handler For `"run_command"` Tool Definition  
- **Issue**: Declared args/schema but no handler wired.
- **Fix Strategy**:
  - Create dedicated shell command handler class implementing interface contract:
    ```python
    @dataclass
    class ShellCommandHandler(ToolHandler):
        def execute(self, command_args: ShellCommandArgs) -> ToolResult[str]:
            proc = subprocess.run(command_args.command.split(), capture_output=True, timeout=command_args.timeout_seconds)
            return ToolResult(success=True, output=proc.stdout.decode()) if proc.returncode == 0 else ToolResult(...error...)
    ```

#### ✅ Task R2 – Add Schema Validation Enforcement Before Routing Execution Calls  
- **Improvement**: Prevent invalid tool invocations early.
- Implementation:
  ```python
  def route_execution(self, request: ToolRequest) -> Awaitable[ToolResult]:
      try:
          validated_args = validate_model(request.tool_name, request.arguments)
      except ValidationError as ve:
          raise ToolExecutionError(str(ve)) from ve
      handler = self._handlers.get(validated_args.__class__)
      if not handler:
          raise NotImplementedError(f"No handler for {validated_args}")
      return handler.execute(validated_args)
  ```

---

## Phase 6: Refactor String-Based Error Returns To Structured Exceptions  
**Estimated Effort:** High | **Files Affected:** All modules returning string errors (`agent.py`, `benchmark.py`) etc.  

### Tasks:
#### ✅ Task S1 – Convert Methods Returning Strings Like `"File not found"` Into Raising Typed Exceptions  
- **Issue**: Brittle programmatic handling due to unstructured returns.
- **Fix Strategy Across Modules**:
  - Identify methods returning string-based failures and refactor to raise appropriate exceptions:
    ```python
    # Before:
    def some_method(self) -> str | None:
        if not exists(path): return "File not found"

    # After:
    def some_method(self) -> Optional[str]:
        if not exists(path):
            raise FileOperationError(f"Path does not exist: {path}")
  ```

Apply this consistently in `agent.py`, `benchmark.py` where applicable.

---

## Phase 7: Address Typing Style Inconsistencies  
**Estimated Effort:** Medium | **Files Affected:** Several files lacking future annotations or PEP604 support  

### Tasks:
#### ✅ Task T1 – Standardize All Modules To Use `from __future__ import annotations` + PEP 604 Unions  
- **Issue**: Mixed typing styles reduce clarity/maintainability.
- **Fix Strategy**:
  - Add at top of every `.py` file needing modern syntax:
    ```python
    from __future__ import annotations
    ```
  - Replace legacy union expressions like `Union[A, B]` with `A | B`.

---

## Phase 8: Improve Benchmark Configuration Flexibility  
**Estimated Effort:** Medium | **Files Affected:** `benchmark.py`  

### Tasks:
#### ✅ Task B1 – Make Scoring Thresholds Configurable Per Model/Category Pair  
- **Issue**: Hardcoded subjective thresholds bias automated metrics.
- **Fix Strategy**:
  - Introduce CLI flag-based overrides for timeouts/scoring weights:
    ```python
    parser.add_argument("--timeout-per-model", nargs="+")
    parser.add_argument("--retry-backoff-factor", default=1.0)
  ```

#### ✅ Task B2 – Enhance Haiku Syllable Estimation Accuracy Using Vowel Grouping Heuristics  
- **Improvement**: More precise syllable detection improves fairness.
- Strategy:
  - Replace simple vowel counting with improved heuristic considering silent 'e':
    ```python
    def estimate_syllables(word: str) -> int:
        vowels = "aeiouy"
        count = sum(1 for c in word if c.lower() in vowels)
        if word.endswith("e") and len(word)>2:
            count -= 1
        return max(count, 1)
    ```

---

## Phase 9: Add Unit Test Coverage For Critical Edge Cases  
**Estimated Effort:** Medium | **Files Affected:** New test suite additions (`tests/test_agent_paths.py`, `tests/test_tool_router.py`) etc.  

### Tasks:
#### ✅ Task U1 – Write Tests Around Path Normalization Edge Cases (`/c/`, `/d/`)  
- Focus Areas:
  - Absolute vs relative paths
  - Windows-style separators conversion
  - Trailing slashes trimming behavior

Example test snippet:
```python
def test_normalize_path_trailing_slash():
    assert normalize_path("/some/dir/") == "/some/dir"
```

#### ✅ Task U2 – Add Tool Dispatch Logic Tests Verifying Correct Handler Routing  
- Scenarios Tested:
  - Valid tool request routed to correct handler instance
  - Invalid schema rejected gracefully via exception raise

---

## Final Compliance Check Requirements (All Must Pass):
Before merging changes, ensure these conditions hold true:

| Requirement | Verification Method |
|------------|---------------------|
| ✅ Full mypy strict type checking passes (`--strict`) | Run `mypy --strict .` post-change |
| ✅ No runtime circular imports introduced | Static import graph check via tooling |
| ✅ Structured exception usage replaces string returns everywhere feasible | Manual audit + grep verification |
| ✅ Consistent typing style enforced across all modules | Confirm presence of `from __future__ import annotations` in all files requiring modern syntax |

---

## Summary Timeline Estimate:
| Phase | Duration Estimate | Priority Level |
|------|------------------|----------------|
| Phases 1–3 | ~2 days | 🔴 High |
| Phases 4–6 | ~1 day | 🟡 Medium |
| Phases 7–9 | ~1.5 days | 🟢 Lower |

Total estimated effort: **~4.5 developer-days** assuming familiarity with codebase structure and existing tooling setup.

Would you like me to generate sample implementations or diffs for any particular phase/task?