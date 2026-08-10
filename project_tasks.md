<tool_call> or XML tags.
- PATH RULES: New files MUST use a sub-package prefix (`agent1/...`, `agent_core/...`). I will create `agent_core/types.py`.
- SIZE RULES: Max 150 lines.
- STDLIB SHADOWING: Avoid names like `logging`, `config`, etc. `types` is fine, but `agent_core/types.py` might shadow `typing`? No, `types` is stdlib but usually safe if prefixed properly or if it's just a module file. To be strictly safe, I'll use `agent_core/shared_types.py`.
- CRITICAL: Do not create files overlapping with existing ones in the same directory. The prompt lists many existing files in `agent_core/`, including `config.py`, `exceptions.py`, etc. It does NOT list `shared_types.py` or `types.py`. I will use `agent_core/shared_types.py`.

Let's extract the types carefully:
- `AppConfig` (from spec)
- `SecurityViolationError` (from analysis/spec)
- `ToolName`, `ToolRequest` (tool routing)
- LLM message/response structures (`ChatMessage`, `Choice`, `ChatCompletionResponse`, etc.)
- Context/State tracking types

I will structure it cleanly, strictly typed, using `pydantic.BaseModel` and standard library. I'll ensure mypy strict compliance with explicit types and `from __future__ import annotations`.

Check existing files in `agent_core/`:
`__init__.py`, `agent.py`, `agent_file_context_interface.py`, `config.py`, `constants.py`, `context_management.py`, `entities.py`, `exceptions.py`, `file_context_retriever.py`, `file_searcher.py`, `file_system.py`, `llm_client.py`

I will create `agent_core/shared_types.py`. It doesn't overlap.
Wait, the prompt says "Extract shared classes/types." I will just output the code block for `agent_core/shared_types.py`.

Let's draft it carefully within 150 lines.
```python
from __future__ import annotations
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class AppConfig:
    workspace_root: str
    lm_studio_url: str = "http://localhost:1234/v1"
    max_iterations: int = 50
    allowed_shell_binaries: list[str] = field(default_factory=lambda: ["python", "ls", "cat"])
    enable_symlink_traversal: bool = False


class SecurityViolationError(Exception):
    pass

DANGEROUS_PATTERN: Final[re.Pattern[str]] = re.compile(r"[;&|`$(){}<>!\\]", re.IGNORECASE)


class ToolName(str, Enum):
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    EXECUTE_SHELL = "execute_shell"


class ToolRequest(BaseModel):
    name: ToolName
    arguments: dict[str, str] = Field(default_factory=dict)


class FunctionCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    function: FunctionCall
    id: str
    type: str = "function"


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(BaseModel):
    index: int
    message: ChatMessage


class ChatCompletionResponse(BaseModel):
    choices: list[Choice]
```

Check constraints:
- Output ONLY Python code — no intro text. Start with ```python.
- All types valid, mypy strict compliant.
- No forward-ref errors (using `from __future__ import annotations` and defining before use).
- Avoid circular imports.
- PATH RULES: `agent_core/shared_types.py` is used as filename context implicitly or I will just output the code block with a comment indicating the path, but the prompt says "Output ONLY Python code". I'll just put the code. To be safe, I'll add a docstring with the intended file path.
- SIZE RULES: 48 lines. Well under 150.
- STDLIB SHADOWING: `shared_types` is fine.

One detail: The prompt says "Extract shared classes/types." It might expect just the code block. I will provide exactly that.
I'll verify mypy strict compatibility: `dict[str, Any]`, `list[Choice]`, etc. are valid Python 3.9+. With `from __future__ import annotations`, it's fully postponed. Pydantic v2 handles these fine.
All good.
Output matches response.✅
Proceeds. 
Wait, should I include `CorrelationIdContext` or similar? The prompt lists existing files that already have some of these. I'll stick to the extracted ones from the spec/analysis that are missing or need consolidation. The provided list shows `agent_core/config.py`, `agent_core/exceptions.py`, etc. exist. So extracting into a new file is fine as long as names don't clash, but I should avoid duplicating existing ones if possible. The prompt says "Extract shared classes/types." from the *spec and analysis*. I'll just output the clean types.
I will ensure no overlap with the listed files in `agent_core/`. `shared_types.py` is safe.
Code block only. No extra text.
Done. 
Output Generation. 
(Self-Correction/Refinement during thought)
The prompt says "Output ONLY Python code — no intro text. Start with ```python."
I will strictly follow that.
One minor thing: `pydantic` might not be installed in the target env, but it's standard for this spec. I'll keep it.
All constraints met.
Proceeds. 
[Output Generation] -> *Proceeds*
</think>

```python
"""Intended file path: agent_core/shared_types.py"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class AppConfig:
    """Validated application settings."""
    workspace_root: str
    lm_studio_url: str = "http://localhost:1234/v1"
    max_iterations: int = 50
    allowed_shell_binaries: list[str] = field(default_factory=lambda: ["python", "ls", "cat", "grep", "find"])
    enable_symlink_traversal: bool = False


class SecurityViolationError(Exception):
    """Raised when a path operation violates workspace boundaries or security policies."""
    pass

DANGEROUS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[;&|`$(){}<>!\\]", re.IGNORECASE
)


class ToolName(str, Enum):
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    EXECUTE_SHELL = "execute_shell"


class ToolRequest(BaseModel):
    """Strictly typed tool invocation schema."""
    name: ToolName
    arguments: dict[str, str] = Field(default_factory=dict)


class FunctionCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    function: FunctionCall
    id: str
    type: str = "function"


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(BaseModel):
    index: int
    message: ChatMessage


class ChatCompletionResponse(BaseModel):
    choices: list[Choice]
```