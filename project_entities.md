<tool_call>` or XML tags.

Looking at the spec and analysis, I can extract these key shared types/classes:
- Configuration models (`AppConfig`, settings)
- Security/Path utilities (`SecurityViolationError`, path validation functions/types)
- Sanitizer types/constants
- LLM client/response types
- Tool routing/validation schemas (`ToolName`, `ToolRequest`)
- Agent orchestrator types/context

I will create a single Python file (or multiple if needed, but the prompt says "Output ONLY Python code", I'll put it in one block or logically separate files within the block) containing these extracted shared classes/types. To be safe and comply with "Extract shared classes/types", I'll define them in a clean, modular way that could be split into `types.py`, `config.py`, `security.py`, etc., but I will output them as a single cohesive Python script or clearly separated modules within the code block. Actually, it's better to structure it as a package-like set of classes/functions in one block for simplicity, or use standard module separation syntax if needed. I'll just define them all in one file for the extraction task, properly typed.

Let's list the types to extract:
1. `AppConfig` (Pydantic settings)
2. `SecurityViolationError`
3. `ToolName` (Enum)
4. `ToolRequest` (Pydantic model)
5. LLM message/response structures
6. Context/State tracking types

I will ensure strict typing, no forward refs issues by using `from __future__ import annotations`, and valid Pydantic v2 syntax. I'll avoid circular imports by keeping it flat or carefully structured.

Structure:
```python
from __future__ import annotations
import re
from enum import Enum
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Configuration ---
class AppConfig(BaseSettings): ...

# --- Security & Paths ---
class SecurityViolationError(Exception): ...
DANGEROUS_PATTERN: Final[re.Pattern[str]] = ...

# --- Tool Routing ---
class ToolName(str, Enum): ...
class ToolRequest(BaseModel): ...

# --- LLM Types ---
class ChatMessage(BaseModel): ...
class FunctionCall(BaseModel): ...
class ToolCall(BaseModel): ...
class Choice(BaseModel): ...
class ChatCompletionResponse(BaseModel): ...
```

Wait, the prompt says "Extract shared classes/types." I will focus on the core domain models that would be shared across modules.
I'll make sure everything is strictly typed and valid.
No intro text. Start directly with ```python.
Check mypy strict compliance: `from __future__ import annotations` handles forward refs. Use explicit types where possible. Pydantic v2 compatible.

Let's draft it carefully.
```python
from __future__ import annotations
import re
from enum import Enum
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class AppConfig(BaseSettings):
    """Validated application settings."""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    workspace_root: str
    lm_studio_url: str = "http://localhost:1234/v1"
    max_iterations: int = 50
    allowed_shell_binaries: list[str] = ["python", "ls", "cat", "grep", "find"]
    enable_symlink_traversal: bool = False


# ---------------------------------------------------------------------------
# Security & Path Utilities
# ---------------------------------------------------------------------------
class SecurityViolationError(Exception):
    """Raised when a path operation violates workspace boundaries or security policies."""
    pass

DANGEROUS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[;&|`$(){}<>!\\]", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Tool Routing & Validation
# ---------------------------------------------------------------------------
class ToolName(str, Enum):
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    EXECUTE_SHELL = "execute_shell"


class ToolRequest(BaseModel):
    """Strictly typed tool invocation schema."""
    name: ToolName
    arguments: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM Interaction Types
# ---------------------------------------------------------------------------
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
- All types must be valid — no unbound TypeVars, no forward-ref errors. (Using `from __future__ import annotations` handles this safely for Pydantic and standard typing)
- Must pass mypy strict. (Types are explicit, `Any` is used appropriately where needed, `Final` for constants, `Field` from pydantic)
- Avoid circular imports. (Flat structure, no cross-references that would cause cycles)
- Never use <tool_call> or XML tags.

One minor thing: `pydantic_settings` might not be strictly necessary if I just use standard Pydantic, but it's fine. I'll stick to `BaseModel` for simplicity and avoid external deps beyond pydantic itself, or keep it as is since it was in the spec. Actually, `BaseSettings` is fine.
I will verify mypy strict compatibility: `dict[str, Any]`, `list[Choice]`, etc. are all valid Python 3.9+ syntax. With `from __future__ import annotations`, everything is postponed evaluation, so no forward ref errors even if I referenced them. But here they are defined before use anyway.
I'll adjust slightly to ensure maximum compatibility and strictness: replace `Any` with concrete types where possible, or keep it. It's fine.
The prompt asks to "Extract shared classes/types." from the provided spec/analysis. The extracted types match exactly what's needed for a clean architecture.

I will output exactly the code block. No extra text.
Ready. 
One check: `pydantic_settings` is an external package. I'll just use standard `BaseModel` and add a comment or keep it simple. Actually, keeping `AppConfig` as `BaseSettings` is fine per spec. I'll stick with it.
All good.
Output matches response.✅
Proceeds. 
[Self-Correction/Verification during thought]
Wait, the prompt says "Extract shared classes/types." It doesn't specify a file name. I'll just output the code block.
I will make sure there are no syntax errors and it strictly follows Python typing conventions.
`from __future__ import annotations` is crucial for mypy strict with forward refs or postponed evaluation, though not strictly needed here since all types are defined before use or are built-ins. I'll keep it for safety.
I'll remove `pydantic_settings` to avoid dependency issues in a pure type extraction context, and just use `BaseModel` with a classmethod or standard init, but the spec explicitly mentioned Pydantic settings. I'll keep it but add a fallback comment if needed? No, just stick to standard libraries + pydantic as requested implicitly by the architecture. Actually, I'll replace `pydantic_settings` with standard `BaseModel` and add a note or just use `dataclass` for config to avoid external deps in type extraction? The prompt says "Extract shared classes/types." I'll keep it clean with just `pydantic.BaseModel`.

Let's refine:
```python
from __future__ import annotations
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AppConfig:
    """Validated application settings."""
    workspace_root: str
    lm_studio_url: str = "http://localhost:1234/v1"
    max_iterations: int = 50
    allowed_shell_binaries: list[str] = field(default_factory=lambda: ["python", "ls", "cat", "grep", "find"])
    enable_symlink_traversal: bool = False


# ---------------------------------------------------------------------------
# Security & Path Utilities
# ---------------------------------------------------------------------------
class SecurityViolationError(Exception):
    """Raised when a path operation violates workspace boundaries or security policies."""
    pass

DANGEROUS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[;&|`$(){}<>!\\]", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Tool Routing & Validation
# ---------------------------------------------------------------------------
class ToolName(str, Enum):
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    EXECUTE_SHELL = "execute_shell"


class ToolRequest(BaseModel):
    """Strictly typed tool invocation schema."""
    name: ToolName
    arguments: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM Interaction Types
# ---------------------------------------------------------------------------
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
This is cleaner, uses only `pydantic` and stdlib, strictly typed, passes mypy strict. No forward refs issues. Matches spec requirements. Output matches exactly.✅
Proceeds. 
Output Generation. 
(Note: I will ensure the output starts immediately with ```python as requested.) 
All constraints met.
Done. 
[Output Generation] -> *Proceeds*
</think>

```python
from __future__ import annotations
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AppConfig:
    """Validated application settings."""
    workspace_root: str
    lm_studio_url: str = "http://localhost:1234/v1"
    max_iterations: int = 50
    allowed_shell_binaries: list[str] = field(default_factory=lambda: ["python", "ls", "cat", "grep", "find"])
    enable_symlink_traversal: bool = False


# ---------------------------------------------------------------------------
# Security & Path Utilities
# ---------------------------------------------------------------------------
class SecurityViolationError(Exception):
    """Raised when a path operation violates workspace boundaries or security policies."""
    pass

DANGEROUS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[;&|`$(){}<>!\\]", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Tool Routing & Validation
# ---------------------------------------------------------------------------
class ToolName(str, Enum):
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    EXECUTE_SHELL = "execute_shell"


class ToolRequest(BaseModel):
    """Strictly typed tool invocation schema."""
    name: ToolName
    arguments: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM Interaction Types
# ---------------------------------------------------------------------------
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