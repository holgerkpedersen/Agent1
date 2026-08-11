import subprocess
import logging
from pathlib import Path
from typing import Any, Dict, List

from agent_core.security.allowlist import is_command_allowed
from agent_core.security.path_utils import SecurityViolationError, normalize_path

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Safely executes tools and commands without using shell=True."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.workspace_root = workspace_root or Path.cwd()

    def execute(self, tool_name: str, args: List[str], **kwargs: Any) -> Dict[str, Any]:
        if not is_command_allowed(tool_name):
            raise SecurityViolationError(f"Command '{tool_name}' is not allowed.")

        safe_args = self._sanitize_path_args(args)

        try:
            proc = subprocess.run(
                [tool_name] + safe_args,
                capture_output=True,
                text=True,
                timeout=kwargs.get("timeout", 30),
                cwd=self.workspace_root,
                check=False,
            )
            return {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
                "success": proc.returncode == 0,
            }
        except subprocess.TimeoutExpired as exc:
            logger.warning("Tool execution timed out: %s", tool_name)
            return {"error": str(exc), "success": False}
        except Exception as exc:
            logger.error("Tool execution failed: %s", exc)
            return {"error": str(exc), "success": False}

    def _sanitize_path_args(self, args: List[str]) -> List[str]:
        sanitized: List[str] = []
        for arg in args:
            if "/" in arg or "\\" in arg or Path(arg).is_absolute():
                try:
                    resolved = normalize_path(self.workspace_root, arg)
                    sanitized.append(str(resolved))
                except SecurityViolationError:
                    raise
                except Exception:
                    sanitized.append(arg)
            else:
                sanitized.append(arg)
        return sanitized