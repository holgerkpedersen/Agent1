from __future__ import annotations

import asyncio
import logging

from .exceptions import ToolExecutionError

logger = logging.getLogger(__name__)

async def run_subprocess_with_timeout(
    cmd: list[str], 
    timeout_sec: float,
    cwd: str | None = None
) -> tuple[int, bytes, bytes]:
    """Run subprocess with timeout and return (returncode, stdout, stderr)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        return (proc.returncode or 0), stdout, stderr
    except asyncio.TimeoutError as e:
        logger.warning("Subprocess timed out after %ss: %s", timeout_sec, " ".join(cmd))
        raise ToolExecutionError("subprocess", f"Timed out after {timeout_sec} seconds") from e

# Type alias for consistent subprocess result handling
SubprocessResult = tuple[int, bytes, bytes]  # (returncode, stdout, stderr)