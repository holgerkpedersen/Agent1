"""Regression: a timed-out subprocess must be killed and reaped, not left running.

Leaving the child alive leaves its asyncio pipe transports open, which surfaces
as PytestUnraisableExceptionWarning / ResourceWarning on Windows.
"""
import asyncio

import pytest

import agent_core.subprocess_utils as _mod
from agent_core.exceptions import ToolExecutionError


def test_timed_out_process_is_killed_and_reaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned: list[asyncio.subprocess.Process] = []
    real_exec = asyncio.create_subprocess_exec

    async def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        proc = await real_exec(*args, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(_mod.asyncio, "create_subprocess_exec", spy)

    with pytest.raises(ToolExecutionError, match="Timed out after 1.0 seconds"):
        asyncio.run(
            _mod.run_subprocess_with_timeout(
                ["python", "-c", "import time; time.sleep(10)"], timeout_sec=1.0
            )
        )

    assert len(spawned) == 1
    # returncode is None while the child still runs; after kill+wait it is set.
    assert spawned[0].returncode is not None
