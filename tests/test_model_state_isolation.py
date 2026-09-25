"""Runtime-state isolation: failover must never clobber the real model.json/.env.

Regression (2026-09-25): a failover test with a stub provider named "go"
triggered ``FailoverProvider.chat``'s persist-the-working-model path, which
rewrote the developer's real ``model.json`` AND ``.env``
(``AGENT_MODEL=go``).  Every failover test module now stubs
``persist_model_choice``; this test proves the sandbox itself protects the
real files, so a future test that forgets to stub cannot silently corrupt the
session's persisted model.
"""

import asyncio
import json
import pathlib
from typing import Any

from agent_core.llm.provider import FailoverProvider


class _Stub:
    def __init__(self, name: str, fail: bool):
        self.name = name
        self.model_name = name
        self.temperature = 0.0
        self.max_tokens = 16
        self._profile_name = None
        self.fail = fail

    def apply_profile(self, *args: Any) -> None:
        pass

    async def chat(self, messages, tools=None, max_tokens=None, disable_thinking=False):
        if self.fail:
            return "[Error: connection refused]"
        return f"ok:{self.name}"


def test_failover_persists_only_in_the_sandbox(tmp_path, monkeypatch):
    import agent_core.constants as constants

    real_model = pathlib.Path(constants.MODEL_JSON_PATH)
    before = real_model.read_text(encoding="utf-8") if real_model.is_file() else None

    monkeypatch.setattr(
        constants, "MODEL_JSON_PATH", str(tmp_path / "model.json"),
    )
    monkeypatch.chdir(tmp_path)  # persist_model_choice writes .env in cwd

    provider = FailoverProvider(
        [_Stub("dead", fail=True), _Stub("working", fail=False)],
        model_name="dead",
    )
    reply = asyncio.run(provider.chat([{"role": "user", "content": "hi"}]))
    assert reply == "ok:working"

    # The working model WAS persisted — into the sandbox.
    sandbox = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
    assert sandbox["model"] == "working"

    # ...and the real developer state is byte-for-byte untouched.
    after = real_model.read_text(encoding="utf-8") if real_model.is_file() else None
    assert after == before
