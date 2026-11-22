"""Regression tests: bare ``/model`` is an interactive picker (opencode parity).

User request: "/model should show the available models and let me pick one
like in the opencode chat".  Before the fix, ``execute`` called
``_list_models(agent, interactive=True)`` but ``_list_models`` IGNORED the
flag — the command dead-ended in a printed catalog with no way to select.

Contract pinned here:

* ``model`` with no args on a TTY lists AND prompts; a number selects the Nth
  LM Studio row (the same numbering ``model list`` prints).
* Empty input / ``q`` / EOF / Ctrl+C cancel without switching or persisting.
* An out-of-range number is rejected without switching.
* ``model pick`` forces the prompt.
* Non-TTY (piped) input only lists — it must never block on stdin.
* ``model list`` stays READ-ONLY and never prompts (multi-shell safety,
  see tests/test_multi_shell_model_isolation.py).

Hermetic: every catalog/provider touchpoint is stubbed; no server or file is
touched.
"""

import asyncio
import sys
import types
from typing import Any

import pytest

_MODELS = [
    {
        "key": "laguna-s-2.1",
        "display_name": "Laguna S 2.1",
        "params_string": "9B",
        "size_bytes": 1_000_000,
        "loaded": True,
        "instance_id": "inst-1",
    },
    {
        "key": "qwen3.5-9b-mtp",
        "display_name": "Qwen3.5 9B MTP",
        "params_string": "9B",
        "size_bytes": 2_000_000,
        "loaded": False,
        "instance_id": None,
    },
]


class _FakeTTY:
    """Stdin stand-in that reports itself as a terminal."""

    def isatty(self) -> bool:
        return True


class _FakePipe:
    """Stdin stand-in for piped/batch input."""

    def isatty(self) -> bool:
        return False


def _make_agent(current: str = "laguna-s-2.1") -> Any:
    provider = types.SimpleNamespace(model_name=current, _profile_name=None)
    llm = types.SimpleNamespace(model_name=current, _provider=provider)
    return types.SimpleNamespace(llm=llm)


@pytest.fixture()
def env(monkeypatch):
    """Stub the model command's I/O + every catalog; record side effects."""
    import agent_core.commands.model_cmd as mc

    state: dict[str, Any] = {
        "persisted": [],
        "loads": [],
        "picked": [],
        "prompts": [],
        "answer": "",
    }

    cmd = mc.ModelCommand()
    monkeypatch.setattr(cmd, "_fetch_models", lambda: (list(_MODELS), []))
    monkeypatch.setattr(cmd, "_get_vram_display", lambda models: "fake vram")
    monkeypatch.setattr(cmd, "_opencode_catalog", lambda agent: ([], [], False))
    monkeypatch.setattr(cmd, "_llama_catalog", lambda agent: [])
    monkeypatch.setattr(cmd, "_openrouter_catalog", lambda agent, free_only=True: [])
    monkeypatch.setattr(cmd, "_lemonade_catalog", lambda agent: [])
    monkeypatch.setattr(
        cmd,
        "_try_load_or_unload",
        lambda key: (state["loads"].append(key) or (True, f"{key} loaded")),
    )
    monkeypatch.setattr(
        cmd, "_rebuild_provider", lambda agent, name, provider_override=None: None
    )
    monkeypatch.setattr(
        mc,
        "persist_model_choice",
        lambda name, provider=None: state["persisted"].append((name, provider)),
    )

    def _fake_read_input(prompt: str = "") -> str:
        state["prompts"].append(prompt)
        return str(state["answer"])

    monkeypatch.setattr(mc, "read_input", _fake_read_input)

    # Capture what the picker hands to _switch_model.
    real_switch = mc.ModelCommand._switch_model

    async def _spy_switch(self, args, agent):
        state["picked"].append(list(args))
        return await real_switch(self, args, agent)

    monkeypatch.setattr(mc.ModelCommand, "_switch_model", _spy_switch)
    return cmd, state


def _run(cmd, agent, args):
    return asyncio.run(cmd.execute(args, agent))


# ---------------------------------------------------------------------------
# The fix: a number selects a model
# ---------------------------------------------------------------------------


class TestPickerSelectsNumberedModel:
    def test_bare_model_prompts_and_a_number_switches(
        self, env, monkeypatch, capsys
    ):
        cmd, state = env
        monkeypatch.setattr(sys, "stdin", _FakeTTY())
        state["answer"] = "2"
        agent = _make_agent("laguna-s-2.1")

        assert _run(cmd, agent, []) is True

        out = capsys.readouterr().out
        # The listing was shown AND a pick was offered.
        assert "Laguna S 2.1" in out
        assert len(state["prompts"]) == 1
        assert "Pick a model number" in state["prompts"][0]
        assert "1-2" in state["prompts"][0]
        # #2 in the listing is the second LM Studio model.
        assert state["picked"] == [["qwen3.5-9b-mtp"]]
        assert agent.llm.model_name == "qwen3.5-9b-mtp"
        assert state["persisted"] == [("qwen3.5-9b-mtp", "lmstudio")]

    def test_numbering_matches_the_listing_order(self, env, monkeypatch):
        cmd, state = env
        monkeypatch.setattr(sys, "stdin", _FakeTTY())
        state["answer"] = "1"

        _run(cmd, _make_agent("qwen3.5-9b-mtp"), [])

        assert state["picked"] == [["laguna-s-2.1"]]

    def test_free_text_name_is_forwarded_to_switch(self, env, monkeypatch):
        cmd, state = env
        monkeypatch.setattr(sys, "stdin", _FakeTTY())
        state["answer"] = "opencode-zen/hy3-free"

        _run(cmd, _make_agent(), [])

        assert state["picked"] == [["opencode-zen/hy3-free"]]

    def test_pick_subcommand_forces_the_prompt(self, env, monkeypatch):
        cmd, state = env
        # Even without a TTY, `model pick` must still prompt.
        monkeypatch.setattr(sys, "stdin", _FakePipe())
        state["answer"] = "2"

        _run(cmd, _make_agent(), ["pick"])

        assert state["picked"] == [["qwen3.5-9b-mtp"]]


# ---------------------------------------------------------------------------
# Cancelling must change nothing
# ---------------------------------------------------------------------------


class TestPickerCancelIsInert:
    @pytest.mark.parametrize("answer", ["", "   ", "q", "quit", "cancel"])
    def test_cancel_leaves_model_untouched(self, env, monkeypatch, answer):
        cmd, state = env
        monkeypatch.setattr(sys, "stdin", _FakeTTY())
        state["answer"] = answer
        agent = _make_agent("laguna-s-2.1")

        _run(cmd, agent, [])

        assert state["picked"] == []
        assert state["persisted"] == []
        assert agent.llm.model_name == "laguna-s-2.1"

    def test_eof_is_treated_as_cancel(self, env, monkeypatch):
        """read_input turns EOFError into "" — the picker treats that as keep."""
        cmd, state = env
        monkeypatch.setattr(sys, "stdin", _FakeTTY())
        state["answer"] = ""  # what read_input returns on EOF

        _run(cmd, _make_agent(), [])

        assert state["picked"] == []
        assert state["persisted"] == []

    def test_out_of_range_number_is_rejected(self, env, monkeypatch, capsys):
        cmd, state = env
        monkeypatch.setattr(sys, "stdin", _FakeTTY())
        state["answer"] = "99"

        _run(cmd, _make_agent(), [])

        assert state["picked"] == []
        assert "No model number 99" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Non-interactive safety
# ---------------------------------------------------------------------------


class TestNonInteractiveStaysReadOnly:
    def test_piped_stdin_lists_without_prompting(self, env, monkeypatch, capsys):
        cmd, state = env
        monkeypatch.setattr(sys, "stdin", _FakePipe())
        state["answer"] = "2"

        _run(cmd, _make_agent(), [])

        out = capsys.readouterr().out
        assert "Laguna S 2.1" in out
        # No prompt was issued, so nothing could be selected.
        assert state["prompts"] == []
        assert state["picked"] == []

    def test_model_list_never_prompts_nor_switches(self, env, monkeypatch):
        cmd, state = env
        monkeypatch.setattr(sys, "stdin", _FakeTTY())
        state["answer"] = "2"

        assert _run(cmd, _make_agent(), ["list"]) is True

        assert state["prompts"] == []
        assert state["picked"] == []
        assert state["persisted"] == []
