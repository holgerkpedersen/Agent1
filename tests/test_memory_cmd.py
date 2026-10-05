"""Regression tests for the ``memory`` REPL command (attribution read path).

The decision-attribution feature stamps every ``experiences`` row with the
model that *decided* it and logs one ``llm_decisions`` provenance row
(``agent_core/memory/attribution.py``).  ``Agent._record_llm_experience``
wired the WRITE path; before this command the four analytics helpers
(``attribution_summary`` / ``experiences_by_llm`` / ``success_rate_by_model``
/ ``decision_latency_histogram``) had no caller outside tests, so the
measurement the feature existed to provide was unreachable from the REPL.

These tests drive the REAL path end to end: a real ``Agent`` writes rows via
``_record_llm_experience``, then ``MemoryCommand`` reads the same DB back.
Nothing is simulated, and every file lands in ``tmp_path``.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import agent as agent_mod
from agent_core.commands.help_cmd import HelpCommand
from agent_core.commands.memory_cmd import MemoryCommand
from agent_core.commands.registry import CommandRegistry


def _make_agent(workspace: Path, tmp_path: Path, monkeypatch) -> "agent_mod.Agent":
    """Build a real Agent whose state files (incl. the memory DB) land in temp."""
    workspace.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        agent_mod, "CHAT_HISTORY_JSON_PATH", str(tmp_path / "chat_history.json")
    )
    monkeypatch.setattr(
        agent_mod, "AGENT_MEMORY_JSON_PATH", str(tmp_path / "agent_memory.json")
    )
    return agent_mod.Agent(workspace=str(workspace))


def _run(args: list[str], bot) -> bool:
    return asyncio.run(MemoryCommand().execute(args, bot))


def _db(tmp_path: Path) -> Path:
    """The SQLite DB ``_record_llm_experience`` derives from the JSON path."""
    return tmp_path / "agent_memory.db"


# ---------------------------------------------------------------------------
# 1. Surface: registration, name, help
# ---------------------------------------------------------------------------

def test_memory_command_is_registered() -> None:
    registry = CommandRegistry()
    agent_mod._register_commands(registry)
    assert "memory" in registry.names(), "the memory command must be dispatchable"

    command = registry.get("memory")
    assert command is not None
    assert command.name == "memory"
    assert "memory" in command.help_text.lower()


def test_help_lists_every_registered_command(capsys, monkeypatch) -> None:
    """``help`` builds its OWN registry, so the two lists can drift.

    ``help_cmd`` mirrors ``_register_commands`` by hand; before this test the
    mirror had silently fallen behind (``habits`` and ``issue`` were
    registered but absent from ``help``).  Pin the two against each other.
    """
    real = CommandRegistry()
    agent_mod._register_commands(real)

    asyncio.run(HelpCommand().execute([], None))
    out = capsys.readouterr().out

    listed = {
        line.split()[0]
        for line in out.splitlines()
        if line.startswith("  ") and line.split()
    }
    missing = real.names() - listed
    assert not missing, f"help does not list registered command(s): {sorted(missing)}"


def test_help_shows_memory_details(capsys) -> None:
    asyncio.run(HelpCommand().execute(["memory"], None))
    out = capsys.readouterr().out
    assert "memory:" in out
    assert "--latency" in out


# ---------------------------------------------------------------------------
# 2. Read-only: a missing DB is a clean message, never a crash or a new file
# ---------------------------------------------------------------------------

def test_missing_db_reports_cleanly_and_creates_nothing(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    bot = _make_agent(tmp_path / "ws_empty", tmp_path, monkeypatch)

    assert _run([], bot) is True, "the command must not exit the REPL"

    out = capsys.readouterr().out
    assert "No memory database" in out, f"expected a clean message, got: {out!r}"
    assert not _db(tmp_path).exists(), (
        "an inspection command must not create the DB as a side effect"
    )


def test_missing_db_json_is_still_valid_json(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    bot = _make_agent(tmp_path / "ws_empty_json", tmp_path, monkeypatch)

    _run(["--json"], bot)

    payload = json.loads(capsys.readouterr().out)
    assert payload["experiences"] == 0
    assert payload["by_llm"] == []


# ---------------------------------------------------------------------------
# 3. Round trip: real writes -> real reads, per model
# ---------------------------------------------------------------------------

def test_summary_reports_each_deciding_model(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    bot = _make_agent(tmp_path / "ws_rt", tmp_path, monkeypatch)

    # model-a: two decisions, one success (outcome 0.9) one failure (0.1).
    assert bot._record_llm_experience(
        "llm_decision", 0.9, decision_llm="model-a",
        decision_prompt="prompt one", run_id="run-1", latency_ms=120.0,
    ) is not None
    assert bot._record_llm_experience(
        "llm_decision", 0.1, decision_llm="model-a",
        decision_prompt="prompt two", run_id="run-1", latency_ms=180.0,
    ) is not None
    # model-b: one successful decision.
    assert bot._record_llm_experience(
        "llm_decision", 1.0, decision_llm="model-b",
        decision_prompt="prompt three", run_id="run-2", latency_ms=50.0,
    ) is not None

    _run([], bot)
    out = capsys.readouterr().out

    assert "model-a" in out and "model-b" in out
    # 3 experiences, all attributed; 3 provenance rows (the exact invariant).
    assert "3" in out
    assert "experiences: 3" in out
    assert "attributed: 3" in out
    assert "unattributed: 0" in out
    assert "llm_decisions: 3" in out


def test_success_rate_is_computed_per_model(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    bot = _make_agent(tmp_path / "ws_rate", tmp_path, monkeypatch)
    bot._record_llm_experience("llm_decision", 0.9, decision_llm="good")
    bot._record_llm_experience("llm_decision", 0.8, decision_llm="good")
    bot._record_llm_experience("llm_decision", 0.1, decision_llm="bad")

    _run(["--json"], bot)
    payload = json.loads(capsys.readouterr().out)

    rates = {r["decision_llm"]: r for r in payload["success_by_model"]}
    assert rates["good"]["successes"] == 2
    assert rates["good"]["total"] == 2
    assert rates["good"]["success_rate"] == 1.0
    assert rates["bad"]["successes"] == 0
    assert rates["bad"]["success_rate"] == 0.0


def test_omitting_decision_llm_falls_back_to_the_active_model(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """``decision_llm=None`` means RESOLVE it, not "unattributed".

    The stamp is designed so a call site cannot forget it: an explicit
    ``None`` is resolved from the agent's active model.  This is the contract
    that made the write path self-maintaining, so pin it from the read side.
    """
    bot = _make_agent(tmp_path / "ws_fallback", tmp_path, monkeypatch)
    assert bot._record_llm_experience("llm_decision", 0.9) is not None

    _run(["--json"], bot)
    payload = json.loads(capsys.readouterr().out)

    assert payload["experiences"] == 1
    assert payload["attributed"] == 1, (
        "an omitted decision_llm must still be attributed to the active model"
    )
    assert payload["by_llm"][0]["decision_llm"], "a concrete model name is recorded"


def test_unattributed_rows_are_not_credited_to_any_model(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """No model identity anywhere -> NULL attribution -> excluded from stats."""
    bot = _make_agent(tmp_path / "ws_unattr", tmp_path, monkeypatch)
    bot._record_llm_experience("llm_decision", 0.9, decision_llm="model-x")

    # Now make the agent genuinely unattributable (no model identity
    # available): the write must leave the attribution columns NULL rather
    # than inventing an author, and must log no provenance row.
    monkeypatch.setattr(
        "agent_core.memory.attribution.resolve_decision_llm", lambda _a: None
    )
    bot._record_llm_experience("llm_decision", 0.9)
    bot._record_llm_experience("llm_decision", 0.9)

    _run(["--json"], bot)
    payload = json.loads(capsys.readouterr().out)

    assert payload["experiences"] == 3
    assert payload["attributed"] == 1
    assert payload["unattributed"] == 2
    assert payload["llm_decisions"] == 1, (
        "the provenance invariant: one llm_decisions row per attributed row"
    )
    assert [r["decision_llm"] for r in payload["by_llm"]] == ["model-x"]


# ---------------------------------------------------------------------------
# 4. Latency histogram
# ---------------------------------------------------------------------------

def test_latency_histogram_buckets_logged_decisions(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    bot = _make_agent(tmp_path / "ws_lat", tmp_path, monkeypatch)
    bot._record_llm_experience("llm_decision", 0.9, decision_llm="m", latency_ms=50.0)
    bot._record_llm_experience("llm_decision", 0.9, decision_llm="m", latency_ms=150.0)
    bot._record_llm_experience("llm_decision", 0.9, decision_llm="m", latency_ms=190.0)

    _run(["--latency", "--bucket-ms", "100", "--json"], bot)
    payload = json.loads(capsys.readouterr().out)

    assert payload["latency_histogram"] == [
        {"bucket_ms": 0, "count": 1},
        {"bucket_ms": 100, "count": 2},
    ]


def test_latency_histogram_renders_a_table(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    bot = _make_agent(tmp_path / "ws_lat_txt", tmp_path, monkeypatch)
    bot._record_llm_experience("llm_decision", 0.9, decision_llm="m", latency_ms=50.0)

    _run(["--latency"], bot)
    out = capsys.readouterr().out

    assert "latency" in out.lower()
    assert "0-99" in out, f"expected a human-readable bucket range, got: {out!r}"


def test_bad_bucket_size_is_rejected(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    bot = _make_agent(tmp_path / "ws_lat_bad", tmp_path, monkeypatch)

    assert _run(["--latency", "--bucket-ms", "0"], bot) is True
    out = capsys.readouterr().out
    assert "bucket-ms" in out and "positive" in out


def test_unknown_flag_prints_usage(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    bot = _make_agent(tmp_path / "ws_usage", tmp_path, monkeypatch)

    assert _run(["--nope"], bot) is True
    out = capsys.readouterr().out
    assert "Usage: memory" in out
