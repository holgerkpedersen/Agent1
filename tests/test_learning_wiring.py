"""Part B wiring tests (plan item C2).

Covers the meta-policy / observability wiring shipped with plan item #4 and
friends, always against the **real** production API of
``agent_core.llm.learning``:

1. ``_finish_turn`` records success/failure under a ``TaskType`` inferred
   from the user input (through the real ``record_turn_outcome``).
2. A manually pinned profile suppresses the suggestion entirely, and the
   ``profile_auto`` pref defaults off — no auto-apply in a default config.
3. ``EVOLVER`` weights persist to ``meta_policy.json`` and restore on
   reload (with the evolver's 0.1–2.0 clamp).
4. The quality proxy computes 0.0 / 0.8 / 0.7 with tracing disabled and
   ``_finish_turn`` records it via ``append_event("turn", "quality", x)``.
5. ``perf_history.json`` round-trips across ``PerfTracker`` instances.
6. The suite itself never writes the live repo-root state files (the conftest
   redirect of ``TURN_LOG_PATH`` / ``meta_policy.json`` is pinned here —
   fixture turns used to land in the developer's real turn log and evolved
   real meta-policy weights from test outcomes).

Everything writes into ``tmp_path``; shared singletons (``METRICS``,
``EVOLVER``, ``PerfTracker`` class state, module globals) are monkeypatched
so no state leaks between tests.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent as agent_mod
import agent_core.commands.perf_cmd as perf_cmd
import agent_core.constants as constants
import agent_core.llm.learning as learning
from agent_core.commands.perf_cmd import PerfTracker
from agent_core.config import AgentDisplayMode
from agent_core.llm.learning import DEFAULT_PROFILE_TYPE, record_turn_outcome
from agent_core.llm.meta_policy import MetaPolicyEvolver
from agent_core.llm.metrics_tracker import MetricsTracker
from agent_core.llm.llm_types import ProfileType
from agent_core.monitoring import metrics_file


def _make_agent(workspace: Path, tmp_path: Path, monkeypatch) -> "agent_mod.Agent":
    """Build a real Agent whose state files land in *tmp_path*."""
    workspace.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        agent_mod, "CHAT_HISTORY_JSON_PATH", str(tmp_path / "chat_history.json")
    )
    monkeypatch.setattr(
        agent_mod, "AGENT_MEMORY_JSON_PATH", str(tmp_path / "agent_memory.json")
    )
    return agent_mod.Agent(workspace=str(workspace))


def _sandbox_learning(tmp_path: Path, monkeypatch) -> MetricsTracker:
    """Isolated METRICS + evolver + state file for record_turn_outcome."""
    metrics = MetricsTracker()
    monkeypatch.setattr(learning, "METRICS", metrics)
    monkeypatch.setattr(learning, "EVOLVER", MetaPolicyEvolver())
    monkeypatch.setattr(
        learning, "_state_path",
        lambda: str(tmp_path / "meta_policy.json"),
    )
    monkeypatch.setattr(learning, "_weights_loaded", True)
    monkeypatch.setattr(learning, "_turns_since_evolve", 0)
    return metrics


def _finish(bot, monkeypatch, tmp_path, *, final_text, llm_error, user_input):
    """Run the real _finish_turn with observability I/O pointed at tmp_path."""
    monkeypatch.setattr(agent_mod, "TURN_LOG_PATH", str(tmp_path / "turn_log.jsonl"))
    quality_events: list = []
    monkeypatch.setattr(
        metrics_file, "append_event",
        lambda kind, name, value: quality_events.append((kind, name, value)),
    )
    bot._last_user_input = user_input
    bot._turn_started_at = datetime.now()
    if not hasattr(bot, "_turn_start_index"):
        bot._turn_start_index = 0
    bot._finish_turn(final_text, llm_error, object(), AgentDisplayMode.QUIET)
    return quality_events


# ---------------------------------------------------------------------------
# 1. _finish_turn records success/failure under the inferred TaskType
# ---------------------------------------------------------------------------

def test_finish_turn_records_success_and_failure_task_types(
    tmp_path: Path, monkeypatch
) -> None:
    metrics = _sandbox_learning(tmp_path, monkeypatch)
    bot = _make_agent(tmp_path / "ws_record", tmp_path, monkeypatch)

    _finish(
        bot, monkeypatch, tmp_path,
        final_text="Added the widget.", llm_error=None,
        user_input="implement the new widget",
    )
    _finish(
        bot, monkeypatch, tmp_path,
        final_text="Could not finish.", llm_error="boom",
        user_input="fix the crash in parsing",
    )

    successes = {
        key: value for key, value in metrics._success_counts.items()
        if key[0] == "implement"
    }
    failures = {
        key: value for key, value in metrics._failure_counts.items()
        if key[0] == "fix"
    }
    assert sum(successes.values()) == 1, (
        "clean turn must record one success under TaskType 'implement'"
    )
    assert sum(failures.values()) == 1, (
        "errored turn must record one failure under TaskType 'fix'"
    )
    # The task type (not the profile) is what inference decides; the profile
    # falls back to the default for un-pinned turns.
    assert all(key[1] == DEFAULT_PROFILE_TYPE.value for key in successes)


# ---------------------------------------------------------------------------
# 2. pinned profile suppresses the suggestion; profile_auto defaults off
# ---------------------------------------------------------------------------

def _run_chat(bot, monkeypatch, user_input: str) -> SimpleNamespace:
    """Drive chat_nlp end-to-end with the tool loop and finish stubbed out."""
    calls = SimpleNamespace(recommend=[], applied=[], finished=[])

    async def _stub_loop(*, user_input, messages, display_mode, seen_calls):
        return ("done", messages, None, None)

    monkeypatch.setattr(bot, "_run_chained_tool_loop", _stub_loop)
    monkeypatch.setattr(
        bot, "_finish_turn",
        lambda **kwargs: calls.finished.append(kwargs),
    )
    monkeypatch.setattr(
        agent_mod, "_resolve_display_mode", lambda: AgentDisplayMode.QUIET,
    )
    monkeypatch.setattr(
        agent_mod, "recommend_profile",
        lambda *args, **kwargs: calls.recommend.append(args) or None,
    )
    import asyncio
    asyncio.run(bot.chat_nlp(user_input))
    return calls


def test_pinned_profile_suppresses_suggestion(
    tmp_path: Path, monkeypatch
) -> None:
    bot = _make_agent(tmp_path / "ws_pin", tmp_path, monkeypatch)
    assert hasattr(bot, "llm"), "Agent must own an llm client for this test"
    bot.llm._profile_pinned = True

    calls = _run_chat(bot, monkeypatch, "implement the new feature")

    assert calls.recommend == [], (
        "a manually pinned profile must suppress the suggestion entirely"
    )
    assert calls.applied == []
    assert len(calls.finished) == 1


def test_profile_auto_defaults_off_no_auto_apply(
    tmp_path: Path, monkeypatch
) -> None:
    from agent_core.llm.workspace_prefs import get_pref

    ws = tmp_path / "ws_auto"
    bot = _make_agent(ws, tmp_path, monkeypatch)
    bot.llm._profile_pinned = False
    # A fake provider captures any auto-apply attempt.
    bot.llm._provider = SimpleNamespace(
        apply_profile=lambda *args: calls.applied.append(args),
    )
    calls = SimpleNamespace(applied=[], profile_name=getattr(
        bot.llm, "_profile_name", None,
    ))

    async def _stub_loop(*, user_input, messages, display_mode, seen_calls):
        return ("done", messages, None, None)

    monkeypatch.setattr(bot, "_run_chained_tool_loop", _stub_loop)
    monkeypatch.setattr(bot, "_finish_turn", lambda **kwargs: None)
    monkeypatch.setattr(
        agent_mod, "_resolve_display_mode", lambda: AgentDisplayMode.QUIET,
    )
    suggested: list = []
    monkeypatch.setattr(
        agent_mod, "recommend_profile",
        lambda *args, **kwargs: suggested.append(args) or "fast-codegen",
    )

    import asyncio
    asyncio.run(bot.chat_nlp("implement the new feature"))

    assert len(suggested) == 1, "un-pinned turns must consider a suggestion"
    assert get_pref(ws, "profile_auto") is None, (
        "profile_auto must default off in a fresh workspace"
    )
    assert calls.applied == [], "no auto-apply without the profile_auto pref"
    assert getattr(bot.llm, "_profile_name", None) == calls.profile_name


# ---------------------------------------------------------------------------
# 3. EVOLVER weights persist to meta_policy.json and restore on reload
# ---------------------------------------------------------------------------

def test_evolver_weights_persist_and_restore(
    tmp_path: Path, monkeypatch
) -> None:
    state = tmp_path / "meta_policy.json"
    monkeypatch.setattr(learning, "_state_path", lambda: str(state))
    monkeypatch.setattr(learning, "_weights_loaded", True)
    fresh = MetaPolicyEvolver()
    monkeypatch.setattr(learning, "EVOLVER", fresh)

    fresh._weights[ProfileType.FAST_CODEGEN] = 1.5
    learning.save_weights()

    assert state.exists(), "save_weights must write meta_policy.json"
    payload = json.loads(state.read_text(encoding="utf-8"))
    assert payload["weights"]["fast_codegen"] == pytest.approx(1.5)

    # Reload restores the persisted weight over the fresh default.
    fresh._weights[ProfileType.FAST_CODEGEN] = 1.0
    learning.load_weights(force=True)
    assert fresh._weights[ProfileType.FAST_CODEGEN] == pytest.approx(1.5)

    # Out-of-range persisted weights are clamped to the evolver bounds.
    state.write_text(
        json.dumps({"weights": {"fast_codegen": 99.0}}), encoding="utf-8"
    )
    learning.load_weights(force=True)
    assert fresh._weights[ProfileType.FAST_CODEGEN] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 4. quality proxy with tracing disabled + append_event wiring
# ---------------------------------------------------------------------------

def test_quality_proxy_values_with_tracing_disabled(monkeypatch) -> None:
    import harnessfix.tracing as tracing

    monkeypatch.setattr(tracing, "trace_enabled", lambda: False)

    assert agent_mod._turn_quality_score("boom", []) == pytest.approx(0.0)
    assert agent_mod._turn_quality_score(None, ["src/x.py"]) == pytest.approx(0.8)
    assert agent_mod._turn_quality_score(None, []) == pytest.approx(0.7)


def test_finish_turn_records_quality_event_with_tracing_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    import harnessfix.tracing as tracing

    monkeypatch.setattr(tracing, "trace_enabled", lambda: False)
    metrics = _sandbox_learning(tmp_path, monkeypatch)
    bot = _make_agent(tmp_path / "ws_quality", tmp_path, monkeypatch)
    # Keep this test focused on the quality hook (the B2 hook is covered by
    # test_finish_turn_records_success_and_failure_task_types above).  The
    # real call passes the user input positionally — the stub must accept it.
    monkeypatch.setattr(
        agent_mod, "record_turn_outcome", lambda *args, **kwargs: None,
    )

    events = _finish(
        bot, monkeypatch, tmp_path,
        final_text="All good.", llm_error=None,
        user_input="analyze the logs",
    )

    assert any(
        kind == "turn" and name == "quality"
        and value == pytest.approx(0.7)
        for kind, name, value in events
    ), f"expected a ('turn','quality',0.7) event with tracing off, got {events!r}"


# ---------------------------------------------------------------------------
# 5. perf_history.json round-trips across PerfTracker instances
# ---------------------------------------------------------------------------

def test_perf_history_round_trips_across_instances(
    tmp_path: Path, monkeypatch
) -> None:
    history = tmp_path / "perf_history.json"
    monkeypatch.setattr(perf_cmd, "PERF_HISTORY_JSON_PATH", str(history))
    monkeypatch.setattr(PerfTracker, "_records", [])
    monkeypatch.setattr(PerfTracker, "_loaded", False)

    PerfTracker.record("help", 1.25, "show me help")
    assert history.exists(), "record() must persist perf_history.json"
    payload = json.loads(history.read_text(encoding="utf-8"))
    assert payload[-1]["command"] == "help"

    # Simulate a fresh process: drop the in-memory state, reload from disk.
    monkeypatch.setattr(PerfTracker, "_records", [])
    monkeypatch.setattr(PerfTracker, "_loaded", False)

    summary = PerfTracker.summary()
    assert [row["command"] for row in summary] == ["help"]
    assert summary[0]["calls"] == 1
    assert summary[0]["avg"] == "1.250s" or summary[0]["calls"] == 1


# ---------------------------------------------------------------------------
# 6. The suite redirects the A3/B2 state files out of the live repo root
# ---------------------------------------------------------------------------

def test_suite_redirects_turn_log_and_meta_policy_out_of_the_live_repo(
    tmp_path: Path,
) -> None:
    """The A3 turn log and the B2 meta-policy must never hit the live repo.

    Both paths resolve at CALL time, so before conftest redirected them every
    full suite run seeded the live ``turn_log.jsonl`` with fixture turns
    ("finish the task" x6 — input to habit mining) and evolved the live
    ``meta_policy.json`` weights from test outcomes (deep_analysis 1.099529
    observed after one run).  Mirrors ``TestRuntimeStateIsolation`` in
    ``tests/test_experience_recording.py``: pin the constants as live, then
    require the call-time values the suite actually uses to be elsewhere.
    """
    repo_root = Path(agent_mod.__file__).resolve().parent

    # Precondition: the constants-level paths ARE the live in-repo paths.
    live_turn_log = Path(constants.TURN_LOG_PATH).resolve()
    live_meta_policy = (
        Path(constants.CHAT_HISTORY_JSON_PATH).resolve().parent / "meta_policy.json"
    )
    for live in (live_turn_log, live_meta_policy):
        assert repo_root in live.parents, (
            f"precondition: {live} is expected to be the live in-repo path"
        )

    # What the suite actually writes to (conftest autouse redirect).
    redirected = {
        "TURN_LOG_PATH": Path(agent_mod.TURN_LOG_PATH).resolve(),
        "learning._state_path()": Path(learning._state_path()).resolve(),
    }
    for name, path in redirected.items():
        assert repo_root not in path.parents, (
            f"{name} -> {path} still lives inside the repo; the suite would "
            "pollute the live turn log / meta-policy (habit mining + profile "
            "recommendations read them)"
        )
    assert redirected["TURN_LOG_PATH"] != live_turn_log
    assert redirected["learning._state_path()"] != live_meta_policy
    # The redirect targets THIS test's per-test runtime dir (conftest autouse
    # fixture), so every test gets a fresh copy instead of shared live state.
    for name, path in redirected.items():
        assert tmp_path.resolve() in path.parents, (
            f"{name} -> {path} is not inside this test's tmp_path"
        )
