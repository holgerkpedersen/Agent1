"""Tests for the systematic issues ledger + detectors (Phase 0 infrastructure).

These cover the *machine* parts (ledger atomicity/idempotency, detector
accuracy, level gating) without invoking the LLM fix path.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from harnessfix import issue_loop as il
from harnessfix import issues as issue_store


def test_issues_ledger_is_valid_json_and_schema() -> None:
    issues = issue_store.load_issues()
    assert isinstance(issues, list)
    required = {
        "id", "title", "category", "severity", "locations", "status",
        "autonomy_level", "evidence", "suggested_approach", "created_at",
        "resolved_at",
    }
    for it in issues:
        assert required <= it.keys(), f"issue {it.get('id')} missing fields"


def test_make_issue_id_is_stable_and_level_defaults() -> None:
    a = issue_store.make_issue("dup", "x", ["agent.py:10"])
    b = issue_store.make_issue("dup", "x", ["agent.py:10"])
    assert a["id"] == b["id"]
    assert a["autonomy_level"] == issue_store.DEFAULT_AUTONOMY_LEVEL == 1
    assert a["status"] == "open"


def test_upsert_is_idempotent(tmp_path: Path) -> None:
    ledger = tmp_path / ".issues.json"
    issues = issue_store.load_issues(ledger)
    one = issue_store.make_issue("dup", "x", ["agent.py:10"])
    assert issue_store.upsert(issues, one) is True
    # Same id again must NOT add a second entry.
    assert issue_store.upsert(issues, issue_store.make_issue("dup", "x", ["agent.py:10"])) is False
    # Different locations -> different id -> added.
    assert issue_store.upsert(issues, issue_store.make_issue("dup", "y", ["agent.py:20"])) is True
    issue_store.save_issues(issues, ledger)
    reloaded = issue_store.load_issues(ledger)
    assert len(reloaded) == 2


def test_collect_issues_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Inject a synthetic finding so collection has something to seed; the real
    # repo is currently clean of these patterns, so we can't rely on it here.
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import logging, traceback\n"
        "logger = logging.getLogger(__name__)\n"
        "try:\n"
        "    risky()\n"
        "except Exception:\n"
        "    logger.warning('boom %s', traceback.format_exc())\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(il, "_iter_py_files", lambda root: [sample])
    ledger = tmp_path / ".issues.json"
    n1 = il.collect_issues(ledger)
    n2 = il.collect_issues(ledger)
    assert n1 == 1
    assert n2 == 0  # re-running adds nothing
    saved = issue_store.load_issues(ledger)
    assert len(saved) == n1


def test_open_issues_respects_level_cap(tmp_path: Path) -> None:
    ledger = tmp_path / ".issues.json"
    issues = issue_store.load_issues(ledger)
    issues.append(issue_store.make_issue("cat", "low", ["a.py:1"], autonomy_level=0))
    issues.append(issue_store.make_issue("cat", "high", ["b.py:1"], autonomy_level=1))
    issues.append(issue_store.make_issue("cat", "bench", ["c.py:1"], autonomy_level=2))
    # level_cap=1: the level-0 (human-only) and level-1 are eligible; level-2 excluded.
    elig = issue_store.open_issues(issues, max_level=1)
    levels = {int(i["autonomy_level"]) for i in elig}
    assert levels == {0, 1}
    # Resolved issues are excluded entirely.
    issue_store.resolve(issues, issues[0]["id"], "resolved")
    assert not any(i["id"] == issues[0]["id"] for i in issue_store.open_issues(issues, max_level=1))


def test_promote_validates_level(tmp_path: Path) -> None:
    ledger = tmp_path / ".issues.json"
    issues = issue_store.load_issues(ledger)
    issues.append(issue_store.make_issue("cat", "x", ["a.py:1"]))
    ok, _ = issue_store.promote(issues, issues[0]["id"], 2)
    assert ok and issues[0]["autonomy_level"] == 2
    bad, msg = issue_store.promote(issues, issues[0]["id"], 9)
    assert not bad and "invalid" in msg


def test_detectors_find_no_remaining_log_swallow_sites() -> None:
    files = il._iter_py_files(il.REPO_ROOT)
    dups = il.find_duplicate_handlers(files)
    swallows = il.find_log_swallow_excepts(files)
    # Every known log-and-swallow site was fixed by the deterministic resolver
    # (agent.py + agent_core/security/shutdown.py); the shared `_suppress_and_log`
    # sink is intentionally excluded from detection.
    assert dups == []
    assert swallows == []


def test_resolve_log_swallow_fixes_and_verifies(tmp_path: Path) -> None:
    """End-to-end: the deterministic resolver removes the flagged pattern and the
    verifier then passes. Re-running on the fixed file is a no-op."""
    mod = tmp_path / "mod.py"
    mod.write_text(
        "import logging, traceback\n"
        "logger = logging.getLogger(__name__)\n"
        "def do():\n"
        "    try:\n"
        "        risky()\n"
        "    except Exception:\n"
        "        logger.warning('boom %s', traceback.format_exc())\n",
        encoding="utf-8",
    )
    issue = issue_store.make_issue("best-effort-except", "x", [f"{mod.as_posix()}:5"])
    files = il._issue_files(issue)
    assert il.verify_issue(issue, files) is False
    assert il._resolve_log_swallow(issue, None) is True
    assert il.find_log_swallow_excepts([mod]) == []
    assert il.verify_issue(issue, files) is True
    # Idempotent: re-running on the fixed file changes nothing.
    assert il._resolve_log_swallow(issue, None) is False


def test_resolve_log_swallow_multistmt_handler(tmp_path: Path) -> None:
    """A handler with a fallback statement after the log call keeps the broad
    `except` (so the fallback still runs on exception) but the leading log+format_exc
    call is replaced by `<logger>.exception(label)`, so the detector no longer flags it
    and the original exception is still logged with its traceback."""
    mod = tmp_path / "mod.py"
    mod.write_text(
        "import logging, traceback\n"
        "logger = logging.getLogger(__name__)\n"
        "val = 1\n"
        "try:\n"
        "    val = risky()\n"
        "except Exception:\n"
        "    logger.warning('boom %s', traceback.format_exc())\n"
        "    val = 0\n",
        encoding="utf-8",
    )
    issue = issue_store.make_issue("best-effort-except", "x", [f"{mod.as_posix()}:5"])
    assert il._resolve_log_swallow(issue, None) is True
    out = mod.read_text(encoding="utf-8")
    assert "val = 0" in out  # fallback preserved
    assert "except Exception" in out  # broad except kept (semantics preserved)
    assert "traceback.format_exc" not in out  # detector no longer flags it
    assert "logger.exception(" in out  # logs the current exception + traceback


def test_resolve_log_swallow_fail_closed_on_branchy_handler(tmp_path: Path) -> None:
    """A handler that branches on the bound exception must NOT be rewritten."""
    mod = tmp_path / "mod.py"
    mod.write_text(
        "import logging, traceback\n"
        "logger = logging.getLogger(__name__)\n"
        "try:\n"
        "    risky()\n"
        "except Exception as e:\n"
        "    if 'x' in str(e):\n"
        "        logger.warning('boom %s', traceback.format_exc())\n"
        "    else:\n"
        "        raise\n",
        encoding="utf-8",
    )
    issue = issue_store.make_issue("best-effort-except", "x", [f"{mod.as_posix()}:4"])
    assert il._resolve_log_swallow(issue, None) is False
    assert "except Exception as e:" in mod.read_text(encoding="utf-8")


def test_resolve_log_swallow_skips_shared_sink() -> None:
    """The canonical `_suppress_and_log` sink must never be rewritten."""
    sink = il.REPO_ROOT / "agent_core" / "suppress_log.py"
    before = sink.read_text(encoding="utf-8")
    issue = issue_store.make_issue("best-effort-except", "x", [f"{sink.as_posix()}:1"])
    assert il._resolve_log_swallow(issue, None) is False
    assert sink.read_text(encoding="utf-8") == before


def test_resolve_issue_accepts_when_generator_fixes_and_gates_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lock the autonomous plumbing: generator fixes the file, verifier then
    passes, gates pass -> accepted. No LLM required."""
    import harnessfix.gates as gates_mod

    mod = tmp_path / "mod.py"
    mod.write_text(
        "import logging, traceback\n"
        "logger = logging.getLogger(__name__)\n"
        "try:\n"
        "    print(1)\n"
        "except Exception:\n"
        "    logger.warning('boom %s', traceback.format_exc())\n",
        encoding="utf-8",
    )
    issue = issue_store.make_issue(
        "best-effort-except", "x", [f"{mod.as_posix()}:5"],
    )

    def fake_gen(iss: dict, agent: object) -> bool:  # simulate the fix command
        text = mod.read_text(encoding="utf-8")
        out, drop_next = [], False
        for ln in text.splitlines():
            if ln.strip().startswith("except Exception:"):
                drop_next = True
                continue
            if drop_next and "logger.warning" in ln and "traceback" in ln:
                drop_next = False
                continue
            out.append(ln)
        mod.write_text("\n".join(out) + "\n", encoding="utf-8")
        return True

    monkeypatch.setattr(gates_mod, "run_test_gate", lambda *a, **k: (True, ""))
    monkeypatch.setattr(gates_mod, "run_security_gate", lambda *a, **k: (True, ""))

    summary = il.resolve_issue(issue, agent=None, level_cap=1, generate_fn=fake_gen)
    assert summary["verdict"] == "accepted"
    assert summary["accepted"] is True
    # The fix actually removed the flagged pattern.
    assert il.find_log_swallow_excepts([mod]) == []


def test_resolve_issue_fails_closed_when_gates_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the generator 'fixes' the file but gates fail, the issue is NOT
    resolved (fail-closed, no merge on ambiguity)."""
    import harnessfix.gates as gates_mod

    mod = tmp_path / "mod.py"
    mod.write_text(
        "import logging, traceback\n"
        "logger = logging.getLogger(__name__)\n"
        "try:\n"
        "    print(1)\n"
        "except Exception:\n"
        "    logger.warning('boom %s', traceback.format_exc())\n",
        encoding="utf-8",
    )
    issue = issue_store.make_issue(
        "best-effort-except", "x", [f"{mod.as_posix()}:5"],
    )

    def fake_gen(iss: dict, agent: object) -> bool:
        mod.write_text(
            "import logging\nlogger = logging.getLogger(__name__)\n"
            "print(1)\n",
            encoding="utf-8",
        )
        return True

    monkeypatch.setattr(gates_mod, "run_test_gate", lambda *a, **k: (False, "tests red"))
    monkeypatch.setattr(gates_mod, "run_security_gate", lambda *a, **k: (True, ""))

    summary = il.resolve_issue(issue, agent=None, level_cap=1, generate_fn=fake_gen)
    assert summary["verdict"] == "gates_failed"
    assert summary["accepted"] is False
    # The flagged pattern is gone from the file, but the ledger must stay open.
    assert il.find_log_swallow_excepts([mod]) == []

