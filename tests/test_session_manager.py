"""Unit tests for the legacy in-process SessionManager.

Target of this suite: ``agent_core/cli/session/session_manager.py`` (class
``SessionManager``), the manager behind the CLI ``clear`` command.  The
named-chat-session layer under ``agent_core/session/`` is a different
subsystem covered by ``tests/test_session_layer.py`` and friends; nothing
here touches it.

The contract pinned here is read from the source and must not drift:

1. ``get_session_stats()`` returns exactly ``message_count``, ``token_usage``
   and ``last_activity``.
2. ``token_usage`` is ``sum(len(str(content)) // 4)`` - integer division per
   message, summed, computed via ``str()`` so non-string or missing content
   cannot raise.
3. ``add_message()`` appends and advances ``last_activity``.
4. ``last_activity`` is a timezone-aware UTC ISO-8601 string set at
   construction, so stats work on an empty manager.
5. ``clear_session()`` empties the message list and leaves the object usable.
"""

from datetime import datetime

from agent_core.cli.session.session_manager import SessionManager


# --------------------------------------------------------------------------
# T1 - empty state and basic counting
# --------------------------------------------------------------------------

def test_sm_empty_stats_expose_the_three_contract_keys():
    sm = SessionManager()
    stats = sm.get_session_stats()
    assert set(stats) == {"message_count", "token_usage", "last_activity"}


def test_sm_empty_manager_reports_zero_counts():
    sm = SessionManager()
    stats = sm.get_session_stats()
    assert stats["message_count"] == 0
    assert stats["token_usage"] == 0


def test_sm_empty_manager_has_iso_last_activity():
    sm = SessionManager()
    parsed = datetime.fromisoformat(sm.get_session_stats()["last_activity"])
    assert parsed.tzinfo is not None


def test_sm_add_message_increments_message_count():
    sm = SessionManager()
    sm.add_message({"role": "user", "content": "hello"})
    assert sm.get_session_stats()["message_count"] == 1
    sm.add_message({"role": "assistant", "content": "hi there"})
    assert sm.get_session_stats()["message_count"] == 2


# --------------------------------------------------------------------------
# T2 - token arithmetic and payload robustness
# --------------------------------------------------------------------------

def test_sm_token_usage_is_summed_integer_division_per_message():
    sm = SessionManager()
    a = "x" * 10   # 10 // 4 == 2
    b = "y" * 9    # 9 // 4 == 2
    sm.add_message({"role": "user", "content": a})
    sm.add_message({"role": "assistant", "content": b})
    assert sm.get_session_stats()["token_usage"] == len(a) // 4 + len(b) // 4


def test_sm_empty_string_content_adds_zero_tokens():
    sm = SessionManager()
    sm.add_message({"role": "user", "content": ""})
    stats = sm.get_session_stats()
    assert stats["message_count"] == 1
    assert stats["token_usage"] == 0


def test_sm_non_string_content_does_not_raise():
    sm = SessionManager()
    sm.add_message({"role": "user", "content": 1234567890})
    stats = sm.get_session_stats()
    assert stats["message_count"] == 1
    assert stats["token_usage"] == len(str(1234567890)) // 4


def test_sm_missing_content_key_does_not_raise():
    sm = SessionManager()
    sm.add_message({"role": "user"})
    stats = sm.get_session_stats()
    assert stats["message_count"] == 1
    assert stats["token_usage"] == 0


# --------------------------------------------------------------------------
# T3 - clock behaviour, reset, instance isolation
# --------------------------------------------------------------------------

def test_sm_last_activity_is_timezone_aware_utc():
    sm = SessionManager()
    parsed = datetime.fromisoformat(sm.get_session_stats()["last_activity"])
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0


def test_sm_last_activity_is_monotonic_across_two_adds():
    sm = SessionManager()
    first = datetime.fromisoformat(sm.get_session_stats()["last_activity"])
    sm.add_message({"role": "user", "content": "one"})
    mid = datetime.fromisoformat(sm.get_session_stats()["last_activity"])
    sm.add_message({"role": "user", "content": "two"})
    last = datetime.fromisoformat(sm.get_session_stats()["last_activity"])
    assert first <= mid <= last


def test_sm_add_message_advances_last_activity():
    sm = SessionManager()
    before = sm.get_session_stats()["last_activity"]
    sm.add_message({"role": "user", "content": "advance me"})
    after = sm.get_session_stats()["last_activity"]
    assert datetime.fromisoformat(after) >= datetime.fromisoformat(before)


def test_sm_clear_session_returns_to_empty_state():
    sm = SessionManager()
    sm.add_message({"role": "user", "content": "x" * 40})
    sm.clear_session()
    stats = sm.get_session_stats()
    assert stats["message_count"] == 0
    assert stats["token_usage"] == 0


def test_sm_manager_is_usable_after_clear():
    sm = SessionManager()
    sm.add_message({"role": "user", "content": "old"})
    sm.clear_session()
    sm.add_message({"role": "user", "content": "new"})
    stats = sm.get_session_stats()
    assert stats["message_count"] == 1
    assert stats["token_usage"] == len("new") // 4


def test_sm_clear_is_idempotent():
    sm = SessionManager()
    sm.add_message({"role": "user", "content": "x"})
    sm.clear_session()
    sm.clear_session()
    assert sm.get_session_stats()["message_count"] == 0


def test_sm_two_instances_never_share_messages():
    a = SessionManager()
    b = SessionManager()
    a.add_message({"role": "user", "content": "only in a"})
    assert a.get_session_stats()["message_count"] == 1
    assert b.get_session_stats()["message_count"] == 0
    a.clear_session()
    assert a.get_session_stats()["message_count"] == 0
    assert b.get_session_stats()["message_count"] == 0
