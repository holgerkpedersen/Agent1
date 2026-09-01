"""Root pytest configuration.

Trace capture in agent.py chat_nlp is opt-out (AGENT_NO_TRACE=1 disables it)
so a real agent session produces a trace corpus by default.  Test runs must
not write reports/traces/ artifacts, so the whole suite runs with tracing
disabled unless a test explicitly enables it.
"""

import os
from pathlib import Path

import pytest

os.environ.setdefault("AGENT_NO_TRACE", "1")


@pytest.fixture(autouse=True)
def _isolate_from_real_tree_and_beacons(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep every test from mutating tracked source or the live beacons.

    Two classes of test previously touched the real repo / live dashboard:

    * The harnessfix repair modules (``stuck_repeat``, ``tool_interface``,
      ``abandonment_resume``) apply/revert via a module-level
      ``_TARGET = Path("agent_core/llm/tool_loop.py")`` resolved relative to
      CWD.  Tests that call ``apply()``/``revert()`` (directly, or through
      ``run_loop`` / ``_reset_repairs``) therefore edit the REAL file.  When
      such a test runs inside an autonomous driver's post-repair ``pytest``
      gate, it reverts the very repair the loop just applied, so the loop
      records ``accepted`` but has nothing to commit and silently spins every
      remaining iteration re-applying a repair that can never persist.
    * ``harnessfix.progress`` writes the live beacon (``run_status.json``) and
      history (``run_history.jsonl``) to ``reports/harnessfix``.  Driver tests
      that call ``main()`` (e.g. ``test_autonomous_driver.py``) and the
      issue-loop test therefore pollute those live files, corrupting the
      dashboard during a real run.

    Both are redirected to a per-test temp dir here, once, for the whole
    suite.  Tests that already sandbox their own edits (e.g.
    ``test_repairs_stuck_repeat.py``) stay compatible: their own monkeypatch
    simply overrides this one for the duration of the test.
    """
    import harnessfix.progress as progress
    import harnessfix.repairs.abandonment_resume as abandonment_resume
    import harnessfix.repairs.stuck_repeat as stuck_repeat
    import harnessfix.repairs.tool_interface as tool_interface

    # Modules that edit a real source file via a module-level ``_TARGET``.
    # The redirected copies live in a SEPARATE temp dir (sibling of
    # ``tmp_path``) — never inside the per-test workspace — so the
    # workspace walkers in ``_suggest_consumers`` and ``_unwired_closure``
    # (``agent_core/commands/implement_cmd.py``) never see the redirected
    # files as "existing modules" or "reference sources".  Tests treat
    # ``tmp_path`` as the user's project; leaking harnessfix internals in
    # there caused false ``b``/``c``-substring pin matches against
    # ``tool_loop.py``'s 900 lines of Python source (every other file is
    # full of those letters) and false ``tool``-token matches in
    # ``_suggest_consumers``.  The regressions are pinned by
    # ``TestSuggestConsumers`` and ``TestUnwiredClosure`` in
    # ``tests/test_implement_safety.py``.
    targets_dir = tmp_path_factory.mktemp("harnessfix_targets_") / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)
    for mod in (stuck_repeat, tool_interface, abandonment_resume):
        real = mod._TARGET
        if not real.exists():
            continue
        local = targets_dir / f"{mod.__name__.split('.')[-1]}_{real.name}"
        local.write_text(real.read_text(encoding="utf-8"), encoding="utf-8")
        monkeypatch.setattr(mod, "_TARGET", local)
        # Revert any repair already applied in the real tree so the
        # sandboxed copy starts from the UN-applied (original) state.
        # Only tool_interface needs this — its loop tests read the target
        # to assert whether the repair was applied; stuck_repeat and
        # abandonment_resume tests create their own isolated copies and
        # apply/revert independently, so reverting their copies here would
        # interfere with their own apply/revert roundtrip assertions.
        if mod is tool_interface:
            try:
                mod.revert()
            except Exception:
                pass

    # Redirect the live autonomous-run beacons to temp so driver/loop tests
    # never write into reports/harnessfix/*.
    beacon_dir = tmp_path / "reports" / "harnessfix"
    monkeypatch.setattr(progress, "OUTPUT_DIR", beacon_dir)
    monkeypatch.setattr(progress, "STATUS_PATH", beacon_dir / "run_status.json")
    monkeypatch.setattr(progress, "HISTORY_PATH", beacon_dir / "run_history.jsonl")
