"""Patch docs for decision #079 (scratch, deleted after use)."""

# --- AGENTS.md: add convention bullet + roadmap entry -----------------------
path = "AGENTS.md"
content = open(path, encoding="utf-8").read()

conv_anchor = (
    "- Windows shell is `cmd.exe` \u2014 no grep/tail; use `python -c \"...\" "
    "one-liners inside\n  the agent REPL; normalize paths via "
    "`to_windows_path`."
)
assert conv_anchor in content, "conventions anchor missing"
addition = conv_anchor + (
    "\n- **No emojis or pictographs in repo text files** (decision #079)."
    " Plain-text\n  status markers instead (`[DONE]`, `[Q]` quick win,"
    " `[S]` strategic). Monochrome\n  CLI glyphs are exempt (check/cross/"
    "warning marks, box drawing \u2014 they are\n  load-bearing terminal"
    " output asserted by tests). Enforced as audit check 6:\n "
    " `python scripts/audit_invariants.py` fails on findings"
    " (`agent_core/text_policy.py`).\n- Runtime state files"
    " (`chat_history.json`, `agent_memory.json`) and the\n  `.docs/`,"
    " `backups/`, `reports/` trees are exempt from content scans."
)
content = content.replace(conv_anchor, addition, 1)

tail_anchor = (
    "  Full suite: 1470 passed / 2 skipped."
)
assert tail_anchor in content, "roadmap tail anchor missing"
roadmap = tail_anchor + (
    "\n- **No-emoji policy + audit gate (decision #079, DONE, 2026-08-25)**:"
    " new\n  `agent_core/text_policy.py` (stdlib-only emoji/pictograph"
    " detector with an explicit\n  monochrome-glyph allowlist;"
    " `scan_tree` skips runtime-state files) wired into\n"
    "  `scripts/audit_invariants.py` as check 6 (findings are ERRORS)."
    " Cleaned AGENTS.md\n  ([DONE] markers), the improvement plan ([Q]/[S]"
    " tags) and repaired a mojibake\n  byte in CHANGES.md. Regression found"
    " by the new tests: `_mutating_files_this_turn`\n  scanned restored"
    " history from previous sessions \u2014 fixed with a per-turn boundary"
    "\n  (`Agent._turn_start_index`). Tests:"
    " `tests/test_text_policy.py` (26),\n"
    "  `tests/test_quickwins_batch2.py::TestTurnBoundaryAfterRestart` (3)."
)
content = content.replace(tail_anchor, roadmap, 1)

with open(path, "w", encoding="utf-8", newline="") as f:
    f.write(content)
print("AGENTS.md patched")

# --- CHANGES.md: prepend the entry ------------------------------------------
entry = """## 2026-08-25 - feat: no-emoji policy with audit enforcement (decision #079); fixes stale self-review listing

**Change**: agent_core/text_policy.py (new: stdlib-only emoji/pictograph detector - Unicode So/Sk categories plus dedicated emoji ranges U+2600-27BF/U+2B00-BFFF/U+1F000-1FAFF/U+FE0F - with ALLOWED_MONO_CHARS exempting the repo's monochrome CLI glyphs (check/cross/warning, box drawing, arrows); scan_tree skips runtime-state chat_history.json/agent_memory.json and the .docs/backups/reports trees), scripts/audit_invariants.py (check 6: scan_tree over the workspace, findings reported as audit ERRORS referencing decision #079), AGENTS.md (10 badge markers -> plain [DONE]-style bold headers; conventions section documents the policy), docs/AGENTIC_IMPROVEMENT_PLAN.md (legend and inline lightning/building tags -> [Q]/[S]), CHANGES.md (one U+FFFD mojibake byte repaired to an em dash, matching analysis_verifier's real output format; git history shows the byte was committed already corrupted, so nothing was lost). Latent bug found by the new tests: Agent._mutating_files_this_turn scanned the ENTIRE persisted history although its contract says "this turn", so after a session restart the [self-review] note listed files written by previous sessions (observed live: a fresh Agent reported 7 stale tmp_* writes from an earlier session). Fixed with Agent._turn_start_index, set in __init__ and reset at the top of every chat_nlp turn before the user message is appended; the scan now starts there.

**Reason**: User directive: avoid emojis in files. Emoji render inconsistently across Windows codepages, CI logs and terminals and can be silently mangled into unrecoverable replacement characters (the CHANGES.md byte proves it happened here). Plain-text markers survive any encoding. The monochrome glyph vocabulary stays exempt because colors.py constants and verifier outputs use them and existing tests assert on them - they are load-bearing output, not decoration. The turn-boundary fix restores the documented behavior of the self-review note.

**Files**: agent_core/text_policy.py (new), scripts/audit_invariants.py (check 6 + None-safe docs-dir name), agent.py (_turn_start_index init/reset, scoped _mutating_files_this_turn), tests/test_text_policy.py (new, 26 tests incl. audit wiring: planted emoji fails, clean tree passes, real repo is clean), tests/test_quickwins_batch2.py (TestTurnBoundaryAfterRestart: restored-history ignored, current-turn still detected, end-to-end restart-then-write prints only the new file).

"""
path = "CHANGES.md"
content = open(path, encoding="utf-8-sig").read()
marker = "## 2026-08-25 - feat: symbol-level NLP tools"
idx = content.find(marker)
assert idx != -1, "symbol-tools entry marker not found"
content = content[:idx] + entry + content[idx:]
with open(path, "w", encoding="utf-8", newline="") as f:
    f.write(content)
print("CHANGES.md patched")
