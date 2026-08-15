"""Tests for the semantic (geometric) module-duplication layer.

Precision-first: true duplicate pairs from the workflow run must be flagged,
unrelated planned modules must not, and test/benchmark modules never enter the
corpus.
"""
import pytest

from agent_core.utils.module_similarity import (
    ModuleSimilarity,
    PlannedModule,
    _is_production_module,
    _meaningful_tokens,
)

_TRUE_PAIRS = [
    (
        "agent_core/security/shell_allowlist.py",
        "Shell command allow-list + hardened blocklist replacing incomplete _DANGEROUS_SHELL_PATTERNS",
        "agent_core/security/allowlist.py",
    ),
    (
        "agent_core/security/sanitizer_fix.py",
        "Corrected forbidden patterns + pipe rejection replacing malformed pattern in security/sanitizer.py:_FORBIDDEN_PATTERNS",
        "agent_core/security/sanitizer.py",
    ),
    (
        "agent_core/security/path_guard.py",
        "Workspace containment enforcement replacing missing _safe_path/_resolve_nlp_path check",
        "agent_core/security/path_utils.py",
    ),
]


@pytest.fixture()
def ws(tmp_path):
    """A small production corpus with real-style docstrings."""
    files = {
        "agent_core/security/allowlist.py": '"""Safe shell command allow-list and validation logic."""\nSAFE = set()\n',
        "agent_core/security/sanitizer.py": '"""Input sanitizer stripping shell-injection payloads."""\nX = 1\n',
        "agent_core/security/path_utils.py": '"""Centralized path normalization and workspace containment."""\nY = 2\n',
        "agent_core/path_utils.py": '"""Path normalization utilities for workspace containment."""\nZ = 3\n',
        "agent_core/llm/tool_loop.py": '"""Tool calling loop orchestrator for LLM conversations."""\nW = 4\n',
        "agent_core/tool_schemas.py": '"""Tool definitions and OpenAI-format schemas."""\nV = 5\n',
        "agent_core/commands/reporting_cmd.py": '"""Command that produces performance reports."""\nU = 6\n',
        "tests/test_sanitizer.py": '"""Tests for sanitizer — MUST NOT be in the corpus."""\nT = 7\n',
        "benchmarks/security_benchmarks.py": '"""Benchmarks — MUST NOT be in the corpus."""\nB = 8\n',
    }
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return tmp_path


class TestCorpus:
    def test_tests_and_benchmarks_excluded(self, ws):
        sim = ModuleSimilarity(str(ws))
        paths = sim.corpus.paths
        assert not any("tests/" in p or "benchmarks/" in p for p in paths)
        assert "agent_core/security/allowlist.py" in paths

    def test_corpus_reused_while_unchanged(self, ws):
        sim1 = ModuleSimilarity(str(ws))
        sim2 = ModuleSimilarity(str(ws))
        assert sim1.corpus is sim2.corpus  # cached — identical precision, faster

    def test_corpus_rebuilt_after_change(self, ws):
        sim1 = ModuleSimilarity(str(ws))
        (ws / "agent_core" / "security" / "allowlist.py").write_text(
            '"""Safe shell command allow-list and validation logic (v2)."""\nSAFE = set()\n',
            encoding="utf-8",
        )
        sim2 = ModuleSimilarity(str(ws))
        assert sim1.corpus is not sim2.corpus


class TestTruePairs:
    def test_top1_points_at_the_duplicate(self, ws):
        """Geometric top-1 must land on the existing module for every true pair."""
        sim = ModuleSimilarity(str(ws))
        for path, desc, expected in _TRUE_PAIRS:
            feature = sim._planned_feature(PlannedModule(path, desc))
            top = sim.corpus.top(_meaningful_tokens(feature), k=3)
            assert top, f"no match for {path}"
            assert top[0][0] == expected, f"{path}: top-1 {top[0][0]}, expected {expected}"

    def test_true_pairs_flagged(self, ws):
        sim = ModuleSimilarity(str(ws))
        planned = [PlannedModule(p, d) for p, d, _ in _TRUE_PAIRS]
        findings = sim.find_duplicates(planned)
        flagged = {f.file: f.existing for f in findings}
        for path, _, expected in _TRUE_PAIRS:
            assert flagged.get(path) == expected, f"{path} not flagged against {expected}"


class TestFalsePairs:
    def test_chain_limiter_not_flagged(self, ws):
        sim = ModuleSimilarity(str(ws))
        findings = sim.find_duplicates([
            PlannedModule("agent_core/nlp/chain_limiter.py", "Bound recursive tool execution depth"),
        ])
        assert findings == []

    def test_self_match_excluded(self, ws):
        sim = ModuleSimilarity(str(ws))
        findings = sim.find_duplicates([
            PlannedModule("agent_core/commands/reporting_cmd.py", "Command that produces performance reports"),
        ])
        assert not any(f.existing == "agent_core/commands/reporting_cmd.py" for f in findings)

    def test_unrelated_module_not_flagged(self, ws):
        sim = ModuleSimilarity(str(ws))
        findings = sim.find_duplicates([
            PlannedModule("agent_core/nlp/prompt_scheduler.py", "Schedule LLM prompts by priority"),
        ])
        assert findings == []


class TestProductionFilter:
    def test_is_production_module(self):
        assert _is_production_module("agent_core/security/allowlist.py")
        assert not _is_production_module("tests/test_sanitizer.py")
        assert not _is_production_module("tests/unit/test_llm_client.py")
        assert not _is_production_module("benchmarks/security_benchmarks.py")
        assert not _is_production_module("test_tool_loop_nlp.py")
        assert not _is_production_module("performance_dashboard/config.py")
