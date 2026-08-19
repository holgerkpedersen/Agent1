"""Regression tests for repeated-section deduplication (2026-08-19 incident).

The workflow analyze gate printed duplicated ``## 7. CLARIFYING QUESTIONS``
sections. Two compounding causes:

1. The console printer sliced the section-6 marker to ``## 8.``, so the
   slice swallowed section 7 and printed it again in the section-7 pass.
2. The model sometimes emits the same section header and body twice (with
   near-identical rewording, e.g. "inline code comments" vs "inline code
   annotations"). ``strip_reasoning`` kept those duplicates in
   ``project_analysis.md`` — files from 2026-08-15/16/18 contain SEC6/SEC7
   repeated up to 4x.

This file pins ``dedupe_repeated_sections`` / ``extract_section`` and the
analysis-mode dedupe inside ``strip_reasoning``.
"""
from agent_core.commands.reasoning_strip import (
    dedupe_repeated_sections,
    extract_section,
    strip_reasoning,
)

BLOCKED_RESPONSE = """# Analysis for workspace C:\\Dev\\Agent1

## 1. SCOPE

- audit task

## 6. MISSING INFORMATION (BLOCKERS)

- No definition of "critical" (security vs. performance vs. correctness weighting).
- No statement of what output format is expected (report, annotations, code comments).

## 7. CLARIFYING QUESTIONS

- What weighting defines "critical" — security severity, blast radius, performance impact, or a mix?
- What is the required output format — a markdown report, inline code comments, or a structured list with rationale per region?

## 8. SUCCESS METRICS & OVERSIGHT

- **Measurable criteria**: concrete verifiable risk.

**BLOCKED:** yes
"""


class TestDedupeRepeatedSections:
    """Repeated section headers keep only their first copy."""

    def test_identical_repeated_section_kept_once(self) -> None:
        r = (
            "## 7. CLARIFYING QUESTIONS\n"
            "- **Q1:** A\n"
            "- **Q2:** B\n"
            "## 7. CLARIFYING QUESTIONS\n"
            "- **Q1:** A\n"
            "- **Q2:** B\n"
        )
        out = dedupe_repeated_sections(r)
        assert out.count("## 7. CLARIFYING QUESTIONS") == 1
        assert out.count("**Q1:**") == 1
        assert out.count("**Q2:**") == 1

    def test_near_identical_reworded_section_drained(self) -> None:
        """2026-08-19 case: model rewrote the section with tiny wording diffs."""
        r = (
            "## 7. CLARIFYING QUESTIONS\n"
            "- ... inline code comments ...\n"
            "## 7. CLARIFYING QUESTIONS\n"
            "- ... inline code annotations ...\n"
        )
        out = dedupe_repeated_sections(r)
        assert out.count("## 7. CLARIFYING QUESTIONS") == 1
        assert "inline code comments" in out
        assert "inline code annotations" not in out

    def test_blockers_suffix_and_plain_header_same_section(self) -> None:
        r = (
            "## 6. MISSING INFORMATION (BLOCKERS)\n"
            "- first copy\n"
            "## 6. MISSING INFORMATION\n"
            "- second copy\n"
        )
        out = dedupe_repeated_sections(r)
        assert out.count("MISSING INFORMATION") == 1
        assert "first copy" in out
        assert "second copy" not in out

    def test_distinct_sections_all_survive(self) -> None:
        out = dedupe_repeated_sections(BLOCKED_RESPONSE)
        for marker in ("## 1. SCOPE", "## 6. MISSING INFORMATION",
                       "## 7. CLARIFYING QUESTIONS", "## 8. SUCCESS METRICS"):
            assert out.count(marker) == 1, marker

    def test_triple_duplicate_kept_once(self) -> None:
        r = ("## 7. CLARIFYING QUESTIONS\n- copy alpha\n"
             "## 7. CLARIFYING QUESTIONS\n- copy beta\n"
             "## 7. CLARIFYING QUESTIONS\n- copy gamma\n")
        out = dedupe_repeated_sections(r)
        assert out.count("## 7. CLARIFYING QUESTIONS") == 1
        assert "copy alpha" in out
        assert "copy beta" not in out
        assert "copy gamma" not in out

    def test_no_sections_unchanged(self) -> None:
        text = "no headers here\njust prose\n"
        assert dedupe_repeated_sections(text) == text

    def test_empty_text(self) -> None:
        assert dedupe_repeated_sections("") == ""


class TestExtractSection:
    """A marker extract never swallows the following sections."""

    def test_section6_extract_excludes_section7(self) -> None:
        deduped = dedupe_repeated_sections(BLOCKED_RESPONSE)
        sec6 = extract_section(deduped, "## 6. MISSING INFORMATION")
        assert sec6.startswith("## 6. MISSING INFORMATION (BLOCKERS)")
        assert "CLARIFYING QUESTIONS" not in sec6

    def test_section7_extract_excludes_section8(self) -> None:
        deduped = dedupe_repeated_sections(BLOCKED_RESPONSE)
        sec7 = extract_section(deduped, "## 7. CLARIFYING QUESTIONS")
        assert sec7.startswith("## 7. CLARIFYING QUESTIONS")
        assert "SUCCESS METRICS" not in sec7
        assert "BLOCKED" not in sec7

    def test_missing_marker_returns_empty(self) -> None:
        assert extract_section("## 1. SCOPE\n- x\n", "## 6. MISSING INFORMATION") == ""

    def test_console_pipeline_prints_each_section_once(self, capsys) -> None:
        """Full blocked-gate print path: 6 once, 7 once, even when the model
        emitted the 7-section twice (2026-08-19 10:15:40 run)."""
        r = BLOCKED_RESPONSE.replace(
            "## 8. SUCCESS METRICS",
            "## 7. CLARIFYING QUESTIONS\n"
            "- What weighting defines \"critical\" — security severity, blast radius, performance impact, or a mix?\n"
            "- What is the required output format — a markdown report, inline code annotations, or a structured list with rationale per region?\n"
            "\n"
            "## 8. SUCCESS METRICS",
        )
        deduped = dedupe_repeated_sections(r)
        printed = []
        for marker in ("## 6. MISSING INFORMATION", "## 7. CLARIFYING QUESTIONS"):
            section_text = extract_section(deduped, marker)
            if not section_text:
                continue
            print(section_text)
            print("")
            printed.append(section_text)
        out = capsys.readouterr().out
        assert out.count("## 6. MISSING INFORMATION (BLOCKERS)") == 1
        assert out.count("## 7. CLARIFYING QUESTIONS") == 1
        assert "inline code comments" in out
        assert "inline code annotations" not in out
        assert len(printed) == 2


class TestStripReasoningAnalysisDedupe:
    """project_analysis.md no longer keeps model-emitted duplicates."""

    def test_line_by_line_path_dedupes(self) -> None:
        """Blocked runs (no Refinement/Verification) use the line-by-line
        branch; repeated sections must still collapse."""
        out = strip_reasoning(BLOCKED_RESPONSE, mode="analysis")
        assert out.count("## 7. CLARIFYING QUESTIONS") == 1

    def test_model_emitted_duplicate_removed_from_file_content(self) -> None:
        r = BLOCKED_RESPONSE.replace(
            "**BLOCKED:** yes",
            "## 7. CLARIFYING QUESTIONS\n"
            "- What weighting defines \"critical\" — security severity, blast radius, performance impact, or a mix?\n"
            "- What is the required output format — a markdown report, inline code annotations, or a structured list with rationale per region?\n"
            "\n"
            "**BLOCKED:** yes",
        )
        out = strip_reasoning(r, mode="analysis")
        assert out.count("## 7. CLARIFYING QUESTIONS") == 1
        assert "inline code comments" in out
        assert "inline code annotations" not in out
        assert "**BLOCKED:** yes" in out

    def test_trailing_duplicate_does_not_swallow_blocked_marker(self) -> None:
        """A repeated section at the end must not drain the BLOCKED line."""
        r = (
            "## 7. CLARIFYING QUESTIONS\n"
            "- first copy\n"
            "## 7. CLARIFYING QUESTIONS\n"
            "- second copy\n"
            "**BLOCKED:** yes\n"
        )
        out = dedupe_repeated_sections(r)
        assert out.count("## 7. CLARIFYING QUESTIONS") == 1
        assert "first copy" in out
        assert "second copy" not in out
        assert "**BLOCKED:** yes" in out

    def test_clean_sections_path_dedupes(self) -> None:
        """Refinement-bearing responses go through _clean_analysis_sections."""
        r = BLOCKED_RESPONSE.replace("**BLOCKED:** yes", "**BLOCKED:** no") + (
            "\n## Refinement (self-critique)\n"
            "- gap one\n"
        )
        out = strip_reasoning(r, mode="analysis")
        assert out.count("## 7. CLARIFYING QUESTIONS") == 1
        assert "## Refinement" in out

    def test_light_mode_untouched(self) -> None:
        """plan/entities/taskplan stripping must not run section dedupe."""
        r = "## 7. CLARIFYING QUESTIONS\n- A\n## 7. CLARIFYING QUESTIONS\n- B\n"
        out = strip_reasoning(r, mode="light")
        assert out.count("## 7. CLARIFYING QUESTIONS") == 2
