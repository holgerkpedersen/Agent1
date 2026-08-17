"""Tests for the tightened benchmark scorer (plan FIX item 21):
exact/word-bounded matching instead of lenient substring/keyword overlap,
plus the optional LLM-judge fallback."""
import asyncio

from benchmark import (
    answers_match,
    judge_answer,
    score_instruction_following,
    score_question,
)


class TestAnswersMatchTightened:
    def test_exact_match(self):
        assert answers_match("Paris", "paris")

    def test_word_bounded_expected_inside_response(self):
        assert answers_match("The answer is 42.", "42")

    def test_no_raw_substring_false_positive(self):
        assert not answers_match("142", "42")
        assert not answers_match("retry the test", "true")
        assert not answers_match("retro", "true")

    def test_no_reverse_containment_credit(self):
        assert not answers_match("42", "The answer is 42.")

    def test_numeric_tolerance(self):
        assert answers_match("3.2", "3")
        assert answers_match("1000", "1,000")

    def test_fraction_equivalence(self):
        assert answers_match("0.5", "1/2")

    def test_keyword_overlap_tightened(self):
        # One keyword out of eight must NOT match anymore.
        assert not answers_match("Paris", "Paris is the capital of France in Europe")
        # Substantial overlap (7/8 words) still matches.
        assert answers_match(
            "Paris is the capital of France Europe",
            "Paris is the capital of France in Europe",
        )

    def test_padded_response_not_credited(self):
        # Same expected words but buried in a much longer response.
        assert not answers_match(
            "The quick brown fox jumps over the lazy dog near the Paris "
            "riverbank on a sunny Sunday afternoon while birds sing",
            "Paris is the capital",
        )


class TestScoreInstructionFollowingTightened:
    def test_nothing_else_exact_echo(self):
        score = score_instruction_following(
            'Respond with "true" and nothing else.', "true"
        )
        assert score == 1.0

    def test_nothing_else_extra_words_penalized(self):
        score = score_instruction_following(
            'Respond with "true" and nothing else.', "true and false"
        )
        assert score < 1.0

    def test_nothing_else_punctuation_tolerant(self):
        score = score_instruction_following(
            'Respond with "true" and nothing else.', "true."
        )
        assert score == 1.0

    def test_exactly_three_words(self):
        assert score_instruction_following(
            "Respond with exactly 3 words.", "one two three"
        ) == 1.0
        assert score_instruction_following(
            "Respond with exactly 3 words.", "one two"
        ) < 1.0

    def test_all_lowercase(self):
        assert score_instruction_following(
            "Write your answer in only lowercase.", "hello world"
        ) == 1.0
        assert score_instruction_following(
            "Write your answer in only lowercase.", "Hello World"
        ) < 1.0


class TestJudgeFallback:
    async def _judge(self, response, expected):
        return "yes they match" if "yes" in expected else "no"

    def test_judge_overrides_heuristic_false(self):
        async def go():
            correct, _ = await score_question(
                "q", "anything", "yes please", "knowledge", judge_fn=self._judge
            )
            return correct

        assert asyncio.run(go()) is True

    def test_heuristic_true_never_consults_judge(self):
        calls = []

        async def judge(response, expected):
            calls.append(1)
            return "no"

        async def go():
            return await score_question(
                "q", "Paris", "paris", "knowledge", judge_fn=judge
            )

        correct, _ = asyncio.run(go())
        assert correct is True
        assert calls == []

    def test_judge_absent_uses_heuristics(self):
        async def go():
            return await score_question("q", "Paris", "paris", "knowledge")

        correct, _ = asyncio.run(go())
        assert correct is True

    def test_judge_answer_parses_verdicts(self):
        async def judge(response, expected):
            return "No, they differ."

        async def go():
            return await judge_answer(judge, "a", "b")

        assert asyncio.run(go()) is False

    def test_judge_answer_ambiguous_returns_none(self):
        async def judge(response, expected):
            return "maybe"

        async def go():
            return await judge_answer(judge, "a", "b")

        assert asyncio.run(go()) is None

    def test_judge_answer_exception_returns_none(self):
        async def judge(response, expected):
            raise RuntimeError("judge down")

        async def go():
            return await judge_answer(judge, "a", "b")

        assert asyncio.run(go()) is None


class TestLooksIncompleteTuning:
    def test_completion_overrides_weak_marker(self):
        from agent import _looks_incomplete
        # "remaining" is a weak marker, but the answer is clearly finished.
        assert not _looks_incomplete("The remaining issue is fixed — all done.")
        assert not _looks_incomplete("Budget spent, task is complete.")

    def test_weak_marker_without_completion_still_incomplete(self):
        from agent import _looks_incomplete
        assert _looks_incomplete("The tool budget is exhausted, I still need more calls.")
        assert _looks_incomplete("I could not complete the analysis yet — would need more budget.")
