"""Accuracy / latency benchmark for OpenAI-compatible LLM servers.

Two modes:

* **standard** (default) — single shot, ``temperature=0.1``, 2048 tokens per
  question.  Cheap pass/fail signal.
* **deep-analysis** (``--deep``) — the agent's own ``deep-analysis`` profile
  (``temperature=0.7``, larger token budget) plus a second, self-checking
  attempt on reasoning/math questions, and a per-question trace in the report.
  Use it to measure the model the way the agent actually drives it.

Results are keyed ``model|profile`` (decision #055) and written to
``reports/models.json``; each run also drops a ``reports/benchmark_<stamp>.json``
snapshot so ``--trend`` has history to read.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_core.config import lmstudio_base_url
from agent_core.constants import DEFAULT_LLAMA_BASE_URL

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions for Benchmarking
# ---------------------------------------------------------------------------


class BenchmarkError(Exception):
    """Base exception for benchmark-related failures."""

    pass


class ModelAPIError(BenchmarkError):
    """Raised when the model API returns an error or fails to respond."""

    pass


# ---------------------------------------------------------------------------
# Question Bank  (~25 per category x 5 categories = ~125 total)
# Each entry: (prompt, expected_answer_or_checker_key)
# ---------------------------------------------------------------------------

REASONING_QUESTIONS = [
    (
        "If all Bloops are Razzies and all Razzies are Lazzies, "
        "are all Bloops definitely Lazzies? Answer yes or no.",
        "yes",
    ),
    (
        "A farmer has 17 sheep. All but 9 run away. "
        "How many sheep does the farmer have left?",
        "9",
    ),
    (
        "If it takes 5 machines 5 minutes to make 5 widgets, "
        "how long would it take 100 machines to make 100 widgets?",
        "5 minutes",
    ),
    ("What comes next in the sequence: 2, 6, 18, 54, ?", "162"),
    (
        "If A is taller than B, and B is taller than "
        "C, who is the shortest? Answer with one letter.",
        "C",
    ),
    (
        "A clock shows 3:15. What is the angle between "
        "the hour hand and the minute hand in degrees?",
        "7.5",
    ),
    (
        "Mary's father has five daughters: Nana, Nene, "
        "Nini, Nono. What is the fifth daughter's name?",
        "Mary",
    ),
    (
        "If you rearrange the letters 'CIFAIPC', you get the "
        "name of a(n): A) City B) Animal C) Ocean D) Country",
        "ocean",
    ),
    ("What number should replace the question mark: 1, 1, 2, 3, 5, 8, ?", "13"),
    (
        "A bat and a ball cost $1.10 in total. The bat costs $1 more than "
        "the ball. How much does the ball cost? Give your answer in dollars.",
        "0.10",
    ),
    ("If yesterday was two days after Monday, what day is tomorrow?", "Thursday"),
    (
        "In a race, you overtake the person in "
        "second place. What place are you now in?",
        "second",
    ),
    ("How many sides does a hexagon have?", "6"),
    (
        "If 3 cats catch 3 mice in 3 minutes, how many "
        "cats are needed to catch 100 mice in 100 minutes?",
        "3",
    ),
    ("What is the next prime number after 29?", "31"),
    (
        "A snail is at the bottom of a 10-meter well. Each day it climbs 3 "
        "meters, but each night it slips back 2 meters. How many days to get out?",
        "8",
    ),
    (
        "If you have a 3x3x3 cube painted red on all outside faces and cut "
        "into 1x1x1 cubes, how many small cubes have exactly two red faces?",
        "12",
    ),
    (
        "Which does not belong: Apple, Banana, Carrot, "
        "Date, Elderberry? Answer with one word.",
        "Carrot",
    ),
    (
        "If you flip a fair coin twice, what is the probability "
        "of getting at least one head? Give answer as a fraction.",
        "3/4",
    ),
    ("What comes next: J, F, M, A, M, J, ?", "J"),
    (
        "A train leaves station A at 60 mph. Another leaves station B (120 "
        "miles away) toward A at 40 mph. How many hours until they meet?",
        "1.2",
    ),
    (
        "If you write out the numbers from 1 to 100, "
        "how many times does the digit '7' appear?",
        "20",
    ),
    (
        "What is the only number that has the same "
        "number of letters as its value in English?",
        "four",
    ),
    (
        "In a group of 30 people, everyone shakes hands "
        "with everyone else once. How many handshakes occur?",
        "435",
    ),
    ("If A=1, B=2, ..., Z=26, what is the sum of the letters in 'CAT'?", "24"),
]

MATH_QUESTIONS = [
    ("What is 17 x 23?", "391"),
    ("Simplify: (12 + 8) / 5", "4"),
    ("What is the square root of 2025?", "45"),
    ("Calculate: 15% of 240", "36"),
    ("If x + 7 = 15, what is x?", "8"),
    ("What is 2^10?", "1024"),
    (
        "A rectangle has length 12 and width 5. What is its diagonal? Give the number.",
        "13",
    ),
    ("Convert 72 km/h to m/s", "20"),
    ("What is the sum of interior angles of a pentagon in degrees?", "540"),
    ("Solve: 3x - 9 = 12. What is x?", "7"),
    ("What is the greatest common divisor of 48 and 36?", "12"),
    (
        "A circle has radius 7. What is its area? Use "
        "pi approx 3.14, round to nearest whole number.",
        "154",
    ),
    ("What is 0.375 as a fraction in simplest form?", "3/8"),
    (
        "If a triangle has sides 5, 12, and 13, what "
        "type of triangle is it? Answer with one word.",
        "right",
    ),
    ("Calculate the factorial of 6 (6!)", "720"),
    ("What is the median of: 3, 7, 8, 5, 12, 15, 9?", "8"),
    (
        "A store offers a 20% discount on an item priced at "
        "$85. What is the sale price? Give just the number.",
        "68",
    ),
    ("What is log base 10 of 1000?", "3"),
    ("How many degrees are in a full circle?", "360"),
    ("If f(x) = 2x^2 + 3, what is f(4)?", "35"),
    ("What is the least common multiple of 8 and 12?", "24"),
    ("A right triangle has legs 9 and 12. What is its area?", "54"),
    ("Convert 3/7 to a decimal, rounded to two decimal places.", "0.43"),
    ("What is the derivative of x^3 at x = 2? Give just the number.", "12"),
    (
        "If you invest $1000 at 5% annual interest compounded "
        "yearly, how much after 2 years? Round to nearest dollar.",
        "1103",
    ),
]

CODING_QUESTIONS = [
    (
        "Write a Python function called 'is_palindrome' that "
        "takes a string and returns True if it reads the same "
        "forwards and backwards. Return only the function code.",
        None,
    ),
    ("Write a one-line Python expression to reverse a list called 'lst'.", "lst[::-1]"),
    ("What does this Python code output? print(len('hello world'))", "11"),
    (
        "Write a Python function 'fibonacci(n)' that returns the "
        "nth Fibonacci number (0-indexed). Return only the function.",
        None,
    ),
    (
        "In Python, what is the difference between "
        "a list and a tuple? Answer in one sentence.",
        None,
    ),
    (
        "What does this code output?\nx = [1, 2, 3]\ny = x\ny.append(4)\nprint(x)",
        "[1, 2, 3, 4]",
    ),
    (
        "Write a Python function 'flatten(lst)' that takes a nested "
        "list and returns a flat list. Return only the function.",
        None,
    ),
    (
        "What is the time complexity of binary search on "
        "a sorted array of n elements? Use Big-O notation.",
        "O(log n)",
    ),
    (
        "Write a Python one-liner to find all even numbers from 1 to 20.",
        "[x for x in range(1, 21) if x % 2 == 0]",
    ),
    ("What does this output?\nprint(bool(''))", "False"),
    (
        "Write a Python function 'count_vowels(s)' that returns "
        "the number of vowels in string s. Return only the function.",
        None,
    ),
    (
        "In Python, what keyword is used to define a "
        "constant? (Hint: there isn't one - explain briefly.)",
        None,
    ),
    (
        "What does this code output?\na = {1: "
        "'one', 2: 'two'}\nprint(a.get(3, 'missing'))",
        "missing",
    ),
    (
        "Write a Python function 'binary_search(arr, "
        "target)' on a sorted list. Return only the function.",
        None,
    ),
    ("What is the output of: print(0.1 + 0.2 == 0.3)", "False"),
    (
        "Write a Python function 'merge_sort(lst)' that "
        "returns a new sorted list. Return only the function.",
        None,
    ),
    ("In SQL, what clause filters groups created by GROUP BY?", "HAVING"),
    ("What does this output?\nprint(type([]) is type(list()))", "True"),
    (
        "Write a Python function 'anagram(s1, s2)' that returns "
        "True if s1 and s2 are anagrams. Return only the function.",
        None,
    ),
    ("What HTTP status code means 'Not Found'?", "404"),
    (
        "Write a Python function 'factorial(n)' "
        "using recursion. Return only the function.",
        None,
    ),
    (
        "In Git, what command creates a new branch and switches to it in one step?",
        "git checkout -b",
    ),
    ("What does this output?\nprint([x**2 for x in range(5)])", "[0, 1, 4, 9, 16]"),
    (
        "Write a Python function 'remove_duplicates(lst)' "
        "that preserves order. Return only the function.",
        None,
    ),
    ("What is the output of: print('abc' * 3)", "abcabcabc"),
]

KNOWLEDGE_QUESTIONS = [
    ("What is the chemical symbol for gold?", "Au"),
    ("Who painted the Mona Lisa? Give just the last name.", "Da Vinci"),
    ("What planet is known as the Red Planet?", "Mars"),
    ("What year did World War II end?", "1945"),
    ("What is the capital of Australia?", "Canberra"),
    ("Which element has atomic number 1?", "Hydrogen"),
    ("Who wrote 'Romeo and Juliet'?", "Shakespeare"),
    ("What is the largest ocean on Earth?", "Pacific"),
    ("In what year did the first Moon landing occur?", "1969"),
    (
        "What gas do plants absorb from the atmosphere during photosynthesis?",
        "Carbon dioxide",
    ),
    ("Who developed the theory of relativity? Give just the last name.", "Einstein"),
    ("What is the hardest natural substance on Earth?", "Diamond"),
    ("Which country has the most population in the world?", "India"),
    (
        "What is the speed of light approximately in km/s? Give a round number.",
        "300000",
    ),
    ("Who was the first President of the United States?", "Washington"),
    ("What is the smallest prime number?", "2"),
    ("Which organ in the human body produces insulin?", "Pancreas"),
    ("What language has the most native speakers worldwide?", "Chinese"),
    ("What is the boiling point of water at sea level in Celsius?", "100"),
    ("Who discovered penicillin? Give just the last name.", "Fleming"),
    ("What is the longest river in Africa?", "Nile"),
    ("Which planet has the most moons in our solar system?", "Saturn"),
    ("What year did the Berlin Wall fall?", "1989"),
    ("What is the currency of Japan?", "Yen"),
    ("Who wrote '1984'?", "Orwell"),
]

INSTRUCTION_FOLLOWING_QUESTIONS = [
    ("Write exactly three words. No more, no less.", None),
    ("List the numbers from 1 to 5, each on its own line, with nothing else.", None),
    ("Respond only with the word 'banana'. Do not add any other text.", "banana"),
    ("Give me a sentence that contains exactly 7 words. Count carefully.", None),
    (
        "Write the alphabet backwards from Z to A "
        "as one continuous string with no spaces.",
        "zyxwvutsrqponmlkjihgfedcba",
    ),
    (
        "List three fruits, each starting with a different "
        "letter of the alphabet, comma-separated only.",
        None,
    ),
    (
        "Do not use the letter 'e' in your "
        "response. Write a short sentence about cats.",
        None,
    ),
    ("Output exactly this text and nothing else: HELLO WORLD", "HELLO WORLD"),
    (
        "Write a haiku (5-7-5 syllables) about "
        "programming. Output only the poem, no explanation.",
        None,
    ),
    (
        "Give me 4 colors in alphabetical order, "
        "one per line, with no numbering or bullets.",
        None,
    ),
    ("Respond with exactly one word that means 'happy'.", None),
    ("Write a sentence where every word starts with the letter 's'.", None),
    (
        "List these numbers from smallest to largest: 42, 7, "
        "103, 28. Output only the numbers separated by commas.",
        "7, 28, 42, 103",
    ),
    (
        "Write a title for an essay about climate "
        "change. The title must be exactly 5 words long.",
        None,
    ),
    ("Output the word 'success' in all capital letters and nothing else.", "SUCCESS"),
    (
        "Give me two synonyms for 'big'. Output "
        "only the two words separated by a comma.",
        None,
    ),
    ("Write a question that ends with the word 'why?'.", None),
    (
        "List 5 countries. Each must have fewer than 7 "
        "letters in its name. One per line, no numbering.",
        None,
    ),
    (
        "Repeat this exact phrase: The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumps over the lazy dog",
    ),
    (
        "Write a sentence that has exactly 10 words. "
        "Count carefully and output only the sentence.",
        None,
    ),
    (
        "Name three planets in our solar system, ordered by "
        "distance from the Sun (closest first). Comma-separated only.",
        None,
    ),
    ("Output the number 42 in Roman numerals and nothing else.", "XLII"),
    ("Write a short greeting. Do not use any punctuation marks at all.", None),
    ("Give me exactly two sentences about water. No more, no less.", None),
    (
        "List the four seasons in order starting from "
        "spring. Use only lowercase letters, comma-separated.",
        "spring, summer, fall, winter",
    ),
]

ALL_CATEGORIES = {
    "reasoning": REASONING_QUESTIONS,
    "math": MATH_QUESTIONS,
    "coding": CODING_QUESTIONS,
    "knowledge": KNOWLEDGE_QUESTIONS,
    "instruction_following": INSTRUCTION_FOLLOWING_QUESTIONS,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class QuestionResult:
    question_idx: int
    prompt: str
    response: str
    correct: bool | None  # None = not auto-scoreable
    score: float  # 0.0-1.0
    latency_ms: float
    tokens_used: int | None = None


@dataclass
class QuestionOutcome:
    """One graded attempt: what the server returned and what it scored.

    ``error`` is set (and the attempt is unscoreable) when the API failed —
    a dead server must never be scored as a wrong answer.
    """

    response: str = ""
    correct: bool | None = None
    score: float = 0.0
    latency_ms: float = 0.0
    tokens_used: int | None = None
    error: str = ""

    @property
    def answered(self) -> bool:
        return not self.error and bool(self.response.strip())


@dataclass
class CategoryResult:
    category: str
    results: list[QuestionResult] = field(default_factory=list)

    @property
    def correct_count(self) -> int:
        return sum(1 for r in self.results if r.correct is True)

    @property
    def scoreable_count(self) -> int:
        return sum(1 for r in self.results if r.correct is not None)

    @property
    def accuracy(self) -> float | None:
        if self.scoreable_count == 0:
            return None
        return round(self.correct_count / self.scoreable_count * 100, 1)

    @property
    def avg_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return round(statistics.mean(r.latency_ms for r in self.results), 1)

    @property
    def p95_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        vals = sorted(r.latency_ms for r in self.results)
        idx = int(len(vals) * 0.95)
        return round(vals[min(idx, len(vals) - 1)], 1)


@dataclass
class ModelResult:
    model: str
    profile: str = ""
    categories: dict[str, CategoryResult] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        """Result key: ``model|profile`` when a profile is set (decision #055)
        — the same model behaves differently under deep-analysis vs
        fast-codegen, so results must be keyed by the full context."""
        return f"{self.model}|{self.profile}" if self.profile else self.model

    @property
    def total_correct(self) -> int:
        return sum(c.correct_count for c in self.categories.values())

    @property
    def total_scoreable(self) -> int:
        return sum(c.scoreable_count for c in self.categories.values())

    @property
    def overall_accuracy(self) -> float | None:
        if self.total_scoreable == 0:
            return None
        return round(self.total_correct / self.total_scoreable * 100, 1)

    @property
    def avg_latency_ms(self) -> float:
        all_lat = [r.latency_ms for c in self.categories.values() for r in c.results]
        if not all_lat:
            return 0.0
        return round(statistics.mean(all_lat), 1)

    @property
    def total_questions(self) -> int:
        return sum(len(c.results) for c in self.categories.values())

    @property
    def total_tokens(self) -> int | None:
        """Sum of reported ``usage.total_tokens``; None when no question
        reported usage (some servers omit it)."""
        vals = [
            r.tokens_used
            for c in self.categories.values()
            for r in c.results
            if r.tokens_used is not None
        ]
        return sum(vals) if vals else None

    def accuracy_for(self, category: str) -> float | None:
        cr = self.categories.get(category)
        return cr.accuracy if cr else None


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


#: LaTeX a reasoning model wraps math answers in: \(x=8\), $\frac{3}{8}$,
#: \[ 42 \].  These are stripped before normalization so a formatted answer
#: grades the same as the plain one the question sheet expects.
_LATEX_DELIMS_RE = re.compile(r"\\[()[\]]|\$+")
_LATEX_SPACING_RE = re.compile(r"\\(?:left|right|!|,|;|:|quad|qquad)")
_LATEX_FRAC_CMDS = ("\\dfrac", "\\tfrac", "\\frac")
_LATEX_SYMBOLS = (
    ("\\times", "*"),
    ("\\cdot", "*"),
    ("\\div", "/"),
    ("\\pi", "pi"),
    ("\\%", "%"),
)


def _read_brace_group(text: str, i: int) -> tuple[str, int]:
    """Return the contents of the brace group starting at *text[i]* and the
    index just past its closing brace.  Unbalanced input degrades to a single
    character so a malformed answer can never raise mid-run."""
    if i >= len(text) or text[i] != "{":
        return "", i
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j], j + 1
    return text[i + 1 :], len(text)


def _convert_latex_fracs(text: str) -> str:
    """Rewrite ``\\frac{a}{b}`` (and dfrac/tfrac) as ``a/b``.

    A plain ``a/b`` matters: normalize_answer keeps "/" but drops brackets, so
    ``(3)/(8)`` would normalize to "3 / 8" and never equal the expected "3/8".
    Nested fractions are handled by rescanning after each rewrite.
    """
    out = text
    for cmd in _LATEX_FRAC_CMDS:
        while cmd in out:
            start = out.find(cmd)
            i = start + len(cmd)
            while i < len(out) and out[i] in " \t":
                i += 1
            num, i = _read_brace_group(out, i)
            while i < len(out) and out[i] in " \t":
                i += 1
            den, i = _read_brace_group(out, i)
            if not num and not den:
                # \frac with no operands: drop the command only.
                out = out[:start] + " " + out[start + len(cmd) :]
                continue
            out = f"{out[:start]}{num.strip()}/{den.strip()}{out[i:]}"
    return out


def latex_to_text(text: str) -> str:
    """Strip LaTeX markup so math answers compare as text.

    Only activates when the text shows LaTeX markers, so ordinary prose is
    passed through untouched.
    """
    if "\\" not in text and "$" not in text:
        return text
    out = _convert_latex_fracs(text)
    for sym, plain in _LATEX_SYMBOLS:
        out = out.replace(sym, plain)
    out = _LATEX_DELIMS_RE.sub(" ", out)
    out = _LATEX_SPACING_RE.sub(" ", out)
    return out.replace("{", " ").replace("}", " ")


def normalize_answer(text: str | None) -> str:
    if not text:
        # None (a null "content" from the provider) means no answer —
        # normalize to "" so answers_match scores it False instead of
        # crashing the whole run on .strip().
        return ""
    text = latex_to_text(text.strip().lower())
    text = re.sub(r"[^\w\s./]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


#: A labelled final answer, e.g. the line the deep-analysis review pass asks
#: the model to end with ("ANSWER: 42").  Graded on the label's payload, so a
#: long self-review is not scored wrong just for containing reasoning text.
_ANSWER_LABEL_RE = re.compile(r"(?im)^\s*(?:final\s+)?answer\s*[:=]\s*(.+?)\s*$")


def extract_final_answer(response: str | None) -> str:
    """Return the model's declared final answer, else *response* unchanged.

    The deep-analysis review prompt ends with "ANSWER: <final answer>"; when
    that label is present the payload is the answer being graded.  The LAST
    label wins (a model that restates itself leaves the newest verdict last).
    """
    if not response:
        return response or ""
    matches = _ANSWER_LABEL_RE.findall(response)
    if not matches:
        return response
    last = matches[-1].strip()
    return last or response


def _to_float(text: str) -> float | None:
    """Parse a plain number, tolerating thousands separators and spaces."""
    try:
        return float(text.replace(",", "").replace(" ", ""))
    except (ValueError, TypeError, AttributeError):
        return None


def _eval_slash_expr(text: str) -> float | None:
    """Evaluate a bare ``a/b[/c...]`` expression left to right.

    Returns None when the text is not a pure slash expression, so callers can
    fall back.  Nested ``\\frac{\\frac{1}{2}}{2}`` flattens to "1/2/2" during
    LaTeX conversion; evaluating left to right yields the right value.
    """
    parts = text.replace(" ", "").split("/")
    if len(parts) < 2:
        return None
    values: list[float] = []
    for part in parts:
        value = _to_float(part)
        if value is None:
            return None
        values.append(value)
    result = values[0]
    for divisor in values[1:]:
        if divisor == 0:
            return None
        result /= divisor
    return result


def _numeric_candidates(text: str) -> list[float]:
    """Return every numeric reading of *text* (plain value and slash value).

    "0.5" -> [0.5];  "1/2" -> [0.5];  "1/2/2" -> [0.25].  A text that is not
    numeric at all contributes nothing, so prose never scores numerically.
    """
    candidates: list[float] = []
    plain = _to_float(text)
    if plain is not None:
        candidates.append(plain)
    slash = _eval_slash_expr(text)
    if slash is not None and slash not in candidates:
        candidates.append(slash)
    return candidates


def answers_match(response: str | None, expected: str) -> bool:
    norm_resp = normalize_answer(extract_final_answer(response))
    norm_exp = normalize_answer(expected)

    if not norm_exp or not norm_resp:
        return False

    if norm_resp == norm_exp:
        return True

    # Word-bounded containment ONLY: raw substring matching produced false
    # positives ("42" inside "142", "true" inside "retry").  The WHOLE
    # expected answer must appear as a word-bounded phrase in the response.
    if len(norm_exp) >= 2 and re.search(
        rf"(?<!\w){re.escape(norm_exp)}(?!\w)", norm_resp
    ):
        return True

    # Short numeric answers: a reasoning model says "x = 8" or "gives 8" where
    # the sheet expects "8".  Compare WHOLE tokens, never substrings, so "5"
    # still cannot match "15" and "42" still cannot match "142".
    if re.fullmatch(r"\d+(?:\.\d+)?", norm_exp) and norm_exp in norm_resp.split():
        return True

    # Numeric tolerance for math answers (also accept "1 000" for "1,000")
    try:
        r_val = float(norm_resp.replace(",", "").replace(" ", ""))
        e_val = float(norm_exp.replace(",", "").replace(" ", ""))
        if abs(r_val - e_val) < 0.5:
            return True
    except (ValueError, TypeError):
        logger.debug(
            "Numeric comparison failed for response=%r expected=%r", norm_resp, norm_exp
        )

    # Fraction equivalence (0.5 == 1/2, and nested \frac{a}{b} forms which
    # flatten to "1/2/2" and evaluate left to right).  Both sides are reduced
    # to candidate numeric values and compared pairwise, so plain decimals and
    # slash expressions match in either direction.
    for r_cand in _numeric_candidates(norm_resp):
        for e_cand in _numeric_candidates(norm_exp):
            if abs(r_cand - e_cand) < 0.01:
                return True

    # Keyword overlap for knowledge questions — TIGHTENED: overlap must be
    # substantial (>= 0.85 of the expected words), at least two words must
    # match, and the response must not pad the expected answer with many
    # extra words (the old >= 0.7 overlap accepted one-keyword guesses).
    resp_words = set(norm_resp.split())
    exp_words = set(norm_exp.split())
    if len(exp_words) > 0:
        overlap = len(resp_words & exp_words) / len(exp_words)
        if (
            overlap >= 0.85
            and len(resp_words & exp_words) >= 2
            and len(resp_words) <= len(exp_words) * 2 + 5
        ):
            return True

    return False


def _count_syllables(word: str) -> int:
    """Improved English syllable estimator using common rules.

    Handles silent-e, vowel clusters, and common exceptions.
    """
    word = word.lower().strip(".,;:!?\"'()-")
    if not word:
        return 0

    # Common single-syllable words that would otherwise be miscounted
    exceptions_map: dict[str, int] = {
        "the": 1,
        "a": 1,
        "i": 1,
        "is": 1,
        "it": 1,
        "to": 1,
        "of": 1,
        "and": 1,
        "in": 1,
        "for": 1,
        "on": 1,
        "be": 1,
        "as": 1,
        "with": 1,
        "that": 1,
        "this": 1,
        "are": 1,
        "was": 1,
        "has": 1,
        "had": 1,
        "but": 1,
        "not": 1,
    }
    if word in exceptions_map:
        return exceptions_map[word]

    # Count vowel groups
    vowels = set("aeiouy")
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel

    # Silent final 'e' (but not '-le', '-ve', '-ne' endings)
    if (
        word.endswith("e")
        and len(word) > 2
        and word[-3:] not in ("le", "ve", "ne")
        and count > 1
    ):
        count -= 1

    # '-ed' adds a syllable only when pronounced as /id/ (after t/d)
    if word.endswith("ed"):
        base = word[:-2]
        if not base:
            pass
        elif base[-1:] in ("t", "d") and count > 1:
            pass  # already counted
        else:
            count -= 1

    return max(1, count)


def _count_line_syllables(line: str) -> int:
    """Count syllables for a single line of text."""
    words = re.findall(r"[a-zA-Z]+", line.lower())
    return sum(_count_syllables(w) for w in words)


def score_instruction_following(prompt: str, response: str) -> float:
    """Score instruction-following on a 0-1 scale based on constraint checks."""
    score = 1.0
    pl = prompt.lower()
    rl = response.lower().strip()

    # "exactly three words" / "exactly N words"
    m = re.search(r"exactly\s+(\d+)\s+word", pl)
    if m:
        n = int(m.group(1))
        actual = len(rl.split())
        if actual != n:
            score -= 0.5

    # "nothing else" / "only" constraints -- penalize extra text.  The old
    # check credited any response containing the target as a substring plus
    # up to 3 extra words ("true and false" for target "true").  Now the
    # response must be an exact echo of the target (punctuation-insensitive):
    # the word sets must be identical.
    if "nothing else" in pl or "and nothing else" in pl:
        expected_match = re.search(
            r"(?:output|respond|repeat)\s+(?:with\s+)?(?:exactly\s+)?"
            r'["\']?([^"\']+?)["\']?\s*(?:\.|and nothing else|$)',
            pl,
            re.IGNORECASE,
        )
        if expected_match:
            target = expected_match.group(1).strip().lower()
            target_clean = target.strip(".,!?;:()\"' ")
            resp_clean = rl.strip(".,!?;:()\"' ")
            target_words = set(target_clean.split())
            resp_words = set(resp_clean.split())
            if resp_clean != target_clean and not (
                target_words and resp_words == target_words
            ):
                score -= 0.5

    # "no punctuation" constraint
    if "do not use any punctuation" in pl:
        punct_chars = set(".,!?;:\"'()-")
        if any(c in punct_chars for c in response):
            score -= 0.3

    # "exactly one word"
    if "exactly one word" in pl or "one word that means" in pl:
        if len(rl.split()) != 1:
            score -= 0.5

    # Syllable count for haiku (improved estimator)
    if "haiku" in pl and "5-7-5" in pl:
        lines = [ln.strip() for ln in rl.split("\n") if ln.strip()]
        if len(lines) == 3:
            syllables = [_count_line_syllables(line) for line in lines]
            if not (
                4 <= syllables[0] <= 6
                and 6 <= syllables[1] <= 8
                and 4 <= syllables[2] <= 6
            ):
                score -= 0.3

    # "no numbering or bullets"
    if "no numbering" in pl or "no bullets" in pl:
        if re.search(r"^\d+[\.\)]", rl, re.MULTILINE) or re.search(
            r"^[-*]", rl, re.MULTILINE
        ):
            score -= 0.3

    # "lowercase only" / "all lowercase" — compare the RAW response; rl is
    # already lowercased so comparing rl to rl.lower() could never fire.
    if "only lowercase" in pl or "all lowercase" in pl:
        raw = response.strip()
        if raw != raw.lower():
            score -= 0.3

    return max(0.0, min(1.0, round(score, 2)))


async def judge_answer(judge_fn: Any, response: str, expected: str) -> bool | None:
    """LLM-judged verdict when heuristics cannot decide.

    *judge_fn* is an async callable ``(response, expected) -> str``.  Returns
    True/False on a clear verdict, None when the judge is absent, raises, or
    is ambiguous.  The judge is a FALLBACK only — exact/numeric/overlap
    matches never consult it.
    """
    if judge_fn is None:
        return None
    try:
        verdict = await judge_fn(response, expected)
    except Exception:
        return None
    low = str(verdict).strip().lower()
    if low.startswith(("yes", "true", "correct", "match", "similar")):
        return True
    if low.startswith(("no", "false", "incorrect", "different", "not")):
        return False
    return None


async def _correct_with_judge(response: str, expected: str, judge_fn: Any) -> bool:
    """answers_match with the optional LLM-judge fallback."""
    correct = answers_match(response, expected)
    if correct or judge_fn is None:
        return correct
    judged = await judge_answer(judge_fn, response, expected)
    return correct if judged is None else judged


async def score_question(
    prompt: str,
    response: str,
    expected: str | None,
    category: str,
    judge_fn: Any = None,
) -> tuple[bool | None, float]:
    """Returns (correct_bool_or_None, score_0_to_1).

    *judge_fn* (async ``(response, expected) -> str``) is consulted only when
    the heuristic scorer returns False — LLM-judged matching for answers the
    exact/numeric/keyword checks cannot decide (plan FIX item 21).
    """

    if category == "instruction_following":
        inst_score = score_instruction_following(prompt, response)
        if expected:
            match_result = await _correct_with_judge(response, expected, judge_fn)
            return match_result, max(inst_score, 1.0 if match_result else 0.0)
        return None, inst_score

    if category == "coding":
        # For code-generation questions (expected is None), check syntax
        if expected is None:
            code_match = re.search(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
            code = code_match.group(1).strip() if code_match else response.strip()
            try:
                compile(code, "<benchmark>", "exec")
                return None, 0.7  # Valid syntax -- partial credit
            except SyntaxError:
                return False, 0.0

        # For output-prediction questions
        matched = await _correct_with_judge(response, expected, judge_fn)
        return matched, 1.0 if matched else 0.0

    # Reasoning / Math / Knowledge
    if expected is None:
        return None, 0.5  # Unscoreable -- give neutral score

    correct = await _correct_with_judge(response, expected, judge_fn)
    return correct, 1.0 if correct else 0.0


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

#: Provider name -> OpenAI-compatible base URL.  ``lmstudio`` is the default;
#: ``llama`` is a local llama.cpp llama-server whose canonical base URL is
#: ``AGENT_LLAMA_URL`` (default ``http://127.0.0.1:8080/v1``, the port the
#: agent framework already uses for llama.cpp -- decision #014).
PROVIDER_BASE_URLS: dict[str, str] = {
    "lmstudio": lmstudio_base_url(),
    "llama": DEFAULT_LLAMA_BASE_URL,
}
DEFAULT_PROVIDER = "lmstudio"

#: Benchmark profiles.  ``standard`` stays the historic single-shot,
#: temperature-0.1, 2048-token run so old snapshots stay comparable.
#: ``deep-analysis`` mirrors agent_core.llm.model_profiles' deep profile
#: (temperature 0.7 = brainstorming/analysis temperature) with a token budget
#: reasoning models need, and re-asks reasoning/math questions so a model that
#: reasons itself into a wrong first answer can correct it on review.
#:
#: ``max_tokens`` is resolved at run time (CLI, then env, then this default) to
#: keep the benchmark usable on servers with a small context window.
PROFILES: dict[str, dict[str, Any]] = {
    "standard": {
        "temperature": 0.1,
        "max_tokens": 2048,
        "verify_round": False,
    },
    "deep-analysis": {
        "temperature": 0.7,
        "max_tokens": 16384,
        "verify_round": True,
    },
}
DEFAULT_BENCH_PROFILE = "standard"

#: Environment override for the deep-analysis token budget (the benchmark CLI
#: cannot know llama-server's -c value, so deployments that run a small context
#: set this instead of editing code).
DEEP_MAX_TOKENS_ENV = "BENCHMARK_DEEP_MAX_TOKENS"

#: Deep-analysis verification pass.  Only the factual categories are re-asked:
#: instruction-following answers are graded on the shape of the first attempt
#: and coding answers are graded by the judge, so a retry would just add
#: latency to the same verdict.
VERIFY_CATEGORIES: frozenset[str] = frozenset({"reasoning", "math"})

#: Upper bound on how much of an answer is kept in the report (a deep run's
#: reasoning can run to tens of KB per question).
REPORT_ANSWER_CHARS = 4000


def resolve_bench_profile(profile: str) -> dict[str, Any]:
    """Return the setting dict for *profile*.

    An unknown profile name is not an error: the user may want a custom tag
    (e.g. ``--profile friday-1``) that only groups results in the report.  Such
    a tag gets the standard settings unless it names one of the built-ins.
    """
    if profile in PROFILES:
        return PROFILES[profile]
    return PROFILES[DEFAULT_BENCH_PROFILE]


def resolve_server_model_id(
    provider: str,
    chat_url: str,
    model: str,
    timeout: float = 3.0,
) -> str:
    """Return the model id the server will actually accept.

    A llama.cpp ``llama-server`` loads exactly one model at startup and
    registers it under its own id (e.g.
    ``deepseek-v4.1-flash-ista-2.00bpw``), so the ``llama/`` routing label the
    agent framework uses (``llama/deepseek-v4.1-flash``) is rejected with
    ``model not found``.  For the ``llama`` provider we therefore ask
    ``GET <base>/models`` which id is served and use that when the requested
    name does not match.  Best effort: any failure keeps the requested name.
    """
    if provider != "llama":
        return model

    base = chat_url[: -len("/chat/completions")] if chat_url.endswith(
        "/chat/completions"
    ) else chat_url
    try:
        import urllib.request

        req = urllib.request.Request(f"{base}/models", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as exc:  # noqa: BLE001 - discovery is best effort
        logger.debug("Could not list llama-server models: %s", exc)
        return model

    items = data.get("data") if isinstance(data, dict) else None
    served = [
        str(m.get("id")).strip()
        for m in (items or [])
        if isinstance(m, dict) and m.get("id")
    ]
    if not served:
        return model

    bare = model[len("llama/") :] if model.startswith("llama/") else model
    for candidate in served:
        if candidate == model or candidate == bare:
            return candidate

    # The requested label is not what the server serves.  Use the server's id
    # and say so: silently benchmarking a different loaded model would make
    # the report a lie.
    chosen = served[0]
    print(
        f"  Note: llama-server serves {chosen!r}; requesting that instead "
        f"of {model!r} (loaded model wins on a single-model server)."
    )
    return chosen


def print_deep_summary(results: list[ModelResult], profile: str) -> None:
    """Print the per-question detail a deep-analysis run exists for.

    The comparison table only shows accuracy; a deep run also cares about how
    the questions it can act on went (reasoning/math network), how much it
    cost, and which answers came in under the budget — i.e. were cut off before
    they could be graded.
    """
    for r in results:
        print(f"  Deep-analysis detail for {r.display_name}")
        print(f"  {'-' * 60}")
        tokens = r.total_tokens
        print(
            f"    Total tokens:  {tokens if tokens is not None else 'not reported'}"
            f"   Questions: {r.total_questions}"
        )
        for cat in ("reasoning", "math", "coding"):
            acc = r.accuracy_for(cat)
            if acc is not None:
                print(f"    {cat:<12} {acc:>6.1f}%")
        truncated = [
            qr
            for c in r.categories.values()
            for qr in c.results
            if qr.correct is False and not qr.response.strip()
        ]
        if truncated:
            print(
                f"    WARNING: {len(truncated)} question(s) returned an empty "
                "answer (token budget exhausted?) — raise --max-tokens."
            )
        print()


def deep_max_tokens(default: int) -> int:
    """Resolve the deep-analysis token budget: env override, else *default*."""
    raw = os.environ.get(DEEP_MAX_TOKENS_ENV)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Ignoring %s=%r: not an integer.", DEEP_MAX_TOKENS_ENV, raw
        )
        return default
    if value <= 0:
        logger.warning(
            "Ignoring %s=%r: must be positive.", DEEP_MAX_TOKENS_ENV, raw
        )
        return default
    return value


#: Default chat-completions URL; set from --provider/--url in main().
BASE_URL = f"{lmstudio_base_url()}/chat/completions"


def resolve_base_url(provider: str, url: str | None = None) -> str:
    """Resolve the chat-completions URL for ``provider``.

    An explicit ``url`` always wins.  Otherwise the provider's canonical
    base URL is used; a bare host:port gets the OpenAI-compatible
    ``/v1/chat/completions`` suffix appended (llama-server serves under
    ``/v1``, LM Studio at the root).
    """
    if url:
        base = url.rstrip("/")
    else:
        base = PROVIDER_BASE_URLS.get(
            provider, PROVIDER_BASE_URLS[DEFAULT_PROVIDER]
        ).rstrip("/")
    if not base.lower().endswith("/chat/completions"):
        if not base.lower().endswith("/v1"):
            base = f"{base}/v1"
        base = f"{base}/chat/completions"
    return base


async def query_model(
    model: str,
    prompt: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_tokens: int = 2048,
    temperature: float = 0.1,
) -> tuple[str, float, int | None]:
    (
        """Returns (response_text, latency_ms, """
        """tokens_used). Raises ModelAPIError on failure."""
    )
    import urllib.request
    import urllib.error

    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            # 0.1 keeps the standard run reproducible; the deep-analysis
            # profile passes 0.7 (see PROFILES) to sample the way the agent's
            # analysis prompts do.
            "temperature": temperature,
            # Reasoning models (e.g. qwen3.8-27b) consume the budget in
            # reasoning_content BEFORE emitting content: with 512 tokens the
            # answer came back empty (finish_reason "length"), scoring a
            # False the model did not deserve.  2048 gives the reasoning
            # room to finish AND the answer room to exist; deep-analysis
            # raises it further.
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        BASE_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_error: str | None = None
    for attempt in range(max_retries):
        start = time.monotonic()
        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None, lambda: urllib.request.urlopen(req, timeout=120)
            )
            latency_ms = (time.monotonic() - start) * 1000
            body = json.loads(resp.read().decode("utf-8"))
            message = body["choices"][0].get("message") or {}
            text = message.get("content")
            if not isinstance(text, str):
                # llama.cpp's OpenAI-compat server returns "content": null
                # for a pure-reasoning turn or when the token budget runs out
                # before any visible text.  There is no answer to score —
                # hand the scorer an empty string instead of None, which
                # crashed normalize_answer mid-run (AttributeError on .strip).
                text = ""
            tokens = body.get("usage", {}).get("total_tokens")
            return text, latency_ms, tokens

        except urllib.error.HTTPError as e:
            last_error = f"HTTP {e.code}"
            if e.code >= 500:
                delay = base_delay * (2**attempt)
                print(
                    f"    [retry {attempt + 1}/{max_retries}] Server error, "
                    f"waiting {delay:.0f}s..."
                )
                await asyncio.sleep(delay)
                continue
            raise ModelAPIError(f"HTTP Error: {e.code}") from e

        except urllib.error.URLError as e:
            last_error = str(e.reason) if hasattr(e, "reason") else str(e)
            delay = base_delay * (2**attempt)
            print(
                f"    [retry {attempt + 1}/{max_retries}] Connection error, "
                f"waiting {delay:.0f}s..."
            )
            await asyncio.sleep(delay)

        except Exception as e:
            raise ModelAPIError(str(e)) from e

    raise ModelAPIError(f"Failed after {max_retries} retries: {last_error}")


def _probe_endpoint(base_url: str, timeout: float = 5.0) -> None:
    """Verify the provider endpoint is usable before running the benchmark.

    Sends a minimal chat request; raises BenchmarkError with a clear message if
    the server is down, rejects the request (e.g. no model loaded), or fails to
    return a valid chat-completion body.  This is the fast path to a "no live
    model" condition so the caller can fall back instead of stalling per-
    question on urlopen(timeout=120) retries.
    """
    import urllib.error
    import urllib.request

    payload = json.dumps(
        {
            "model": "_probe_",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        base_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        # A 4xx with a "no model loaded" / "model not found" body is the
        # common autonomous failure mode: the server is up but no model is
        # loaded, so every one of the ~125 real queries would fail slowly.
        # Surface it immediately so the caller can fall back fast.
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 - best-effort
            pass
        lowered = detail.lower()
        if (
            "no models loaded" in lowered
            or "model not found" in lowered
            or "does not exist" in lowered
        ):
            raise BenchmarkError(
                f"Server has no model available at {base_url}: {detail[:200]}"
            ) from exc
        if exc.code >= 500:
            raise BenchmarkError(
                f"Endpoint {base_url} returned {exc.code} on probe"
            ) from exc
        return
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise BenchmarkError(f"Endpoint unreachable at {base_url}: {exc}") from exc
    # A server that accepts the connection but never returns valid chat JSON
    # (e.g. a stub/health endpoint) would otherwise make every real query
    # block on urlopen(timeout=120).  Treat a non-conforming body as a probe
    # failure so the caller can fall back quickly.
    try:
        data = json.loads(body)
        if not isinstance(data, dict) or "choices" not in data:
            raise ValueError("response missing 'choices'")
    except ValueError as exc:
        raise BenchmarkError(
            f"Endpoint at {base_url} did not return a chat response: {exc}"
        ) from exc


async def evaluate_question(
    model: str,
    prompt: str,
    expected: str | None,
    category: str,
    judge_fn: Any = None,
    max_tokens: int = 2048,
    temperature: float = 0.1,
) -> QuestionOutcome:
    """Ask one question once and grade it.

    API failures come back as an *unscoreable* outcome carrying the error text,
    never as a wrong answer: a dead endpoint must not depress accuracy.
    """
    try:
        response, latency_ms, tokens = await query_model(
            model,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except ModelAPIError as exc:
        return QuestionOutcome(error=f"{exc} (after {time.strftime('%H:%M:%S')})")

    correct, score = await score_question(
        prompt, response, expected, category, judge_fn=judge_fn
    )
    return QuestionOutcome(
        response=response,
        correct=correct,
        score=score,
        latency_ms=latency_ms,
        tokens_used=tokens,
    )


async def verify_question(
    model: str,
    prompt: str,
    expected: str | None,
    category: str,
    first: QuestionOutcome,
    judge_fn: Any = None,
    max_tokens: int = 2048,
    temperature: float = 0.1,
) -> QuestionOutcome:
    """Deep-analysis second pass: shows the model its own answer for review.

    The model gets the original question plus its first attempt and is asked to
    re-check it — the same "look again before you commit" step the agent's
    analysis loop performs.  The verdict is the reviewed answer (up if the model
    corrects itself, down if it talks itself out of a right answer); latency and
    tokens are the sum of both attempts, so a deep run's cost stays visible.

    Returns *first* unchanged when the retry failed, so a flaky second call can
    only ever leave the score where it was.
    """
    review_prompt = (
        f"{prompt}\n\n"
        f"Your previous answer was:\n{first.response.strip()}\n\n"
        "Re-check the reasoning above step by step. If it is correct, restate "
        "the final answer. If it is wrong, give the corrected answer. End your "
        "reply with the line 'ANSWER: <final answer>'."
    )
    second = await evaluate_question(
        model,
        review_prompt,
        expected,
        category,
        judge_fn=judge_fn,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if not second.answered:
        return first
    return QuestionOutcome(
        response=second.response,
        correct=second.correct,
        score=second.score,
        latency_ms=first.latency_ms + second.latency_ms,
        tokens_used=((first.tokens_used or 0) + (second.tokens_used or 0)) or None,
    )


async def run_category(
    model: str,
    category: str,
    questions: list[tuple[str, str | None]],
    repetition: int = 0,
    judge_fn: Any = None,
    max_tokens: int = 2048,
    temperature: float = 0.1,
    verify_round: bool = False,
    response_chars: int = 500,
) -> CategoryResult:
    """Run one category.

    ``verify_round`` is the deep-analysis pass: reasoning/math answers get a
    second, self-reviewing attempt (see ``verify_question``).  ``response_chars``
    caps how much of each answer is kept — 500 for the console-oriented standard
    run, much more when the report is meant to be read.
    """
    cat_result = CategoryResult(category=category)
    total = len(questions)
    verify = verify_round and category in VERIFY_CATEGORIES

    print(f"  [{model}] {category}: running {total} questions...")
    if verify:
        print("    (deep-analysis: second self-review pass on each answer)")
    for idx, (prompt, expected) in enumerate(questions, 1):
        sys.stdout.write(f"\r    Question {idx}/{total}")
        sys.stdout.flush()

        outcome = await evaluate_question(
            model,
            prompt,
            expected,
            category,
            judge_fn=judge_fn,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if verify and outcome.answered:
            outcome = await verify_question(
                model,
                prompt,
                expected,
                category,
                outcome,
                judge_fn=judge_fn,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        if outcome.error:
            # Keep the transport error in the response field so a reader of the
            # report can see why the question is unscoreable.
            response = outcome.error
            correct = None
            score = 0.0
        else:
            response = outcome.response
            correct = outcome.correct
            score = outcome.score

        cat_result.results.append(
            QuestionResult(
                question_idx=idx - 1,
                prompt=prompt,
                response=response[:response_chars],
                correct=correct,
                score=score,
                latency_ms=outcome.latency_ms,
                tokens_used=outcome.tokens_used,
            )
        )

    print(f"\r    Question {total}/{total} -- done              ")
    return cat_result


async def run_benchmark(
    models: list[str],
    categories: dict[str, list[tuple[str, str | None]]],
    repetitions: int = 1,
    judge_fn: Any = None,
    profile: str = "",
    max_tokens: int = 2048,
    temperature: float = 0.1,
    verify_round: bool = False,
    response_chars: int = 500,
) -> list[ModelResult]:
    results = []
    run_kwargs: dict[str, Any] = {
        "judge_fn": judge_fn,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "verify_round": verify_round,
        "response_chars": response_chars,
    }

    for model in models:
        print(f"\n{'=' * 60}")
        print(f"  Benchmarking: {model}")
        if profile:
            print(f"  Profile:      {profile}")
        print(f"{'=' * 60}")

        model_result = ModelResult(model=model, profile=profile)
        for cat_name, questions in categories.items():
            if repetitions == 1:
                cat_res = await run_category(
                    model, cat_name, questions, **run_kwargs
                )
                model_result.categories[cat_name] = cat_res
            else:
                # Run multiple times and average scores
                all_cat_results: list[CategoryResult] = []
                for rep in range(repetitions):
                    print(f"\n  --- Repetition {rep + 1}/{repetitions} ---")
                    cr = await run_category(
                        model,
                        cat_name,
                        questions,
                        repetition=rep,
                        **run_kwargs,
                    )
                    all_cat_results.append(cr)

                # Merge: average scores per question. Precompute the
                # per-question repetition data once (cached outside any
                # further iteration), then build results via comprehension.
                n_questions = len(questions)
                rep_data = [
                    [
                        cr.results[q_idx]
                        for cr in all_cat_results
                        if q_idx < len(cr.results)
                    ]
                    for q_idx in range(n_questions)
                ]

                merged = CategoryResult(category=cat_name)
                merged.results = [
                    QuestionResult(
                        question_idx=q_idx,
                        prompt=reps[0].prompt if reps else "",
                        response=(reps[0].response[:500] if reps else ""),
                        correct=(
                            all(
                                r.correct is True for r in reps if r.correct is not None
                            )
                            if any(r.correct is True for r in reps)
                            else (
                                False
                                if any(r.correct is not None for r in reps)
                                else None
                            )
                        ),
                        score=round(statistics.mean(r.score for r in reps), 2)
                        if reps
                        else 0.0,
                        latency_ms=statistics.mean(r.latency_ms for r in reps)
                        if reps
                        else 0.0,
                    )
                    for q_idx, reps in enumerate(rep_data)
                ]
                model_result.categories[cat_name] = merged

        results.append(model_result)
        print(
            f"\n  {model} summary: "
            f"{model_result.total_correct}/{model_result.total_scoreable} correct "
            f"({model_result.overall_accuracy}%), "
            f"avg latency {model_result.avg_latency_ms}ms\n"
        )

    return results


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


def print_report(results: list[ModelResult]) -> None:
    """Print a formatted comparison table to the console."""
    # Build ordered, de-duplicated category names via dict.fromkeys (preserves
    # insertion order without repeated append + membership checks).
    cat_names = list(dict.fromkeys(c for r in results for c in r.categories))

    header = f"{'Model':<30}"
    for cat in cat_names:
        short = cat.replace("_", " ").title()[:12]
        header += f"  {short:<14}"
    header += f"  {'Overall':<10}  {'Avg Latency':>12}"
    print(f"\n{'=' * len(header)}")
    print(header)
    print("-" * len(header))

    for r in results:
        line = f"{r.display_name:<30}"
        for cat in cat_names:
            cr = r.categories.get(cat)
            if cr and cr.accuracy is not None:
                line += f"  {cr.accuracy:>6.1f}%{'':<7}"
            elif cr:
                line += f"  {'N/A':>9}{'':<5}"
            else:
                line += f"  {'---':>9}{'':<5}"

        overall = r.overall_accuracy
        if overall is not None:
            line += f"  {overall:>6.1f}%{'':<3}"
        else:
            line += f"  {'N/A':>9}"
        line += f"  {r.avg_latency_ms:>8.0f}ms"
        print(line)

    print()


def build_model_json(model_result: ModelResult) -> dict[str, Any]:
    """Build a JSON-serializable dict for a single model's results."""
    model_data: dict[str, Any] = {
        "model": model_result.model,
        "profile": model_result.profile,
        "display_name": model_result.display_name,
        "overall_accuracy": model_result.overall_accuracy,
        "total_correct": model_result.total_correct,
        "total_scoreable": model_result.total_scoreable,
        "avg_latency_ms": model_result.avg_latency_ms,
        "categories": {},
    }

    for cat_name, cr in model_result.categories.items():
        # Build the questions list with a comprehension instead of repeated
        # append calls (avoids O(n) intermediate growth overhead).
        cat_data: dict[str, Any] = {
            "accuracy": cr.accuracy,
            "correct_count": cr.correct_count,
            "scoreable_count": cr.scoreable_count,
            "avg_latency_ms": cr.avg_latency_ms,
            "p95_latency_ms": cr.p95_latency_ms,
            "questions": [
                {
                    "index": qr.question_idx,
                    "prompt": qr.prompt,
                    "response": qr.response,
                    "correct": qr.correct,
                    "score": qr.score,
                    "latency_ms": qr.latency_ms,
                    "tokens_used": qr.tokens_used,
                }
                for qr in cr.results
            ],
        }

        model_data["categories"][cat_name] = cat_data

    return model_data


def save_models_json(results: list[ModelResult], path: str) -> None:
    """Save all model results to a single models.json file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, Any] = {}
    if out_path.exists():
        # Read the cached content once via an explicit context manager so the
        # handle is never leaked; reuse it across all model writes below.
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                file_content = f.read()
            existing = json.loads(file_content)
        except (json.JSONDecodeError, OSError):
            logger.debug(
                "Could not parse existing models.json at %s; starting fresh.", out_path
            )
            existing = {}

    for r in results:
        existing[r.display_name] = build_model_json(r)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"  models.json saved to: {out_path} ({len(existing)} model(s) total)")


def load_models_json(path: str = "reports/models.json") -> dict[str, Any]:
    """Load the aggregate models.json (model name -> latest result entry).

    Returns an empty dict when the file is missing or unreadable.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def load_report_snapshots(
    reports_dir: str = "reports",
) -> list[tuple[str, dict[str, Any]]]:
    """Load every ``benchmark_*.json`` snapshot as ``(snapshot_id, model)``.

    Snapshot files are produced by ``save_json_report`` (list of model dicts
    under ``models``); the snapshot id is the file stem (timestamp), sorted
    chronologically so trend helpers see the oldest run first.
    """
    snapshots: list[tuple[str, dict[str, Any]]] = []
    d = Path(reports_dir)
    if not d.is_dir():
        return snapshots
    for fp in sorted(d.glob("benchmark_*.json")):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        models = data.get("models") if isinstance(data, dict) else None
        if isinstance(models, list):
            for m in models:
                if isinstance(m, dict) and m.get("model"):
                    snapshots.append((fp.stem, m))
    return snapshots


def model_accuracy_history(
    model: str, snapshots: list[tuple[str, dict[str, Any]]]
) -> list[tuple[str, float | None]]:
    """Per-model accuracy history: ``[(snapshot_id, overall_accuracy)]``.

    Missing / unscored runs appear with ``None`` accuracy so callers can see
    gaps instead of misreading a skip as zero.
    """
    return [
        (sid, m.get("overall_accuracy"))
        for sid, m in snapshots
        if m.get("model") == model
    ]


def accuracy_delta(
    model: str, snapshots: list[tuple[str, dict[str, Any]]]
) -> float | None:
    """Latest vs previous accuracy delta in percentage points, or None when
    fewer than two scored runs exist."""
    scored = [
        acc for _, acc in model_accuracy_history(model, snapshots) if acc is not None
    ]
    if len(scored) < 2:
        return None
    return round(scored[-1] - scored[-2], 1)


def trend_summary(
    model: str, snapshots: list[tuple[str, dict[str, Any]]]
) -> dict[str, Any]:
    """Compact per-model trend summary: history, delta, direction, and a
    confidence flag (True only when >= 3 scored runs agree on the direction)."""
    history = model_accuracy_history(model, snapshots)
    delta = accuracy_delta(model, snapshots)
    scored = [acc for _, acc in history if acc is not None]

    direction: str = "flat"
    if delta is not None:
        direction = "improved" if delta > 0 else ("regressed" if delta < 0 else "flat")

    confidence = False
    if len(scored) >= 3:
        diffs = [
            b - a for a, b in zip(scored, scored[1:]) if b is not None and a is not None
        ]
        if diffs:
            confidence = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)

    return {
        "model": model,
        "history": [{"snapshot": sid, "overall_accuracy": acc} for sid, acc in history],
        "delta": delta,
        "direction": direction,
        "confidence": confidence,
    }


def print_trend(model: str, snapshots: list[tuple[str, dict[str, Any]]]) -> None:
    """Print a human-readable trend report for *model*."""
    summary = trend_summary(model, snapshots)
    print(f"\n  Trend for '{model}' over {len(summary['history'])} snapshot(s):")
    for entry in summary["history"]:
        acc = entry["overall_accuracy"]
        acc_str = f"{acc:.1f}%" if acc is not None else "N/A"
        print(f"    {entry['snapshot']:<28} {acc_str}")
    delta = summary["delta"]
    print(
        f"  Delta: {delta:+.1f}pp ({summary['direction']})"
        if delta is not None
        else "  Delta: N/A (need >= 2 scored runs)"
    )
    print(
        f"  Confidence: "
        f"{'high' if summary['confidence'] else 'low (needs >= 3 consistent runs)'}"
    )


def save_json_report(results: list[ModelResult], path: str) -> None:
    """Save detailed results to a JSON file (all models combined)."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": [build_model_json(r) for r in results],
    }

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  JSON report saved to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark local LLMs via LM Studio / llama.cpp API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python benchmark.py --model qwen2.5-coder-7b\n"
            "  python benchmark.py --model m1 m2 --repetitions 3\n"
            "  python benchmark.py --provider llama --model qwen3-8b\n"
            "  python benchmark.py --deep --provider llama "
            "--model llama/deepseek-v4.1-flash\n"
            "  python benchmark.py --trend qwen2.5-coder-7b\n"
        ),
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=DEFAULT_PROVIDER,
        choices=sorted(PROVIDER_BASE_URLS),
        help=f"OpenAI-compatible server to benchmark against "
        f"(default: {DEFAULT_PROVIDER}). 'llama' targets a local "
        f"llama.cpp llama-server at AGENT_LLAMA_URL "
        f"(default {DEFAULT_LLAMA_BASE_URL}).",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        required=False,
        help="Model name(s) as registered in LM Studio / llama-server. A "
        "llama/ routing prefix is accepted and stripped for llama-server.",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Deep-analysis run: deep-analysis profile (temperature 0.7, "
        "larger token budget), a second self-review pass on every "
        "reasoning/math question, and longer answers kept in the report. "
        "Equivalent to --profile deep-analysis --verify-round.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="",
        help="Profile tag for the run (e.g. deep-analysis); results are keyed "
        "model|profile so the same model under different profiles is "
        "compared like with like (decision #055). Defaults to the profile "
        "implied by --deep (deep-analysis) or standard.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Token budget per question. Default: the profile's budget "
        "(standard 2048, deep-analysis 16384 or $BENCHMARK_DEEP_MAX_TOKENS). "
        "Reasoning models consume the budget in reasoning_content before "
        "emitting the answer; 512 starves it into empty responses.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. Default: the profile's value "
        "(standard 0.1, deep-analysis 0.7).",
    )
    parser.add_argument(
        "--verify-round",
        action="store_true",
        help="Re-ask reasoning/math questions with the model's own answer for "
        "review (implied by --deep).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(ALL_CATEGORIES.keys()),
        choices=list(ALL_CATEGORIES.keys()),
        help="Categories to run (default: all)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of times to repeat each test (default: 1)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to save the JSON snapshot (default: "
        "reports/benchmark_<timestamp>.json)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help=f"Override the provider base URL (default per provider: "
        f"lmstudio={PROVIDER_BASE_URLS['lmstudio']}, "
        f"llama={PROVIDER_BASE_URLS['llama']})",
    )
    parser.add_argument(
        "--trend",
        type=str,
        default=None,
        metavar="MODEL",
        help="Print the accuracy trend for MODEL over "
        "reports/benchmark_*.json snapshots and exit",
    )
    return parser.parse_args(argv)


async def main() -> None:
    args = parse_args()

    if args.trend:
        snapshots = load_report_snapshots()
        print_trend(args.trend, snapshots)
        return

    if not args.model:
        print("Error: --model is required unless --trend is used.")
        return

    global BASE_URL  # noqa: PLW0603
    BASE_URL = resolve_base_url(args.provider, args.url)

    # Fail fast: probe the endpoint with a short timeout before fanning out
    # across ~125 questions.  Without this, an unreachable LM Studio server
    # (or one with no model loaded) makes every query_model() call block on
    # urlopen(timeout=120) and retry, so the whole benchmark (and the
    # autonomous driver's gate) stalls for the full gate timeout instead of
    # reporting the problem immediately.
    try:
        _probe_endpoint(BASE_URL)
    except BenchmarkError as exc:
        print(f"Error: {exc}")
        print("Benchmark aborted: no usable LM Studio endpoint/model.")
        return

    categories = {k: v for k, v in ALL_CATEGORIES.items() if k in args.categories}
    total_q = (
        sum(len(qs) for qs in categories.values()) * len(args.model) * args.repetitions
    )

    # ---- profile resolution: --deep is sugar for the deep-analysis profile ----
    profile = args.profile or ("deep-analysis" if args.deep else "")
    settings = resolve_bench_profile(profile or DEFAULT_BENCH_PROFILE)
    max_tokens = (
        args.max_tokens
        if args.max_tokens is not None
        else deep_max_tokens(settings["max_tokens"])
    )
    temperature = (
        args.temperature if args.temperature is not None else settings["temperature"]
    )
    verify_round = bool(args.verify_round or args.deep or settings["verify_round"])
    deep_run = verify_round or profile == "deep-analysis"
    # A deep report is read, not skimmed: keep enough of each answer to see the
    # reasoning that produced the verdict.
    response_chars = REPORT_ANSWER_CHARS if deep_run else 500

    # llama-server serves exactly one model under its own id; translate the
    # agent's routing label (llama/deepseek-v4.1-flash) to that id.
    models = [
        resolve_server_model_id(args.provider, BASE_URL, m) for m in args.model
    ]

    print("\n  Benchmark Configuration")
    print(f"  {'-' * 40}")
    print(f"  Models:        {', '.join(models)}")
    print(f"  Profile:       {profile or DEFAULT_BENCH_PROFILE}")
    print(f"  Categories:    {', '.join(categories.keys())}")
    print(
        "  Questions:     "
        f"{total_q // (len(models) * args.repetitions)} per model "
        f"({total_q} total with repetitions)"
    )
    print(f"  Repetitions:   {args.repetitions}")
    print(f"  Max tokens:    {max_tokens} per question")
    print(f"  Temperature:   {temperature}")
    if verify_round:
        print(
            "  Verify round:  on (second self-review pass for "
            f"{', '.join(sorted(VERIFY_CATEGORIES))})"
        )
    print(f"  Provider:      {args.provider}")
    print(f"  API URL:       {BASE_URL}")
    print()

    results = await run_benchmark(
        models,
        categories,
        args.repetitions,
        profile=profile,
        max_tokens=max_tokens,
        temperature=temperature,
        verify_round=verify_round,
        response_chars=response_chars,
    )

    print_report(results)
    if deep_run:
        print_deep_summary(results, profile)

    if args.output:
        snapshot_path = args.output
    else:
        # Every run leaves a snapshot so --trend has history to read; the
        # timestamp keeps them distinct and sortable.
        snapshot_path = str(
            Path("reports") / f"benchmark_{time.strftime('%Y%m%d-%H%M%S')}.json"
        )
    save_json_report(results, snapshot_path)

    models_path = Path("reports") / "models.json"
    save_models_json(results, str(models_path))

    print("\n  Benchmark complete.")


if __name__ == "__main__":
    asyncio.run(main())
