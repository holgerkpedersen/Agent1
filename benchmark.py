import argparse
import asyncio
import json
import logging
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_core.config import lmstudio_base_url

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
    ("If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies? Answer yes or no.", "yes"),
    ("A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?", "9"),
    ("If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?", "5 minutes"),
    ("What comes next in the sequence: 2, 6, 18, 54, ?", "162"),
    ("If A is taller than B, and B is taller than C, who is the shortest? Answer with one letter.", "C"),
    ("A clock shows 3:15. What is the angle between the hour hand and the minute hand in degrees?", "7.5"),
    ("Mary's father has five daughters: Nana, Nene, Nini, Nono. What is the fifth daughter's name?", "Mary"),
    ("If you rearrange the letters 'CIFAIPC', you get the name of a(n): A) City B) Animal C) Ocean D) Country", "ocean"),
    ("What number should replace the question mark: 1, 1, 2, 3, 5, 8, ?", "13"),
    ("A bat and a ball cost $1.10 in total. The bat costs $1 more than the ball. How much does the ball cost? Give your answer in dollars.", "0.10"),
    ("If yesterday was two days after Monday, what day is tomorrow?", "Thursday"),
    ("In a race, you overtake the person in second place. What place are you now in?", "second"),
    ("How many sides does a hexagon have?", "6"),
    ("If 3 cats catch 3 mice in 3 minutes, how many cats are needed to catch 100 mice in 100 minutes?", "3"),
    ("What is the next prime number after 29?", "31"),
    ("A snail is at the bottom of a 10-meter well. Each day it climbs 3 meters, but each night it slips back 2 meters. How many days to get out?", "8"),
    ("If you have a 3x3x3 cube painted red on all outside faces and cut into 1x1x1 cubes, how many small cubes have exactly two red faces?", "12"),
    ("Which does not belong: Apple, Banana, Carrot, Date, Elderberry? Answer with one word.", "Carrot"),
    ("If you flip a fair coin twice, what is the probability of getting at least one head? Give answer as a fraction.", "3/4"),
    ("What comes next: J, F, M, A, M, J, ?", "J"),
    ("A train leaves station A at 60 mph. Another leaves station B (120 miles away) toward A at 40 mph. How many hours until they meet?", "1.2"),
    ("If you write out the numbers from 1 to 100, how many times does the digit '7' appear?", "20"),
    ("What is the only number that has the same number of letters as its value in English?", "four"),
    ("In a group of 30 people, everyone shakes hands with everyone else once. How many handshakes occur?", "435"),
    ("If A=1, B=2, ..., Z=26, what is the sum of the letters in 'CAT'?", "24"),
]

MATH_QUESTIONS = [
    ("What is 17 x 23?", "391"),
    ("Simplify: (12 + 8) / 5", "4"),
    ("What is the square root of 2025?", "45"),
    ("Calculate: 15% of 240", "36"),
    ("If x + 7 = 15, what is x?", "8"),
    ("What is 2^10?", "1024"),
    ("A rectangle has length 12 and width 5. What is its diagonal? Give the number.", "13"),
    ("Convert 72 km/h to m/s", "20"),
    ("What is the sum of interior angles of a pentagon in degrees?", "540"),
    ("Solve: 3x - 9 = 12. What is x?", "7"),
    ("What is the greatest common divisor of 48 and 36?", "12"),
    ("A circle has radius 7. What is its area? Use pi approx 3.14, round to nearest whole number.", "154"),
    ("What is 0.375 as a fraction in simplest form?", "3/8"),
    ("If a triangle has sides 5, 12, and 13, what type of triangle is it? Answer with one word.", "right"),
    ("Calculate the factorial of 6 (6!)", "720"),
    ("What is the median of: 3, 7, 8, 5, 12, 15, 9?", "8"),
    ("A store offers a 20% discount on an item priced at $85. What is the sale price? Give just the number.", "68"),
    ("What is log base 10 of 1000?", "3"),
    ("How many degrees are in a full circle?", "360"),
    ("If f(x) = 2x^2 + 3, what is f(4)?", "35"),
    ("What is the least common multiple of 8 and 12?", "24"),
    ("A right triangle has legs 9 and 12. What is its area?", "54"),
    ("Convert 3/7 to a decimal, rounded to two decimal places.", "0.43"),
    ("What is the derivative of x^3 at x = 2? Give just the number.", "12"),
    ("If you invest $1000 at 5% annual interest compounded yearly, how much after 2 years? Round to nearest dollar.", "1103"),
]

CODING_QUESTIONS = [
    ("Write a Python function called 'is_palindrome' that takes a string and returns True if it reads the same forwards and backwards. Return only the function code.", None),
    ("Write a one-line Python expression to reverse a list called 'lst'.", "lst[::-1]"),
    ("What does this Python code output? print(len('hello world'))", "11"),
    ("Write a Python function 'fibonacci(n)' that returns the nth Fibonacci number (0-indexed). Return only the function.", None),
    ("In Python, what is the difference between a list and a tuple? Answer in one sentence.", None),
    ("What does this code output?\nx = [1, 2, 3]\ny = x\ny.append(4)\nprint(x)", "[1, 2, 3, 4]"),
    ("Write a Python function 'flatten(lst)' that takes a nested list and returns a flat list. Return only the function.", None),
    ("What is the time complexity of binary search on a sorted array of n elements? Use Big-O notation.", "O(log n)"),
    ("Write a Python one-liner to find all even numbers from 1 to 20.", "[x for x in range(1, 21) if x % 2 == 0]"),
    ("What does this output?\nprint(bool(''))", "False"),
    ("Write a Python function 'count_vowels(s)' that returns the number of vowels in string s. Return only the function.", None),
    ("In Python, what keyword is used to define a constant? (Hint: there isn't one - explain briefly.)", None),
    ("What does this code output?\na = {1: 'one', 2: 'two'}\nprint(a.get(3, 'missing'))", "missing"),
    ("Write a Python function 'binary_search(arr, target)' on a sorted list. Return only the function.", None),
    ("What is the output of: print(0.1 + 0.2 == 0.3)", "False"),
    ("Write a Python function 'merge_sort(lst)' that returns a new sorted list. Return only the function.", None),
    ("In SQL, what clause filters groups created by GROUP BY?", "HAVING"),
    ("What does this output?\nprint(type([]) is type(list()))", "True"),
    ("Write a Python function 'anagram(s1, s2)' that returns True if s1 and s2 are anagrams. Return only the function.", None),
    ("What HTTP status code means 'Not Found'?", "404"),
    ("Write a Python function 'factorial(n)' using recursion. Return only the function.", None),
    ("In Git, what command creates a new branch and switches to it in one step?", "git checkout -b"),
    ("What does this output?\nprint([x**2 for x in range(5)])", "[0, 1, 4, 9, 16]"),
    ("Write a Python function 'remove_duplicates(lst)' that preserves order. Return only the function.", None),
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
    ("What gas do plants absorb from the atmosphere during photosynthesis?", "Carbon dioxide"),
    ("Who developed the theory of relativity? Give just the last name.", "Einstein"),
    ("What is the hardest natural substance on Earth?", "Diamond"),
    ("Which country has the most population in the world?", "India"),
    ("What is the speed of light approximately in km/s? Give a round number.", "300000"),
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
    ("Write the alphabet backwards from Z to A as one continuous string with no spaces.", "zyxwvutsrqponmlkjihgfedcba"),
    ("List three fruits, each starting with a different letter of the alphabet, comma-separated only.", None),
    ("Do not use the letter 'e' in your response. Write a short sentence about cats.", None),
    ("Output exactly this text and nothing else: HELLO WORLD", "HELLO WORLD"),
    ("Write a haiku (5-7-5 syllables) about programming. Output only the poem, no explanation.", None),
    ("Give me 4 colors in alphabetical order, one per line, with no numbering or bullets.", None),
    ("Respond with exactly one word that means 'happy'.", None),
    ("Write a sentence where every word starts with the letter 's'.", None),
    ("List these numbers from smallest to largest: 42, 7, 103, 28. Output only the numbers separated by commas.", "7, 28, 42, 103"),
    ("Write a title for an essay about climate change. The title must be exactly 5 words long.", None),
    ("Output the word 'success' in all capital letters and nothing else.", "SUCCESS"),
    ("Give me two synonyms for 'big'. Output only the two words separated by a comma.", None),
    ("Write a question that ends with the word 'why?'.", None),
    ("List 5 countries. Each must have fewer than 7 letters in its name. One per line, no numbering.", None),
    ("Repeat this exact phrase: The quick brown fox jumps over the lazy dog", "The quick brown fox jumps over the lazy dog"),
    ("Write a sentence that has exactly 10 words. Count carefully and output only the sentence.", None),
    ("Name three planets in our solar system, ordered by distance from the Sun (closest first). Comma-separated only.", None),
    ("Output the number 42 in Roman numerals and nothing else.", "XLII"),
    ("Write a short greeting. Do not use any punctuation marks at all.", None),
    ("Give me exactly two sentences about water. No more, no less.", None),
    ("List the four seasons in order starting from spring. Use only lowercase letters, comma-separated.", "spring, summer, fall, winter"),
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
    score: float           # 0.0-1.0
    latency_ms: float
    tokens_used: int | None = None


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


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s./]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def answers_match(response: str, expected: str) -> bool:
    norm_resp = normalize_answer(response)
    norm_exp = normalize_answer(expected)

    if not norm_exp or not norm_resp:
        return False

    if norm_resp == norm_exp:
        return True

    # Word-bounded containment ONLY: raw substring matching produced false
    # positives ("42" inside "142", "true" inside "retry").  The WHOLE
    # expected answer must appear as a word-bounded phrase in the response.
    if (
        len(norm_exp) >= 2
        and re.search(rf"(?<!\w){re.escape(norm_exp)}(?!\w)", norm_resp)
    ):
        return True

    # Numeric tolerance for math answers (also accept "1 000" for "1,000")
    try:
        r_val = float(norm_resp.replace(",", "").replace(" ", ""))
        e_val = float(norm_exp.replace(",", "").replace(" ", ""))
        if abs(r_val - e_val) < 0.5:
            return True
    except (ValueError, TypeError):
        logger.debug("Numeric comparison failed for response=%r expected=%r", norm_resp, norm_exp)

    # Fraction equivalence
    for frac in [norm_resp, norm_exp]:
        if "/" in frac:
            try:
                num, den = frac.split("/")
                fval = float(num) / float(den)
                other = normalize_answer(
                    norm_exp if frac == norm_resp else norm_resp
                )
                oval = float(other.replace(",", "").replace(" ", ""))
                if abs(fval - oval) < 0.01:
                    return True
            except (ValueError, ZeroDivisionError):
                logger.debug("Fraction comparison failed for %r", frac)

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
        "the": 1, "a": 1, "i": 1, "is": 1, "it": 1, "to": 1,
        "of": 1, "and": 1, "in": 1, "for": 1, "on": 1, "be": 1,
        "as": 1, "with": 1, "that": 1, "this": 1, "are": 1,
        "was": 1, "has": 1, "had": 1, "but": 1, "not": 1,
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
        and not word[-3:] in ("le", "ve", "ne")
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
        lines = [l.strip() for l in rl.split("\n") if l.strip()]
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


async def judge_answer(
    judge_fn: Any, response: str, expected: str
) -> bool | None:
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


async def _correct_with_judge(
    response: str, expected: str, judge_fn: Any
) -> bool:
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
            code_match = re.search(
                r"```(?:python)?\s*(.*?)```", response, re.DOTALL
            )
            code = (
                code_match.group(1).strip() if code_match else response.strip()
            )
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

BASE_URL = f"{lmstudio_base_url()}/chat/completions"


async def query_model(
    model: str,
    prompt: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_tokens: int = 2048,
) -> tuple[str, float, int | None]:
    """Returns (response_text, latency_ms, tokens_used). Raises ModelAPIError on failure."""
    import urllib.request
    import urllib.error

    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            # Reasoning models (e.g. qwen3.8-27b) consume the budget in
            # reasoning_content BEFORE emitting content: with 512 tokens the
            # answer came back empty (finish_reason "length"), scoring a
            # False the model did not deserve.  2048 gives the reasoning
            # room to finish AND the answer room to exist.
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
            text = body["choices"][0]["message"]["content"]
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
    """Verify the LM Studio endpoint is usable before running the benchmark.

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
        if ("no models loaded" in lowered or "model not found" in lowered
                or "does not exist" in lowered):
            raise BenchmarkError(
                f"LM Studio has no model available at {base_url}: {detail[:200]}"
            ) from exc
        if exc.code >= 500:
            raise BenchmarkError(
                f"LM Studio endpoint returned {exc.code} on probe: {base_url}"
            ) from exc
        return
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise BenchmarkError(
            f"LM Studio endpoint unreachable at {base_url}: {exc}"
        ) from exc
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
            f"LM Studio endpoint at {base_url} did not return a chat response: {exc}"
        ) from exc


async def run_category(
    model: str,
    category: str,
    questions: list[tuple[str, str | None]],
    repetition: int = 0,
    judge_fn: Any = None,
    max_tokens: int = 2048,
) -> CategoryResult:
    cat_result = CategoryResult(category=category)
    total = len(questions)

    print(f"  [{model}] {category}: running {total} questions...")
    for idx, (prompt, expected) in enumerate(questions, 1):
        sys.stdout.write(f"\r    Question {idx}/{total}")
        sys.stdout.flush()

        try:
            response, latency_ms, tokens = await query_model(model, prompt, max_tokens=max_tokens)
            correct, score = await score_question(
                prompt, response, expected, category, judge_fn=judge_fn
            )
        except ModelAPIError as e:
            response = str(e)
            latency_ms = 0.0
            tokens = None
            correct = None
            score = 0.0

        cat_result.results.append(
            QuestionResult(
                question_idx=idx - 1,
                prompt=prompt,
                response=response[:500],
                correct=correct,
                score=score,
                latency_ms=latency_ms,
                tokens_used=tokens,
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
) -> list[ModelResult]:
    results = []

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
                    model, cat_name, questions, judge_fn=judge_fn, max_tokens=max_tokens
                )
                model_result.categories[cat_name] = cat_res
            else:
                # Run multiple times and average scores
                all_cat_results: list[CategoryResult] = []
                for rep in range(repetitions):
                    print(f"\n  --- Repetition {rep + 1}/{repetitions} ---")
                    cr = await run_category(
                        model, cat_name, questions, repetition=rep, judge_fn=judge_fn,
                        max_tokens=max_tokens
                    )
                    all_cat_results.append(cr)

                # Merge: average scores per question. Precompute the
                # per-question repetition data once (cached outside any
                # further iteration), then build results via comprehension.
                n_questions = len(questions)
                rep_data = [
                    [cr.results[q_idx] for cr in all_cat_results if q_idx < len(cr.results)]
                    for q_idx in range(n_questions)
                ]

                merged = CategoryResult(category=cat_name)
                merged.results = [
                    QuestionResult(
                        question_idx=q_idx,
                        prompt=reps[0].prompt if reps else "",
                        response=(reps[0].response[:500] if reps else ""),
                        correct=(
                            all(r.correct is True for r in reps if r.correct is not None)
                            if any(r.correct is True for r in reps)
                            else (False if any(r.correct is not None for r in reps) else None)
                        ),
                        score=round(statistics.mean(r.score for r in reps), 2) if reps else 0.0,
                        latency_ms=statistics.mean(r.latency_ms for r in reps) if reps else 0.0,
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
    cat_names = list(dict.fromkeys(
        c for r in results for c in r.categories
    ))

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
            logger.debug("Could not parse existing models.json at %s; starting fresh.", out_path)
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
        diffs = [b - a for a, b in zip(scored, scored[1:]) if b is not None and a is not None]
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
    print(f"  Confidence: {'high' if summary['confidence'] else 'low (needs >= 3 consistent runs)'}")


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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark local LLMs via LM Studio API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python benchmark.py --model qwen2.5-coder-7b\n'
            '  python benchmark.py --model m1 m2 --repetitions 3\n'
            '  python benchmark.py --trend qwen2.5-coder-7b\n'
        ),
    )
    parser.add_argument(
        "--model", nargs="+", required=False,
        help="Model name(s) as registered in LM Studio",
    )
    parser.add_argument(
        "--profile", type=str, default="",
        help="Profile tag for the run (e.g. deep-analysis); results are keyed "
             "model|profile so the same model under different profiles is "
             "compared like with like (decision #055)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048,
        help="Token budget per question (default: 2048). Reasoning models "
             "consume the budget in reasoning_content before emitting the "
             "answer; 512 starves it into empty responses.",
    )
    parser.add_argument(
        "--categories", nargs="+", default=list(ALL_CATEGORIES.keys()),
        choices=list(ALL_CATEGORIES.keys()),
        help="Categories to run (default: all)",
    )
    parser.add_argument(
        "--repetitions", type=int, default=1,
        help="Number of times to repeat each test (default: 1)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to save JSON report",
    )
    parser.add_argument(
        "--url", type=str, default=BASE_URL,
        help=f"LM Studio API URL (default: {BASE_URL})",
    )
    parser.add_argument(
        "--trend", type=str, default=None, metavar="MODEL",
        help="Print the accuracy trend for MODEL over reports/benchmark_*.json snapshots and exit",
    )
    return parser.parse_args()


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
    BASE_URL = args.url.rstrip("/")

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
    total_q = sum(len(qs) for qs in categories.values()) * len(
        args.model
    ) * args.repetitions

    print("\n  Benchmark Configuration")
    print(f"  {'-' * 40}")
    print(f"  Models:        {', '.join(args.model)}")
    if args.profile:
        print(f"  Profile:       {args.profile}")
    print(f"  Categories:    {', '.join(categories.keys())}")
    print(
        "  Questions:     "
        f"{total_q // (len(args.model) * args.repetitions)} per model "
        f"({total_q} total with repetitions)"
    )
    print(f"  Repetitions:   {args.repetitions}")
    print(f"  Max tokens:    {args.max_tokens} per question")
    print(f"  API URL:       {BASE_URL}")
    print()

    results = await run_benchmark(
        args.model, categories, args.repetitions,
        profile=args.profile, max_tokens=args.max_tokens,
    )

    print_report(results)

    if args.output:
        save_json_report(results, args.output)

    models_path = Path("reports") / "models.json"
    save_models_json(results, str(models_path))

    print("\n  Benchmark complete.")


if __name__ == "__main__":
    asyncio.run(main())