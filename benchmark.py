import argparse
import asyncio
import json
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Question Bank  (~25 per category × 5 categories = ~125 total)
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
    ("If you have a 3×3×3 cube painted red on all outside faces and cut into 1×1×1 cubes, how many small cubes have exactly two red faces?", "12"),
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
    ("What is 17 × 23?", "391"),
    ("Simplify: (12 + 8) ÷ 5", "4"),
    ("What is the square root of 2025?", "45"),
    ("Calculate: 15% of 240", "36"),
    ("If x + 7 = 15, what is x?", "8"),
    ("What is 2^10?", "1024"),
    ("A rectangle has length 12 and width 5. What is its diagonal? Give the number.", "13"),
    ("Convert 72 km/h to m/s", "20"),
    ("What is the sum of interior angles of a pentagon in degrees?", "540"),
    ("Solve: 3x - 9 = 12. What is x?", "7"),
    ("What is the greatest common divisor of 48 and 36?", "12"),
    ("A circle has radius 7. What is its area? Use π ≈ 3.14, round to nearest whole number.", "154"),
    ("What is 0.375 as a fraction in simplest form?", "3/8"),
    ("If a triangle has sides 5, 12, and 13, what type of triangle is it? Answer with one word.", "right"),
    ("Calculate the factorial of 6 (6!)", "720"),
    ("What is the median of: 3, 7, 8, 5, 12, 15, 9?", "8"),
    ("A store offers a 20% discount on an item priced at $85. What is the sale price? Give just the number.", "68"),
    ("What is log base 10 of 1000?", "3"),
    ("How many degrees are in a full circle?", "360"),
    ("If f(x) = 2x² + 3, what is f(4)?", "35"),
    ("What is the least common multiple of 8 and 12?", "24"),
    ("A right triangle has legs 9 and 12. What is its area?", "54"),
    ("Convert 3/7 to a decimal, rounded to two decimal places.", "0.43"),
    ("What is the derivative of x³ at x = 2? Give just the number.", "12"),
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
    ("In Python, what keyword is used to define a constant? (Hint: there isn't one — explain briefly.)", None),
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
    categories: dict[str, CategoryResult] = field(default_factory=dict)

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
    text = re.sub(r'[^\w\s./]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def answers_match(response: str, expected: str) -> bool:
    norm_resp = normalize_answer(response)
    norm_exp = normalize_answer(expected)

    if not norm_exp or not norm_resp:
        return False

    if norm_resp == norm_exp:
        return True

    if norm_exp in norm_resp or norm_resp in norm_exp:
        return True

    # Numeric tolerance for math answers
    try:
        r_val = float(norm_resp.replace(',', ''))
        e_val = float(norm_exp.replace(',', ''))
        if abs(r_val - e_val) < 0.5:
            return True
    except (ValueError, TypeError):
        pass

    # Fraction equivalence
    for frac in [norm_resp, norm_exp]:
        if '/' in frac:
            try:
                num, den = frac.split('/')
                fval = float(num) / float(den)
                other = normalize_answer(norm_exp if frac == norm_resp else norm_resp)
                oval = float(other.replace(',', ''))
                if abs(fval - oval) < 0.01:
                    return True
            except (ValueError, ZeroDivisionError):
                pass

    # Keyword overlap for knowledge questions
    resp_words = set(norm_resp.split())
    exp_words = set(norm_exp.split())
    if len(exp_words) > 0 and len(resp_words & exp_words) / len(exp_words) >= 0.7:
        return True

    return False


def score_instruction_following(prompt: str, response: str) -> float:
    """Score instruction-following on a 0-1 scale based on constraint checks."""
    score = 1.0
    pl = prompt.lower()
    rl = response.lower().strip()

    # "exactly three words" / "exactly N words"
    m = re.search(r'exactly\s+(\d+)\s+word', pl)
    if m:
        n = int(m.group(1))
        actual = len(rl.split())
        if actual != n:
            score -= 0.5

    # "nothing else" / "only" constraints — penalize extra text
    if 'nothing else' in pl or 'and nothing else' in pl:
        expected_match = re.search(r'(?:output|respond|repeat)\s+(?:with\s+)?(?:exactly\s+)?["\']?([^"\']+?)["\']?\s*(?:\.|$)', pl, re.IGNORECASE)
        if expected_match:
            target = expected_match.group(1).strip().lower()
            if rl != target and not (target in rl and len(rl.split()) <= 3):
                score -= 0.5

    # "no punctuation" constraint
    if 'do not use any punctuation' in pl:
        punct_chars = set('.,!?;:"\'()-')
        if any(c in punct_chars for c in response):
            score -= 0.3

    # "exactly one word"
    if 'exactly one word' in pl or 'one word that means' in pl:
        if len(rl.split()) != 1:
            score -= 0.5

    # Syllable count for haiku (rough check)
    if 'haiku' in pl and '5-7-5' in pl:
        lines = [l.strip() for l in rl.split('\n') if l.strip()]
        if len(lines) == 3:
            syllables = []
            vowels = 'aeiouy'
            for line in lines:
                count = sum(1 for c in line.lower() if c in vowels)
                syllables.append(count)
            # Rough check: each line should have ~5,7,5 vowel counts (imperfect but indicative)
            if not (4 <= syllables[0] <= 6 and 6 <= syllables[1] <= 8 and 4 <= syllables[2] <= 6):
                score -= 0.3

    # "no numbering or bullets"
    if 'no numbering' in pl or 'no bullets' in pl:
        if re.search(r'^\d+[\.\)]', rl, re.MULTILINE) or re.search(r'^[-*•]', rl, re.MULTILINE):
            score -= 0.3

    # "lowercase only" / "all lowercase"
    if 'only lowercase' in pl:
        if rl != rl.lower():
            score -= 0.3

    return max(0.0, min(1.0, round(score, 2)))


def score_question(prompt: str, response: str, expected: str | None, category: str) -> tuple[bool | None, float]:
    """Returns (correct_bool_or_None, score_0_to_1)."""

    if category == "instruction_following":
        inst_score = score_instruction_following(prompt, response)
        if expected:
            match = answers_match(response, expected)
            return match, max(inst_score, 1.0 if match else 0.0)
        return None, inst_score

    if category == "coding":
        # For code-generation questions (expected is None), check syntax
        if expected is None:
            code_match = re.search(r'```(?:python)?\s*(.*?)```', response, re.DOTALL)
            code = code_match.group(1).strip() if code_match else response.strip()
            try:
                compile(code, '<benchmark>', 'exec')
                return None, 0.7  # Valid syntax — partial credit
            except SyntaxError:
                return False, 0.0

        # For output-prediction questions
        return answers_match(response, expected), 1.0 if answers_match(response, expected) else 0.0

    # Reasoning / Math / Knowledge
    if expected is None:
        return None, 0.5  # Unscoreable — give neutral score

    correct = answers_match(response, expected)
    return correct, 1.0 if correct else 0.0


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:1234/v1/chat/completions"


async def query_model(
    model: str,
    prompt: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> tuple[str, float, int | None]:
    """Returns (response_text, latency_ms, tokens_used)."""
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 512,
    }).encode("utf-8")

    req = urllib.request.Request(
        BASE_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_error = None
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
                delay = base_delay * (2 ** attempt)
                print(f"    [retry {attempt+1}/{max_retries}] Server error, waiting {delay:.0f}s...")
                await asyncio.sleep(delay)
                continue
            return f"[Error: HTTP {e.code}]", latency_ms, None

        except urllib.error.URLError as e:
            last_error = str(e.reason)
            delay = base_delay * (2 ** attempt)
            print(f"    [retry {attempt+1}/{max_retries}] Connection error, waiting {delay:.0f}s...")
            await asyncio.sleep(delay)

        except Exception as e:
            return f"[Error: {e}]", latency_ms, None

    return f"[Error after {max_retries} retries: {last_error}]", 0.0, None


async def run_category(
    model: str,
    category: str,
    questions: list[tuple[str, str | None]],
    repetition: int = 0,
) -> CategoryResult:
    cat_result = CategoryResult(category=category)
    total = len(questions)

    print(f"  [{model}] {category}: running {total} questions...")
    for idx, (prompt, expected) in enumerate(questions, 1):
        sys.stdout.write(f"\r    Question {idx}/{total}")
        sys.stdout.flush()

        response, latency_ms, tokens = await query_model(model, prompt)
        correct, score = score_question(prompt, response, expected, category)

        cat_result.results.append(QuestionResult(
            question_idx=idx - 1,
            prompt=prompt,
            response=response[:500],
            correct=correct,
            score=score,
            latency_ms=latency_ms,
            tokens_used=tokens,
        ))

    print(f"\r    Question {total}/{total} — done              ")
    return cat_result


async def run_benchmark(
    models: list[str],
    categories: dict[str, list[tuple[str, str | None]]],
    repetitions: int = 1,
) -> list[ModelResult]:
    results = []

    for model in models:
        print(f"\n{'='*60}")
        print(f"  Benchmarking: {model}")
        print(f"{'='*60}")

        model_result = ModelResult(model=model)
        for cat_name, questions in categories.items():
            if repetitions == 1:
                cat_res = await run_category(model, cat_name, questions)
                model_result.categories[cat_name] = cat_res
            else:
                # Run multiple times and average scores
                all_cat_results = []
                for rep in range(repetitions):
                    print(f"\n  --- Repetition {rep+1}/{repetitions} ---")
                    cr = await run_category(model, cat_name, questions, repetition=rep)
                    all_cat_results.append(cr)

                # Merge: average scores per question
                merged = CategoryResult(category=cat_name)
                n_questions = len(questions)
                for q_idx in range(n_questions):
                    reps = [cr.results[q_idx] for cr in all_cat_results if q_idx < len(cr.results)]
                    avg_score = statistics.mean(r.score for r in reps)
                    avg_latency = statistics.mean(r.latency_ms for r in reps)
                    any_correct = any(r.correct is True for r in reps)
                    all_correct = all(r.correct is True for r in reps)
                    merged.results.append(QuestionResult(
                        question_idx=q_idx,
                        prompt=reps[0].prompt,
                        response=reps[0].response[:500],
                        correct=all_correct if any(r.correct is not None for r in reps) else None,
                        score=round(avg_score, 2),
                        latency_ms=avg_latency,
                    ))
                model_result.categories[cat_name] = merged

        results.append(model_result)
        print(f"\n  {model} summary: {model_result.total_correct}/{model_result.total_scoreable} correct "
              f"({model_result.overall_accuracy}%), avg latency {model_result.avg_latency_ms}ms\n")

    return results


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

def print_report(results: list[ModelResult]):
    """Print a formatted comparison table to the console."""
    cat_names = []
    for r in results:
        for c in r.categories:
            if c not in cat_names:
                cat_names.append(c)

    header = f"{'Model':<30}"
    for cat in cat_names:
        short = cat.replace('_', ' ').title()[:12]
        header += f"  {short:<14}"
    header += f"  {'Overall':<10}  {'Avg Latency':>12}"
    print(f"\n{'='*len(header)}")
    print(header)
    print('-' * len(header))

    for r in results:
        line = f"{r.model:<30}"
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


def build_model_json(model_result: ModelResult) -> dict:
    """Build a JSON-serializable dict for a single model's results."""
    model_data = {
        "model": model_result.model,
        "overall_accuracy": model_result.overall_accuracy,
        "total_correct": model_result.total_correct,
        "total_scoreable": model_result.total_scoreable,
        "avg_latency_ms": model_result.avg_latency_ms,
        "categories": {},
    }

    for cat_name, cr in model_result.categories.items():
        cat_data = {
            "accuracy": cr.accuracy,
            "correct_count": cr.correct_count,
            "scoreable_count": cr.scoreable_count,
            "avg_latency_ms": cr.avg_latency_ms,
            "p95_latency_ms": cr.p95_latency_ms,
            "questions": [],
        }

        for qr in cr.results:
            cat_data["questions"].append({
                "index": qr.question_idx,
                "prompt": qr.prompt,
                "response": qr.response,
                "correct": qr.correct,
                "score": qr.score,
                "latency_ms": qr.latency_ms,
                "tokens_used": qr.tokens_used,
            })

        model_data["categories"][cat_name] = cat_data

    return model_data


def save_models_json(results: list[ModelResult], path: str):
    """Save all model results to a single models.json file.
    Each model is stored under its own top-level key for easy lookup."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing file to append/merge new results
    existing: dict[str, Any] = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}

    for r in results:
        existing[r.model] = build_model_json(r)

    out_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  models.json saved to: {out_path} ({len(existing)} model(s) total)")


def save_json_report(results: list[ModelResult], path: str):
    """Save detailed results to a JSON file (all models combined)."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": [build_model_json(r) for r in results],
    }

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  JSON report saved to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark local LLMs via LM Studio API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python benchmark.py --model qwen2.5-coder-7b\n'
            '  python benchmark.py --model qwen2.5-coder-7b llama3.1-8b --categories reasoning math\n'
            '  python benchmark.py --model m1 m2 --repetitions 3 --output report.json\n'
        ),
    )
    parser.add_argument(
        "--model", nargs="+", required=True,
        help="Model name(s) as registered in LM Studio (e.g., qwen2.5-coder-7b-instruct)",
    )
    parser.add_argument(
        "--categories", nargs="+", default=list(ALL_CATEGORIES.keys()),
        choices=list(ALL_CATEGORIES.keys()),
        help="Categories to run (default: all). Options: reasoning math coding knowledge instruction_following",
    )
    parser.add_argument(
        "--repetitions", type=int, default=1,
        help="Number of times to repeat each test for consistency measurement (default: 1)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to save JSON report (e.g., reports/benchmark.json)",
    )
    parser.add_argument(
        "--url", type=str, default=BASE_URL,
        help=f"LM Studio API URL (default: {BASE_URL})",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    global BASE_URL
    BASE_URL = args.url.rstrip("/")

    categories = {k: v for k, v in ALL_CATEGORIES.items() if k in args.categories}
    total_q = sum(len(qs) for qs in categories.values()) * len(args.model) * args.repetitions

    print(f"\n  Benchmark Configuration")
    print(f"  {'─'*40}")
    print(f"  Models:        {', '.join(args.model)}")
    print(f"  Categories:    {', '.join(categories.keys())}")
    print(f"  Questions:     {total_q // (len(args.model) * args.repetitions)} per model ({total_q} total with repetitions)")
    print(f"  Repetitions:   {args.repetitions}")
    print(f"  API URL:       {BASE_URL}")
    print()

    results = await run_benchmark(args.model, categories, args.repetitions)

    print_report(results)

    if args.output:
        save_json_report(results, args.output)

    # Always save to models.json (per-model keys, accumulates over runs)
    models_path = Path("reports") / "models.json"
    save_models_json(results, str(models_path))

    print("\n  Benchmark complete.")


if __name__ == "__main__":
    asyncio.run(main())
