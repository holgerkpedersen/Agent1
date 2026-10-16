"""Build an all-model benchmark comparison table from reports/models.json.

Reads the persistent per-model results database written by ``python benchmark.py``
(loaded at benchmark.py via load_models_json(); key format is "model|profile") and
prints a cross-model comparison of per-category accuracy, overall accuracy, latency
and tokens/question, sorted best-overall first. Read-only: this script writes nothing.

Usage:  python bench_compare.py
"""
from __future__ import annotations

import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "reports" / "models.json"

CATS = ["reasoning", "math", "coding", "knowledge", "instruction_following"]


def acc(entry: dict, cat: str) -> str:
    c = entry.get("categories", {}).get(cat)
    if not c:
        return "---"
    a = c.get("accuracy")
    return f"{a:>5.1f}%" if isinstance(a, (int, float)) else "  N/A"


def main() -> None:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))

    rows = sorted(
        data.items(),
        key=lambda kv: -(kv[1].get("overall_accuracy") or -1),
    )

    hdr = f"{'Model':<40}" + "".join(
        f"{c.replace('_', ' ').title()[:12]:>13}" for c in CATS
    )
    hdr += f"{'Overall':>9}{'AvgLat':>9}{'tok/q':>8}"
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for name, entry in rows:
        line = f"{name:<40}"
        for cat in CATS:
            line += f"{acc(entry, cat):>13}"
        ov = entry.get("overall_accuracy")
        lat = entry.get("avg_latency_ms")
        # avg tokens per question (sum of tokens_used / number of questions)
        tot_tok, n_q = 0, 0
        for cat in CATS:
            for q in entry.get("categories", {}).get(cat, {}).get("questions", []):
                n_q += 1
                if q.get("tokens_used"):
                    tot_tok += q["tokens_used"]
        tokq = f"{tot_tok / n_q:.0f}" if n_q and tot_tok else "---"
        ov_str = f"{ov:.1f}%" if isinstance(ov, (int, float)) else "N/A"
        lat_str = f"{lat:.0f}ms" if isinstance(lat, (int, float)) else "---"
        line += f"{ov_str:>9}{lat_str:>9}{tokq:>8}"
        print(line)
    print("=" * len(hdr))

    # per-model question counts / scoreable coverage
    print("\nCoverage:")
    for name, entry in rows:
        cats = entry.get("categories", {})
        n_q = sum(len(cats.get(c, {}).get("questions", [])) for c in CATS)
        scoreable = entry.get("total_scoreable")
        correct = entry.get("total_correct")
        print(f"  {name:<40} questions={n_q:<4} scoreable={scoreable}"
              f"  correct={correct}")


if __name__ == "__main__":
    main()
