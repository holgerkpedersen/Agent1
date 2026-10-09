"""Scratch: inspect the review ledger's suitability for precision measurement."""
import json
from collections import Counter

d = json.load(open("reports/harnessfix/review.json"))
lab = [r for r in d if r["disposition"] != "unreviewed"]
print("total records:", len(d))
print("labeled:", len(lab))
print("by disposition:", Counter(r["disposition"] for r in lab))
print("by source:", Counter(r["source"] for r in lab))
print("source x disposition:", Counter((r["source"], r["disposition"]) for r in lab))
print()
print("root_layer distribution (labeled):", Counter(r["root_layer"] for r in lab))
print()
print("mechanism distribution (labeled):")
for m, c in Counter(r["mechanism"][:70] for r in lab).most_common():
    print(f"  {c:4d}  {m}")
print()
print("note prefixes (first 45 chars):")
for n, c in Counter(r["note"][:45] for r in lab).most_common(12):
    print(f"  {c:4d}  {n}")
