"""Self-test for eval metric primitives. Run: python evals/test_metrics.py"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.common import (
    answer_variants,
    jaccard,
    markdown_table,
    normalize_text,
    precision_recall_f1,
)

# --- normalize_text ---
assert normalize_text("  Hello,   World!  ") == "hello world"
assert normalize_text("Café — Déjà vu") == "café déjà vu"
assert normalize_text("U.S.A.") == "u s a"
assert normalize_text("") == ""

# --- answer_variants: numbers ---
v = answer_variants("30,000", "Number")
assert "30 000" in v and "30000" in v, v

# --- answer_variants: dates reorder and match across formats ---
v = answer_variants("May 5, 2001", "Date")
assert "may 5 2001" in v, v
assert "5 may 2001" in v, v
# a page saying "5 May 2001" must match gold "May 5, 2001"
page = normalize_text("The event happened on 5 May 2001 in Paris.")
assert any(variant in page for variant in v), v

# --- no bare-year variant for full dates ---
assert "2001" not in v, v

# --- non-dates pass through untouched ---
v = answer_variants("Marie Curie", "Person")
assert v == ["marie curie"], v

# --- jaccard ---
assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
assert jaccard({"a"}, {"b"}) == 0.0
assert jaccard(set(), set()) == 0.0
assert abs(jaccard({"a", "b", "c"}, {"b", "c", "d"}) - 0.5) < 1e-9

# --- precision/recall/f1 on a hand-computed confusion matrix ---
# tp=8, fp=2, fn=4 -> p=0.8, r=2/3, f1=2*0.8*(2/3)/(0.8+2/3)
p, r, f1 = precision_recall_f1(tp=8, fp=2, fn=4)
assert abs(p - 0.8) < 1e-9
assert abs(r - 8 / 12) < 1e-9
assert abs(f1 - (2 * 0.8 * (8 / 12)) / (0.8 + 8 / 12)) < 1e-9
assert precision_recall_f1(0, 0, 0) == (0.0, 0.0, 0.0)

# --- markdown_table shape ---
t = markdown_table(["a", "b"], [[1, 2], [3, 4]])
assert t.splitlines()[0] == "| a | b |"
assert len(t.splitlines()) == 4

print("OK - all metric self-tests passed")
