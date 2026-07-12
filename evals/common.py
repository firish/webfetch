"""
Shared utilities for the eval harness.

These scripts are standalone (not part of the webfetch package) and use only
deps the library already requires. Metric primitives live here so both eval
layers grade answers identically and test_metrics.py can verify them once.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

# Date formats seen in SimpleQA-style gold answers. Parsed after
# normalize_text(), so no punctuation remains ("May 5, 2001" -> "may 5 2001").
_DATE_FORMATS: tuple[str, ...] = (
    "%B %d %Y",   # may 5 2001
    "%d %B %Y",   # 5 may 2001
    "%Y %m %d",   # 2001 05 05
    "%m %d %Y",   # 05 05 2001
    "%B %Y",      # may 2001
    "%Y",         # 2001
)

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text for substring answer matching.

    Lowercase, NFKC-fold unicode, replace punctuation with spaces, collapse
    whitespace. Applied to BOTH gold answers and page text so matching is
    symmetric.

    Args:
        text: Raw text.

    Returns:
        Normalized text.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _date_variants(normalized: str) -> list[str]:
    """Return re-orderings of a normalized date string, or [] if not a date."""
    for fmt in _DATE_FORMATS[:4]:  # only full dates get reordered
        try:
            dt = datetime.strptime(normalized, fmt)
        except ValueError:
            continue
        month = dt.strftime("%B").lower()
        # Unpadded day forms ("5 may") alongside strftime's padded "05 may" -
        # web pages overwhelmingly write dates unpadded.
        return [
            f"{month} {dt.day} {dt.year}",
            f"{dt.day} {month} {dt.year}",
            normalize_text(dt.strftime("%B %d %Y")),
            normalize_text(dt.strftime("%d %B %Y")),
            normalize_text(dt.strftime("%Y %m %d")),
        ]
    return []


def answer_variants(answer: str, answer_type: str | None = None) -> list[str]:
    """Build acceptable normalized variants of a gold answer.

    Always includes the normalized answer itself. Numbers get a
    comma-stripped variant ("30,000" -> "30000"). Full dates get common
    re-orderings. Deliberately NO bare-year variant for full dates - a lone
    year matches almost any page and would inflate recall.

    Args:
        answer: The gold answer string.
        answer_type: Optional type hint from the dataset (e.g. "Date",
            "Number"). Used to trigger extra variants; date parsing is also
            attempted regardless.

    Returns:
        Deduplicated list of normalized variants, base form first.
    """
    base = normalize_text(answer)
    variants = [base]

    no_commas = normalize_text(answer.replace(",", ""))
    if no_commas != base:
        variants.append(no_commas)

    if answer_type is None or answer_type.lower() == "date":
        variants.extend(_date_variants(base))

    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def jaccard(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Jaccard similarity of two token sets (0.0 when both empty)."""
    if not tokens_a and not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write dicts to a JSONL file (stable key order for diff-ability)."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """Render a simple GitHub-flavored markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall, F1 (0.0 where denominators are zero)."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1
