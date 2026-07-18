"""
List-query eval: which lever fixes the "top 10 X" failure mode?

Compression selects the best-scoring sentences, but list items are
parallel structure - it keeps 2-3 items and drops the rest. The ranker's
top-5 chunk cap can also cut lists that span chunks. This eval measures
ITEM RECALL (fraction of gold list items present in the final tool-result
string) under candidate configurations, on identical captured chunks:

  prod                 compress, top-5 chunks, 8k budget (today's output)
  nocompress-top5      compression off, everything else prod
  compress-top10       compression on, twice the chunks
  nocompress-top10     both levers
  nocompress-top12-16k all captured chunks, larger budget

Capture runs ONCE live (12 queries, ~5 min): production search/fetch, but
the cross-encoder keeps 12 chunks instead of 5 so the sweep can slice.
Variants are then pure offline formatting - same philosophy as
run_compression_eval.py.

Run: python evals/run_list_eval.py [--capture] [--provider multi]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.common import markdown_table, normalize_text, read_jsonl, write_jsonl

DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "list_queries.jsonl"
CHUNKS_PATH = Path(__file__).resolve().parent / "datasets" / "list_chunks.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

CAPTURE_TOP_K = 12   # bank more than production so variants can slice
BUDGET = 8000        # DEFAULT_TOOL_RESULT_BUDGET
CHARS_PER_TOKEN = 4

VARIANTS = [
    # (name, n_chunks, compress, budget)
    ("prod", 5, True, 8000),
    ("nocompress-top5", 5, False, 8000),
    ("compress-top10", 10, True, 8000),
    ("nocompress-top10", 10, False, 8000),
    ("nocompress-top12-16k", 12, False, 16000),
]


def capture(provider: str) -> list[dict]:
    """Run each list query through production search/fetch/rank once,
    keeping CAPTURE_TOP_K chunks (eval-owned cross-encoder instance)."""
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    from webfetch import Pipeline
    from webfetch.rank.crossencoder import CrossEncoderRanker
    from webfetch.rank.hybrid import HybridRanker
    from webfetch.search import get_search_adapter

    pipeline = Pipeline(
        search=get_search_adapter(provider),
        rankers=[HybridRanker(), CrossEncoderRanker(top_k=CAPTURE_TOP_K)],
        cache=None,  # capture must not touch the production cache
    )
    rows = []
    for i, q in enumerate(read_jsonl(DATASET_PATH), 1):
        try:
            result = pipeline.search_chunks(q["query"])
            rows.append({**q, "chunks": [
                {"text": c.text, "url": c.url, "title": c.title,
                 "score": c.score} for c in result.chunks], "error": None})
            print(f"  [{i}/12] {q['id']}: {len(result.chunks)} chunks "
                  f"{result.elapsed_secs:.0f}s", flush=True)
        except Exception as exc:
            rows.append({**q, "chunks": [],
                         "error": f"{type(exc).__name__}: {exc}"})
            print(f"  [{i}/12] {q['id']}: ERROR {exc}", flush=True)
        time.sleep(2)
    return rows


def item_recall(row: dict, context: str) -> float:
    """Fraction of gold items with any variant present in the context."""
    hay = normalize_text(context)
    hits = sum(1 for variants in row["items"]
               if any(normalize_text(v) in hay for v in variants))
    return hits / len(row["items"])


def render(row: dict, n_chunks: int, compress: bool, budget: int) -> str:
    """Format captured chunks exactly as the tool would."""
    from webfetch.compress import compress_chunks
    from webfetch.extract.base import build_context
    from webfetch.rank.base import Chunk
    chunks = [Chunk(**c) for c in row["chunks"][:n_chunks]]
    if compress:
        chunks = compress_chunks(row["query"], chunks)
    return build_context(chunks, budget_chars=budget, merge_sources=True,
                        header_style="domain")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--provider", default="multi")
    args = parser.parse_args()

    if args.capture or not CHUNKS_PATH.exists():
        print(f"## Capture (provider={args.provider})", flush=True)
        rows = capture(args.provider)
        write_jsonl(CHUNKS_PATH, rows)
    else:
        rows = read_jsonl(CHUNKS_PATH)
        print(f"Reusing {CHUNKS_PATH}")
    rows = [r for r in rows if not r.get("error") and r["chunks"]]
    n = len(rows)

    print(f"\n## Sweep on {n} queries")
    table, results = [], {}
    for name, n_chunks, compress, budget in VARIANTS:
        recalls, tokens, perfect = [], [], 0
        for row in rows:
            ctx = render(row, n_chunks, compress, budget)
            r = item_recall(row, ctx)
            recalls.append(r)
            tokens.append(len(ctx) // CHARS_PER_TOKEN)
            perfect += r == 1.0
        results[name] = {"item_recall": round(statistics.mean(recalls), 3),
                         "perfect_lists": perfect,
                         "tokens_mean": round(statistics.mean(tokens))}
        table.append([name, f"{statistics.mean(recalls):.1%}",
                      f"{perfect}/{n}", round(statistics.mean(tokens))])
    print(markdown_table(
        ["variant", "item recall", "perfect lists", "tok/result"], table))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"list_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n": n, "variants": results}, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
