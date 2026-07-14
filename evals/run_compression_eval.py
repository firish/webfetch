"""
Compression eval: tokens-vs-recall frontier for sentence-level compression.

Two phases:

A) Capture (live, run once): the 50 base queries from live_queries.jsonl go
   through Pipeline.search_chunks (production tool-mode path, multi-engine
   fusion, fresh cache) and the final ranked chunks are banked to a JSONL
   artifact. Reused on later runs - compression is a pure function of
   (query, chunks), so candidates sweep on identical captured chunks (same
   controlled-experiment design as the ranking gap-1 experiment).

B) Offline sweep: every (scorer x policy x param) config from
   webfetch/compress.py - the eval imports the SHIPPED module, so what is
   measured is exactly what ships - plus guard ablations on the winner,
   crossed with context FORMATS (source-header merging + header styles,
   from build_context). Headers are ~26% of an uncompressed result and
   sentence selection cannot touch them, so formatting is part of the
   same tokens-vs-recall frontier.
   Metrics per config:
     - tokens/result (mean, median) of the final tool-result context
     - retrieval recall (gold answer variant in the compressed context)
     - answer survival: among queries the RAW chunks answered, how many
       still answer after compression - the direct measure of what
       compression destroys
   Gate (from ROADMAP): PASS = tokens <= 50% of baseline AND recall drop
   <= 2 queries of 50 (4 pts); target <= 1 (2 pts). Winner = lexicographic
   (smallest recall drop, then fewest tokens) among passers - precision
   first, same ethos as the semantic-cache thresholds.

Run: python evals/run_compression_eval.py [--capture] [--provider multi]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.common import (
    answer_variants,
    markdown_table,
    normalize_text,
    read_jsonl,
    write_jsonl,
)

DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "live_queries.jsonl"
CHUNKS_PATH = Path(__file__).resolve().parent / "datasets" / "compression_chunks.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RECOMMENDATION_PATH = RESULTS_DIR / "compression_recommendation.json"

DEFAULT_SLEEP_SECS = 2.0
ERROR_BACKOFF_SECS = 30.0
CHARS_PER_TOKEN = 4
BUDGET_CHARS = 8000  # DEFAULT_TOOL_RESULT_BUDGET - production tool budget

# Hard gate (ROADMAP): tokens halved, recall drop under 4 pts on n=50.
GATE_MAX_TOKEN_FRACTION = 0.50
GATE_MAX_RECALL_DROP = 2   # queries lost, of 50 (= 4 pts)
TARGET_RECALL_DROP = 1     # preferred (= 2 pts)

# Sweep grid. Policies/params are compress_chunks arguments; guards default
# ON here and are ablated separately on the winner.
SWEEP_CONFIGS: list[dict] = [
    {"scorer": s, "policy": "ratio", "param": p}
    for s in ("biencoder", "crossencoder", "lexical", "lead")
    for p in (0.3, 0.4, 0.5, 0.6)
] + [
    {"scorer": s, "policy": "topk", "param": k}
    for s in ("biencoder", "crossencoder", "lexical", "lead")
    for k in (1, 2)
] + [
    # Absolute-threshold policy only makes sense for cosine scores.
    {"scorer": "biencoder", "policy": "threshold", "param": t}
    for t in (0.30, 0.40, 0.50)
]

# Context formats: (merge_sources, header_style). "plain/full" is today's
# production format - the gate baseline. "merged/domain" is the proposed
# format: one header per source URL, hostname instead of the full URL
# (titles stay - the extraction-hardening eval showed answers live in them).
FORMATS: dict[str, dict] = {
    "plain": {"merge_sources": False, "header_style": "full"},
    "merged": {"merge_sources": True, "header_style": "full"},
    "merged+domain": {"merge_sources": True, "header_style": "domain"},
}
PROD_FORMAT = "merged+domain"


def capture(provider: str, sleep_secs: float, limit: int | None) -> list[dict]:
    """Phase A: run base queries live and bank the final ranked chunks."""
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    from webfetch import Pipeline, SqliteCache
    from webfetch.search import get_search_adapter

    # Plain exact cache in a temp dir: every query is distinct, semantic
    # matching would only add model load time to the capture run.
    cache_db = str(Path(tempfile.mkdtemp(prefix="webfetch_compress_")) / "cache.db")
    pipeline = Pipeline(search=get_search_adapter(provider),
                        cache=SqliteCache(db_path=cache_db))

    base = [q for q in read_jsonl(DATASET_PATH) if q.get("paraphrase_of") is None]
    if limit:
        base = base[:limit]
    rows: list[dict] = []
    for i, q in enumerate(base, 1):
        try:
            result = pipeline.search_chunks(q["query"])
            rows.append({
                "id": q["id"], "query": q["query"], "answers": q["answers"],
                "answer_type": q.get("answer_type"),
                "chunks": [{"text": c.text, "url": c.url, "title": c.title,
                            "score": c.score} for c in result.chunks],
                "elapsed_secs": round(result.elapsed_secs, 1), "error": None,
            })
            print(f"  [{i}/{len(base)}] {q['id']}: {len(result.chunks)} chunks "
                  f"{result.elapsed_secs:.0f}s", flush=True)
            time.sleep(sleep_secs)
        except Exception as exc:
            rows.append({"id": q["id"], "query": q["query"],
                         "answers": q["answers"],
                         "answer_type": q.get("answer_type"), "chunks": [],
                         "elapsed_secs": 0.0,
                         "error": f"{type(exc).__name__}: {exc}"})
            print(f"  [{i}/{len(base)}] {q['id']}: ERROR {exc}", flush=True)
            time.sleep(ERROR_BACKOFF_SECS)
    return rows


def _context_metrics(row: dict, chunks: list, fmt: dict) -> tuple[int, bool]:
    """(est_tokens, recall_hit) of the tool-result context for these chunks."""
    from webfetch.extract.base import build_context
    if not chunks:
        return 0, False
    ctx = build_context(chunks, budget_chars=BUDGET_CHARS, **fmt)
    haystack = normalize_text(ctx)
    hit = any(v in haystack
              for a in row["answers"]
              for v in answer_variants(a, row.get("answer_type")))
    return len(ctx) // CHARS_PER_TOKEN, hit


def evaluate_config(rows: list[dict], config: dict) -> dict:
    """Run one compression config over all rows, measured under each format.

    Compression runs ONCE per row; the (cheap) formatting variants reuse the
    compressed chunks. config == {} means uncompressed (formatting-only).
    """
    from webfetch.compress import compress_chunks
    from webfetch.rank.base import Chunk

    per_format = {name: {"tokens": [], "hits": []} for name in FORMATS}
    for row in rows:
        chunks = [Chunk(**c) for c in row["chunks"]]
        if config:
            chunks = compress_chunks(row["query"], chunks, **config)
        for name, fmt in FORMATS.items():
            t, h = _context_metrics(row, chunks, fmt)
            per_format[name]["tokens"].append(t)
            per_format[name]["hits"].append(h)
    out = {**config}
    for name, m in per_format.items():
        out[name] = {"tokens": m["tokens"], "hits": m["hits"],
                     "tokens_mean": round(statistics.mean(m["tokens"]), 1),
                     "tokens_median": statistics.median(m["tokens"]),
                     "recall_hits": sum(m["hits"])}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true",
                        help="Force a fresh live capture run")
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--provider", default="multi")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    if args.capture or not args.chunks.exists():
        print(f"## Phase A: live capture (provider={args.provider})", flush=True)
        rows = capture(args.provider, args.sleep, args.limit)
        write_jsonl(args.chunks, rows)
        print(f"Wrote {args.chunks}")
    else:
        rows = read_jsonl(args.chunks)
        print(f"Reusing captured chunks: {args.chunks} ({len(rows)} rows)")

    errored = [r for r in rows if r["error"]]
    rows = [r for r in rows if not r["error"] and r["chunks"]]
    print(f"\n## Phase B: offline sweep on {len(rows)} queries "
          f"({len(errored)} capture errors excluded)")

    # Baseline: uncompressed chunks under every format. "plain" is today's
    # production output - the gate anchor for both tokens and recall.
    baseline = evaluate_config(rows, {})
    base = baseline["plain"]
    base_hits = base["hits"]
    n = len(rows)
    print("\n## Baseline (uncompressed) by format")
    print(markdown_table(
        ["format", "recall", "tok mean", "tok med", "vs plain"],
        [[name, f"{baseline[name]['recall_hits']}/{n}",
          baseline[name]["tokens_mean"], baseline[name]["tokens_median"],
          f"{baseline[name]['tokens_mean'] / base['tokens_mean'] * 100:.0f}%"]
         for name in FORMATS]))

    def survival(hits: list[bool]) -> str:
        alive = sum(1 for h, b in zip(hits, base_hits) if h and b)
        return f"{alive}/{sum(base_hits)}"

    def gate(m: dict) -> tuple[bool, bool]:
        """Gate a config's PROD_FORMAT metrics against today's production."""
        ok_tokens = m["tokens_mean"] <= GATE_MAX_TOKEN_FRACTION * base["tokens_mean"]
        drop = base["recall_hits"] - m["recall_hits"]
        return (ok_tokens and drop <= GATE_MAX_RECALL_DROP,
                ok_tokens and drop <= TARGET_RECALL_DROP)

    results = [evaluate_config(rows, cfg) for cfg in SWEEP_CONFIGS]

    print(f"\n## Frontier ({PROD_FORMAT} format, guards ON; gate vs plain "
          f"baseline: tokens <= {GATE_MAX_TOKEN_FRACTION:.0%}, "
          f"drop <= {GATE_MAX_RECALL_DROP})")
    table_rows = []
    for r in sorted(results, key=lambda r: r[PROD_FORMAT]["tokens_mean"]):
        m = r[PROD_FORMAT]
        passed, target = gate(m)
        table_rows.append([
            r["scorer"], r["policy"], r["param"],
            f"{m['recall_hits']}/{n}", survival(m["hits"]),
            m["tokens_mean"], m["tokens_median"],
            f"{m['tokens_mean'] / base['tokens_mean'] * 100:.0f}%",
            "TARGET" if target else ("PASS" if passed else "-"),
        ])
    print(markdown_table(
        ["scorer", "policy", "param", "recall", "survival", "tok mean",
         "tok med", "vs base", "gate"], table_rows))

    passers = [r for r in results if gate(r[PROD_FORMAT])[0]]
    winner = None
    if passers:
        # Precision-first: smallest recall drop wins, tokens break ties.
        winner = min(passers, key=lambda r: (
            base["recall_hits"] - r[PROD_FORMAT]["recall_hits"],
            r[PROD_FORMAT]["tokens_mean"]))
        wm = winner[PROD_FORMAT]
        print(f"\nWinner: {winner['scorer']}/{winner['policy']}/"
              f"{winner['param']} + {PROD_FORMAT} - recall "
              f"{wm['recall_hits']}/{n} (baseline {base['recall_hits']}/{n}), "
              f"tokens {wm['tokens_mean']} vs {base['tokens_mean']} "
              f"({wm['tokens_mean'] / base['tokens_mean'] * 100:.0f}%)")
    else:
        print("\nNO config passed the gate - per project practice, ship "
              "nothing (or flag-off) rather than a feature that fails its "
              "own eval.")

    # Guard ablations on the winner: each guard off individually, then all.
    ablations = []
    if winner is not None:
        base_cfg = {k: winner[k] for k in ("scorer", "policy", "param")}
        variants = [("all guards on", {}),
                    ("anaphora off", {"anaphora_guard": False}),
                    ("table-skip off", {"table_guard": False}),
                    ("dedup off", {"dedup": False}),
                    ("all guards off", {"anaphora_guard": False,
                                        "table_guard": False, "dedup": False})]
        print("\n## Guard ablations (winner config)")
        ab_rows = []
        for name, overrides in variants:
            m = evaluate_config(rows, {**base_cfg, **overrides})[PROD_FORMAT]
            ablations.append({"name": name, **{k: m[k] for k in
                              ("tokens_mean", "tokens_median", "recall_hits")}})
            ab_rows.append([name, f"{m['recall_hits']}/{n}", survival(m["hits"]),
                            m["tokens_mean"], m["tokens_median"]])
        print(markdown_table(
            ["guards", "recall", "survival", "tok mean", "tok med"], ab_rows))

    def strip(r: dict | None) -> dict | None:
        if r is None:
            return None
        return {k: ({kk: vv for kk, vv in v.items()
                     if kk not in ("tokens", "hits")}
                    if isinstance(v, dict) else v)
                for k, v in r.items()}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"compression_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n": n, "prod_format": PROD_FORMAT,
                   "baseline": strip(baseline),
                   "configs": [strip(r) for r in results],
                   "ablations": ablations,
                   "winner": strip(winner)}, f, indent=2)
    print(f"\nWrote {out}")

    if winner is not None:
        rec = {**{k: winner[k] for k in ("scorer", "policy", "param")},
               "anaphora_guard": True, "table_guard": True, "dedup": True,
               **FORMATS[PROD_FORMAT],
               "measured": strip(winner)[PROD_FORMAT],
               "baseline_plain": strip(baseline)["plain"], "n": n}
        with open(RECOMMENDATION_PATH, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        print(f"Wrote {RECOMMENDATION_PATH}")


if __name__ == "__main__":
    main()
