"""
Layer 2: live pipeline eval (DDG network, no LLM calls).

Runs diverse factoid queries with known short answers through
Pipeline.search_chunks (the production tool-mode path) and measures:

- retrieval recall: does any gold-answer variant appear in the returned chunks
- tokens per tool result (the cost claim), latency, failed fetches
- cache diagnostics:
  * exact-replay hit rate (same query twice - MUST be 100%)
  * paraphrase hit rate under the current exact-match cache (baseline ~0)
  * semantic-cache OPPORTUNITY rate: paraphrases whose similarity to their
    base query clears the Layer 1 matcher config - what a semantic cache
    WOULD serve, measured before the feature exists
  * false-hit risk: distinct query pairs in the suite that would wrongly match

Run: python evals/run_pipeline_eval.py [--limit N] [--sleep 2.0]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time

# Tokenizers' thread pool interacts badly with forked/threaded native code
# (observed: one SIGABRT mid-suite with fork warnings). Pin it off before
# any transformers import.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.common import answer_variants, markdown_table, normalize_text, read_jsonl

from webfetch import Pipeline, SemanticSqliteCache, SqliteCache, get_search_adapter
from webfetch.config import DEFAULT_SEARCH_PROVIDER, DEFAULT_TOOL_RESULT_BUDGET
from webfetch.extract.base import build_context
from webfetch.rank.base import Chunk

DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "live_queries.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RECOMMENDATION_PATH = RESULTS_DIR / "matcher_recommendation.json"

DEFAULT_SLEEP_SECS = 2.0
RATE_LIMIT_BACKOFF_SECS = 30.0
CHARS_PER_TOKEN = 4
# Used only if no matcher_recommendation.json exists (run Layer 1 first).
FALLBACK_BI_THRESHOLD = 0.80
# A cache replay must be near-instant; anything slower means the cache path
# was not actually exercised.
MAX_REPLAY_SECS = 0.5


@dataclass
class QueryRunRecord:
    """One pipeline invocation and its measurements."""

    id: str
    query: str
    kind: str  # "base" | "exact_replay" | "paraphrase"
    recall_hit: bool | None  # None when the run errored
    n_chunks: int
    n_failed_urls: int
    elapsed_secs: float
    est_tokens: int
    from_cache: bool
    error: str | None
    cache_kind: str | None = None  # "exact" | "semantic" | None
    matched_query: str | None = None


def retrieval_recall(gold_answers: list[str], answer_type: str | None,
                     chunks: list[Chunk]) -> bool:
    """True if any gold-answer variant appears in the normalized chunk text."""
    haystack = normalize_text(" ".join(c.text for c in chunks))
    for answer in gold_answers:
        for variant in answer_variants(answer, answer_type):
            if variant in haystack:
                return True
    return False


def estimate_tokens(chunks: list[Chunk], budget_chars: int) -> int:
    """Estimated tokens of the tool-result string built from these chunks."""
    if not chunks:
        return 0
    return len(build_context(chunks, budget_chars=budget_chars)) // CHARS_PER_TOKEN


def run_one(pipeline: Pipeline, row: dict, kind: str, budget_chars: int) -> QueryRunRecord:
    """Run one query through the pipeline, never raising."""
    try:
        result = pipeline.search_chunks(row["query"])
        return QueryRunRecord(
            id=row["id"], query=row["query"], kind=kind,
            recall_hit=retrieval_recall(row["answers"], row.get("answer_type"), result.chunks),
            n_chunks=len(result.chunks), n_failed_urls=len(result.failed_urls),
            elapsed_secs=round(result.elapsed_secs, 3),
            est_tokens=estimate_tokens(result.chunks, budget_chars),
            from_cache=result.from_cache, error=None,
            cache_kind=result.cache_kind, matched_query=result.matched_query,
        )
    except Exception as exc:
        return QueryRunRecord(
            id=row["id"], query=row["query"], kind=kind, recall_hit=None,
            n_chunks=0, n_failed_urls=0, elapsed_secs=0.0, est_tokens=0,
            from_cache=False, error=f"{type(exc).__name__}: {exc}",
        )


def run_suite(queries: list[dict], pipeline: Pipeline, limit: int | None,
              sleep_secs: float, budget_chars: int) -> list[QueryRunRecord]:
    """Run base queries (cold + exact replay) then their paraphrases."""
    base = [q for q in queries if q.get("paraphrase_of") is None]
    paras = [q for q in queries if q.get("paraphrase_of") is not None]
    if limit is not None:
        base = base[:limit]
        base_ids = {q["id"] for q in base}
        paras = [p for p in paras if p["paraphrase_of"] in base_ids]

    records: list[QueryRunRecord] = []
    total = len(base) + len(paras)
    done = 0
    for row in base:
        base_rec = run_one(pipeline, row, "base", budget_chars)
        records.append(base_rec)
        done += 1
        print(f"  [{done}/{total}] base         {base_rec.id}: "
              f"recall={base_rec.recall_hit} chunks={base_rec.n_chunks} "
              f"{base_rec.elapsed_secs:.1f}s"
              + (f" ERROR {base_rec.error}" if base_rec.error else ""))
        if base_rec.error:
            # Nothing was cached, so an exact replay would measure nothing
            # (it would just be a second cold run) - skip it.
            time.sleep(RATE_LIMIT_BACKOFF_SECS)
            continue
        time.sleep(sleep_secs)
        replay_rec = run_one(pipeline, row, "exact_replay", budget_chars)
        records.append(replay_rec)
        if replay_rec.error:
            time.sleep(RATE_LIMIT_BACKOFF_SECS)
        elif not replay_rec.from_cache:
            time.sleep(sleep_secs)
    for row in paras:
        rec = run_one(pipeline, row, "paraphrase", budget_chars)
        records.append(rec)
        done += 1
        print(f"  [{done}/{total}] paraphrase   {rec.id}: "
              f"recall={rec.recall_hit} {rec.elapsed_secs:.1f}s"
              + (f" ERROR {rec.error}" if rec.error else ""))
        if rec.error:
            time.sleep(RATE_LIMIT_BACKOFF_SECS)
        elif not rec.from_cache:
            time.sleep(sleep_secs)
    return records


def semantic_opportunity(base_queries: list[dict], paraphrases: list[dict],
                         matcher: dict) -> dict:
    """Measure what a semantic cache WOULD do, using the Layer 1 config.

    A paraphrase counts as an opportunity hit when it matches its own base
    query under the recommended matcher (bi cosine, optionally cross-encoder
    verified). False-hit risk counts distinct-base pairs that wrongly match.

    Args:
        base_queries: Rows with paraphrase_of == None.
        paraphrases: Rows with paraphrase_of set.
        matcher: Layer 1 recommendation (mode, threshold, ce_model, ...).

    Returns:
        Dict with opportunity/false-hit details.
    """
    from sentence_transformers import SentenceTransformer

    # Reuse the harness's head-type-aware CE scoring (handles sigmoid heads,
    # raw logits, and NLI bidirectional entailment).
    from evals.run_matcher_eval import ce_scores

    bi_threshold = matcher["threshold"]
    texts = [q["query"] for q in base_queries] + [p["query"] for p in paraphrases]
    model = SentenceTransformer(matcher.get("bi_model", "all-MiniLM-L6-v2"))
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    base_vecs = vecs[: len(base_queries)]
    para_vecs = vecs[len(base_queries):]

    use_ce = matcher.get("mode") == "bi+ce" and matcher.get("ce_model")
    _prob_cache: dict[tuple[str, str], float] = {}

    def batch_probs(pair_list: list[tuple[str, str]]) -> None:
        todo = [pq for pq in pair_list if pq not in _prob_cache]
        if not todo:
            return
        scores = ce_scores([{"q1": a, "q2": b} for a, b in todo], matcher["ce_model"])
        _prob_cache.update(dict(zip(todo, scores)))

    def matches(q1: str, q2: str, cos: float) -> bool:
        if cos < bi_threshold:
            return False
        if use_ce:
            batch_probs([(q1, q2)])
            return _prob_cache[(q1, q2)] >= matcher["ce_threshold"]
        return True

    base_by_id = {q["id"]: i for i, q in enumerate(base_queries)}

    # Pre-batch every pair the loops below could ask about (one model load).
    if use_ce:
        candidates: list[tuple[str, str]] = []
        for p, pv in zip(paraphrases, para_vecs):
            sims = base_vecs @ pv
            own = base_by_id.get(p["paraphrase_of"])
            if own is not None and float(sims[own]) >= bi_threshold:
                candidates.append((base_queries[own]["query"], p["query"]))
            best = int(np.argmax(sims))
            if best != own and float(sims[best]) >= bi_threshold:
                candidates.append((base_queries[best]["query"], p["query"]))
        sims_bb_pre = base_vecs @ base_vecs.T
        for i in range(len(base_queries)):
            for j in range(i + 1, len(base_queries)):
                if float(sims_bb_pre[i][j]) >= bi_threshold:
                    candidates.append((base_queries[i]["query"], base_queries[j]["query"]))
        batch_probs(candidates)

    hits: list[dict] = []
    wrong_target: list[dict] = []
    for p, pv in zip(paraphrases, para_vecs):
        sims = base_vecs @ pv
        own = base_by_id.get(p["paraphrase_of"])
        if own is None:
            continue
        own_cos = float(sims[own])
        own_match = matches(base_queries[own]["query"], p["query"], own_cos)
        hits.append({"id": p["id"], "own_cosine": round(own_cos, 3), "hit": own_match})
        best = int(np.argmax(sims))
        if best != own and matches(base_queries[best]["query"], p["query"], float(sims[best])):
            wrong_target.append({
                "id": p["id"], "wrongly_matched": base_queries[best]["id"],
                "cosine": round(float(sims[best]), 3),
            })

    base_false: list[dict] = []
    sims_bb = base_vecs @ base_vecs.T
    for i in range(len(base_queries)):
        for j in range(i + 1, len(base_queries)):
            cos = float(sims_bb[i][j])
            if cos >= bi_threshold and matches(
                base_queries[i]["query"], base_queries[j]["query"], cos
            ):
                base_false.append({
                    "a": base_queries[i]["id"], "b": base_queries[j]["id"],
                    "cosine": round(cos, 3),
                })

    n_hit = sum(1 for h in hits if h["hit"])
    return {
        "matcher": matcher,
        "n_paraphrases": len(hits),
        "opportunity_hits": n_hit,
        "opportunity_rate": round(n_hit / len(hits), 3) if hits else None,
        "per_paraphrase": hits,
        "wrong_target_matches": wrong_target,
        "base_false_matches": base_false,
    }


def aggregate(records: list[QueryRunRecord]) -> dict:
    """Aggregate per-kind metrics from run records."""
    base = [r for r in records if r.kind == "base" and r.error is None]
    replays = [r for r in records if r.kind == "exact_replay" and r.error is None]
    paras = [r for r in records if r.kind == "paraphrase" and r.error is None]
    errors = [r for r in records if r.error is not None]

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"mean": None, "median": None, "p95": None}
        s = sorted(vals)
        return {
            "mean": round(statistics.mean(s), 2),
            "median": round(statistics.median(s), 2),
            "p95": round(s[min(len(s) - 1, int(0.95 * len(s)))], 2),
        }

    return {
        "n_base": len(base), "n_errors": len(errors),
        "recall_hits": sum(1 for r in base if r.recall_hit),
        "recall_rate": round(sum(1 for r in base if r.recall_hit) / len(base), 3) if base else None,
        "latency_secs": _stats([r.elapsed_secs for r in base]),
        "tokens": _stats([float(r.est_tokens) for r in base]),
        "failed_urls_mean": round(statistics.mean([r.n_failed_urls for r in base]), 2) if base else None,
        "exact_replay_hit_rate": round(
            sum(1 for r in replays if r.from_cache) / len(replays), 3) if replays else None,
        "exact_replay_all_exact_kind": all(r.cache_kind == "exact" for r in replays if r.from_cache),
        "exact_replay_max_secs": max((r.elapsed_secs for r in replays), default=None),
        # Base queries are all distinct - a semantic hit here is a live
        # false positive of the semantic cache.
        "base_semantic_hits": sum(1 for r in base if r.cache_kind == "semantic"),
        "paraphrase_semantic_hits": sum(1 for r in paras if r.cache_kind == "semantic"),
        "paraphrase_exact_cache_hits": sum(1 for r in paras if r.cache_kind == "exact"),
        "n_paraphrases": len(paras),
    }


def main() -> None:
    """Run the Layer 2 suite and print the report."""
    # Standalone script convenience: pick up provider API keys from .env
    # (the library itself never loads .env by design).
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECS)
    parser.add_argument("--semantic-threshold", type=float, default=None)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--budget-chars", type=int, default=DEFAULT_TOOL_RESULT_BUDGET)
    parser.add_argument("--provider", default=DEFAULT_SEARCH_PROVIDER,
                        choices=["ddg", "brave", "serper", "tavily", "multi"],
                        help="Search provider (keyed engines need API keys; "
                             "ddg rate-limits long suites; multi = RRF "
                             "fusion of all engines with credentials)")
    parser.add_argument("--exact-cache-only", action="store_true",
                        help="Use the exact-match SqliteCache (pre-semantic "
                             "baseline behavior)")
    args = parser.parse_args()

    if args.semantic_threshold is not None:
        matcher = {"mode": "bi", "threshold": args.semantic_threshold,
                   "ce_model": None, "ce_threshold": None}
    elif RECOMMENDATION_PATH.exists():
        with open(RECOMMENDATION_PATH, encoding="utf-8") as f:
            matcher = json.load(f)
        print(f"Matcher config from Layer 1: mode={matcher['mode']} "
              f"bi>={matcher['threshold']:.2f}"
              + (f" ce>={matcher['ce_threshold']:.2f} ({matcher['ce_model']})"
                 if matcher.get("ce_model") else ""))
    else:
        matcher = {"mode": "bi", "threshold": FALLBACK_BI_THRESHOLD,
                   "ce_model": None, "ce_threshold": None}
        print(f"WARNING: no matcher_recommendation.json - using bi>={FALLBACK_BI_THRESHOLD}")

    cache_dir = args.cache_dir or Path(tempfile.mkdtemp(prefix="webfetch_eval_"))
    cache_db = str(cache_dir / "cache.db")
    cache = (SqliteCache(db_path=cache_db) if args.exact_cache_only
             else SemanticSqliteCache(db_path=cache_db))
    print(f"Cache db: {cache_db} | provider: {args.provider} | "
          f"cache: {type(cache).__name__}")
    pipeline = Pipeline(
        search=get_search_adapter(args.provider),
        cache=cache,
    )

    queries = read_jsonl(args.dataset)
    records = run_suite(queries, pipeline, args.limit, args.sleep, args.budget_chars)
    agg = aggregate(records)

    base_rows = [q for q in queries if q.get("paraphrase_of") is None]
    if args.limit is not None:
        base_rows = base_rows[:args.limit]
    base_ids = {q["id"] for q in base_rows}
    para_rows = [q for q in queries
                 if q.get("paraphrase_of") is not None and q["paraphrase_of"] in base_ids]
    opportunity = semantic_opportunity(base_rows, para_rows, matcher) if para_rows else None

    print("\n## Per-query")
    print(markdown_table(
        ["id", "kind", "recall", "chunks", "failed", "secs", "tokens", "cache", "error"],
        [[r.id, r.kind, r.recall_hit, r.n_chunks, r.n_failed_urls,
          f"{r.elapsed_secs:.1f}", r.est_tokens, r.cache_kind or "-", r.error or "-"]
         for r in records],
    ))

    print(f"\n## Aggregate (base queries, n={agg['n_base']})")
    print(f"retrieval recall: {agg['recall_hits']}/{agg['n_base']} "
          f"({(agg['recall_rate'] or 0) * 100:.1f}%)")
    print(f"latency secs: {agg['latency_secs']}")
    print(f"tokens/result: {agg['tokens']}")
    print(f"failed urls mean: {agg['failed_urls_mean']} | errors: {agg['n_errors']}")

    # Wrong-target audit: a semantic hit must have matched its OWN base query.
    base_query_by_id = {q["id"]: q["query"] for q in base_rows}
    wrong_targets = [
        {"id": r.id, "matched": r.matched_query}
        for r in records
        if r.kind == "paraphrase" and r.cache_kind == "semantic"
        and r.matched_query != base_query_by_id.get(
            next((q["paraphrase_of"] for q in para_rows if q["id"] == r.id), None))
    ]

    print("\n## Cache diagnostics")
    replay_ok = (agg["exact_replay_hit_rate"] == 1.0
                 and agg["exact_replay_all_exact_kind"])
    replay_fast = (agg["exact_replay_max_secs"] or 0) <= MAX_REPLAY_SECS
    base_fresh_ok = agg["base_semantic_hits"] == 0
    targets_ok = not wrong_targets
    print(f"exact-replay hit rate: {agg['exact_replay_hit_rate']} "
          f"(all kind=exact: {agg['exact_replay_all_exact_kind']}) "
          + ("[OK]" if replay_ok else "[FAIL - MUST BE 1.0 and exact]"))
    print(f"exact-replay max secs: {agg['exact_replay_max_secs']} "
          + ("[OK]" if replay_fast else f"[FAIL - must be <= {MAX_REPLAY_SECS}]"))
    print(f"semantic hits on distinct base queries: {agg['base_semantic_hits']} "
          + ("[OK]" if base_fresh_ok else "[FAIL - live false positives!]"))
    print(f"paraphrase SEMANTIC hit rate: "
          f"{agg['paraphrase_semantic_hits']}/{agg['n_paraphrases']} "
          f"(exact hits: {agg['paraphrase_exact_cache_hits']})")
    print(f"wrong-target semantic hits: {len(wrong_targets)} "
          + ("[OK]" if targets_ok else f"[FAIL] {wrong_targets}"))
    if opportunity:
        print(f"semantic-cache OPPORTUNITY rate: "
              f"{opportunity['opportunity_hits']}/{opportunity['n_paraphrases']} "
              f"({(opportunity['opportunity_rate'] or 0) * 100:.0f}%)")
        if opportunity["wrong_target_matches"]:
            print(f"WRONG-TARGET matches: {opportunity['wrong_target_matches']}")
        else:
            print("wrong-target matches: none")
        if opportunity["base_false_matches"]:
            print(f"BASE-PAIR false matches: {opportunity['base_false_matches']}")
        else:
            print("base-pair false matches: none")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.out_dir / f"pipeline_eval_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"dataset": str(args.dataset), "limit": args.limit,
                       "sleep_secs": args.sleep, "budget_chars": args.budget_chars,
                       "cache_db": cache_db, "provider": args.provider,
                       "matcher": matcher},
            "records": [asdict(r) for r in records],
            "aggregate": agg,
            "wrong_targets": wrong_targets,
            "semantic_opportunity": opportunity,
        }, f, indent=2)
    print(f"\nWrote {out_path}")

    if not (replay_ok and replay_fast and base_fresh_ok and targets_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
