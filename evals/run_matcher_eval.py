"""
Layer 1: offline semantic-matcher eval (no network beyond model downloads).

Answers the question the semantic cache needs answered before it exists:
at what cosine-similarity threshold can we treat two queries as "the same
question" - and how much does a cross-encoder verification stage help?

Sweeps:
1. Bi-encoder cosine alone across thresholds 0.50-0.99.
2. Cascade: bi-encoder shortlist threshold x cross-encoder (sigmoid) threshold.

The recommendation is precision-first: a semantic-cache false hit silently
serves wrong search results, so we pick the highest-recall threshold whose
precision >= --min-precision.

Run: python evals/run_matcher_eval.py [--min-precision 0.98] [--ce-model NAME]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.common import markdown_table, precision_recall_f1, read_jsonl

# Mirrors webfetch.config.BIENCODER_MODEL - the semantic cache will reuse the
# same model the ranking cascade already loads.
BI_MODEL = "all-MiniLM-L6-v2"
# Two candidate verifiers: the library's ranking cross-encoder (zero new
# downloads) and a duplicate-question model purpose-trained on QQP.
CE_MODELS = (
    "cross-encoder/nli-deberta-v3-base",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/quora-distilroberta-base",
)
THRESHOLDS = [i / 100 for i in range(50, 100)]
SHORTLIST_THRESHOLDS = (0.60, 0.65, 0.70, 0.75, 0.80)
CE_THRESHOLDS = [i / 100 for i in range(5, 100)]
DEFAULT_MIN_PRECISION = 0.98
# If nothing reaches the primary bar, retry here before giving up. QQP label
# noise depresses measured precision by a few points (audited: several
# "false positives" are duplicates mislabeled as non-duplicates).
FALLBACK_MIN_PRECISION = 0.95

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PAIRS_PATH = Path(__file__).resolve().parent / "datasets" / "matcher_pairs.jsonl"


@dataclass
class ThresholdMetrics:
    """Confusion-matrix metrics at one decision threshold."""

    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float


def embed_unique(texts: list[str], model_name: str) -> dict[str, np.ndarray]:
    """Embed unique strings once with normalized embeddings.

    Args:
        texts: Strings to embed (duplicates fine).
        model_name: sentence-transformers model name.

    Returns:
        Mapping of text -> unit-normalized embedding vector.
    """
    from sentence_transformers import SentenceTransformer

    unique = list(dict.fromkeys(texts))
    model = SentenceTransformer(model_name)
    vectors = model.encode(unique, normalize_embeddings=True, show_progress_bar=False)
    return dict(zip(unique, vectors))


def cosine_for_pairs(pairs: list[dict], emb: dict[str, np.ndarray]) -> list[float]:
    """Cosine similarity per pair (dot product of normalized vectors)."""
    return [float(np.dot(emb[p["q1"]], emb[p["q2"]])) for p in pairs]


def sweep(
    labels: list[int],
    scores: list[float],
    thresholds: Sequence[float],
) -> list[ThresholdMetrics]:
    """Compute confusion metrics at each threshold (predict 1 iff score >= t)."""
    out: list[ThresholdMetrics] = []
    n = len(labels)
    for t in thresholds:
        tp = fp = fn = tn = 0
        for label, score in zip(labels, scores):
            pred = score >= t
            if pred and label == 1:
                tp += 1
            elif pred and label == 0:
                fp += 1
            elif not pred and label == 1:
                fn += 1
            else:
                tn += 1
        assert tp + fp + fn + tn == n
        p, r, f1 = precision_recall_f1(tp, fp, fn)
        out.append(ThresholdMetrics(t, tp, fp, fn, tn, p, r, f1))
    return out


def recommend(metrics: list[ThresholdMetrics], min_precision: float) -> ThresholdMetrics | None:
    """Highest-recall row with precision >= min_precision (ties: lowest threshold)."""
    eligible = [m for m in metrics if m.precision >= min_precision]
    if not eligible:
        return None
    return max(eligible, key=lambda m: (m.recall, -m.threshold))


def ce_scores(pairs: list[dict], model_name: str) -> list[float]:
    """Cross-encoder scores per pair, mapped to (0, 1).

    Handles three head types:
    - sigmoid-head models (quora, stsb): already emit probabilities -
      sigmoid-ing again would compress the threshold range, so use as-is
    - raw-logit single-score models (ms-marco): squash through sigmoid
    - multi-class NLI heads (contradiction/entailment/neutral): paraphrase
      is approximated as BIDIRECTIONAL entailment - score both directions
      and take the min of P(entailment)
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)
    tuples = [(p["q1"], p["q2"]) for p in pairs]
    raw = np.asarray(model.predict(tuples, show_progress_bar=False))

    if raw.ndim == 2:
        label2id = {k.lower(): v for k, v in model.model.config.label2id.items()}
        ent = label2id.get("entailment", 1)
        rev = np.asarray(model.predict([(b, a) for a, b in tuples],
                                       show_progress_bar=False))

        def p_entail(mat: np.ndarray) -> np.ndarray:
            rows_are_probs = np.all(mat >= 0) and np.allclose(mat.sum(axis=1), 1.0, atol=1e-3)
            if not rows_are_probs:
                e = np.exp(mat - mat.max(axis=1, keepdims=True))
                mat = e / e.sum(axis=1, keepdims=True)
            return mat[:, ent]

        return [float(min(f, r)) for f, r in zip(p_entail(raw), p_entail(rev))]

    if float(raw.min()) >= 0.0 and float(raw.max()) <= 1.0:
        return [float(s) for s in raw]
    return [float(1 / (1 + np.exp(-s))) for s in raw]


def sweep_cascade(
    labels: list[int],
    cos_scores: list[float],
    ce_probs: list[float],
    shortlist_thresholds: Sequence[float],
    ce_thresholds: Sequence[float],
) -> list[dict]:
    """Sweep the two-stage cascade: match iff cosine >= bi_t AND ce >= ce_t."""
    rows: list[dict] = []
    for bi_t in shortlist_thresholds:
        for ce_t in ce_thresholds:
            tp = fp = fn = 0
            for label, cos, ce in zip(labels, cos_scores, ce_probs):
                pred = cos >= bi_t and ce >= ce_t
                if pred and label == 1:
                    tp += 1
                elif pred and label == 0:
                    fp += 1
                elif not pred and label == 1:
                    fn += 1
            p, r, f1 = precision_recall_f1(tp, fp, fn)
            rows.append({
                "bi_threshold": bi_t, "ce_threshold": ce_t,
                "tp": tp, "fp": fp, "fn": fn,
                "precision": p, "recall": r, "f1": f1,
            })
    return rows


def sweep_or_ensemble(labels: list[int], cos: list[float], p1: list[float],
                      p2: list[float], bi_ts, ce_ts) -> list[dict]:
    """Sweep OR-ensemble configs: cos >= bi AND (p1 >= t1 OR p2 >= t2).

    Motivated by the 2026-07-18 live run: the NLI verifier scores certain
    question<->keyword-form paraphrases near zero (out-of-distribution),
    while a duplicate-question model handles them - and vice versa for the
    entity-swap traps NLI kills natively. The OR lets each verifier cover
    the other's blind spot; precision discipline comes from the sweep.
    """
    y = np.asarray(labels, dtype=bool)
    c = np.asarray(cos)
    a1 = np.asarray(p1)
    a2 = np.asarray(p2)
    rows: list[dict] = []
    for bi_t in bi_ts:
        base = c >= bi_t
        for t1 in ce_ts:
            m1 = a1 >= t1
            for t2 in ce_ts:
                pred = base & (m1 | (a2 >= t2))
                tp = int(np.sum(pred & y))
                fp = int(np.sum(pred & ~y))
                fn = int(np.sum(~pred & y))
                precision, recall, f1 = precision_recall_f1(tp, fp, fn)
                rows.append({"bi_threshold": bi_t, "t1": t1, "t2": t2,
                             "tp": tp, "fp": fp, "fn": fn,
                             "precision": precision, "recall": recall,
                             "f1": f1})
    return rows


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    """Run the full Layer 1 sweep and write recommendation + results JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=PAIRS_PATH)
    parser.add_argument("--min-precision", type=float, default=DEFAULT_MIN_PRECISION)
    parser.add_argument("--ce-models", nargs="*", default=list(CE_MODELS))
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    pairs = read_jsonl(args.pairs)
    labels = [p["label"] for p in pairs]
    print(f"Loaded {len(pairs)} pairs ({sum(labels)} positive)")

    texts = [p["q1"] for p in pairs] + [p["q2"] for p in pairs]
    emb = embed_unique(texts, BI_MODEL)
    cos_scores = cosine_for_pairs(pairs, emb)

    # Sanity: identical strings must score ~1.0.
    probe = pairs[0]["q1"]
    assert float(np.dot(emb[probe], emb[probe])) >= 0.999

    bi_metrics = sweep(labels, cos_scores, THRESHOLDS)
    rec = recommend(bi_metrics, args.min_precision)

    print(f"\n## Bi-encoder cosine sweep ({BI_MODEL}, n={len(pairs)})")
    display_rows = [m for m in bi_metrics if round(m.threshold * 100) % 5 == 0]
    if rec and rec not in display_rows:
        display_rows = sorted(display_rows + [rec], key=lambda m: m.threshold)
    print(markdown_table(
        ["threshold", "TP", "FP", "FN", "TN", "precision", "recall", "F1"],
        [[_fmt(m.threshold), m.tp, m.fp, m.fn, m.tn,
          _fmt(m.precision), _fmt(m.recall), _fmt(m.f1)] for m in display_rows],
    ))
    if rec:
        print(f"\nRecommended (bi-only): {rec.threshold:.2f} "
              f"(recall {rec.recall:.3f} at precision {rec.precision:.3f} "
              f">= {args.min_precision})")
    else:
        print(f"\nNo bi-only threshold reaches precision >= {args.min_precision}")

    # Hard-negative difficulty check: mean jaccard of qqp-hard-negative pairs.
    from evals.common import jaccard, normalize_text
    hard = [p for p in pairs if p["source"] == "qqp-hard-negative"]
    jac = [jaccard(set(normalize_text(p["q1"]).split()),
                   set(normalize_text(p["q2"]).split())) for p in hard]
    print(f"Hard-negative mean jaccard: {np.mean(jac):.2f} (expect > 0.5)")

    cascade_results: dict[str, list[dict]] = {}
    best_cascades: dict[str, dict] = {}
    ce_probs_by_model: dict[str, list[float]] = {}
    for ce_model in args.ce_models:
        print(f"\n## Cascade sweep with {ce_model}")
        t0 = time.perf_counter()
        probs = ce_scores(pairs, ce_model)
        ce_probs_by_model[ce_model] = probs
        print(f"(scored {len(pairs)} pairs in {time.perf_counter() - t0:.1f}s)")
        rows = sweep_cascade(labels, cos_scores, probs, SHORTLIST_THRESHOLDS, CE_THRESHOLDS)
        cascade_results[ce_model] = rows
        eligible = [r for r in rows if r["precision"] >= args.min_precision]
        best = max(eligible, key=lambda r: r["recall"]) if eligible else None
        if best:
            best_cascades[ce_model] = best
            print(f"Best: bi>={best['bi_threshold']:.2f} + ce>={best['ce_threshold']:.2f} "
                  f"-> recall {best['recall']:.3f} at precision {best['precision']:.3f}")
        else:
            print(f"No cascade config reaches precision >= {args.min_precision}")

    # OR-ensemble sweeps over every verifier pair.
    or_results: dict[tuple[str, str], list[dict]] = {}
    model_list = list(ce_probs_by_model)
    for i in range(len(model_list)):
        for j in range(i + 1, len(model_list)):
            m1, m2 = model_list[i], model_list[j]
            rows = sweep_or_ensemble(labels, cos_scores,
                                     ce_probs_by_model[m1],
                                     ce_probs_by_model[m2],
                                     SHORTLIST_THRESHOLDS, CE_THRESHOLDS)
            or_results[(m1, m2)] = rows
            eligible = [r for r in rows if r["precision"] >= args.min_precision]
            best = max(eligible, key=lambda r: r["recall"]) if eligible else None
            short1, short2 = m1.split("/")[-1], m2.split("/")[-1]
            if best:
                print(f"\n## OR-ensemble {short1} | {short2}: "
                      f"bi>={best['bi_threshold']:.2f}, "
                      f"{short1}>={best['t1']:.2f} OR {short2}>={best['t2']:.2f} "
                      f"-> recall {best['recall']:.3f} at precision "
                      f"{best['precision']:.3f}")
            else:
                print(f"\n## OR-ensemble {short1} | {short2}: no config "
                      f"reaches precision >= {args.min_precision}")

    # Pick overall winner: best recall among (bi-only rec, each cascade best),
    # at the primary precision bar; fall back to FALLBACK_MIN_PRECISION.
    def candidates_at(bar: float) -> list[dict]:
        out: list[dict] = []
        bi_rec = recommend(bi_metrics, bar)
        if bi_rec:
            out.append({
                "mode": "bi", "threshold": bi_rec.threshold, "ce_model": None,
                "ce_threshold": None, "precision": bi_rec.precision,
                "recall": bi_rec.recall,
            })
        for ce_model, rows in cascade_results.items():
            eligible = [r for r in rows if r["precision"] >= bar]
            if eligible:
                best = max(eligible, key=lambda r: r["recall"])
                out.append({
                    "mode": "bi+ce", "threshold": best["bi_threshold"],
                    "ce_model": ce_model, "ce_threshold": best["ce_threshold"],
                    "precision": best["precision"], "recall": best["recall"],
                })
        for (m1, m2), rows in or_results.items():
            eligible = [r for r in rows if r["precision"] >= bar]
            if eligible:
                best = max(eligible, key=lambda r: r["recall"])
                out.append({
                    "mode": "bi+ce_or", "threshold": best["bi_threshold"],
                    "ce_model": m1, "ce_threshold": best["t1"],
                    "ce_model_2": m2, "ce_threshold_2": best["t2"],
                    "precision": best["precision"], "recall": best["recall"],
                })
        return out

    bar_used: float | None = args.min_precision
    candidates = candidates_at(args.min_precision)
    if not candidates and FALLBACK_MIN_PRECISION < args.min_precision:
        bar_used = FALLBACK_MIN_PRECISION
        candidates = candidates_at(FALLBACK_MIN_PRECISION)
        if candidates:
            print(f"\nNOTE: primary bar {args.min_precision} unreachable - "
                  f"recommending at fallback bar {bar_used}")
    if not candidates:
        # No config clears either bar. Still emit a recommendation (Layer 2
        # depends on the JSON handoff): take the highest-precision config
        # with recall >= 0.5 (or the max-precision one outright), flagged
        # met_bar=false. The per-source breakdown below shows whether the
        # residual FPs come from trusted negatives or noisy QQP labels.
        bar_used = None
        pool: list[dict] = []
        bi_best = max(bi_metrics, key=lambda m: (m.precision, m.recall))
        pool.append({"mode": "bi", "threshold": bi_best.threshold, "ce_model": None,
                     "ce_threshold": None, "precision": bi_best.precision,
                     "recall": bi_best.recall})
        for ce_model, rows in cascade_results.items():
            useful = [r for r in rows if r["recall"] >= 0.5] or rows
            best = max(useful, key=lambda r: (r["precision"], r["recall"]))
            pool.append({"mode": "bi+ce", "threshold": best["bi_threshold"],
                         "ce_model": ce_model, "ce_threshold": best["ce_threshold"],
                         "precision": best["precision"], "recall": best["recall"]})
        candidates = pool
        print(f"\nNOTE: no config clears even the fallback bar "
              f"{FALLBACK_MIN_PRECISION} - emitting best-available config "
              f"flagged met_bar=false")
    winner = max(candidates, key=lambda c: (c["precision"], c["recall"])) \
        if bar_used is None else max(candidates, key=lambda c: c["recall"])
    winner["met_bar"] = bar_used is not None
    winner["min_precision"] = bar_used
    winner["primary_bar"] = args.min_precision
    winner["bi_model"] = BI_MODEL
    winner["n_pairs"] = len(pairs)

    # Predictions for every pair at the winning config (reusing stored CE
    # probs - the winner's model was already scored during the sweep).
    win_probs = (ce_probs_by_model.get(winner["ce_model"])
                 if winner["mode"] in ("bi+ce", "bi+ce_or") else None)
    win_probs2 = (ce_probs_by_model.get(winner.get("ce_model_2"))
                  if winner["mode"] == "bi+ce_or" else None)

    def predict(i: int) -> bool:
        if cos_scores[i] < winner["threshold"]:
            return False
        if winner["mode"] == "bi+ce_or":
            return (win_probs[i] >= winner["ce_threshold"]
                    or win_probs2[i] >= winner["ce_threshold_2"])
        if win_probs is not None:
            return win_probs[i] >= winner["ce_threshold"]
        return True

    # Per-source breakdown: positives report recall, negatives report FPs.
    # This is where domain gaps show up (e.g. QQP-tuned thresholds vs the
    # factoid queries the cache actually serves).
    print("\n## Per-source breakdown at winning config "
          f"(mode={winner['mode']}, bi>={winner['threshold']:.2f}"
          + (f", ce>={winner['ce_threshold']:.2f}" if win_probs is not None else "")
          + ")")
    per_source: dict[str, dict] = {}
    source_rows = []
    for source in sorted({p["source"] for p in pairs}):
        idxs = [i for i, p in enumerate(pairs) if p["source"] == source]
        pos = [i for i in idxs if pairs[i]["label"] == 1]
        neg = [i for i in idxs if pairs[i]["label"] == 0]
        pos_hit = sum(1 for i in pos if predict(i))
        neg_fp = sum(1 for i in neg if predict(i))
        per_source[source] = {"n": len(idxs), "pos": len(pos), "pos_hit": pos_hit,
                              "neg": len(neg), "neg_fp": neg_fp}
        source_rows.append([
            source, len(idxs),
            f"{pos_hit}/{len(pos)}" if pos else "-",
            f"{neg_fp}/{len(neg)}" if neg else "-",
        ])
    print(markdown_table(["source", "n", "recall (pos)", "false pos (neg)"], source_rows))

    # Trusted-negative precision: QQP labels are noisy (audited mislabels
    # depress measured precision), so ALSO report FPs on the hand-written
    # negative sources only - the number that actually maps to wrong-answer
    # risk in production.
    trusted_sources = {"handwritten", "factoid-negative", "cross-form"}
    t_neg = [i for i, p in enumerate(pairs)
             if p["source"] in trusted_sources and p["label"] == 0]
    t_fp = sum(1 for i in t_neg if predict(i))
    print(f"Trusted-negative FPs at winning config: {t_fp}/{len(t_neg)}")
    winner["trusted_negative_fp"] = t_fp
    winner["trusted_negative_n"] = len(t_neg)

    # Detailed audit of the hand-crafted negatives (must all be no-match).
    adv = [(i, p) for i, p in enumerate(pairs)
           if p["source"] in ("handwritten", "factoid-negative")]
    print("\n## Hand-crafted pairs at winning config")
    adv_rows = []
    adv_fp = 0
    for i, p in adv:
        pred = predict(i)
        wrong = pred != bool(p["label"])
        if wrong and p["label"] == 0:
            adv_fp += 1
        adv_rows.append([
            p["id"], (p["note"] or "")[:40], _fmt(cos_scores[i]),
            _fmt(win_probs[i]) if win_probs is not None else "-",
            "MATCH" if pred else "no-match",
            "WRONG" if wrong else "ok",
        ])
    print(markdown_table(["id", "note", "cosine", "ce_prob", "predicted", "verdict"], adv_rows))
    print(f"Hand-crafted false positives: {adv_fp}")
    winner["per_source"] = per_source

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    full_path = args.out_dir / f"matcher_eval_{ts}.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"bi_model": BI_MODEL, "min_precision": args.min_precision,
                       "n_pairs": len(pairs)},
            "bi_sweep": [asdict(m) for m in bi_metrics],
            "cascade_sweeps": cascade_results,
            "or_ensemble_bests": {
                f"{m1}|{m2}": max(rows, key=lambda r: (r["precision"] >= args.min_precision, r["recall"]))
                for (m1, m2), rows in or_results.items()
            },
            "recommendation": winner,
        }, f, indent=2)
    rec_path = args.out_dir / "matcher_recommendation.json"
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(winner, f, indent=2)
    print(f"\nWrote {full_path}\nWrote {rec_path}")


if __name__ == "__main__":
    main()
