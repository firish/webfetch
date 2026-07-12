"""
Volatility classifier eval: pick the fallback freshness classifier.

FreshQA queries carry gold volatility labels which map onto the cache's
freshness classes: fast-changing -> realtime, slow-changing -> recent,
never-changing -> stable.

Candidates:
1. Keyword heuristics (zero deps) - hand-written cue rules, evaluated on
   the full set (rules were written blind to the data beyond 3 examples).
2. Embedding nearest-centroid - per-class centroids from a seeded 60/40
   train/test split using the bi-encoder the library already loads.
3. kNN (k=5) over the same split.

The metric that matters is cost-weighted: classifying a fast-changing query
as stable serves stale results for months (expensive); classifying a stable
query as realtime just costs extra searches (cheap). So the winner is picked
by realtime-class recall first, accuracy second. If nothing beats
MIN_REALTIME_RECALL, the recommendation is to ship WITHOUT a library
classifier and rely on the model hint + DEFAULT_FRESHNESS.

Run: python evals/run_volatility_eval.py
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.common import markdown_table, read_jsonl

DATASET = Path(__file__).resolve().parent / "datasets" / "freshqa_queries.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

SEED = 42
TEST_FRACTION = 0.4
KNN_K = 5
BI_MODEL = "all-MiniLM-L6-v2"
MIN_REALTIME_RECALL = 0.6

LABEL_MAP = {
    "fast-changing": "realtime",
    "slow-changing": "recent",
    "never-changing": "stable",
}
CLASSES = ("realtime", "recent", "stable")

# Cue rules, checked in order. Realtime first (the expensive class to miss),
# then stable (clearly historical), else recent (the safe default).
_REALTIME_RE = re.compile(
    r"\b(latest|newest|most recent|current|currently|today|tonight|right now|"
    r"this (week|month|year|season)|so far|as of|price|stock|score|standings|"
    r"ranking|rank|record for|now)\b", re.IGNORECASE)
_STABLE_RE = re.compile(
    r"\b(first|invented|founded|discovered|original|originally|history of|"
    r"was born|born in|died|in \d{4}|of \d{4}|\d{4}s)\b", re.IGNORECASE)


def classify_keywords(query: str) -> str:
    """Rule-based freshness class for a query."""
    if _REALTIME_RE.search(query):
        return "realtime"
    if _STABLE_RE.search(query):
        return "stable"
    return "recent"


def confusion(gold: list[str], pred: list[str]) -> dict[str, dict[str, int]]:
    """Gold-class -> predicted-class counts."""
    m = {g: {p: 0 for p in CLASSES} for g in CLASSES}
    for g, p in zip(gold, pred):
        m[g][p] += 1
    return m


def report(name: str, gold: list[str], pred: list[str]) -> dict:
    """Print and return metrics for one candidate."""
    m = confusion(gold, pred)
    n = len(gold)
    acc = sum(1 for g, p in zip(gold, pred) if g == p) / n
    recalls = {
        c: (m[c][c] / sum(m[c].values())) if sum(m[c].values()) else 0.0
        for c in CLASSES
    }
    fast_as_stable = m["realtime"]["stable"] / max(1, sum(m["realtime"].values()))
    print(f"\n## {name} (n={n})")
    print(markdown_table(
        ["gold \\ pred"] + list(CLASSES),
        [[g] + [m[g][p] for p in CLASSES] for g in CLASSES],
    ))
    print(f"accuracy: {acc:.3f} | realtime recall: {recalls['realtime']:.3f} | "
          f"realtime->stable (worst error): {fast_as_stable:.3f}")
    return {"name": name, "n": n, "accuracy": round(acc, 3),
            "recalls": {k: round(v, 3) for k, v in recalls.items()},
            "realtime_as_stable_rate": round(fast_as_stable, 3),
            "confusion": m}


def main() -> None:
    """Run all candidates and write the recommendation."""
    rows = read_jsonl(DATASET)
    queries = [r["query"] for r in rows]
    gold = [LABEL_MAP[r["volatility"]] for r in rows]
    print(f"Loaded {len(rows)} FreshQA queries "
          f"({', '.join(f'{c}={gold.count(c)}' for c in CLASSES)})")

    results = []

    # 1. Keyword heuristics - full set.
    results.append(report("keyword-heuristics", gold,
                          [classify_keywords(q) for q in queries]))

    # 2/3. Embedding candidates - seeded split.
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer(BI_MODEL)
    vecs = np.asarray(model.encode(queries, normalize_embeddings=True,
                                   show_progress_bar=False))
    idx = list(range(len(rows)))
    random.Random(SEED).shuffle(idx)
    n_test = int(len(idx) * TEST_FRACTION)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    test_gold = [gold[i] for i in test_idx]

    centroids = {}
    for c in CLASSES:
        members = np.stack([vecs[i] for i in train_idx if gold[i] == c])
        centroid = members.mean(axis=0)
        centroids[c] = centroid / np.linalg.norm(centroid)

    centroid_pred = []
    for i in test_idx:
        sims = {c: float(vecs[i] @ centroids[c]) for c in CLASSES}
        centroid_pred.append(max(sims, key=sims.get))
    results.append(report(f"nearest-centroid ({BI_MODEL}, test 40%)",
                          test_gold, centroid_pred))

    train_matrix = np.stack([vecs[i] for i in train_idx])
    train_gold = [gold[i] for i in train_idx]
    knn_pred = []
    for i in test_idx:
        sims = train_matrix @ vecs[i]
        top = np.argsort(-sims)[:KNN_K]
        votes = [train_gold[int(t)] for t in top]
        knn_pred.append(max(CLASSES, key=votes.count))
    results.append(report(f"kNN k={KNN_K} (test 40%)", test_gold, knn_pred))

    # 4. Hybrid: keyword cue wins outright, centroid decides uncued queries.
    # Degrades to keywords-only without sentence-transformers.
    def centroid_cls(i: int) -> str:
        sims = {c: float(vecs[i] @ centroids[c]) for c in CLASSES}
        return max(sims, key=sims.get)

    hybrid_pred = []
    for i in test_idx:
        kw = classify_keywords(queries[i])
        hybrid_pred.append(kw if kw != "recent" else centroid_cls(i))
    results.append(report("hybrid: keywords -> centroid fallback (test 40%)",
                          test_gold, hybrid_pred))

    # Export production centroids computed on the FULL dataset (the split
    # exists only for honest measurement). The library ships this ~5KB
    # artifact instead of the dataset.
    full_centroids = {}
    for c in CLASSES:
        members = np.stack([vecs[i] for i in range(len(rows)) if gold[i] == c])
        centroid = members.mean(axis=0)
        full_centroids[c] = (centroid / np.linalg.norm(centroid)).tolist()
    centroid_path = RESULTS_DIR / "volatility_centroids.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(centroid_path, "w", encoding="utf-8") as f:
        json.dump({"model": BI_MODEL, "source": "FreshQA (CC), full 377 rows",
                   "classes": full_centroids}, f)
    print(f"\nExported production centroids -> {centroid_path}")

    # Winner: realtime recall first (cost-weighted), accuracy tiebreak.
    winner = max(results, key=lambda r: (r["recalls"]["realtime"], r["accuracy"]))
    ship_classifier = winner["recalls"]["realtime"] >= MIN_REALTIME_RECALL
    recommendation = {
        "ship_classifier": ship_classifier,
        "winner": winner["name"] if ship_classifier else None,
        "min_realtime_recall": MIN_REALTIME_RECALL,
        "winner_metrics": winner,
    }
    print(f"\n## Recommendation")
    if ship_classifier:
        print(f"SHIP: {winner['name']} - realtime recall "
              f"{winner['recalls']['realtime']:.3f} >= {MIN_REALTIME_RECALL}, "
              f"accuracy {winner['accuracy']:.3f}")
    else:
        print(f"DO NOT ship a library classifier - best realtime recall "
              f"{winner['recalls']['realtime']:.3f} < {MIN_REALTIME_RECALL}. "
              f"Rely on model hint + DEFAULT_FRESHNESS.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"volatility_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"candidates": results, "recommendation": recommendation},
                  f, indent=2)
    with open(RESULTS_DIR / "volatility_recommendation.json", "w",
              encoding="utf-8") as f:
        json.dump(recommendation, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
