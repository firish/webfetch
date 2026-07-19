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
# Hand-labeled tech-release/listicle/spec slice added 2026-07-18 after a
# live run showed the classifier bucketing versioned-release and listicle
# queries as realtime (15-min TTL = cache effectively dead for them).
# FreshQA's fast-changing class is news/prices/sports; this covers the
# tech distribution the tool actually serves.
TECH_DATASET = Path(__file__).resolve().parent / "datasets" / "tech_volatility_queries.jsonl"
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

# FIXED rule variant (candidate, 2026-07-18). Two repairs measured against
# the live-run failures: (1) drop "this year/this season" and bare
# "ranking|rank" from realtime cues - they fire on listicles ("top ANC
# headphones this year") that tolerate days-long TTLs; (2) year cues only
# count as STABLE for PAST years - "in 2026" during 2026 is a currency
# marker, not a history marker (real bug: "tools to use instead of Docker
# in 2026" got stable/90d).
import time as _time
_CURRENT_YEAR = _time.localtime().tm_year
_REALTIME_FIXED_RE = re.compile(
    r"\b(latest|newest|most recent|current|currently|today|tonight|"
    r"right now|this (week|month)|so far|as of|price|stock|score|"
    r"standings|record for|now)\b", re.IGNORECASE)
_STABLE_FIXED_RE = re.compile(
    r"\b(first|invented|founded|discovered|original|originally|history of|"
    r"was born|born in|died|\d{4}s)\b", re.IGNORECASE)
# Year mentions are matched separately and compared in code - a regex
# alternation for "past years" breaks at decade boundaries.
_YEAR_MENTION_RE = re.compile(r"\b(?:in|of) (\d{4})\b", re.IGNORECASE)


def classify_keywords(query: str) -> str:
    """Rule-based freshness class for a query (current shipped rules)."""
    if _REALTIME_RE.search(query):
        return "realtime"
    if _STABLE_RE.search(query):
        return "stable"
    return "recent"


def classify_keywords_fixed(query: str) -> str:
    """Rule-based class with the two 2026-07-18 cue repairs."""
    if _REALTIME_FIXED_RE.search(query):
        return "realtime"
    if _STABLE_FIXED_RE.search(query):
        return "stable"
    m = _YEAR_MENTION_RE.search(query)
    if m and int(m.group(1)) < _CURRENT_YEAR:
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
    for r in rows:
        r["slice"] = "freshqa"
    tech = read_jsonl(TECH_DATASET) if TECH_DATASET.exists() else []
    for r in tech:
        r["slice"] = "tech"
    rows = rows + tech
    queries = [r["query"] for r in rows]
    gold = [LABEL_MAP[r["volatility"]] for r in rows]
    slices = [r["slice"] for r in rows]
    print(f"Loaded {len(rows)} queries (freshqa={len(rows) - len(tech)}, "
          f"tech={len(tech)}; "
          f"{', '.join(f'{c}={gold.count(c)}' for c in CLASSES)})")

    def slice_acc(gold_l, pred_l, slice_l, name):
        idxs = [i for i, sl in enumerate(slice_l) if sl == name]
        if not idxs:
            return None
        return round(sum(1 for i in idxs if gold_l[i] == pred_l[i]) / len(idxs), 3)

    results = []

    def add(name, gold_l, pred_l, slice_l):
        r = report(name, gold_l, pred_l)
        r["tech_accuracy"] = slice_acc(gold_l, pred_l, slice_l, "tech")
        if r["tech_accuracy"] is not None:
            print(f"tech-slice accuracy: {r['tech_accuracy']:.3f}")
        results.append(r)
        return r

    # 1. Keyword heuristics - current shipped rules vs fixed rules.
    add("keyword-heuristics (current)", gold,
        [classify_keywords(q) for q in queries], slices)
    add("keyword-heuristics (fixed cues)", gold,
        [classify_keywords_fixed(q) for q in queries], slices)

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
    add(f"nearest-centroid ({BI_MODEL}, test 40%)",
        test_gold, centroid_pred, [slices[i] for i in test_idx])

    train_matrix = np.stack([vecs[i] for i in train_idx])
    train_gold = [gold[i] for i in train_idx]
    knn_pred = []
    for i in test_idx:
        sims = train_matrix @ vecs[i]
        top = np.argsort(-sims)[:KNN_K]
        votes = [train_gold[int(t)] for t in top]
        knn_pred.append(max(CLASSES, key=votes.count))
    add(f"kNN k={KNN_K} (test 40%)", test_gold, knn_pred,
        [slices[i] for i in test_idx])

    # 4. Hybrid: keyword cue wins outright, centroid decides uncued queries.
    # Degrades to keywords-only without sentence-transformers.
    def centroid_cls(i: int) -> str:
        sims = {c: float(vecs[i] @ centroids[c]) for c in CLASSES}
        return max(sims, key=sims.get)

    test_slices = [slices[i] for i in test_idx]
    hybrid_pred = []
    for i in test_idx:
        kw = classify_keywords(queries[i])
        hybrid_pred.append(kw if kw != "recent" else centroid_cls(i))
    add("hybrid: current keywords -> centroid (test 40%)",
        test_gold, hybrid_pred, test_slices)

    hybrid_fixed_pred = []
    for i in test_idx:
        kw = classify_keywords_fixed(queries[i])
        hybrid_fixed_pred.append(kw if kw != "recent" else centroid_cls(i))
    add("hybrid: FIXED keywords -> centroid (test 40%)",
        test_gold, hybrid_fixed_pred, test_slices)

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
        json.dump({"model": BI_MODEL,
                   "source": f"FreshQA (CC) + tech slice, {len(rows)} rows",
                   "classes": full_centroids}, f)
    print(f"\nExported production centroids -> {centroid_path}")

    # Winner: realtime recall >= bar is a CONSTRAINT (the expensive class
    # to miss), best accuracy among those wins. Recall-first selection is
    # gameable by over-predicting realtime - exactly the failure the tech
    # slice documents.
    eligible = [r for r in results
                if r["recalls"]["realtime"] >= MIN_REALTIME_RECALL]
    winner = (max(eligible, key=lambda r: r["accuracy"]) if eligible
              else max(results, key=lambda r: (r["recalls"]["realtime"],
                                               r["accuracy"])))
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
