"""
Build checked-in eval datasets from raw downloads (deterministic, SEED=42).

Outputs (small JSONL files, committed to the repo):
- datasets/matcher_pairs.jsonl: Layer 1 - QQP paraphrase positives, QQP hard
  lexical-overlap negatives, and hand-written adversarial negatives that
  target known semantic-cache failure modes.
- datasets/live_queries.jsonl: Layer 2 - topic-stratified SimpleQA factoid
  queries plus hand-written paraphrases for a subset (used to measure the
  semantic-cache opportunity rate).
- datasets/freshqa_queries.jsonl: optional, volatility-labeled queries for
  the future TTL feature; built only if the raw FreshQA export downloaded.

Each output has a .meta.json sidecar recording source, license, seed, counts.

Run: python evals/fetch_datasets.py && python evals/build_datasets.py
"""

from __future__ import annotations

import ast
import json
import random
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.common import jaccard, normalize_text, write_jsonl

RAW_DIR = Path(__file__).resolve().parent / "datasets" / "raw"
OUT_DIR = Path(__file__).resolve().parent / "datasets"

SEED = 42
N_QQP_POS = 100
N_QQP_NEG = 100
N_SIMPLEQA = 50
# Substring matching against page text is brittle for long answers - keep
# gold answers short so recall measures retrieval, not string luck.
MAX_ANSWER_CHARS = 50

# Hand-written negatives (label 0) targeting semantic-cache failure modes:
# high embedding similarity, different answers. A semantic cache that matches
# any of these serves a silently wrong result.
ADVERSARIAL_PAIRS: list[dict] = [
    {"id": "adv-entity-01", "q1": "what is the latest React version", "q2": "what is the latest Vue version", "note": "entity swap"},
    {"id": "adv-entity-02", "q1": "who is the CEO of OpenAI", "q2": "who is the CEO of Anthropic", "note": "entity swap"},
    {"id": "adv-entity-03", "q1": "population of Norway", "q2": "population of Sweden", "note": "entity swap"},
    {"id": "adv-attr-01", "q1": "iPhone 15 battery life", "q2": "iPhone 15 price", "note": "attribute swap"},
    {"id": "adv-attr-02", "q1": "when was the Eiffel Tower built", "q2": "how tall is the Eiffel Tower", "note": "attribute swap"},
    {"id": "adv-attr-03", "q1": "Tesla Model 3 range", "q2": "Tesla Model 3 top speed", "note": "attribute swap"},
    {"id": "adv-neg-01", "q1": "is aspartame safe to consume", "q2": "is aspartame dangerous to consume", "note": "opposite framing, same info need - same search results serve both (positive control)", "label": 1},
    {"id": "adv-neg-02", "q1": "foods that are high in iron", "q2": "foods that are low in iron", "note": "polarity swap"},
    {"id": "adv-time-01", "q1": "population of Japan in 1990", "q2": "population of Japan", "note": "time scope"},
    {"id": "adv-time-02", "q1": "who won the FIFA World Cup in 2010", "q2": "who won the FIFA World Cup in 2014", "note": "time scope"},
    {"id": "adv-time-03", "q1": "latest iPhone model", "q2": "first iPhone model", "note": "time scope"},
    {"id": "adv-unit-01", "q1": "how many miles is 5 km", "q2": "how many miles is 50 km", "note": "quantity change"},
    {"id": "adv-unit-02", "q1": "convert 100 USD to EUR", "q2": "convert 100 USD to GBP", "note": "unit swap"},
    {"id": "adv-version-01", "q1": "python 3.12 release date", "q2": "python 3.13 release date", "note": "version swap"},
    {"id": "adv-version-02", "q1": "what's new in Node 20", "q2": "what's new in Node 22", "note": "version swap"},
    {"id": "adv-para-01", "q1": "what is the newest React release", "q2": "latest React version", "note": "true paraphrase (positive control)", "label": 1},
    {"id": "adv-para-02", "q1": "how tall is Mount Everest", "q2": "what is the height of Mount Everest", "note": "true paraphrase (positive control)", "label": 1},
]

# Hand-written FACTOID negatives (label 0) in SimpleQA style. The QQP
# negatives cover conversational questions; these cover the entity-heavy
# factoid distribution the semantic cache actually serves, where a one-token
# difference (year, entity, role) changes the answer entirely.
FACTOID_NEGATIVE_PAIRS: list[dict] = [
    {"id": "fact-neg-01", "q1": "Who received the IEEE Frank Rosenblatt Award in 2010?", "q2": "Who received the IEEE Frank Rosenblatt Award in 2011?", "note": "year swap"},
    {"id": "fact-neg-02", "q1": "Who won the Nobel Prize in Physics in 1965?", "q2": "Who won the Nobel Prize in Chemistry in 1965?", "note": "category swap"},
    {"id": "fact-neg-03", "q1": "In what year was the city of Medellin founded?", "q2": "In what year was the city of Bogota founded?", "note": "entity swap"},
    {"id": "fact-neg-04", "q1": "Who directed the movie Alien (1979)?", "q2": "Who wrote the movie Alien (1979)?", "note": "role swap"},
    {"id": "fact-neg-05", "q1": "What is the capital of North Dakota?", "q2": "What is the capital of South Dakota?", "note": "entity swap"},
    {"id": "fact-neg-06", "q1": "When did construction of the Golden Gate Bridge begin?", "q2": "When did construction of the Golden Gate Bridge finish?", "note": "attribute swap"},
    {"id": "fact-neg-07", "q1": "Who was the first woman to win the Fields Medal?", "q2": "Who was the first woman to win the Turing Award?", "note": "award swap"},
    {"id": "fact-neg-08", "q1": "What year did the Berlin Wall go up?", "q2": "What year did the Berlin Wall come down?", "note": "event polarity"},
    {"id": "fact-neg-09", "q1": "Who played the Joker in The Dark Knight (2008)?", "q2": "Who played the Joker in Joker (2019)?", "note": "film swap"},
    {"id": "fact-neg-10", "q1": "What is the population of Portland, Oregon?", "q2": "What is the population of Portland, Maine?", "note": "place disambiguation"},
    {"id": "fact-neg-11", "q1": "When was Albert Einstein born?", "q2": "When did Albert Einstein die?", "note": "attribute swap"},
    {"id": "fact-neg-12", "q1": "Which team won the 2016 NBA Finals?", "q2": "Which team won the 2016 Stanley Cup?", "note": "league swap"},
    {"id": "fact-neg-13", "q1": "Who composed the soundtrack for Interstellar?", "q2": "Who composed the soundtrack for Inception?", "note": "film swap"},
    {"id": "fact-neg-14", "q1": "How many moons does Jupiter have?", "q2": "How many moons does Saturn have?", "note": "entity swap"},
    {"id": "fact-neg-15", "q1": "Who was the CEO of Twitter in 2020?", "q2": "Who was the CEO of Twitter in 2023?", "note": "time swap"},
]

# Cross-form pairs: question form <-> keyword/noun-phrase form. Added
# 2026-07-18 after a 31-call live run measured 40% semantic recall on
# genuinely-equivalent rewordings - all misses died at the NLI gate, and
# the original 247-pair set (QQP question<->question + factoid slices)
# underrepresented this pair FORM. The first ten positives are the real
# pairs from that run (4 hits, 6 misses), verbatim. Negatives are
# cross-form entity/version/attribute swaps: any config that recovers the
# positives must still kill these.
CROSS_FORM_PAIRS: list[dict] = [
    # --- real pairs from the 2026-07-18 live run (label 1) ---
    {"id": "xform-pos-01", "q1": "React 19 new features server components", "q2": "what's new in React 19 with server components", "label": 1, "note": "real miss"},
    {"id": "xform-pos-02", "q1": "Rust 1.80 release notes highlights", "q2": "what changed in the Rust 1.80 release", "label": 1, "note": "real miss"},
    {"id": "xform-pos-03", "q1": "best noise cancelling headphones 2026", "q2": "top ANC headphones this year 2026", "label": 1, "note": "real miss, vocab swap"},
    {"id": "xform-pos-04", "q1": "PostgreSQL 17 performance improvements", "q2": "how much faster is Postgres 17", "label": 1, "note": "real miss"},
    {"id": "xform-pos-05", "q1": "TypeScript 5.9 new features", "q2": "what did TypeScript 5.9 add", "label": 1, "note": "real miss"},
    {"id": "xform-pos-06", "q1": "Docker alternatives comparison 2026", "q2": "tools to use instead of Docker in 2026", "label": 1, "note": "real miss, vocab swap"},
    {"id": "xform-pos-07", "q1": "current price of Bitcoin in USD", "q2": "how much is Bitcoin worth right now in dollars", "label": 1, "note": "real hit"},
    {"id": "xform-pos-08", "q1": "SpaceX Starship latest launch news", "q2": "most recent SpaceX Starship flight update", "label": 1, "note": "real hit"},
    {"id": "xform-pos-09", "q1": "Mediterranean diet health benefits research", "q2": "studies on health effects of the Mediterranean diet", "label": 1, "note": "real hit"},
    {"id": "xform-pos-10", "q1": "Apple M5 chip specifications", "q2": "specs of the Apple M5 processor", "label": 1, "note": "real hit"},
    # --- synthesized cross-form positives ---
    {"id": "xform-pos-11", "q1": "Kubernetes 1.31 release changes", "q2": "what is new in Kubernetes 1.31", "label": 1},
    {"id": "xform-pos-12", "q1": "Django 5.1 new features", "q2": "what does Django 5.1 introduce", "label": 1},
    {"id": "xform-pos-13", "q1": "iPhone 16 battery capacity mAh", "q2": "how big is the battery in the iPhone 16", "label": 1},
    {"id": "xform-pos-14", "q1": "cheapest electric cars 2026", "q2": "what are the most affordable EVs this year", "label": 1},
    {"id": "xform-pos-15", "q1": "Node 22 performance benchmarks", "q2": "how fast is Node 22", "label": 1},
    {"id": "xform-pos-16", "q1": "Linux kernel 6.10 changelog", "q2": "what changed in Linux kernel 6.10", "label": 1},
    {"id": "xform-pos-17", "q1": "M4 MacBook Air reviews", "q2": "is the M4 MacBook Air any good", "label": 1},
    {"id": "xform-pos-18", "q1": "intermittent fasting weight loss studies", "q2": "does intermittent fasting help you lose weight", "label": 1},
    {"id": "xform-pos-19", "q1": "US federal funds rate today", "q2": "what is the current Fed interest rate", "label": 1},
    {"id": "xform-pos-20", "q1": "best programming languages to learn 2026", "q2": "which coding language should I learn this year", "label": 1},
    {"id": "xform-pos-21", "q1": "Ozempic side effects list", "q2": "what are the side effects of Ozempic", "label": 1},
    {"id": "xform-pos-22", "q1": "Ethereum staking yield current", "q2": "how much can you earn staking Ethereum right now", "label": 1},
    {"id": "xform-pos-23", "q1": "remote work productivity research", "q2": "does working from home make people more productive", "label": 1},
    {"id": "xform-pos-24", "q1": "GPT-5.6 context window size", "q2": "how long is the context window of GPT-5.6", "label": 1},
    # --- cross-form negatives: swaps across the form boundary (label 0) ---
    {"id": "xform-neg-01", "q1": "how much faster is Postgres 17", "q2": "PostgreSQL 16 performance improvements", "label": 0, "note": "version swap"},
    {"id": "xform-neg-02", "q1": "what did TypeScript 5.9 add", "q2": "TypeScript 5.8 new features", "label": 0, "note": "version swap"},
    {"id": "xform-neg-03", "q1": "what's new in React 19 with server components", "q2": "Vue 3 new features", "label": 0, "note": "framework swap"},
    {"id": "xform-neg-04", "q1": "what changed in the Rust 1.80 release", "q2": "Rust 1.80 install instructions", "label": 0, "note": "attribute swap"},
    {"id": "xform-neg-05", "q1": "specs of the Apple M5 processor", "q2": "Apple M4 chip specifications", "label": 0, "note": "version swap"},
    {"id": "xform-neg-06", "q1": "how much is Bitcoin worth right now in dollars", "q2": "Ethereum price today USD", "label": 0, "note": "entity swap"},
    {"id": "xform-neg-07", "q1": "top ANC headphones this year 2026", "q2": "best wireless earbuds 2026", "label": 0, "note": "category swap"},
    {"id": "xform-neg-08", "q1": "tools to use instead of Docker in 2026", "q2": "Docker installation guide 2026", "label": 0, "note": "attribute swap"},
    {"id": "xform-neg-09", "q1": "what is the current Fed interest rate", "q2": "federal funds rate 2020 historical", "label": 0, "note": "time swap"},
    {"id": "xform-neg-10", "q1": "how big is the battery in the iPhone 16", "q2": "iPhone 16 screen size specs", "label": 0, "note": "attribute swap"},
    {"id": "xform-neg-11", "q1": "what is new in Kubernetes 1.31", "q2": "Kubernetes 1.30 release changes", "label": 0, "note": "version swap"},
    {"id": "xform-neg-12", "q1": "does intermittent fasting help you lose weight", "q2": "keto diet weight loss results", "label": 0, "note": "method swap"},
    {"id": "xform-neg-13", "q1": "how fast is Node 22", "q2": "Deno 2 performance benchmarks", "label": 0, "note": "runtime swap"},
    {"id": "xform-neg-14", "q1": "what are the side effects of Ozempic", "q2": "Ozempic dosage guide", "label": 0, "note": "attribute swap"},
]

# Hand-written paraphrases for a subset of the sampled SimpleQA queries,
# keyed by the sampled row id (stable because sampling is seeded). Written
# by hand: programmatic rewording stays too lexically close and would
# inflate the semantic-cache opportunity metric.
PARAPHRASES: dict[str, str] = {
    "simpleqa-4207": "Which constellation contains the Black Widow Pulsar PSR B1957+20?",
    "simpleqa-1591": "When in 1996 did Puntsagiin Jasrai stop being Mongolia's Prime Minister?",
    "simpleqa-0721": "Which pitcher was the Red Sox starter in Game 5 of the 2004 ALCS?",
    "simpleqa-0719": "When was the town of Jardin in Antioquia, Colombia established?",
    "simpleqa-0225": "Who voices Mob's brother in the Spanish dub of Mob Psycho 100?",
    "simpleqa-2505": "What year was the line of control drawn separating the Indian and Pakistani parts?",
    "simpleqa-0415": "The Dragonborn DLC is set off which coast of Morrowind?",
    "simpleqa-2009": "Which city hosted the sixth ASEM Education Ministers' Meeting?",
    "simpleqa-2355": "Which ship is named after Habba Khatoon?",
    "simpleqa-2287": "Who won the Roebling Medal in 1968?",
    "simpleqa-0737": "Which Hezbollah leader is the song 'The Hawk of Lebanon' about?",
    "simpleqa-3672": "Who finished runner-up in Group C of the 2018-19 Champions League?",
    "simpleqa-2377": "Name the twins on The Cosby Show.",
    "simpleqa-3027": "What number of sons did Charles Wheatstone father?",
    "simpleqa-2616": "Who was awarded the 2003 Benjamin Franklin Medal in Computer and Cognitive Science?",
}


def sample_qqp_pairs(raw_path: Path, n_pos: int, n_neg: int, seed: int) -> list[dict]:
    """Sample paraphrase positives and hard lexical-overlap negatives from QQP.

    Positives are a seeded random sample of duplicate pairs. Negatives are
    the non-duplicate pairs with the HIGHEST token-set Jaccard overlap, so
    they are lexically confusable - exactly the pairs a naive matcher gets
    wrong.

    Args:
        raw_path: Path to qqp_validation.parquet.
        n_pos: Number of positive pairs.
        n_neg: Number of hard negative pairs.
        seed: RNG seed.

    Returns:
        List of pair dicts (id, q1, q2, label, source, note).
    """
    df = pd.read_parquet(raw_path)
    rng = random.Random(seed)

    pos = df[df["label"] == 1].sort_values("idx")
    pos_rows = pos.to_dict("records")
    sampled_pos = rng.sample(pos_rows, n_pos)

    neg = df[df["label"] == 0].sort_values("idx")
    scored: list[tuple[float, dict]] = []
    for row in neg.to_dict("records"):
        j = jaccard(
            set(normalize_text(row["question1"]).split()),
            set(normalize_text(row["question2"]).split()),
        )
        scored.append((j, row))

    # Stratified across lexical-overlap bands rather than top-k by jaccard.
    # Top-k selects bag-of-words-identical pairs (word-order swaps, one-token
    # substitutions) where QQP label noise concentrates and which no
    # embedding can separate - the handwritten adversarial set covers that
    # class deliberately. Bands approximate realistic near-miss queries.
    bands = ((0.35, 0.55, int(n_neg * 0.3)),
             (0.55, 0.75, int(n_neg * 0.4)),
             (0.75, 0.92, n_neg - int(n_neg * 0.3) - int(n_neg * 0.4)))
    hard_neg: list[tuple[float, dict]] = []
    for lo, hi, count in bands:
        band = sorted(
            (t for t in scored if lo <= t[0] < hi),
            key=lambda t: t[1]["idx"],
        )
        hard_neg.extend(rng.sample(band, min(count, len(band))))

    pairs: list[dict] = []
    for row in sampled_pos:
        pairs.append({
            "id": f"qqp-pos-{row['idx']:06d}", "q1": row["question1"],
            "q2": row["question2"], "label": 1, "source": "qqp", "note": None,
        })
    for j, row in hard_neg:
        pairs.append({
            "id": f"qqp-neg-{row['idx']:06d}", "q1": row["question1"],
            "q2": row["question2"], "label": 0, "source": "qqp-hard-negative",
            "note": f"jaccard={j:.2f}",
        })
    return pairs


def sample_simpleqa(raw_path: Path, n_queries: int, seed: int) -> list[dict]:
    """Topic-stratified sample of short-answer SimpleQA questions.

    Args:
        raw_path: Path to simple_qa_test_set.csv.
        n_queries: Total queries to sample.
        seed: RNG seed.

    Returns:
        List of query dicts (id, query, answers, topic, answer_type,
        paraphrase_of, source).
    """
    df = pd.read_csv(raw_path)
    rows: list[dict] = []
    for i, row in enumerate(df.to_dict("records")):
        # metadata is a python-repr dict string (single quotes) - literal_eval
        # handles it; json.loads is the fallback in case the format changes.
        try:
            meta = ast.literal_eval(row["metadata"])
        except (ValueError, SyntaxError):
            meta = json.loads(row["metadata"])
        answer = str(row["answer"]).strip()
        if len(answer) > MAX_ANSWER_CHARS:
            continue
        rows.append({
            "id": f"simpleqa-{i:04d}",
            "query": str(row["problem"]).strip(),
            "answers": [answer],
            "topic": meta.get("topic", "unknown"),
            "answer_type": meta.get("answer_type", "unknown"),
            "paraphrase_of": None,
            "source": "simpleqa",
        })

    # Proportional round-robin across sorted topics, seeded shuffle per topic.
    by_topic: dict[str, list[dict]] = {}
    for r in rows:
        by_topic.setdefault(r["topic"], []).append(r)
    rng = random.Random(seed)
    for topic in sorted(by_topic):
        rng.shuffle(by_topic[topic])

    sampled: list[dict] = []
    topics = sorted(by_topic, key=lambda t: -len(by_topic[t]))
    i = 0
    while len(sampled) < n_queries and any(by_topic.values()):
        topic = topics[i % len(topics)]
        if by_topic[topic]:
            sampled.append(by_topic[topic].pop())
        i += 1
    return sampled[:n_queries]


def build_freshqa(raw_path: Path) -> list[dict] | None:
    """Parse the FreshQA sheet export if present (volatility-labeled queries).

    Returns None when the raw file is missing. Kept minimal - these queries
    feed the FUTURE volatility-TTL feature, not this session's evals.
    """
    if not raw_path.exists():
        return None
    # The sheet export has a warning banner line and a blank line before the
    # real header, so the header is on file line 3 (header=2).
    df = pd.read_csv(raw_path, header=2)
    out: list[dict] = []
    for _, row in df.iterrows():
        if str(row.get("split", "")).upper() != "TEST":
            continue
        if str(row.get("false_premise", "")).upper() == "TRUE":
            continue
        answers = [
            str(row[f"answer_{k}"]).strip()
            for k in range(10)
            if f"answer_{k}" in row and pd.notna(row[f"answer_{k}"])
        ]
        if not answers:
            continue
        out.append({
            "id": f"freshqa-{int(row['id']):04d}",
            "query": str(row["question"]).strip(),
            "answers": answers,
            "volatility": str(row.get("fact_type", "unknown")).strip(),
            "effective_year": str(row.get("effective_year", "")).strip(),
            "source": "freshqa",
        })
    return out


def _write_meta(path: Path, meta: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    """Build all datasets and their meta sidecars."""
    today = date.today().isoformat()

    # SimpleQA sample first: the matcher set's factoid-positive slice is
    # derived from these queries plus their hand-written paraphrases.
    queries = sample_simpleqa(RAW_DIR / "simple_qa_test_set.csv", N_SIMPLEQA, SEED)
    by_id = {q["id"]: q for q in queries}

    adversarial = [
        {"id": p["id"], "q1": p["q1"], "q2": p["q2"],
         "label": p.get("label", 0), "source": "handwritten", "note": p["note"]}
        for p in ADVERSARIAL_PAIRS
    ]
    factoid_neg = [
        {"id": p["id"], "q1": p["q1"], "q2": p["q2"],
         "label": 0, "source": "factoid-negative", "note": p["note"]}
        for p in FACTOID_NEGATIVE_PAIRS
    ]
    # Factoid positives: the live-suite paraphrase pairs. This is the
    # on-domain slice - thresholds tuned only on QQP misjudge entity-heavy
    # factoid queries (measured: quora CE underconfident on these).
    factoid_pos = [
        {"id": f"fact-pos-{base_id}", "q1": by_id[base_id]["query"], "q2": text,
         "label": 1, "source": "simpleqa-paraphrase", "note": None}
        for base_id, text in sorted(PARAPHRASES.items()) if base_id in by_id
    ]
    cross_form = [
        {"id": p["id"], "q1": p["q1"], "q2": p["q2"], "label": p["label"],
         "source": "cross-form", "note": p.get("note")}
        for p in CROSS_FORM_PAIRS
    ]
    qqp_pairs = sample_qqp_pairs(RAW_DIR / "qqp_validation.parquet", N_QQP_POS, N_QQP_NEG, SEED)
    matcher_pairs = qqp_pairs + adversarial + factoid_neg + factoid_pos + cross_form
    write_jsonl(OUT_DIR / "matcher_pairs.jsonl", matcher_pairs)
    _write_meta(OUT_DIR / "matcher_pairs.meta.json", {
        "sources": {
            "qqp": "https://huggingface.co/datasets/nyu-mll/glue (validation split)",
            "handwritten": "adversarial pairs written for this repo",
            "factoid-negative": "factoid entity/year/role swaps written for this repo",
            "simpleqa-paraphrase": "SimpleQA queries (MIT) + paraphrases written for this repo",
            "cross-form": "question<->keyword-form pairs; first 10 positives are real pairs from a 2026-07-18 live run",
        },
        "license": "QQP per original Quora terms; SimpleQA MIT; handwritten pairs same license as repo",
        "retrieved": today, "seed": SEED,
        "counts": {
            "qqp_pos": N_QQP_POS, "qqp_hard_neg": N_QQP_NEG,
            "handwritten": len(adversarial), "factoid_neg": len(factoid_neg),
            "factoid_pos": len(factoid_pos), "cross_form": len(cross_form),
            "total": len(matcher_pairs),
        },
        "build_command": "python evals/build_datasets.py",
    })
    print(f"matcher_pairs.jsonl: {len(matcher_pairs)} pairs")
    missing = sorted(set(PARAPHRASES) - set(by_id))
    if missing:
        print(f"WARNING: PARAPHRASES references unsampled ids: {missing}")
    for base_id, text in sorted(PARAPHRASES.items()):
        base = by_id.get(base_id)
        if base is None:
            continue
        queries.append({
            "id": f"{base_id}-p1", "query": text, "answers": base["answers"],
            "topic": base["topic"], "answer_type": base["answer_type"],
            "paraphrase_of": base_id, "source": "handwritten-paraphrase",
        })
    write_jsonl(OUT_DIR / "live_queries.jsonl", queries)
    _write_meta(OUT_DIR / "live_queries.meta.json", {
        "sources": {
            "simpleqa": "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv",
            "handwritten-paraphrase": "paraphrases written for this repo",
        },
        "license": "SimpleQA: MIT (openai/simple-evals)",
        "retrieved": today, "seed": SEED,
        "counts": {
            "base": N_SIMPLEQA, "paraphrases": len(PARAPHRASES),
            "total": len(queries),
        },
        "build_command": "python evals/build_datasets.py",
    })
    print(f"live_queries.jsonl: {len(queries)} rows ({len(PARAPHRASES)} paraphrases)")

    freshqa = build_freshqa(RAW_DIR / "freshqa.csv")
    if freshqa is not None:
        write_jsonl(OUT_DIR / "freshqa_queries.jsonl", freshqa)
        _write_meta(OUT_DIR / "freshqa_queries.meta.json", {
            "sources": {"freshqa": "https://github.com/freshllms/freshqa (sheet snapshot)"},
            "license": "FreshQA: Creative Commons (see repo)",
            "retrieved": today, "seed": SEED,
            "counts": {"total": len(freshqa)},
            "build_command": "python evals/build_datasets.py",
        })
        print(f"freshqa_queries.jsonl: {len(freshqa)} rows (for future TTL feature)")
    else:
        print("freshqa raw file missing - skipped (non-fatal)")


if __name__ == "__main__":
    main()
