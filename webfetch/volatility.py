"""
Freshness (volatility) classifier for cache TTL selection.

Hybrid classifier picked by the eval bake-off (evals/run_volatility_eval.py):
keyword cue rules decide when a cue fires, a nearest-centroid embedding
classifier decides the uncued remainder. Re-baked 2026-07-18 on FreshQA +
a hand-labeled tech-release/listicle slice (413 queries) after a live run
showed versioned-release and listicle queries bucketed realtime: hybrid
won again (accuracy 0.588, realtime recall 0.705 on the merged test
split), and the retrained centroids now classify release-notes queries
stable and listicles recent instead of realtime.

The centroids ship as a ~25KB JSON artifact (webfetch/data/) derived from
FreshQA (CC) + the tech slice - the library never ships the datasets. Without the
rerank extra the centroid stage drops out and uncued queries fall back to
DEFAULT_FRESHNESS, which is exactly the measured keywords-only behavior.

This classifier is only the FALLBACK: in tool mode the calling model's
`freshness` hint takes precedence (it has conversation context we do not).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path

from webfetch.config import BIENCODER_MODEL, DEFAULT_FRESHNESS

logger = logging.getLogger(__name__)

_CENTROIDS_PATH = Path(__file__).resolve().parent / "data" / "volatility_centroids.json"

# Cue rules, checked in order: realtime first (the expensive class to miss),
# then stable (clearly historical). Mirrors evals/run_volatility_eval.py.
# 2026-07-18 repair: year mentions ("in 2026") only count as STABLE cues
# for PAST years - during 2026 they are currency markers, and the old rule
# gave listicles like "tools to use instead of Docker in 2026" a 90-day
# TTL. Measured on FreshQA+tech (413 queries): fixes the one query it
# targets, changes nothing else. Known limitation kept deliberately:
# "this year" stays a realtime cue because FreshQA news/sports queries
# need it - a listicle carrying that phrase gets a shorter TTL than ideal.
_REALTIME_RE = re.compile(
    r"\b(latest|newest|most recent|current|currently|today|tonight|right now|"
    r"this (week|month|year|season)|so far|as of|price|stock|score|standings|"
    r"ranking|rank|record for|now)\b", re.IGNORECASE)
_STABLE_RE = re.compile(
    r"\b(first|invented|founded|discovered|original|originally|history of|"
    r"was born|born in|died|\d{4}s)\b", re.IGNORECASE)
# Compared in code - a "past years" regex alternation breaks at decade
# boundaries.
_YEAR_MENTION_RE = re.compile(r"\b(?:in|of) (\d{4})\b", re.IGNORECASE)

_bi_model = None
_centroids: dict[str, list[float]] | None = None
_centroid_unavailable = False
# Lazy init must be locked: concurrent pipeline calls (fetch threads, eval
# workers) racing SentenceTransformer construction crash inside torch.
_init_lock = threading.Lock()


def _centroid_class(query: str) -> str | None:
    """Nearest-centroid class, or None when the rerank extra is missing."""
    global _bi_model, _centroids, _centroid_unavailable
    if _centroid_unavailable:
        return None
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning(
            "sentence-transformers not installed - freshness classification "
            "degrades to keyword rules only. Install webfetch-llm[rerank]."
        )
        _centroid_unavailable = True
        return None
    with _init_lock:
        if _centroids is None:
            with open(_CENTROIDS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("model") != BIENCODER_MODEL:
                logger.warning("volatility centroids were built with %s, config "
                               "uses %s - regenerate via evals/run_volatility_eval.py",
                               data.get("model"), BIENCODER_MODEL)
            _centroids = data["classes"]
        if _bi_model is None:
            _bi_model = SentenceTransformer(BIENCODER_MODEL)
    vec = _bi_model.encode([query], normalize_embeddings=True,
                           show_progress_bar=False)[0]
    sims = {c: float(np.dot(vec, np.asarray(v, dtype=np.float32)))
            for c, v in _centroids.items()}
    return max(sims, key=sims.get)


def classify_freshness(query: str) -> str:
    """Classify a query's answer volatility for TTL selection.

    Args:
        query: The search query.

    Returns:
        One of "realtime", "recent", "stable". Falls back to
        DEFAULT_FRESHNESS when no cue fires and embeddings are unavailable.
    """
    if _REALTIME_RE.search(query):
        return "realtime"
    if _STABLE_RE.search(query):
        return "stable"
    year = _YEAR_MENTION_RE.search(query)
    if year and int(year.group(1)) < time.localtime().tm_year:
        return "stable"
    return _centroid_class(query) or DEFAULT_FRESHNESS


__all__ = ["classify_freshness"]
