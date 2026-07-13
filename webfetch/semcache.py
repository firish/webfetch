"""
Semantic query cache: serve cached results for paraphrased queries.

Extends the exact-match SqliteCache with a two-stage paraphrase matcher,
mirroring the ranking cascade's cheap-then-precise philosophy:

1. Bi-encoder shortlist: cosine similarity between the incoming query's
   embedding and every cached query's stored embedding. Cheap (one dot
   product) but imprecise - it cannot tell "python 3.12" from "python 3.13"
   (cosine 0.91). Gate at SEMCACHE_BI_THRESHOLD keeps paraphrase recall high.
2. NLI verification: the candidate pair runs jointly through an NLI
   cross-encoder; a paraphrase must be BIDIRECTIONALLY entailing (min of
   P(entailment) in both directions >= SEMCACHE_CE_THRESHOLD). Entailment
   training natively rejects entity/number/time swaps - the failure class
   that makes naive semantic caches dangerous.

Thresholds and the verifier model were selected by the eval harness
(evals/run_matcher_eval.py): precision 0.955 with zero trusted-negative
false positives at bi>=0.60 + NLI>=0.97. A lexical conflict veto was tested
and rejected (removed no false positives, killed true paraphrases).

Requires the optional rerank extra (sentence-transformers). Without it, the
cache logs one warning and degrades to exact-match behavior - the pipeline
works identically either way.
"""

from __future__ import annotations

import json
import logging
import threading
import time

# numpy ships with the optional rerank extra (alongside sentence-transformers).
# Import lazily-tolerant so `import webfetch` works on core deps only - the
# cache degrades to exact-match when the extra is missing.
try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only without [rerank]
    np = None

from webfetch.cache import CacheMatch, SqliteCache, make_query_key
from webfetch.config import (
    BIENCODER_MODEL,
    DEFAULT_CACHE_DB,
    DEFAULT_CACHE_TTL_DAYS,
    SEMCACHE_BI_THRESHOLD,
    SEMCACHE_CE_MODEL,
    SEMCACHE_CE_THRESHOLD,
    SEMCACHE_MAX_CANDIDATES,
)
from webfetch.rank.base import Chunk

logger = logging.getLogger(__name__)


class SemanticSqliteCache(SqliteCache):
    """SqliteCache with semantic (paraphrase) fallback on query lookups.

    The embedding matrix for cached queries is held in memory (loaded from
    sqlite lazily, appended on store) so a lookup is one matrix-vector dot
    product. Assumes a single process owns the cache file - concurrent
    writers from other processes will not be visible until restart.

    Args:
        db_path: Sqlite file path (shared with the exact/page layers).
        ttl_days: Row TTL, as in SqliteCache.
        bi_threshold: Cosine gate for the shortlist stage.
        ce_threshold: Bidirectional-entailment gate for verification.
        bi_model: sentence-transformers bi-encoder name (shared with the
            ranking cascade's default).
        ce_model: NLI cross-encoder name.
        max_candidates: How many shortlist candidates to verify, best first.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_CACHE_DB,
        ttl_days: int = DEFAULT_CACHE_TTL_DAYS,
        bi_threshold: float = SEMCACHE_BI_THRESHOLD,
        ce_threshold: float = SEMCACHE_CE_THRESHOLD,
        bi_model: str = BIENCODER_MODEL,
        ce_model: str = SEMCACHE_CE_MODEL,
        max_candidates: int = SEMCACHE_MAX_CANDIDATES,
    ) -> None:
        super().__init__(db_path=db_path, ttl_days=ttl_days)
        self._bi_threshold = bi_threshold
        self._ce_threshold = ce_threshold
        self._bi_model_name = bi_model
        self._ce_model_name = ce_model
        self._max_candidates = max_candidates
        self._bi_model = None
        self._ce_model = None
        self._models_unavailable = False
        self._model_lock = threading.Lock()
        # In-memory mirror of query_embeddings: parallel lists + row matrix.
        self._emb_keys: list[str] | None = None
        self._emb_queries: list[str] = []
        self._emb_matrix: np.ndarray | None = None
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS query_embeddings ("
                "key TEXT PRIMARY KEY, query TEXT NOT NULL, "
                "vector BLOB NOT NULL, created_at REAL NOT NULL)"
            )
            self._conn.commit()

    # --- model loading (lazy, degrade gracefully like the rankers) ---

    def _load_models(self) -> bool:
        """Load bi-encoder + NLI verifier once; False if deps are missing.

        Locked: concurrent lookups racing lazy model construction crash
        inside torch. Separate lock from the sqlite one so a slow model
        load never blocks page-cache writes from fetch workers.
        """
        if self._models_unavailable:
            return False
        if self._bi_model is not None and self._ce_model is not None:
            return True
        try:
            if np is None:
                raise ImportError("numpy not installed")
            from sentence_transformers import CrossEncoder, SentenceTransformer
        except ImportError:
            logger.warning(
                "sentence-transformers not installed - semantic cache "
                "degrades to exact-match only. Install webfetch[rerank]."
            )
            self._models_unavailable = True
            return False
        with self._model_lock:
            if self._bi_model is None:
                self._bi_model = SentenceTransformer(self._bi_model_name)
            if self._ce_model is None:
                self._ce_model = CrossEncoder(self._ce_model_name)
        return True

    def _embed(self, query: str) -> np.ndarray:
        vec = self._bi_model.encode([query], normalize_embeddings=True,
                                    show_progress_bar=False)[0]
        return np.asarray(vec, dtype=np.float32)

    def _entailment_prob(self, q1: str, q2: str) -> float:
        """Min of P(entailment) across both directions, in (0, 1).

        Paraphrase is approximated as bidirectional entailment: each query
        must entail the other. NLI heads emit per-class logits; softmax only
        when the output is not already a probability distribution.
        """
        raw = np.asarray(self._ce_model.predict(
            [(q1, q2), (q2, q1)], show_progress_bar=False))
        if raw.ndim != 2:
            # Single-score head (not an NLI model): use the score directly,
            # sigmoid-squashing raw logits.
            scores = raw if (raw.min() >= 0 and raw.max() <= 1) \
                else 1 / (1 + np.exp(-raw))
            return float(scores.min())
        label2id = {k.lower(): v
                    for k, v in self._ce_model.model.config.label2id.items()}
        ent = label2id.get("entailment", 1)
        rows_are_probs = np.all(raw >= 0) and np.allclose(raw.sum(axis=1), 1.0, atol=1e-3)
        if not rows_are_probs:
            e = np.exp(raw - raw.max(axis=1, keepdims=True))
            raw = e / e.sum(axis=1, keepdims=True)
        return float(raw[:, ent].min())

    # --- embedding matrix bookkeeping ---

    def _ensure_matrix(self) -> None:
        """Load the cached-query embedding matrix from sqlite once."""
        if self._emb_keys is not None:
            return
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, query, vector FROM query_embeddings"
            ).fetchall()
        self._emb_keys = [r[0] for r in rows]
        self._emb_queries = [r[1] for r in rows]
        self._emb_matrix = (
            np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
            if rows else None
        )

    def _append_embedding(self, key: str, query: str, vec: np.ndarray) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO query_embeddings "
                "(key, query, vector, created_at) VALUES (?, ?, ?, ?)",
                (key, query, vec.tobytes(), time.time()),
            )
            self._conn.commit()
        if self._emb_keys is None:
            return  # matrix not loaded yet; sqlite is the source of truth
        if key in self._emb_keys:
            i = self._emb_keys.index(key)
            self._emb_matrix[i] = vec
            self._emb_queries[i] = query
        else:
            self._emb_keys.append(key)
            self._emb_queries.append(query)
            self._emb_matrix = (
                np.vstack([self._emb_matrix, vec])
                if self._emb_matrix is not None else vec.reshape(1, -1)
            )

    def _drop_embedding(self, key: str) -> None:
        """Remove an embedding whose chunks row expired."""
        with self._lock:
            self._conn.execute("DELETE FROM query_embeddings WHERE key = ?", (key,))
            self._conn.commit()
        if self._emb_keys is not None and key in self._emb_keys:
            i = self._emb_keys.index(key)
            self._emb_keys.pop(i)
            self._emb_queries.pop(i)
            self._emb_matrix = np.delete(self._emb_matrix, i, axis=0)

    # --- public interface ---

    def store(self, query: str, provider: str, n_results: int,
              chunks: list[Chunk], freshness: str | None = None) -> None:
        """Store chunks (exact layer, with class) plus the query's embedding."""
        super().store(query, provider, n_results, chunks, freshness=freshness)
        if not self._load_models():
            return
        key = make_query_key(query, provider, n_results)
        self._append_embedding(key, query, self._embed(query))

    def lookup(self, query: str, provider: str, n_results: int,
               freshness: str | None = None) -> CacheMatch | None:
        """Exact match first; on miss, verified semantic match.

        The freshness hint tightens expiry on both paths - a semantic hit
        against a stale-for-its-class row is a miss.
        """
        exact = super().lookup(query, provider, n_results, freshness=freshness)
        if exact is not None:
            return exact
        if not self._load_models():
            return None
        self._ensure_matrix()
        if self._emb_matrix is None or len(self._emb_keys) == 0:
            return None

        vec = self._embed(query)
        sims = self._emb_matrix @ vec
        order = np.argsort(-sims)[: self._max_candidates]
        for i in order:
            cos = float(sims[i])
            if cos < self._bi_threshold:
                break  # candidates are sorted - nothing below clears the gate
            cached_query = self._emb_queries[i]
            key = self._emb_keys[i]
            if self._entailment_prob(query, cached_query) < self._ce_threshold:
                continue
            hit = self._get_query_row(key, hint=freshness)
            if hit is None:
                # Chunks row expired - clean up the orphaned embedding.
                self._drop_embedding(key)
                continue
            return CacheMatch(
                chunks=[Chunk(**d) for d in json.loads(hit[0])],
                kind="semantic", matched_query=cached_query,
                similarity=cos, age_secs=hit[1], freshness=hit[2],
            )
        return None


__all__ = ["SemanticSqliteCache"]
