"""
Hybrid first-stage ranker: BM25 + bi-encoder fused via RRF.

Replaces the BM25-first cascade gate. The old cascade dropped chunks that
were semantically relevant but shared no keywords with the query (measured:
3 of 10 diagnosed recall misses died at the BM25 top-20 gate despite the
answer being in our chunks). Fusing full-list BM25 and bi-encoder rankings
with Reciprocal Rank Fusion means a chunk only needs to rank well under
EITHER signal to survive to the cross-encoder.

Measured on the gap-1 experiment (50 queries, identical chunks):
recall 46% -> 54% at identical token cost, +~2.2s ranking latency from
bi-encoding all chunks (fetch latency still dominates end to end).

Degrades to BM25-only ordering without sentence-transformers, mirroring
the other encoder stages.
"""

from __future__ import annotations

import logging

from webfetch.config import HYBRID_FUSION_TOP_K
from webfetch.rank.base import AbstractRanker, Chunk
from webfetch.rank.biencoder import BiEncoderRanker
from webfetch.rank.bm25 import BM25Ranker
from webfetch.rank.rrf import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

# Effectively "no trim" for the full-list component rankings.
_ALL = 10**9

_embeddings_available: bool | None = None


def _check_embeddings() -> bool:
    """One-time check for the optional rerank extra."""
    global _embeddings_available
    if _embeddings_available is None:
        try:
            import sentence_transformers  # noqa: F401
            _embeddings_available = True
        except ImportError:
            logger.warning("sentence-transformers not installed - hybrid "
                           "ranking degrades to BM25 only. Install webfetch-llm[rerank].")
            _embeddings_available = False
    return _embeddings_available


class HybridRanker(AbstractRanker):
    """Fuse full-list BM25 and bi-encoder rankings, keep the top fused chunks.

    Args:
        top_k: Fused chunks to pass on (to the cross-encoder stage).
        bi_model: Bi-encoder model name (defaults to the shared config model).
    """

    def __init__(self, top_k: int = HYBRID_FUSION_TOP_K,
                 bi_model: str | None = None) -> None:
        self._top_k = top_k
        self._bm25 = BM25Ranker(top_k=_ALL)
        self._bi = (BiEncoderRanker(model_name=bi_model, top_k=_ALL)
                    if bi_model else BiEncoderRanker(top_k=_ALL))

    def rank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Return the top fused chunks for the query.

        Args:
            query: The user query.
            chunks: All candidate chunks.

        Returns:
            Top chunks by RRF score over (BM25 rank, bi-encoder rank),
            or BM25-only top chunks when embeddings are unavailable.
        """
        if not chunks:
            return []
        bm_ranked = self._bm25.rank(query, chunks)
        if not _check_embeddings():
            # Fusing BM25 with raw input order would dilute the lexical
            # signal - degrade to pure BM25 ordering instead.
            return bm_ranked[: self._top_k]
        bi_ranked = self._bi.rank(query, chunks)
        return reciprocal_rank_fusion([bm_ranked, bi_ranked])[: self._top_k]


__all__ = ["HybridRanker"]
