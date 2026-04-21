"""
Bi-encoder semantic ranking - Stage 2 of the reranker cascade.

Encodes the query and each chunk into dense vectors separately, then scores
by cosine similarity. Catches semantic matches that BM25 misses - e.g. query
"uncertainty" matching chunk "measurement tolerance".

Why bi-encoder, not cross-encoder here:
  - Query and chunks are encoded independently: O(1) + O(N) model calls.
  - Cross-encoder would require O(N) pairwise model calls which is ~20x slower.
  - Good enough accuracy for mid-cascade - the cross-encoder stage cleans up
    ordering at the top.

Optional dep: pip install webfetch[rerank]
"""

import logging

from webfetch.config import BIENCODER_MODEL, BIENCODER_TOP_K
from webfetch.rank.base import AbstractRanker, Chunk

logger = logging.getLogger(__name__)


class BiEncoderRanker(AbstractRanker):
    """Bi-encoder (sentence-transformers) semantic ranker.

    The default model (all-MiniLM-L6-v2) is ~80MB, runs CPU-only, and is the
    standard general-purpose bi-encoder. Swap for bge-small-en or similar via
    config if you need higher accuracy at the cost of a larger download.

    Args:
        model_name: HuggingFace model name for the bi-encoder.
        top_k: Number of chunks to keep after ranking.
    """

    def __init__(
        self,
        model_name: str = BIENCODER_MODEL,
        top_k: int = BIENCODER_TOP_K,
    ) -> None:
        self._model_name = model_name
        self._top_k = top_k
        # Lazy-loaded: sentence-transformers import + model load is ~5s
        # and shouldn't happen just because someone imported the class.
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def rank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []

        try:
            model = self._load_model()
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Run: pip install 'webfetch[rerank]'"
            )
            # Passthrough on missing deps - preserves the cascade rather than
            # crashing, and BM25 results are still usable if noisier.
            return chunks[: self._top_k]

        # Normalize embeddings so dot product == cosine similarity.
        query_embedding = model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        chunk_embeddings = model.encode(
            [c.text for c in chunks],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        scores = (chunk_embeddings @ query_embedding).tolist()

        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        return sorted(chunks, key=lambda c: c.score, reverse=True)[: self._top_k]
