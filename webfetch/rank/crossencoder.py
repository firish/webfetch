"""
Cross-encoder pairwise reranking - Stage 3 of the reranker cascade.

Unlike bi-encoders (which encode query and chunks separately), cross-encoders
score (query, chunk) pairs jointly through the model. This sees the two texts
attending to each other token-by-token, which is much more accurate for
semantic relevance but O(N) model passes instead of O(1) encoding of the query.

Only ever runs on the small top-K from stage 2 because of the cost: scoring
100 chunks is ~100x slower than the bi-encoder on the same set.

Optional dep: pip install webfetch[rerank]
"""

import logging

from webfetch.config import CROSSENCODER_MODEL, CROSSENCODER_TOP_K
from webfetch.rank.base import AbstractRanker, Chunk

logger = logging.getLogger(__name__)


class CrossEncoderRanker(AbstractRanker):
    """Cross-encoder pairwise reranker.

    Default model (ms-marco-MiniLM-L-6-v2) is trained on the MS MARCO passage
    ranking dataset and is the standard lightweight cross-encoder for retrieval.

    Args:
        model_name: HuggingFace model name for the cross-encoder.
        top_k: Number of chunks to keep after ranking.
    """

    def __init__(
        self,
        model_name: str = CROSSENCODER_MODEL,
        top_k: int = CROSSENCODER_TOP_K,
    ) -> None:
        self._model_name = model_name
        self._top_k = top_k
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
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
            return chunks[: self._top_k]

        # Cross-encoder scores each (query, chunk) pair as a single forward pass.
        pairs = [(query, c.text) for c in chunks]
        scores = model.predict(pairs, show_progress_bar=False)

        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        return sorted(chunks, key=lambda c: c.score, reverse=True)[: self._top_k]
