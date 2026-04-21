"""
Ranking stage public interface.

`rank()` runs the full cascade: BM25 (always) -> bi-encoder (optional) ->
cross-encoder (optional). Each stage trims the candidate list before passing
the survivors to the (more expensive) next stage.

Typical usage:

    from webfetch.rank import chunk_text, rank

    chunks = []
    for url, page_text, title in fetched_pages:
        chunks.extend(chunk_text(page_text, url=url, title=title))

    top = rank("Fluke 87V accuracy spec", chunks)
"""

from webfetch.rank.base import AbstractRanker, Chunk
from webfetch.rank.biencoder import BiEncoderRanker
from webfetch.rank.bm25 import BM25Ranker
from webfetch.rank.chunker import chunk_text
from webfetch.rank.crossencoder import CrossEncoderRanker
from webfetch.rank.rrf import reciprocal_rank_fusion


def rank(
    query: str,
    chunks: list[Chunk],
    use_biencoder: bool = True,
    use_crossencoder: bool = True,
) -> list[Chunk]:
    """Run the ranking cascade and return the final top chunks.

    Args:
        query: The user query.
        chunks: All candidate chunks (typically hundreds from fetched pages).
        use_biencoder: Run the bi-encoder stage. Requires `webfetch[rerank]`.
                       Silently skipped if the dep is missing.
        use_crossencoder: Run the cross-encoder stage. Requires `webfetch[rerank]`.
                          Silently skipped if the dep is missing.

    Returns:
        The top chunks from the final stage of the cascade, ordered best-first.
    """
    # Stage 1: BM25 always runs - it's cheap and gives a huge pre-filter win.
    chunks = BM25Ranker().rank(query, chunks)

    if use_biencoder:
        chunks = BiEncoderRanker().rank(query, chunks)

    if use_crossencoder:
        chunks = CrossEncoderRanker().rank(query, chunks)

    return chunks


__all__ = [
    "rank",
    "chunk_text",
    "reciprocal_rank_fusion",
    "Chunk",
    "AbstractRanker",
    "BM25Ranker",
    "BiEncoderRanker",
    "CrossEncoderRanker",
]
