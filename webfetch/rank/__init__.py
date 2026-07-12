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

# Shared default instances. Constructors are cheap, but the encoder models
# lazy-load on first use and are cached ON THE INSTANCE - constructing new
# rankers per call (the old behavior) silently re-downloaded/reloaded ~80MB
# models on every query. Reusing these keeps models warm across calls.
_BM25 = BM25Ranker()
_BIENCODER = BiEncoderRanker()
_CROSSENCODER = CrossEncoderRanker()


def default_rankers(
    use_biencoder: bool = True,
    use_crossencoder: bool = True,
) -> list[AbstractRanker]:
    """Return the default ranking cascade as shared ranker instances.

    Args:
        use_biencoder: Include the bi-encoder stage. Requires `webfetch[rerank]`.
        use_crossencoder: Include the cross-encoder stage. Requires `webfetch[rerank]`.

    Returns:
        Rankers in cascade order (BM25 first), ready to apply sequentially.
    """
    rankers: list[AbstractRanker] = [_BM25]
    if use_biencoder:
        rankers.append(_BIENCODER)
    if use_crossencoder:
        rankers.append(_CROSSENCODER)
    return rankers


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
    for ranker in default_rankers(use_biencoder, use_crossencoder):
        chunks = ranker.rank(query, chunks)
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
