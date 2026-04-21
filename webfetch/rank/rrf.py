"""
Reciprocal Rank Fusion (RRF).

Combines multiple ranked lists into one without requiring score normalization
between rankers. Each chunk's fused score is:

    rrf(d) = sum over rankers of 1 / (k + rank(d))

where rank(d) is the 1-based position of chunk d in each ranker's output and
k is a smoothing constant (60, from Cormack et al. 2009, widely used in prod).

Why RRF over raw-score fusion:
  - BM25 scores are unbounded positive numbers (log-scale term frequencies).
  - Cosine similarities are bounded in [-1, 1].
  - Cross-encoder scores are logits with model-specific ranges.
  Normalizing these to be comparable is fragile. RRF sidesteps the problem
  entirely by using only ranks, not scores. Used in production at Elastic
  and Cohere.
"""

from webfetch.rank.base import Chunk

# Smoothing constant from the original Cormack et al. 2009 paper.
# Higher k reduces the dominance of the top-ranked items across lists.
RRF_K: int = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[Chunk]],
    k: int = RRF_K,
) -> list[Chunk]:
    """Fuse multiple ranked chunk lists into one using RRF.

    Args:
        ranked_lists: Each list is a ranking of chunks (best first). Chunks
                      are deduplicated by `.text` - if the same chunk appears
                      in two lists its RRF scores sum.
        k: Smoothing constant. Default 60 is the standard value.

    Returns:
        A single ranked list sorted by descending fused RRF score. Each
        returned chunk has `.score` set to its RRF score.
    """
    # Dedupe by chunk text - two rankers producing the same chunk should
    # reinforce each other, not double-count as separate results.
    fused: dict[str, tuple[Chunk, float]] = {}

    for ranked in ranked_lists:
        for rank_idx, chunk in enumerate(ranked, start=1):
            key = chunk.text
            rrf_contribution = 1.0 / (k + rank_idx)
            if key in fused:
                existing_chunk, existing_score = fused[key]
                fused[key] = (existing_chunk, existing_score + rrf_contribution)
            else:
                fused[key] = (chunk, rrf_contribution)

    results: list[Chunk] = []
    for chunk, score in fused.values():
        chunk.score = score
        results.append(chunk)

    results.sort(key=lambda c: c.score, reverse=True)
    return results
