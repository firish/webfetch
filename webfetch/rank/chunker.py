"""
Character-based chunker with whitespace-aware boundaries.

Splits fetched page text into Chunk objects sized for semantic reranking.
400-char chunks (~100 tokens) are small enough for the cross-encoder to score
precisely, but big enough to carry a complete fact like a spec line.

Why character-based rather than sentence or token-based:
  - Sentence splitters (nltk, spacy) add heavy deps and mishandle spec tables
    where "lines" are not sentences (e.g. "Accuracy: +/-0.1%").
  - Token-based requires a tokenizer that matches the downstream reranker.
  - Character-based with whitespace snapping is good enough and zero-deps.
"""

from webfetch.config import CHUNK_OVERLAP_RATIO, DEFAULT_CHUNK_SIZE
from webfetch.rank.base import Chunk

# How far back we're willing to walk to find a whitespace break when the
# raw boundary lands mid-word. 50 chars ~ one long word or a few short ones.
_WHITESPACE_SEARCH_WINDOW: int = 50


def chunk_text(
    text: str,
    url: str,
    title: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_ratio: float = CHUNK_OVERLAP_RATIO,
) -> list[Chunk]:
    """Split text into overlapping Chunk objects.

    Args:
        text: The extracted page text to chunk.
        url: Source URL (propagated to every Chunk for provenance).
        title: Source page title (propagated to every Chunk).
        chunk_size: Target chunk length in characters.
        overlap_ratio: Fraction of chunk_size to overlap between neighbors.
                       10% means the last ~40 chars of chunk N repeat as the
                       first ~40 chars of chunk N+1 - avoids a key phrase
                       being split across the chunk boundary and ranked poorly.

    Returns:
        List of Chunks. Empty list if `text` is empty or all whitespace.
    """
    if not text or not text.strip():
        return []

    overlap = int(chunk_size * overlap_ratio)
    chunks: list[Chunk] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Snap `end` back to the nearest whitespace, but only within a small
        # window - we don't want to shrink the chunk dramatically.
        if end < text_len:
            window_start = max(start + 1, end - _WHITESPACE_SEARCH_WINDOW)
            space_idx = text.rfind(" ", window_start, end)
            if space_idx != -1:
                end = space_idx

        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, url=url, title=title))

        if end >= text_len:
            break

        next_start = end - overlap
        # Guard: if overlap or whitespace-snap math ever fails to advance,
        # force progress by 1 char to avoid an infinite loop.
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks
