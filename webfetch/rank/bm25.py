"""
BM25 keyword ranking - Stage 1 of the reranker cascade.

Cheap, fast, CPU-only. Scores every chunk against the query using the BM25
Okapi bag-of-words algorithm and trims to the top K for semantic reranking.

Why BM25 first, not a semantic model:
  - Runs in milliseconds over thousands of chunks. A bi-encoder would take
    seconds; a cross-encoder would take minutes.
  - Catches the easy wins: exact model numbers, unit strings, brand names.
    For T&M specs, an exact match on "87V" is the strongest possible signal
    and BM25 handles that trivially.
  - Anything BM25 misses semantically (e.g. query "uncertainty" vs chunk
    "accuracy tolerance") gets recovered by the bi-encoder stage.
"""

import logging
import re

from rank_bm25 import BM25Okapi

from webfetch.config import BM25_TOP_K
from webfetch.rank.base import AbstractRanker, Chunk

logger = logging.getLogger(__name__)

# Token regex: alphanumeric runs with internal `.`, `-`, `_`, `/` preserved.
# This keeps tokens like "0.1", "87V", "ip67", "mitutoyo-500", "+/-0.05" mostly
# intact, which matters for T&M pages where numeric ranges and part numbers
# are the most distinctive query terms. A plain \w+ tokenizer would split
# "0.1" into "0" and "1", destroying the signal.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-_/][a-z0-9]+)*", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Ranker(AbstractRanker):
    """BM25 Okapi keyword ranker.

    Args:
        top_k: Number of chunks to keep after ranking.
    """

    def __init__(self, top_k: int = BM25_TOP_K) -> None:
        self._top_k = top_k

    def rank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            # Degenerate query (e.g. all punctuation) - BM25 would error.
            # Return the first K chunks unchanged so the cascade continues.
            logger.warning("Empty query tokens after tokenization: %r", query)
            return chunks[: self._top_k]

        tokenized_chunks = [_tokenize(c.text) for c in chunks]

        # rank-bm25 builds its index in-place during construction. Cheap to
        # rebuild per query because the chunk set changes per query anyway.
        bm25 = BM25Okapi(tokenized_chunks)
        scores = bm25.get_scores(query_tokens)

        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        return sorted(chunks, key=lambda c: c.score, reverse=True)[: self._top_k]
