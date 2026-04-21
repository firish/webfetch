"""
Shared types and interfaces for the ranking pipeline.

Defined here (not in pipeline.py) to avoid circular imports: rank submodules
import Chunk/AbstractRanker, and pipeline.py imports both rank submodules and
these types - keeping them in a leaf module breaks the cycle.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single piece of extracted page text, ready for ranking.

    Attributes:
        text: The extracted text content of this chunk.
        url: Source URL this chunk was extracted from.
        title: Page title of the source URL.
        score: Relevance score assigned by the most recent ranking stage.
               Higher is better. Reset by each stage that scores.
    """

    text: str
    url: str
    title: str
    score: float = field(default=0.0)


class AbstractRanker(ABC):
    """Base class for all ranking stages (BM25, bi-encoder, cross-encoder).

    A ranker takes a query and list of chunks, scores each chunk, and returns
    the top-K chunks ordered by relevance (highest score first). Each stage
    mutates `chunk.score` before returning so downstream stages can inspect
    the previous stage's signal if needed.
    """

    @abstractmethod
    def rank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Score chunks by relevance to the query and return top-K.

        Args:
            query: The user query.
            chunks: The candidate chunks to score.

        Returns:
            A new list of the top-K chunks, ordered by descending score.
            Input list is not mutated (except chunk.score fields).
        """
        ...
