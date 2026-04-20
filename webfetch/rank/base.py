"""
Shared data types for the ranking pipeline.

Defined here (not in pipeline.py) to avoid circular imports:
rank submodules import Chunk, and pipeline.py imports both rank submodules
and Chunk - keeping Chunk in a leaf module breaks the cycle.
"""

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
