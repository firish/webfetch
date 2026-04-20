"""
Abstract interface for page fetchers.

Every concrete fetcher (HTML, PDF) implements AbstractFetcher.fetch().
The pipeline calls this interface only - swapping fetcher implementations
requires zero changes to pipeline.py.
"""

from abc import ABC, abstractmethod


class AbstractFetcher(ABC):
    """Base class for all page fetchers.

    A fetcher takes a URL and returns extracted plain text (or None if
    the page could not be fetched or yielded no usable content).
    """

    @abstractmethod
    def fetch(self, url: str) -> str | None:
        """Fetch a URL and return extracted plain text.

        Args:
            url: The page URL to fetch and extract.

        Returns:
            Extracted text as a string, or None if the page could not be
            fetched or contained no extractable content.
        """
        ...
