"""
Abstract interface for page fetchers.

Every concrete fetcher (HTML, PDF) implements AbstractFetcher.fetch().
The pipeline calls this interface only - swapping fetcher implementations
requires zero changes to pipeline.py.
"""

from abc import ABC, abstractmethod

# Shared by all fetchers. A bare "Mozilla/5.0" UA is trivially fingerprinted
# and got us 403'd by fandom/oup/tiktok in eval runs - a full browser header
# set recovers a measurable share of those pages.
BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0.0.0 Safari/537.36"),
    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
               "image/avif,image/webp,*/*;q=0.8"),
    "Accept-Language": "en-US,en;q=0.9",
}


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
