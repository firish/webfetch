"""
Abstract interface for search adapters.

Every concrete adapter (DuckDuckGo, Brave, Serper) must implement
AbstractSearchAdapter.search(). The pipeline only ever calls this interface,
so swapping providers requires zero changes to pipeline.py.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SearchResult:
    """A single result returned by a search adapter.

    Attributes:
        url: The page URL.
        title: Page title as returned by the search engine.
        snippet: Short excerpt shown in search results (not full page text).
        rank: 1-based position in the original search result list.
    """

    url: str
    title: str
    snippet: str
    rank: int


class AbstractSearchAdapter(ABC):
    """Base class for all search provider adapters.

    Subclasses must implement `search()`. All other pipeline stages
    consume `list[SearchResult]` regardless of which adapter produced them.
    """

    @abstractmethod
    def search(self, query: str, n_results: int = 10) -> list[SearchResult]:
        """Run a search query and return the top results.

        Args:
            query: The search query string.
            n_results: Maximum number of results to return.

        Returns:
            A list of SearchResult objects, ordered by relevance (rank=1 is best).
        """
        ...

    @property
    def provider_name(self) -> str:
        """Identifier used as part of the cache key.

        Defaults to the class name. Override if the class name is not stable.
        """
        return self.__class__.__name__
