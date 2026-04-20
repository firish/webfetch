"""
DuckDuckGo search adapter.

Uses the `duckduckgo-search` library (no API key needed). This is the default
adapter for dev use and low-volume production. It hits DDG's internal API, not
an official endpoint, so it can rate-limit under heavy load - use Brave/Serper
for high-volume production.
"""

from duckduckgo_search import DDGS

from webfetch.search.base import AbstractSearchAdapter, SearchResult


class DDGSearchAdapter(AbstractSearchAdapter):
    """Search adapter backed by DuckDuckGo (free, no API key).

    Args:
        region: DDG region code, e.g. "us-en", "wt-wt" (worldwide).
        safesearch: "on", "moderate", or "off".
        timeout: Seconds before the DDG request times out.
    """

    def __init__(
        self,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        timeout: int = 10,
    ) -> None:
        self._region = region
        self._safesearch = safesearch
        self._timeout = timeout

    def search(self, query: str, n_results: int = 10) -> list[SearchResult]:
        """Run a DuckDuckGo text search and return the top results.

        Args:
            query: The search query string.
            n_results: Maximum number of results to return.

        Returns:
            A list of SearchResult objects ordered by relevance (rank=1 is best).
            May return fewer than n_results if DDG returns fewer hits.
        """
        with DDGS(timeout=self._timeout) as ddgs:
            raw = list(
                ddgs.text(
                    query,
                    region=self._region,
                    safesearch=self._safesearch,
                    max_results=n_results,
                )
            )

        return [
            SearchResult(
                url=hit["href"],
                title=hit["title"],
                snippet=hit["body"],
                rank=i + 1,  # 1-based rank, position in DDG result list
            )
            for i, hit in enumerate(raw)
        ]

    @property
    def provider_name(self) -> str:
        return "ddg"
