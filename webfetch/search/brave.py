"""
Brave Search API adapter.

Requires a Brave Search API key: https://brave.com/search/api
Set via env var BRAVE_API_KEY or pass directly to the constructor.

Free tier: 2k queries/month. Paid: ~$5/1k queries.
Recommended for low-to-medium volume production use.
"""

import os

import requests

from webfetch.search.base import AbstractSearchAdapter, SearchResult

_BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"


class BraveSearchAdapter(AbstractSearchAdapter):
    """Search adapter backed by the Brave Search API.

    Args:
        api_key: Brave API key. Defaults to BRAVE_API_KEY env var.
        country: 2-letter country code for result localisation, e.g. "US".
        search_lang: Language code for results, e.g. "en".
        timeout: Seconds before the request times out.

    Raises:
        ValueError: If no API key is provided or found in the environment.
    """

    def __init__(
        self,
        api_key: str | None = None,
        country: str = "US",
        search_lang: str = "en",
        timeout: int = 10,
    ) -> None:
        resolved_key = api_key or os.environ.get("BRAVE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Brave API key required. Pass api_key= or set BRAVE_API_KEY env var."
            )
        self._api_key = resolved_key
        self._country = country
        self._search_lang = search_lang
        self._timeout = timeout

    def search(self, query: str, n_results: int = 10) -> list[SearchResult]:
        """Run a Brave web search and return the top results.

        Args:
            query: The search query string.
            n_results: Maximum number of results to return (Brave max is 20).

        Returns:
            A list of SearchResult objects ordered by relevance (rank=1 is best).
        """
        response = requests.get(
            _BRAVE_API_URL,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self._api_key,
            },
            params={
                "q": query,
                "count": min(n_results, 20),  # Brave caps at 20 per request
                "country": self._country,
                "search_lang": self._search_lang,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()

        web_results = response.json().get("web", {}).get("results", [])

        return [
            SearchResult(
                url=hit["url"],
                title=hit["title"],
                snippet=hit.get("description", ""),
                rank=i + 1,
            )
            for i, hit in enumerate(web_results)
        ]

    @property
    def provider_name(self) -> str:
        return "brave"
