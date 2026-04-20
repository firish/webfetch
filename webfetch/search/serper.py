"""
Serper (Google Search) adapter.

Requires a Serper API key: https://serper.dev
Set via env var SERPER_API_KEY or pass directly to the constructor.

Free tier: 2500 queries one-time on signup. Paid: ~$1/1k queries.
Recommended when Google result quality matters (e.g. obscure T&M equipment specs).
"""

import os

import requests

from webfetch.search.base import AbstractSearchAdapter, SearchResult

_SERPER_API_URL = "https://google.serper.dev/search"


class SerperSearchAdapter(AbstractSearchAdapter):
    """Search adapter backed by Serper (Google Search).

    Args:
        api_key: Serper API key. Defaults to SERPER_API_KEY env var.
        country: 2-letter country code for result localisation, e.g. "us".
        language: Language code for results, e.g. "en".
        timeout: Seconds before the request times out.

    Raises:
        ValueError: If no API key is provided or found in the environment.
    """

    def __init__(
        self,
        api_key: str | None = None,
        country: str = "us",
        language: str = "en",
        timeout: int = 10,
    ) -> None:
        resolved_key = api_key or os.environ.get("SERPER_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Serper API key required. Pass api_key= or set SERPER_API_KEY env var."
            )
        self._api_key = resolved_key
        self._country = country
        self._language = language
        self._timeout = timeout

    def search(self, query: str, n_results: int = 10) -> list[SearchResult]:
        """Run a Google search via Serper and return the top results.

        Args:
            query: The search query string.
            n_results: Maximum number of results to return (Serper max is 100).

        Returns:
            A list of SearchResult objects ordered by relevance (rank=1 is best).
        """
        response = requests.post(
            _SERPER_API_URL,
            headers={
                "X-API-KEY": self._api_key,
                "Content-Type": "application/json",
            },
            json={
                "q": query,
                "num": min(n_results, 100),
                "gl": self._country,
                "hl": self._language,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()

        organic = response.json().get("organic", [])

        return [
            SearchResult(
                url=hit["link"],
                title=hit["title"],
                snippet=hit.get("snippet", ""),
                rank=i + 1,
            )
            for i, hit in enumerate(organic)
        ]

    @property
    def provider_name(self) -> str:
        return "serper"
