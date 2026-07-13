"""
Tavily search adapter (https://tavily.com).

LLM-focused search API with a recurring free tier. Chosen as the third
fusion engine: independent of Brave/DDG, reliable REST API, results map
directly onto SearchResult.
"""

from __future__ import annotations

import os

import requests

from webfetch.search.base import AbstractSearchAdapter, SearchResult

_TAVILY_API_URL = "https://api.tavily.com/search"


class TavilySearchAdapter(AbstractSearchAdapter):
    """Search adapter for the Tavily API.

    Args:
        api_key: Tavily API key. Falls back to the TAVILY_API_KEY env var.
        timeout: HTTP timeout in seconds.

    Raises:
        ValueError: If no API key is provided or found in the environment.
    """

    def __init__(self, api_key: str | None = None, timeout: int = 10) -> None:
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Tavily API key required: pass api_key or set TAVILY_API_KEY"
            )
        self._timeout = timeout

    def search(self, query: str, n_results: int = 10) -> list[SearchResult]:
        """Run a Tavily search and return the top results.

        Args:
            query: The search query string.
            n_results: Maximum number of results to return (Tavily max 20).

        Returns:
            SearchResults ordered by Tavily's relevance (rank=1 is best).
        """
        resp = requests.post(
            _TAVILY_API_URL,
            json={
                "api_key": self._api_key,
                "query": query,
                "max_results": min(n_results, 20),
                # Raw content is fetched by our own fetch stage - snippets only.
                "include_raw_content": False,
            },
            timeout=self._timeout,
        )
        resp.raise_for_status()
        hits = resp.json().get("results", [])
        return [
            SearchResult(
                url=h["url"],
                title=h.get("title", ""),
                snippet=h.get("content", ""),
                rank=i + 1,
            )
            for i, h in enumerate(hits[:n_results])
        ]

    @property
    def provider_name(self) -> str:
        return "tavily"
