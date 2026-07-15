"""
Top-level pipeline orchestrator: search -> fetch -> chunk -> rank [-> extract].

Two entry points:

- Pipeline.search_chunks(query): tool mode. Returns ranked, source-labeled
  chunks for a calling LLM to reason over (the model IS the extractor).
- Pipeline.run(query, keys, extractor): structured extraction mode. Runs
  search_chunks() then a cheap LLM extraction call returning JSON.

Every stage is injected as an abstract interface (composition), so swapping
a search provider, ranker cascade, or cache requires zero changes here.
Error boundary: search adapter exceptions PROPAGATE from this layer -
callers embedding this in an agent loop should catch at the tool handler
(see webfetch/tool.py). Per-URL fetch failures are skipped and reported in
SearchChunksResult.failed_urls instead of raising.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from webfetch.cache import AbstractCache
from webfetch.config import (
    DEFAULT_FETCH_WORKERS,
    DEFAULT_N_RESULTS,
    DEFAULT_TOKEN_BUDGET,
)
from webfetch.extract.base import AbstractExtractor
from webfetch.fetch import fetch_all
from webfetch.rank import AbstractRanker, Chunk, chunk_text, default_rankers
from webfetch.search import AbstractSearchAdapter, SearchResult, get_search_adapter

logger = logging.getLogger(__name__)


@dataclass
class SearchChunksResult:
    """Everything produced by one search_chunks() run.

    Attributes:
        query: The query that was searched.
        chunks: Final ranked chunks, best first.
        results: Raw search hits (empty on a query-level cache hit - the
            cached chunks already carry their source url/title).
        failed_urls: URLs whose fetch returned no text and were skipped.
        from_cache: True if the whole result came from the query-level cache.
        elapsed_secs: Wall time for the run.
    """

    query: str
    chunks: list[Chunk]
    results: list[SearchResult] = field(default_factory=list)
    failed_urls: list[str] = field(default_factory=list)
    from_cache: bool = False
    elapsed_secs: float = 0.0
    # Cache provenance: kind is "exact" | "semantic" | None (fresh run);
    # matched_query is the cached query a semantic hit matched against.
    cache_kind: str | None = None
    matched_query: str | None = None
    cache_age_secs: float | None = None
    # Volatility class used for this run (hint, classifier, or stored).
    freshness: str | None = None


class Pipeline:
    """Orchestrates search -> fetch -> chunk -> rank with optional caching.

    All collaborators are injected; defaults come from config. The cache is
    transparent - passing cache=None gives identical results, just slower.

    Args:
        search: Search adapter. Defaults to get_search_adapter() (config
            provider, currently DDG).
        rankers: Ranking cascade applied in order. Defaults to the shared
            default cascade controlled by use_biencoder/use_crossencoder.
        cache: Optional cache layer (page text + final chunks).
        n_results: Search results to retrieve per query.
        max_workers: Thread pool size for concurrent URL fetching.
        use_biencoder: Include the bi-encoder stage in the default cascade.
            Ignored if `rankers` is given.
        use_crossencoder: Include the cross-encoder stage in the default
            cascade. Ignored if `rankers` is given.
    """

    def __init__(
        self,
        search: AbstractSearchAdapter | None = None,
        rankers: Sequence[AbstractRanker] | None = None,
        cache: AbstractCache | None = None,
        n_results: int = DEFAULT_N_RESULTS,
        max_workers: int = DEFAULT_FETCH_WORKERS,
        use_biencoder: bool = True,
        use_crossencoder: bool = True,
    ) -> None:
        self._search = search if search is not None else get_search_adapter()
        self._rankers = (
            list(rankers)
            if rankers is not None
            else default_rankers(use_biencoder, use_crossencoder)
        )
        self._cache = cache
        self._n_results = n_results
        self._max_workers = max_workers

    def bump_stats(self, **deltas: float) -> None:
        """Accumulate usage counters for cost receipts (webfetch.receipts).

        No-op without a stats-capable cache - receipts are an optional
        byproduct of the cache file, never a pipeline requirement.
        """
        bump = getattr(self._cache, "bump_stats", None)
        if bump is not None:
            bump(**deltas)

    def search_chunks(self, query: str, n_results: int | None = None,
                      use_cache: bool = True,
                      freshness: str | None = None) -> SearchChunksResult:
        """Run search -> fetch (concurrent, cached) -> chunk -> rank.

        Args:
            query: The search query.
            n_results: Override the instance default for this call.
            use_cache: When False, skip the query-cache lookup but still
                store the fresh result (so a forced refresh updates the
                cache). Page-level caching is unaffected.
            freshness: Volatility class ("realtime" | "recent" | "stable")
                controlling cache TTL. When None, the library classifier
                decides. A caller hint is authoritative - in tool mode the
                calling model has context the classifier lacks.

        Returns:
            A SearchChunksResult with ranked chunks and run metadata.

        Raises:
            Whatever the search adapter raises on network/API failure -
            this is the library layer; agent-loop callers should catch at
            the tool handler.
        """
        start = time.perf_counter()
        n = n_results if n_results is not None else self._n_results

        if freshness is None and self._cache is not None:
            from webfetch.volatility import classify_freshness
            freshness = classify_freshness(query)

        if self._cache is not None and use_cache:
            match = self._cache.lookup(query, self._search.provider_name, n,
                                       freshness=freshness)
            if match is not None:
                self.bump_stats(searches_total=1,
                                **{f"cache_hits_{match.kind}": 1})
                return SearchChunksResult(
                    query=query,
                    chunks=match.chunks,
                    from_cache=True,
                    elapsed_secs=time.perf_counter() - start,
                    cache_kind=match.kind,
                    matched_query=match.matched_query,
                    cache_age_secs=match.age_secs,
                    freshness=match.freshness or freshness,
                )

        results = self._search.search(query, n_results=n)
        by_url = {r.url: r for r in results}

        # Page-level cache: only fetch URLs we have not seen before.
        pages: dict[str, str] = {}
        to_fetch: list[str] = []
        for url in by_url:
            text = self._cache.get_page(url) if self._cache is not None else None
            if text is not None:
                pages[url] = text
            else:
                to_fetch.append(url)
        cached_pages = len(pages)

        fetched = fetch_all(to_fetch, max_workers=self._max_workers)
        failed_urls: list[str] = []
        for url, text in fetched.items():
            if text is None:
                failed_urls.append(url)
                continue
            pages[url] = text
            if self._cache is not None:
                self._cache.set_page(url, text)

        all_chunks: list[Chunk] = []
        for url, text in pages.items():
            r = by_url[url]
            all_chunks.extend(chunk_text(text, url=url, title=r.title))

        chunks = all_chunks
        for ranker in self._rankers:
            chunks = ranker.rank(query, chunks)

        if self._cache is not None:
            self._cache.store(query, self._search.provider_name, n, chunks,
                              freshness=freshness)
        self.bump_stats(searches_total=1, fresh_searches=1,
                        pages_fetched=len(to_fetch),
                        pages_from_cache=cached_pages)

        logger.info(
            "search_chunks(%r): %d results, %d fetched (%d cached, %d failed), "
            "%d chunks -> %d ranked",
            query, len(results), len(to_fetch), cached_pages,
            len(failed_urls), len(all_chunks), len(chunks),
        )
        return SearchChunksResult(
            query=query,
            chunks=chunks,
            results=results,
            failed_urls=failed_urls,
            elapsed_secs=time.perf_counter() - start,
            freshness=freshness,
        )

    def run(
        self,
        query: str,
        keys: dict[str, str],
        extractor: AbstractExtractor,
        budget_chars: int = DEFAULT_TOKEN_BUDGET,
    ) -> dict[str, str | None]:
        """Full pipeline: search_chunks() then structured LLM extraction.

        Args:
            query: The search query.
            keys: Mapping of field_name -> short description of what to
                extract, e.g. {"accuracy": "measurement accuracy with units"}.
            extractor: LLM extraction adapter (Claude, GPT, Gemini, Groq).
            budget_chars: Max characters of ranked context sent to the LLM.

        Returns:
            Dict mapping each requested key to its extracted value (or None).
        """
        result = self.search_chunks(query)
        return extractor.extract(result.chunks, keys, budget_chars=budget_chars)


__all__ = ["Pipeline", "SearchChunksResult"]
