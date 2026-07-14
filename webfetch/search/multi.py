"""
Multi-engine search fusion via Reciprocal Rank Fusion.

Queries several search adapters in parallel and fuses their ranked URL
lists: score(url) = sum over engines of 1 / (RRF_K + rank). URLs that
multiple engines agree on rise; the union catches what any single engine
misses. Same RRF as rank/rrf.py (Cormack et al. 2009), keyed by URL
instead of chunk text.

Second payoff is resilience: an engine that errors or rate-limits is
logged and skipped, so fused search degrades instead of failing - it only
raises when EVERY engine fails.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit, urlunsplit

from webfetch.config import FUSION_PEER_EMPTY_MIN
from webfetch.rank.rrf import RRF_K
from webfetch.search.base import AbstractSearchAdapter, SearchResult
from webfetch.search.resilience import CircuitBreaker

logger = logging.getLogger(__name__)


def _normalize_url(url: str) -> str:
    """Canonical URL form for cross-engine deduplication.

    Lowercases scheme/host, drops fragments and trailing slashes - engines
    disagree on these cosmetics for the same page.
    """
    parts = urlsplit(url)
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path,
                       parts.query, ""))


class MultiSearchAdapter(AbstractSearchAdapter):
    """Fan out one query to several adapters and RRF-fuse the results.

    Args:
        adapters: The engines to query. Order only matters for tie-breaks.
        max_workers: Thread pool size for the parallel fan-out.

    Raises:
        ValueError: If constructed with no adapters.
    """

    def __init__(self, adapters: Sequence[AbstractSearchAdapter],
                 max_workers: int = 4) -> None:
        if not adapters:
            raise ValueError("MultiSearchAdapter needs at least one adapter")
        self._adapters = list(adapters)
        self._max_workers = max_workers
        # Per-engine circuit breakers: a repeatedly failing/blocked engine
        # gets benched instead of adding its timeout to every query.
        self._breakers = {a.provider_name: CircuitBreaker()
                          for a in self._adapters}

    def search(self, query: str, n_results: int = 10) -> list[SearchResult]:
        """Query all healthy engines in parallel and return the fused top.

        Each engine is asked for n_results; the fused list is trimmed back
        to n_results. Engine failures are logged and skipped. Breaker
        bookkeeping: an exception is always a failure; an EMPTY response
        counts as a failure only when a peer returned results for the same
        query (silent-block signature, e.g. DDG's fingerprint-block 202) -
        a hard query that empties every engine benches nobody.

        Args:
            query: The search query string.
            n_results: Maximum number of fused results to return.

        Returns:
            RRF-fused SearchResults, best first (rank re-assigned 1..n).

        Raises:
            RuntimeError: If every engine failed or is benched.
        """
        active = [a for a in self._adapters
                  if self._breakers[a.provider_name].allow()]
        for a in self._adapters:
            if a not in active:
                logger.info("fusion: %s benched (breaker open) - skipping",
                            a.provider_name)
        if not active:
            # Every breaker open: probe everyone rather than fail the query.
            active = self._adapters

        def _one(adapter: AbstractSearchAdapter) -> list[SearchResult] | None:
            try:
                return adapter.search(query, n_results=n_results)
            except Exception as exc:
                logger.warning("fusion: %s failed (%s: %s) - skipping",
                               adapter.provider_name, type(exc).__name__, exc)
                return None  # None = raised; [] = clean empty

        with ThreadPoolExecutor(
            max_workers=min(self._max_workers, len(active))
        ) as pool:
            raw = list(pool.map(_one, active))

        peer_max = max((len(r) for r in raw if r), default=0)
        for adapter, results in zip(active, raw):
            breaker = self._breakers[adapter.provider_name]
            if results:
                breaker.record_success()
            elif results is None or peer_max >= FUSION_PEER_EMPTY_MIN:
                breaker.record_failure()
                if results is not None:
                    logger.warning(
                        "fusion: %s returned 0 results while peers had %d - "
                        "likely silent block", adapter.provider_name, peer_max)

        per_engine = [r for r in raw if r]
        if not per_engine:
            raise RuntimeError(
                f"all {len(active)} search engines failed for {query!r}"
            )

        scores: dict[str, float] = {}
        best: dict[str, SearchResult] = {}
        for results in per_engine:
            for r in results:
                key = _normalize_url(r.url)
                scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + r.rank)
                # Keep the representative from the engine that ranked it best.
                if key not in best or r.rank < best[key].rank:
                    best[key] = r

        fused = sorted(scores, key=lambda k: -scores[k])[:n_results]
        return [
            SearchResult(url=best[k].url, title=best[k].title,
                         snippet=best[k].snippet, rank=i + 1)
            for i, k in enumerate(fused)
        ]

    @property
    def provider_name(self) -> str:
        """Stable cache-key component listing the fused engines."""
        return "multi(" + "+".join(a.provider_name for a in self._adapters) + ")"
