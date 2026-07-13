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

from webfetch.rank.rrf import RRF_K
from webfetch.search.base import AbstractSearchAdapter, SearchResult

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

    def search(self, query: str, n_results: int = 10) -> list[SearchResult]:
        """Query all engines in parallel and return the fused top results.

        Each engine is asked for n_results; the fused list is trimmed back
        to n_results. Engine failures are logged and skipped.

        Args:
            query: The search query string.
            n_results: Maximum number of fused results to return.

        Returns:
            RRF-fused SearchResults, best first (rank re-assigned 1..n).

        Raises:
            RuntimeError: If every engine failed.
        """
        def _one(adapter: AbstractSearchAdapter) -> list[SearchResult]:
            try:
                return adapter.search(query, n_results=n_results)
            except Exception as exc:
                logger.warning("fusion: %s failed (%s: %s) - skipping",
                               adapter.provider_name, type(exc).__name__, exc)
                return []

        with ThreadPoolExecutor(
            max_workers=min(self._max_workers, len(self._adapters))
        ) as pool:
            per_engine = list(pool.map(_one, self._adapters))

        if not any(per_engine):
            raise RuntimeError(
                f"all {len(self._adapters)} search engines failed for {query!r}"
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
