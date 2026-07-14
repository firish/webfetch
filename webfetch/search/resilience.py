"""
Search-engine resilience: circuit breakers and priority failover.

Motivated by DDG's fingerprint-blocking behavior: blocked clients get an
empty HTTP 202 that surfaces as zero results with NO exception, so naive
callers silently return "no results" to the model. (Our TLS story is
handled by the ddgs/primp dependency, which impersonates a Chrome
handshake; this module handles what remains - detecting blocks and routing
around a dying engine.)

Two pieces:
  - CircuitBreaker: after N consecutive failures an engine is benched for a
    cooldown, so callers stop burning latency on timeouts against a dead or
    blocking engine. Thread-safe - fusion calls engines from a pool.
  - FallbackSearchAdapter: tries engines in priority order and fails over
    on exception OR empty results. Empty-as-failure is deliberate: for a
    web-scale engine, zero hits for a reasonable query is far more likely a
    block than a genuinely resultless query - and a true zero-result query
    returns [] from the LAST engine anyway, preserving correct behavior.

MultiSearchAdapter uses the breaker with a stricter failure signal (empty
only counts when peer engines returned results for the same query), since
fusion sees peers and can tell a blocked engine from a hard query.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence

from webfetch.config import (
    SEARCH_BREAKER_COOLDOWN_SECS,
    SEARCH_BREAKER_THRESHOLD,
)
from webfetch.search.base import AbstractSearchAdapter, SearchResult

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Consecutive-failure breaker with a cooldown bench.

    Args:
        threshold: Consecutive failures that open the breaker.
        cooldown_secs: How long an open breaker stays open. After the
            cooldown one call is let through (half-open); its outcome
            closes or re-opens the breaker.
    """

    def __init__(self, threshold: int = SEARCH_BREAKER_THRESHOLD,
                 cooldown_secs: float = SEARCH_BREAKER_COOLDOWN_SECS) -> None:
        self._threshold = threshold
        self._cooldown = cooldown_secs
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    def allow(self) -> bool:
        """Whether a call may proceed (closed, or half-open after cooldown)."""
        with self._lock:
            if self._opened_at is None:
                return True
            if time.monotonic() - self._opened_at >= self._cooldown:
                return True  # half-open probe
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold:
                self._opened_at = time.monotonic()


class FallbackSearchAdapter(AbstractSearchAdapter):
    """Priority-ordered failover across engines with per-engine breakers.

    Unlike MultiSearchAdapter (parallel fan-out + fusion, best quality),
    this calls ONE engine at a time and only moves on when it fails or
    returns nothing - the cheap/free-tier configuration where you want DDG
    to serve everything it can and a keyed engine to catch its blocks.

    Args:
        adapters: Engines in priority order (first = preferred).

    Raises:
        ValueError: If constructed with no adapters.
    """

    def __init__(self, adapters: Sequence[AbstractSearchAdapter]) -> None:
        if not adapters:
            raise ValueError("FallbackSearchAdapter needs at least one adapter")
        self._adapters = list(adapters)
        self._breakers = {a.provider_name: CircuitBreaker() for a in self._adapters}

    def search(self, query: str, n_results: int = 10) -> list[SearchResult]:
        """Return results from the first healthy engine that produces any.

        Args:
            query: The search query string.
            n_results: Maximum number of results to return.

        Returns:
            Results from the first engine that yields any; [] only when the
            last attempted engine legitimately returned zero results.

        Raises:
            RuntimeError: If every engine raised or was benched.
        """
        last_exc: Exception | None = None
        saw_clean_empty = False
        for adapter in self._adapters:
            breaker = self._breakers[adapter.provider_name]
            if not breaker.allow():
                logger.info("fallback: %s benched (breaker open) - skipping",
                            adapter.provider_name)
                continue
            try:
                results = adapter.search(query, n_results=n_results)
            except Exception as exc:
                breaker.record_failure()
                last_exc = exc
                logger.warning("fallback: %s failed (%s: %s) - trying next",
                               adapter.provider_name, type(exc).__name__, exc)
                continue
            if results:
                breaker.record_success()
                return results
            # Empty with no exception: likely a silent block (DDG's 202
            # pattern) - counts as failure and falls through.
            breaker.record_failure()
            saw_clean_empty = True
            logger.warning("fallback: %s returned 0 results - trying next",
                           adapter.provider_name)
        if saw_clean_empty:
            return []
        raise RuntimeError(
            f"all {len(self._adapters)} fallback engines failed for {query!r}"
        ) from last_exc

    @property
    def provider_name(self) -> str:
        """Stable cache-key component - lists the chain, not the server."""
        return "fallback(" + ">".join(a.provider_name for a in self._adapters) + ")"


__all__ = ["CircuitBreaker", "FallbackSearchAdapter"]
