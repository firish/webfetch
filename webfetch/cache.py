"""
Transparent cache layer for the webfetch pipeline.

Two cache layers, both in one sqlite database:

- pages:   url -> extracted page text. The big win - fetching is the slowest
           pipeline stage and page text is reusable across different queries
           that hit the same URL.
- queries: query key -> final ranked chunks. Makes repeated identical queries
           (same query + provider + n_results) near-instant.

The pipeline works identically with or without a cache (pass cache=None), so
this layer is purely an optimization - never a behavior change. Failed
fetches are NOT cached: a dead URL today might work tomorrow, and caching
None would hide recoverable failures for the whole TTL.

Backend is stdlib sqlite3 rather than diskcache - one less dependency, and
the access pattern (key lookup, small rows, low write volume) needs nothing
more. WAL mode + a single lock keeps it safe under the concurrent fetch pool.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

from webfetch.config import (
    DEFAULT_CACHE_DB,
    DEFAULT_CACHE_TTL_DAYS,
    DEFAULT_FRESHNESS,
    TTL_BY_FRESHNESS,
)
from webfetch.rank.base import Chunk


@dataclass
class CacheMatch:
    """A successful query-cache lookup with provenance.

    Attributes:
        chunks: The cached ranked chunks.
        kind: "exact" (same normalized query) or "semantic" (paraphrase
            match via the semantic cache).
        matched_query: The cached query text that matched. None for exact
            matches (it is the incoming query itself).
        similarity: Bi-encoder cosine to the matched query (semantic only).
        age_secs: Seconds since the cached entry was stored, if known.
    """

    chunks: list[Chunk]
    kind: str
    matched_query: str | None = None
    similarity: float | None = None
    age_secs: float | None = None
    freshness: str | None = None


def make_query_key(query: str, provider: str, n_results: int) -> str:
    """Build a deterministic cache key for a search query.

    Args:
        query: The search query string (whitespace-normalized, lowercased -
            "Fluke 87V" and "fluke 87v " should hit the same entry).
        provider: Search provider name (different providers return different
            URLs, so results are not interchangeable).
        n_results: Number of search results requested.

    Returns:
        A sha256 hex digest usable as a primary key.
    """
    normalized = " ".join(query.lower().split())
    raw = f"{normalized}|{provider}|{n_results}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AbstractCache(ABC):
    """Interface for pipeline caches.

    Two layers: page text (keyed by URL) and final ranked chunks (keyed by
    a query key from `make_query_key`). Implementations decide storage and
    expiry; the pipeline only calls these four methods.
    """

    @abstractmethod
    def get_page(self, url: str) -> str | None:
        """Return cached extracted text for a URL, or None on miss/expiry."""
        ...

    @abstractmethod
    def set_page(self, url: str, text: str) -> None:
        """Cache extracted text for a URL."""
        ...

    @abstractmethod
    def get_chunks(self, key: str) -> list[Chunk] | None:
        """Return cached ranked chunks for a query key, or None on miss/expiry."""
        ...

    @abstractmethod
    def set_chunks(self, key: str, chunks: list[Chunk]) -> None:
        """Cache the final ranked chunks for a query key."""
        ...

    def lookup(self, query: str, provider: str, n_results: int,
               freshness: str | None = None) -> CacheMatch | None:
        """Look up cached chunks for a query, with provenance.

        Default implementation is exact-match only with flat TTL (the
        freshness hint is ignored). SqliteCache and semantic caches override
        this with per-class TTLs - the pipeline calls only this method, so
        cache implementations stay swappable.

        Args:
            query: The incoming search query.
            provider: Search provider name (part of the cache key).
            n_results: Requested result count (part of the cache key).
            freshness: Optional caller-side volatility class; implementations
                may use it to tighten expiry (min of stored and hinted TTL).

        Returns:
            A CacheMatch on hit, None on miss.
        """
        chunks = self.get_chunks(make_query_key(query, provider, n_results))
        if chunks is None:
            return None
        return CacheMatch(chunks=chunks, kind="exact")

    def store(self, query: str, provider: str, n_results: int,
              chunks: list[Chunk], freshness: str | None = None) -> None:
        """Store the final ranked chunks for a query.

        Args:
            query: The search query that produced these chunks.
            provider: Search provider name.
            n_results: Requested result count.
            chunks: Final ranked chunks to cache.
            freshness: Volatility class to store with the row (drives its
                TTL on later reads). Ignored by the default implementation.
        """
        self.set_chunks(make_query_key(query, provider, n_results), chunks)


class SqliteCache(AbstractCache):
    """Sqlite-backed cache with lazy TTL expiry.

    Expired rows are deleted on read rather than by a background sweeper -
    simpler, and stale rows that are never read again cost only disk space.

    Args:
        db_path: Path to the sqlite file. "~" is expanded; parent dirs are
            created. Defaults to config.DEFAULT_CACHE_DB.
        ttl_days: Rows older than this are treated as misses and deleted.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_CACHE_DB,
        ttl_days: int = DEFAULT_CACHE_TTL_DAYS,
    ) -> None:
        path = os.path.expanduser(db_path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        self._ttl_secs = ttl_days * 86400
        # check_same_thread=False + one lock: fetch workers write pages from
        # pool threads. A single connection behind a lock is plenty for this
        # write volume and avoids per-thread connection management.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            # WAL lets readers proceed during writes and is more crash-safe.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS pages ("
                "url TEXT PRIMARY KEY, text TEXT NOT NULL, created_at REAL NOT NULL)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS queries ("
                "key TEXT PRIMARY KEY, chunks_json TEXT NOT NULL, "
                "created_at REAL NOT NULL, freshness TEXT)"
            )
            # Migration for cache dbs created before volatility-aware TTLs:
            # NULL freshness rows are treated as DEFAULT_FRESHNESS on read.
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(queries)")}
            if "freshness" not in cols:
                self._conn.execute("ALTER TABLE queries ADD COLUMN freshness TEXT")
            # Lifetime usage counters (cost receipts) - live with the cache
            # because it is the one durable file webfetch owns.
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS stats ("
                "key TEXT PRIMARY KEY, value REAL NOT NULL)"
            )
            self._conn.commit()

    def bump_stats(self, **deltas: float) -> None:
        """Accumulate usage counters (see webfetch.receipts).

        Args:
            **deltas: Counter name -> increment, e.g.
                bump_stats(searches_total=1, cache_hits_exact=1).
        """
        with self._lock:
            for key, delta in deltas.items():
                self._conn.execute(
                    "INSERT INTO stats (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = value + ?",
                    (key, float(delta), float(delta)),
                )
            self._conn.commit()

    def get_stats(self) -> dict[str, float]:
        """Return all lifetime usage counters."""
        with self._lock:
            return dict(self._conn.execute("SELECT key, value FROM stats"))

    def _get_with_age(self, table: str, key_col: str, value_col: str,
                      key: str) -> tuple[str, float] | None:
        """Shared read path: (value, age_secs) or None, deleting expired rows."""
        with self._lock:
            row = self._conn.execute(
                f"SELECT {value_col}, created_at FROM {table} WHERE {key_col} = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            value, created_at = row
            age = time.time() - created_at
            if age > self._ttl_secs:
                self._conn.execute(f"DELETE FROM {table} WHERE {key_col} = ?", (key,))
                self._conn.commit()
                return None
            return value, age

    def _get(self, table: str, key_col: str, value_col: str, key: str) -> str | None:
        """Shared read path: return the value or None, deleting expired rows."""
        hit = self._get_with_age(table, key_col, value_col, key)
        return hit[0] if hit is not None else None

    def _effective_ttl_secs(self, stored: str | None, hint: str | None) -> float:
        """Per-class TTL: min(stored class, caller hint, flat ceiling).

        Resolved at read time so retuning TTL_BY_FRESHNESS applies to rows
        already cached. Legacy rows (NULL freshness) use DEFAULT_FRESHNESS.
        """
        ttl = float(TTL_BY_FRESHNESS.get(stored or DEFAULT_FRESHNESS, self._ttl_secs))
        if hint in TTL_BY_FRESHNESS:
            ttl = min(ttl, TTL_BY_FRESHNESS[hint])
        return min(ttl, self._ttl_secs)

    def _get_query_row(self, key: str, hint: str | None = None,
                       ) -> tuple[str, float, str | None] | None:
        """Read a queries row with per-class expiry.

        A row is DELETED only when expired for its OWN stored class - a
        stricter caller hint produces a miss but leaves the row intact,
        since other callers may still be served by it.

        Returns:
            (chunks_json, age_secs, stored_freshness) or None on miss/expiry.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT chunks_json, created_at, freshness FROM queries WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            value, created_at, stored = row
            age = time.time() - created_at
            if age > self._effective_ttl_secs(stored, None):
                self._conn.execute("DELETE FROM queries WHERE key = ?", (key,))
                self._conn.commit()
                return None
            if age > self._effective_ttl_secs(stored, hint):
                return None
            return value, age, stored

    def _set(self, table: str, key_col: str, value_col: str, key: str,
             value: str) -> None:
        """Shared write path: upsert with a fresh timestamp."""
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO {table} ({key_col}, {value_col}, created_at) "
                "VALUES (?, ?, ?)",
                (key, value, time.time()),
            )
            self._conn.commit()

    def get_page(self, url: str) -> str | None:
        """Return cached extracted text for a URL, or None on miss/expiry."""
        return self._get("pages", "url", "text", url)

    def set_page(self, url: str, text: str) -> None:
        """Cache extracted text for a URL."""
        self._set("pages", "url", "text", url, text)

    def get_chunks(self, key: str) -> list[Chunk] | None:
        """Return cached ranked chunks for a query key, or None on miss/expiry."""
        hit = self._get_query_row(key)
        if hit is None:
            return None
        return [Chunk(**d) for d in json.loads(hit[0])]

    def lookup(self, query: str, provider: str, n_results: int,
               freshness: str | None = None) -> CacheMatch | None:
        """Exact-match lookup with per-class TTL and age provenance."""
        hit = self._get_query_row(
            make_query_key(query, provider, n_results), hint=freshness)
        if hit is None:
            return None
        raw, age, stored = hit
        return CacheMatch(
            chunks=[Chunk(**d) for d in json.loads(raw)],
            kind="exact", age_secs=age, freshness=stored,
        )

    def store(self, query: str, provider: str, n_results: int,
              chunks: list[Chunk], freshness: str | None = None) -> None:
        """Store chunks with their volatility class."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO queries "
                "(key, chunks_json, created_at, freshness) VALUES (?, ?, ?, ?)",
                (make_query_key(query, provider, n_results),
                 json.dumps([asdict(c) for c in chunks]), time.time(), freshness),
            )
            self._conn.commit()

    def set_chunks(self, key: str, chunks: list[Chunk]) -> None:
        """Cache the final ranked chunks for a query key."""
        self._set("queries", "key", "chunks_json", key,
                  json.dumps([asdict(c) for c in chunks]))

    def close(self) -> None:
        """Close the underlying sqlite connection."""
        with self._lock:
            self._conn.close()


__all__ = ["AbstractCache", "CacheMatch", "SqliteCache", "make_query_key"]
