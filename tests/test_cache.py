"""SqliteCache: roundtrips, volatility TTLs, legacy rows, stats counters."""

import pytest

from webfetch.cache import SqliteCache
from webfetch.config import TTL_BY_FRESHNESS
from webfetch.rank.base import Chunk


@pytest.fixture()
def cache(tmp_path):
    return SqliteCache(db_path=str(tmp_path / "cache.db"))


def _chunks():
    return [Chunk(text="the answer is 42", url="https://x.com", title="X")]


def _backdate(cache, secs: float) -> None:
    """Age every cached query row by `secs` seconds."""
    with cache._lock:
        cache._conn.execute(
            "UPDATE queries SET created_at = created_at - ?", (secs,))
        cache._conn.commit()


def test_exact_roundtrip(cache):
    cache.store("q", "ddg", 10, _chunks(), freshness="recent")
    match = cache.lookup("q", "ddg", 10)
    assert match is not None and match.kind == "exact"
    assert match.chunks[0].text == "the answer is 42"
    assert match.freshness == "recent"
    # Key includes provider and n_results.
    assert cache.lookup("q", "brave", 10) is None
    assert cache.lookup("q", "ddg", 5) is None


def test_pages_roundtrip(cache):
    cache.set_page("https://x.com", "page text")
    assert cache.get_page("https://x.com") == "page text"
    assert cache.get_page("https://y.com") is None


def test_per_class_ttl_expiry(cache):
    cache.store("q", "ddg", 10, _chunks(), freshness="realtime")
    _backdate(cache, TTL_BY_FRESHNESS["realtime"] + 1)
    assert cache.lookup("q", "ddg", 10) is None


def test_stricter_hint_is_a_miss_but_preserves_row(cache):
    cache.store("q", "ddg", 10, _chunks(), freshness="stable")
    _backdate(cache, TTL_BY_FRESHNESS["realtime"] + 1)
    # A realtime hint tightens expiry: min(stored, hint) - miss.
    assert cache.lookup("q", "ddg", 10, freshness="realtime") is None
    # But the row is still valid for its OWN class.
    assert cache.lookup("q", "ddg", 10) is not None


def test_legacy_null_freshness_behaves_as_default(cache):
    cache.store("q", "ddg", 10, _chunks(), freshness=None)
    _backdate(cache, TTL_BY_FRESHNESS["recent"] - 60)
    assert cache.lookup("q", "ddg", 10) is not None
    _backdate(cache, 120)
    assert cache.lookup("q", "ddg", 10) is None


def test_stats_counters(cache):
    assert cache.get_stats() == {}
    cache.bump_stats(searches_total=1, fresh_searches=1)
    cache.bump_stats(searches_total=1, cache_hits_exact=1)
    stats = cache.get_stats()
    assert stats["searches_total"] == 2
    assert stats["fresh_searches"] == 1
    assert stats["cache_hits_exact"] == 1
