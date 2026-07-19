"""Cost receipts: counter math and report formatting."""

from webfetch.cache import SqliteCache
from webfetch.receipts import CHARS_PER_TOKEN, get_counters, savings_report


def test_missing_db_returns_empty_and_notice(tmp_path):
    db = str(tmp_path / "nope.db")
    assert get_counters(db) == {}
    assert "no searches recorded" in savings_report(db_path=db)


def test_report_math(tmp_path):
    db = str(tmp_path / "cache.db")
    c = SqliteCache(db_path=db)
    c.bump_stats(searches_total=1, fresh_searches=1)
    c.bump_stats(searches_total=1, cache_hits_exact=1)
    c.bump_stats(searches_total=1, cache_hits_semantic=1)
    c.bump_stats(tool_chars_returned=3 * 1400 * CHARS_PER_TOKEN)

    rep = savings_report(db_path=db, hosted_fee=0.01,
                         hosted_tokens_per_call=17400,
                         token_price_per_mtok=5.0)
    assert "searches served:      3" in rep
    assert "exact 1, semantic 1" in rep
    assert "= $0.03" in rep  # 3 x $0.01 search fees
    tok_saving = (3 * 17400 - 3 * 1400) * 5.0 / 1e6
    assert f"~${tok_saving:.2f}" in rep


def test_session_baseline_delta(tmp_path):
    db = str(tmp_path / "cache.db")
    c = SqliteCache(db_path=db)
    c.bump_stats(searches_total=10, fresh_searches=8, cache_hits_exact=2,
                 tool_chars_returned=10 * 1400 * CHARS_PER_TOKEN)
    baseline = get_counters(db)
    c.bump_stats(searches_total=3, fresh_searches=1, cache_hits_exact=1,
                 cache_hits_semantic=1,
                 tool_chars_returned=3 * 1400 * CHARS_PER_TOKEN)

    rep = savings_report(db_path=db, baseline=baseline)
    assert "(this session)" in rep
    assert "searches served:      3" in rep
    assert "exact 1, semantic 1" in rep  # key absent in baseline -> full value
    assert "lifetime: 13 searches" in rep

    # No activity since the snapshot -> quiet session, lifetime intact.
    rep2 = savings_report(db_path=db, baseline=get_counters(db))
    assert "no searches recorded this session" in rep2
