"""Freshness keyword rules: cue behavior without model downloads.

The centroid stage is monkeypatched out so these run offline - they pin
the KEYWORD layer's contracts, including the 2026-07-18 year-mention
repair (past years are historical, current/future years are not).
"""

import pytest

import webfetch.volatility as vol


@pytest.fixture(autouse=True)
def no_centroids(monkeypatch):
    monkeypatch.setattr(vol, "_centroid_class", lambda q: None)


def test_realtime_cues():
    assert vol.classify_freshness("current price of Bitcoin in USD") == "realtime"
    assert vol.classify_freshness("latest SpaceX launch news") == "realtime"


def test_stable_cues():
    assert vol.classify_freshness("who invented the telephone") == "stable"
    assert vol.classify_freshness("music of the 1980s") == "stable"


def test_past_year_is_stable_current_year_is_not():
    assert vol.classify_freshness("who won the World Cup in 2010") == "stable"
    # Current/future years are currency markers, not history markers -
    # the old rule gave this a 90-day TTL (real bug, 2026-07-18 run).
    import time
    year = time.localtime().tm_year
    q = f"tools to use instead of Docker in {year}"
    assert vol.classify_freshness(q) != "stable"
    assert vol.classify_freshness(f"best laptops of {year + 1}") != "stable"


def test_uncued_falls_back_to_default():
    assert vol.classify_freshness("Docker alternatives comparison") == "recent"
