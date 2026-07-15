"""Search resilience: circuit breaker, fallback failover, fusion benching."""

import pytest

from webfetch.search.base import AbstractSearchAdapter, SearchResult
from webfetch.search.multi import MultiSearchAdapter
from webfetch.search.resilience import CircuitBreaker, FallbackSearchAdapter


def results(n: int) -> list[SearchResult]:
    return [SearchResult(url=f"https://x.com/{i}", title="t", snippet="s",
                         rank=i + 1) for i in range(n)]


class FakeAdapter(AbstractSearchAdapter):
    """Scripted adapter: each call pops the next scripted response."""

    def __init__(self, name: str, script: list, default=None):
        self.name = name
        self.script = list(script)
        self.default = default if default is not None else results(5)
        self.calls = 0

    def search(self, query: str, n_results: int = 10):
        self.calls += 1
        out = self.script.pop(0) if self.script else self.default
        if isinstance(out, Exception):
            raise out
        return out

    @property
    def provider_name(self) -> str:
        return self.name


def test_breaker_opens_after_threshold_and_closes_on_success():
    b = CircuitBreaker(threshold=3, cooldown_secs=60)
    assert b.allow()
    b.record_failure()
    b.record_failure()
    assert b.allow()
    b.record_failure()
    assert not b.allow()
    b.record_success()
    assert b.allow()


def test_breaker_half_open_after_cooldown():
    b = CircuitBreaker(threshold=1, cooldown_secs=0.0)
    b.record_failure()
    assert b.allow()  # cooldown elapsed -> probe allowed


def test_fallback_on_exception():
    f = FallbackSearchAdapter([FakeAdapter("a", [RuntimeError("boom")]),
                               FakeAdapter("b", [results(4)])])
    assert len(f.search("q")) == 4


def test_fallback_on_silent_empty():
    f = FallbackSearchAdapter([FakeAdapter("a", [[]]),
                               FakeAdapter("b", [results(3)])])
    assert len(f.search("q")) == 3


def test_all_clean_empty_returns_empty_not_raise():
    f = FallbackSearchAdapter([FakeAdapter("a", [[]]), FakeAdapter("b", [[]])])
    assert f.search("q") == []


def test_all_raise_raises():
    f = FallbackSearchAdapter([FakeAdapter("a", [RuntimeError()],
                                           default=RuntimeError())])
    with pytest.raises(RuntimeError):
        f.search("q")


def test_fallback_benches_failing_engine():
    a = FakeAdapter("a", [[], [], [], results(9)])
    b = FakeAdapter("b", [], default=results(2))
    f = FallbackSearchAdapter([a, b])
    for _ in range(3):
        f.search("q")
    assert a.calls == 3
    f.search("q")
    assert a.calls == 3  # benched: 4th query skips it


def test_fusion_benches_peer_relative_silent_block():
    blocked = FakeAdapter("blk", [[], [], [], results(9)])
    healthy = FakeAdapter("ok", [], default=results(6))
    m = MultiSearchAdapter([blocked, healthy], max_workers=2)
    for _ in range(3):
        assert m.search("q")
    assert blocked.calls == 3
    m.search("q")
    assert blocked.calls == 3  # benched in fusion
    assert m.provider_name == "multi(blk+ok)"


def test_fallback_provider_name_lists_chain():
    f = FallbackSearchAdapter([FakeAdapter("a", []), FakeAdapter("b", [])])
    assert f.provider_name == "fallback(a>b)"
