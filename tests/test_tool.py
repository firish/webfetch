"""handle_web_search: provenance headers, error contract, stub pipeline.

Compression is disabled via monkeypatch so these tests run offline with
core deps only - the compression logic has its own suite.
"""

import pytest

import webfetch.tool as tool
from webfetch.pipeline import SearchChunksResult
from webfetch.rank.base import Chunk


class StubPipeline:
    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc
        self.calls = []

    def search_chunks(self, query, use_cache=True, freshness=None):
        self.calls.append({"query": query, "use_cache": use_cache,
                           "freshness": freshness})
        if self.exc:
            raise self.exc
        return self.result

    def bump_stats(self, **deltas):
        pass


def _result(**kw):
    defaults = dict(query="q", chunks=[Chunk(text="The answer is 42.",
                                             url="https://x.com", title="X")])
    defaults.update(kw)
    return SearchChunksResult(**defaults)


@pytest.fixture(autouse=True)
def no_compression(monkeypatch):
    monkeypatch.setattr(tool, "COMPRESSION_ENABLED", False)


def test_fresh_search_header():
    out = tool.handle_web_search({"query": "q"}, pipeline=StubPipeline(_result()))
    assert out.startswith("[fresh search]\n")
    assert "The answer is 42." in out


def test_exact_cache_header_with_age_and_freshness():
    r = _result(from_cache=True, cache_kind="exact", cache_age_secs=7200,
                freshness="recent")
    out = tool.handle_web_search({"query": "q"}, pipeline=StubPipeline(r))
    assert out.startswith("[cache: exact match, 2h old, recent]")


def test_semantic_cache_header_names_matched_query():
    r = _result(from_cache=True, cache_kind="semantic",
                matched_query="original q", cache_age_secs=60,
                freshness="stable")
    out = tool.handle_web_search({"query": "q2"}, pipeline=StubPipeline(r))
    assert '[cache: semantic match to "original q", 1m old, stable]' in out


def test_force_fresh_and_freshness_passthrough():
    stub = StubPipeline(_result())
    tool.handle_web_search({"query": "q", "force_fresh": True,
                            "freshness": "realtime"}, pipeline=stub)
    assert stub.calls[0]["use_cache"] is False
    assert stub.calls[0]["freshness"] == "realtime"
    # Invalid freshness is tolerated, not an error.
    tool.handle_web_search({"query": "q", "freshness": "bogus"}, pipeline=stub)
    assert stub.calls[1]["freshness"] is None


def test_empty_query_and_no_results_messages():
    out = tool.handle_web_search({"query": "  "}, pipeline=StubPipeline())
    assert out.startswith("web_search error: empty query")
    out = tool.handle_web_search({"query": "q"},
                                 pipeline=StubPipeline(_result(chunks=[])))
    assert "No results found" in out


def test_never_raises_on_pipeline_exception():
    out = tool.handle_web_search(
        {"query": "q"}, pipeline=StubPipeline(exc=RuntimeError("engine down")))
    assert out.startswith("web_search error: RuntimeError")
    assert "engine down" in out
