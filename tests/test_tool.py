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


def test_full_results_skips_compression_and_raises_budget(monkeypatch):
    # Compression enabled and booby-trapped: the flag must never call it.
    monkeypatch.setattr(tool, "COMPRESSION_ENABLED", True)

    def boom(*a, **k):
        raise AssertionError("compress_chunks called despite full_results")
    monkeypatch.setattr(tool, "compress_chunks", boom)

    long_text = "Item one. " * 300  # ~3000 chars, exceeds the 8000 default
    r = _result(chunks=[Chunk(text=long_text, url=f"https://x.com/{i}",
                              title="X") for i in range(4)])
    out = tool.handle_web_search({"query": "top 10 things",
                                  "full_results": True},
                                 pipeline=StubPipeline(r))
    # Larger budget: all four 3k-char chunks fit (12k > 8000 default).
    assert out.count("Item one.") >= 4 * 100


def test_save_finding_roundtrip_shows_distrust_header():
    class RecordingPipeline(StubPipeline):
        def store_chunks(self, query, chunks, freshness=None):
            self.stored = (query, chunks, freshness)

    pipe = RecordingPipeline()
    msg = tool.handle_save_finding(
        {"query": "capital of atlantis", "content": "It is Poseidonia.",
         "source_url": "https://example.com/atlantis"}, pipeline=pipe)
    assert "unverified" in msg
    query, chunks, freshness = pipe.stored
    assert chunks[0].url.startswith("model-finding://")
    assert freshness == tool.SAVE_FINDING_FRESHNESS

    # A later lookup that returns this entry gets the distrust header,
    # and the content is served verbatim (no compression).
    r = _result(from_cache=True, cache_kind="exact", cache_age_secs=120,
                freshness="recent", chunks=chunks)
    out = tool.handle_web_search({"query": "capital of atlantis"},
                                 pipeline=StubPipeline(r))
    assert "model-contributed finding" in out
    assert "UNVERIFIED" in out and "force_fresh" in out
    assert "Poseidonia" in out


def test_save_finding_validation_and_kill_switch(monkeypatch):
    out = tool.handle_save_finding({"query": "", "content": "x"},
                                   pipeline=StubPipeline())
    assert out.startswith("save_finding error")
    monkeypatch.setattr(tool, "SAVE_FINDING_ENABLED", False)
    out = tool.handle_save_finding({"query": "q", "content": "x"},
                                   pipeline=StubPipeline())
    assert "disabled" in out
