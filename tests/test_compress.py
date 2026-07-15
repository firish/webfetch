"""Compression: splitter, scorers, guards, safety contracts.

Uses the lexical scorer throughout so no model download is needed - the
selection/guard logic under test is scorer-independent.
"""

import copy

from webfetch.compress import (
    compress_chunks,
    is_table_like,
    score_lexical,
    split_sentences,
    starts_anaphoric,
)
from webfetch.rank.base import Chunk


def test_split_sentences_newlines_and_punctuation():
    sents = split_sentences(
        "First sentence. Second one here.\n| a | 1 |\nfragment without end")
    assert [s for _, s in sents] == [
        "First sentence.", "Second one here.", "| a | 1 |",
        "fragment without end"]
    assert [line for line, _ in sents] == [0, 0, 1, 2]


def test_table_like_detection():
    assert is_table_like("| spec | 42 |")
    assert is_table_like("Accuracy: +/-0.1% at 23C 2024 rev 3")
    assert not is_table_like("The quick brown fox jumps over the lazy dog.")


def test_anaphoric_starts():
    assert starts_anaphoric("It was founded in 1990.")
    assert starts_anaphoric("However, the results differ.")
    assert not starts_anaphoric("Paris is the capital.")


def test_lexical_scorer_prefers_query_terms():
    scores = score_lexical("zephyr capital", [
        "the zephyr project began", "the capital of France",
        "unrelated text entirely"])
    assert scores[0] > scores[2]
    assert scores[1] > scores[2]


def test_keeps_relevant_drops_noise_never_mutates():
    c1 = Chunk(text="The Eiffel Tower is 330 metres tall. Cookies help us "
                    "serve ads. Subscribe to our newsletter today please.",
               url="u1", title="t1", score=0.9)
    c2 = Chunk(text="Totally unrelated filler sentence here. More filler "
                    "that says nothing useful at all.",
               url="u2", title="t2", score=0.5)
    originals = copy.deepcopy([c1, c2])
    out = compress_chunks("how tall is the eiffel tower", [c1, c2],
                          scorer="lexical", policy="topk", param=1,
                          dedup=False)
    assert "Eiffel Tower is 330" in out[0].text
    assert "Cookies" not in out[0].text
    assert len(out) == 2 and out[1].text  # >= 1 sentence per chunk
    assert [c.text for c in (c1, c2)] == [c.text for c in originals]
    assert out[0] is not c1


def test_anaphora_chain_pulls_predecessors():
    c = Chunk(text="Alice founded Acme. This company grew fast. It reached "
                   "the zenith valuation quickly.", url="u", title="t")
    out = compress_chunks("zenith valuation", [c], scorer="lexical",
                          policy="topk", param=1, dedup=False)
    assert "Alice founded Acme" in out[0].text
    assert "This company" in out[0].text


def test_table_guard_keeps_data_lines():
    c = Chunk(text="Some prose about a sensor product line here.\n"
                   "| range | 0-100C |\n| accuracy | 0.1% |",
              url="u", title="t")
    out = compress_chunks("sensor accuracy", [c], scorer="lexical",
                          policy="topk", param=1)
    assert "| accuracy | 0.1% |" in out[0].text


def test_dedup_across_overlapping_chunks():
    dup = "This exact boundary sentence is repeated across chunk overlaps."
    c1 = Chunk(text="Alpha fact about zebras roaming. " + dup, url="u", title="t")
    c2 = Chunk(text=dup + " Beta fact about zebra stripe patterns.",
               url="u", title="t")
    out = compress_chunks("zebra facts", [c1, c2], scorer="lexical",
                          policy="ratio", param=1.0)
    joined = " ".join(c.text for c in out)
    assert joined.count("repeated across chunk overlaps") == 1


def test_edge_cases_never_raise():
    assert compress_chunks("q", []) == []
    assert compress_chunks("q", [Chunk(text="", url="u", title="t")],
                           scorer="lexical") == []
    # Unknown scorer degrades to passthrough - tool-path safety contract.
    chunks = [Chunk(text="Hello world. Bye now.", url="u", title="t")]
    assert compress_chunks("q", chunks, scorer="nonsense") == chunks
