"""Chunker: character windows, whitespace snapping, overlap."""

from webfetch.rank.chunker import chunk_text


def test_empty_and_whitespace_input():
    assert chunk_text("", url="u", title="t") == []
    assert chunk_text("   \n  ", url="u", title="t") == []


def test_short_text_single_chunk():
    chunks = chunk_text("hello world", url="u", title="t")
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].url == "u" and chunks[0].title == "t"


def test_chunks_cover_text_with_overlap():
    words = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(words, url="u", title="t", chunk_size=400,
                        overlap_ratio=0.10)
    assert len(chunks) > 1
    # Every chunk respects the size bound (plus slack for word snapping).
    assert all(len(c.text) <= 400 for c in chunks)
    # Consecutive chunks share content (the overlap).
    for a, b in zip(chunks, chunks[1:]):
        tail = a.text[-20:]
        assert tail.split()[-1] in b.text


def test_chunk_ends_snap_to_whitespace():
    # Ends snap back to a space (never split mid-word); starts may land
    # mid-word by design - the 10% overlap is character-based.
    text = "supercalifragilistic " * 100
    for c in chunk_text(text, url="u", title="t"):
        assert c.text.split()[-1] == "supercalifragilistic"
