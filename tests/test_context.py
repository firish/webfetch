"""build_context: budgets, source merging, header styles."""

from webfetch.extract.base import build_context
from webfetch.rank.base import Chunk


def _chunks():
    return [
        Chunk(text="alpha text", url="https://a.com/page", title="A"),
        Chunk(text="beta text", url="https://b.com/other", title="B"),
        Chunk(text="alpha more", url="https://a.com/page", title="A"),
    ]


def test_default_format_one_header_per_chunk():
    ctx = build_context(_chunks(), budget_chars=10_000)
    assert ctx.count("[Source:") == 3
    assert "https://a.com/page" in ctx


def test_merge_sources_groups_same_url():
    ctx = build_context(_chunks(), budget_chars=10_000, merge_sources=True)
    assert ctx.count("[Source:") == 2  # a.com merged
    # Group keeps both texts, in rank order.
    assert ctx.index("alpha text") < ctx.index("alpha more")


def test_domain_header_style_drops_path_keeps_title():
    ctx = build_context(_chunks(), budget_chars=10_000, header_style="domain")
    assert "a.com" in ctx and "/page" not in ctx
    assert "[Source: A |" in ctx


def test_budget_truncates_but_never_empty():
    ctx = build_context(_chunks(), budget_chars=30)
    assert ctx  # always at least one chunk, even over budget
    assert "alpha text" in ctx
    assert "beta text" not in ctx
