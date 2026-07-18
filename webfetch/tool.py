"""
Client-side web_search tool for LLM agent loops.

This is the integration layer between webfetch and a frontier-model API:
give the model WEB_SEARCH_TOOL in the `tools` list, and when it emits a
tool_use block, pass the block's input to handle_web_search() and return
the string as the tool_result. The model formulates queries and reasons
over ranked, source-labeled excerpts - webfetch does everything in between.

This replaces hosted web-search tools (which charge ~$10/1k searches plus
retrieved-content tokens) with a local pipeline whose only marginal cost is
the ~2k tokens of ranked context per call.

Error boundary: handle_web_search() NEVER raises. An agent loop dies if a
tool handler throws mid-conversation, so every failure comes back as an
error string the model can read and react to (retry, rephrase, or answer
without search).
"""

from __future__ import annotations

import logging

from webfetch.compress import compress_chunks
from webfetch.config import (
    COMPRESSION_ENABLED,
    DEFAULT_TOOL_RESULT_BUDGET,
    FINDING_URL_SCHEME,
    FULL_RESULTS_BUDGET,
    SAVE_FINDING_ENABLED,
    SAVE_FINDING_FRESHNESS,
    TOOL_HEADER_STYLE,
    TOOL_MERGE_SOURCES,
)

# Import from extract.base (a leaf module) rather than the extract package,
# whose __init__ pulls in every provider adapter.
from webfetch.extract.base import build_context
from webfetch.pipeline import Pipeline, SearchChunksResult
from webfetch.rank.base import Chunk
from webfetch.semcache import SemanticSqliteCache

logger = logging.getLogger(__name__)

# Anthropic Messages API custom tool schema. Provider-agnostic in spirit -
# OpenAI function-calling needs only a mechanical reshape of this dict.
WEB_SEARCH_TOOL: dict = {
    "name": "web_search",
    "description": (
        "Search the web and return the most relevant text excerpts, each "
        "labeled with its source title and URL. Use this for facts past your "
        "training cutoff, current events, prices, product specs, or anything "
        "you are unsure about. Results are ranked excerpts from multiple "
        "pages, not full pages - issue focused queries, and call the tool "
        "again with a different query if the first results are not enough. "
        "Results may be served from a cache; the first line is a "
        "[cache: ...] provenance header showing what matched and how old "
        "it is."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A focused web search query.",
            },
            "force_fresh": {
                "type": "boolean",
                "description": (
                    "Set true to bypass the result cache. Use when you need "
                    "live or very recent data, or when a cached result "
                    "(shown in the [cache: ...] header) looked wrong or "
                    "stale. Default false."
                ),
            },
            "freshness": {
                "type": "string",
                "enum": ["realtime", "recent", "stable"],
                "description": (
                    "How fast this query's answer changes - controls cache "
                    "lifetime. realtime: prices, scores, breaking news "
                    "(minutes). recent: things that change over weeks or "
                    "months (days). stable: historical or definitional "
                    "facts, product specs (months). Omit if unsure."
                ),
            },
            "full_results": {
                "type": "boolean",
                "description": (
                    "Set true when the answer is a LIST, ranking, table, or "
                    "enumeration (e.g. 'top 10 X', 'all members of Y'), or "
                    "when a previous result seemed to be missing detail. "
                    "Returns the ranked excerpts UNCOMPRESSED so parallel "
                    "list items are not trimmed - it does NOT return full "
                    "pages; results are still excerpts. If the items you "
                    "need are absent entirely, rephrase the query toward "
                    "the list content itself instead. Default false."
                ),
            },
        },
        "required": ["query"],
    },
}

# Companion tool: let the model contribute something it learned OUTSIDE
# webfetch (e.g. via a hosted-search fallback) to the cache, so the next
# similar query benefits. Entries are marked and served with a distrust
# header - see _provenance_header.
SAVE_FINDING_TOOL: dict = {
    "name": "save_finding",
    "description": (
        "Save a fact you learned from a source OTHER than web_search (for "
        "example another search tool) into the local search cache, so "
        "repeat questions can be answered without re-searching. The entry "
        "is marked as model-contributed and future readers will see it is "
        "unverified. Use sparingly and only for content you actually "
        "observed, never for guesses."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": ("The search query this finding answers, as "
                                "you would have phrased it for web_search."),
            },
            "content": {
                "type": "string",
                "description": ("The factual content to cache, in a few "
                                "sentences. Include concrete values."),
            },
            "source_url": {
                "type": "string",
                "description": "URL of the original source, if you know it.",
            },
        },
        "required": ["query", "content"],
    },
}

# Lazily created so importing this module stays free - the pipeline spins up
# search adapters and a cache file only when the first tool call arrives.
_default_pipeline: Pipeline | None = None


def get_default_pipeline() -> Pipeline:
    """Return the shared default Pipeline for tool serving.

    Module-level singleton: encoder models stay warm and the sqlite cache
    stays open across every tool call in an agent loop.

    Returns:
        A Pipeline with multi-engine fusion over every search engine that
        has credentials in the environment (just DDG when none do - still
        zero-config), the full ranking cascade, and a SemanticSqliteCache
        (which degrades to exact-match caching without webfetch-llm[rerank]).
    """
    global _default_pipeline
    if _default_pipeline is None:
        from webfetch.search import get_search_adapter
        _default_pipeline = Pipeline(search=get_search_adapter("multi"),
                                     cache=SemanticSqliteCache())
    return _default_pipeline


def _humanize_age(age_secs: float | None) -> str:
    """Coarse human-readable age for the provenance header."""
    if age_secs is None:
        return "age unknown"
    if age_secs < 3600:
        return f"{max(1, int(age_secs // 60))}m old"
    if age_secs < 86400:
        return f"{int(age_secs // 3600)}h old"
    return f"{int(age_secs // 86400)}d old"


def _is_model_finding(chunks: list[Chunk]) -> bool:
    """True when the cached entry was contributed via save_finding."""
    return any(c.url.startswith(FINDING_URL_SCHEME) for c in chunks)


def _provenance_header(result: SearchChunksResult) -> str:
    """One-line cache provenance shown to the calling model.

    Showing the matched query is the model-side safety net for semantic
    cache hits: if the match looks wrong, the model can re-call with
    force_fresh=true. Model-contributed findings get an explicit distrust
    header - they were never fetched or ranked by webfetch.
    """
    suffix = f", {result.freshness}" if result.freshness else ""
    if _is_model_finding(result.chunks):
        return (f"[cache: model-contributed finding, "
                f"{_humanize_age(result.cache_age_secs)}{suffix} - "
                "UNVERIFIED: saved by a model, not fetched by webfetch. "
                "Call again with force_fresh=true to run a real search.]")
    if result.cache_kind == "semantic":
        return (f"[cache: semantic match to \"{result.matched_query}\", "
                f"{_humanize_age(result.cache_age_secs)}{suffix}]")
    if result.cache_kind == "exact":
        return (f"[cache: exact match, "
                f"{_humanize_age(result.cache_age_secs)}{suffix}]")
    return "[fresh search]"


def handle_web_search(
    tool_input: dict,
    pipeline: Pipeline | None = None,
    budget_chars: int = DEFAULT_TOOL_RESULT_BUDGET,
) -> str:
    """Execute a web_search tool call and format the result for the model.

    Args:
        tool_input: The tool_use block's input, e.g. {"query": "..."}.
        pipeline: Pipeline to use. Defaults to the shared singleton.
        budget_chars: Max characters of ranked context to return.

    Returns:
        Source-labeled excerpts on success, or a human-readable error /
        no-results message. Never raises.
    """
    try:
        query = str(tool_input.get("query", "")).strip()
        if not query:
            return "web_search error: empty query. Provide a non-empty 'query' string."
        force_fresh = bool(tool_input.get("force_fresh", False))
        full_results = bool(tool_input.get("full_results", False))
        freshness = tool_input.get("freshness")
        if freshness not in (None, "realtime", "recent", "stable"):
            freshness = None  # tolerate a confused model rather than erroring

        pipe = pipeline if pipeline is not None else get_default_pipeline()
        result = pipe.search_chunks(query, use_cache=not force_fresh,
                                    freshness=freshness)

        if not result.chunks:
            return (
                f"No results found for query: {query!r}. "
                "Try a different or broader phrasing."
                + _save_finding_nudge()
            )

        # Compression happens HERE - after the cache read - so cached rows
        # stay full-text and retuning compression never invalidates them.
        # full_results skips it: compression selects the best-scoring
        # sentences, which trims parallel list items (measured on the list
        # eval: item recall 71.9% -> 76.5% uncompressed). Model-contributed
        # findings are also served verbatim - they are already condensed.
        chunks = result.chunks
        if full_results:
            budget_chars = max(budget_chars, FULL_RESULTS_BUDGET)
        elif COMPRESSION_ENABLED and not _is_model_finding(chunks):
            chunks = compress_chunks(query, chunks)
        out = (_provenance_header(result) + "\n"
               + build_context(chunks, budget_chars=budget_chars,
                               merge_sources=TOOL_MERGE_SOURCES,
                               header_style=TOOL_HEADER_STYLE))
        pipe.bump_stats(tool_chars_returned=len(out))
        return out
    except Exception as exc:
        logger.warning("web_search tool call failed", exc_info=True)
        return (
            f"web_search error: {type(exc).__name__}: {exc}. "
            "Try again, rephrase the query, or answer without search."
            + _save_finding_nudge()
        )


def _save_finding_nudge() -> str:
    """One-line reminder appended to web_search failure exits.

    Models rarely volunteer maintenance calls, so the hint sits exactly at
    the moment a fallback path begins (real-usage observation, 2026-07-18).
    Silent when the feature is disabled - never advertise a dead tool.
    """
    if not SAVE_FINDING_ENABLED:
        return ""
    return (" If you answer this from another source instead, consider "
            "calling save_finding to cache it for next time.")


def handle_save_finding(
    tool_input: dict,
    pipeline: Pipeline | None = None,
) -> str:
    """Store a model-contributed finding in the query cache. Never raises.

    The entry is a single chunk whose URL carries FINDING_URL_SCHEME, which
    is what triggers the distrust header on every future read. Stored with
    SAVE_FINDING_FRESHNESS so unverified content ages out on the normal
    TTL rules, and disabled entirely by SAVE_FINDING_ENABLED=False.

    Args:
        tool_input: {"query": ..., "content": ..., "source_url": optional}.
        pipeline: Pipeline whose cache receives the entry. Defaults to the
            shared singleton.

    Returns:
        A short confirmation or error string for the model.
    """
    try:
        if not SAVE_FINDING_ENABLED:
            return ("save_finding is disabled on this deployment. The "
                    "finding was not saved; nothing else to do.")
        query = str(tool_input.get("query", "")).strip()
        content = str(tool_input.get("content", "")).strip()
        if not query or not content:
            return ("save_finding error: both 'query' and 'content' are "
                    "required.")
        source_url = str(tool_input.get("source_url", "") or "").strip()
        marker_url = FINDING_URL_SCHEME + (source_url or "unattributed")

        pipe = pipeline if pipeline is not None else get_default_pipeline()
        chunk = Chunk(text=content, url=marker_url,
                      title="model-contributed finding")
        pipe.store_chunks(query, [chunk], freshness=SAVE_FINDING_FRESHNESS)
        pipe.bump_stats(findings_saved=1)
        return (f"Saved (marked model-contributed, unverified) under query "
                f"{query!r}. Future matching searches will surface it with "
                "a provenance warning.")
    except Exception as exc:
        logger.warning("save_finding tool call failed", exc_info=True)
        return f"save_finding error: {type(exc).__name__}: {exc}."


__all__ = ["WEB_SEARCH_TOOL", "SAVE_FINDING_TOOL", "handle_web_search",
           "handle_save_finding", "get_default_pipeline"]
