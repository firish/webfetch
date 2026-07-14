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
    TOOL_HEADER_STYLE,
    TOOL_MERGE_SOURCES,
)
# Import from extract.base (a leaf module) rather than the extract package,
# whose __init__ pulls in every provider adapter.
from webfetch.extract.base import build_context
from webfetch.pipeline import Pipeline, SearchChunksResult
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
        },
        "required": ["query"],
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
        A Pipeline with the full ranking cascade and a SemanticSqliteCache
        (which degrades to exact-match caching without webfetch[rerank]).
    """
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = Pipeline(cache=SemanticSqliteCache())
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


def _provenance_header(result: SearchChunksResult) -> str:
    """One-line cache provenance shown to the calling model.

    Showing the matched query is the model-side safety net for semantic
    cache hits: if the match looks wrong, the model can re-call with
    force_fresh=true.
    """
    suffix = f", {result.freshness}" if result.freshness else ""
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
            )

        # Compression happens HERE - after the cache read - so cached rows
        # stay full-text and retuning compression never invalidates them.
        chunks = result.chunks
        if COMPRESSION_ENABLED:
            chunks = compress_chunks(query, chunks)
        return (_provenance_header(result) + "\n"
                + build_context(chunks, budget_chars=budget_chars,
                                merge_sources=TOOL_MERGE_SOURCES,
                                header_style=TOOL_HEADER_STYLE))
    except Exception as exc:
        logger.warning("web_search tool call failed", exc_info=True)
        return (
            f"web_search error: {type(exc).__name__}: {exc}. "
            "Try again, rephrase the query, or answer without search."
        )


__all__ = ["WEB_SEARCH_TOOL", "handle_web_search", "get_default_pipeline"]
