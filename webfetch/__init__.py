"""
webfetch: replicate LLM web-search tool calls cheaply by owning the full
search -> fetch -> rank pipeline.

Quick start (tool mode - the calling model is the extractor):

    from webfetch import Pipeline, SqliteCache, WEB_SEARCH_TOOL, handle_web_search

    pipeline = Pipeline(cache=SqliteCache())
    text = handle_web_search({"query": "Fluke 87V DC voltage accuracy"}, pipeline=pipeline)

Structured extraction mode (webfetch calls a cheap LLM for JSON output):

    from webfetch.extract import ClaudeExtractor
    specs = pipeline.run(query, keys={"accuracy": "..."}, extractor=ClaudeExtractor())

The extract package is intentionally NOT imported here - the root import
must work with core dependencies only.
"""

from webfetch.cache import AbstractCache, CacheMatch, SqliteCache
from webfetch.pipeline import Pipeline, SearchChunksResult
from webfetch.rank import Chunk
from webfetch.receipts import get_counters, savings_report
from webfetch.search import SearchResult, get_search_adapter
from webfetch.semcache import SemanticSqliteCache
from webfetch.tool import WEB_SEARCH_TOOL, get_default_pipeline, handle_web_search

__version__ = "0.1.0"

__all__ = [
    "get_counters",
    "savings_report",
    "Pipeline",
    "SearchChunksResult",
    "AbstractCache",
    "CacheMatch",
    "SqliteCache",
    "SemanticSqliteCache",
    "Chunk",
    "SearchResult",
    "get_search_adapter",
    "WEB_SEARCH_TOOL",
    "handle_web_search",
    "get_default_pipeline",
    "__version__",
]
