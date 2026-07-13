"""
Search package: pluggable search provider adapters.

Exposes a small factory, `get_search_adapter()`, so callers (pipeline, tool
handler) can pick a provider by name from config without importing concrete
adapter classes. Adding a new provider = new adapter file + one registry entry
below - pipeline.py never changes.
"""

import logging

from webfetch.config import DEFAULT_SEARCH_PROVIDER
from webfetch.search.base import AbstractSearchAdapter, SearchResult
from webfetch.search.brave import BraveSearchAdapter
from webfetch.search.ddg import DDGSearchAdapter
from webfetch.search.multi import MultiSearchAdapter
from webfetch.search.serper import SerperSearchAdapter
from webfetch.search.tavily import TavilySearchAdapter

logger = logging.getLogger(__name__)

# Registry mapping provider names (as used in config/env) to adapter classes.
_PROVIDERS: dict[str, type[AbstractSearchAdapter]] = {
    "ddg": DDGSearchAdapter,
    "brave": BraveSearchAdapter,
    "serper": SerperSearchAdapter,
    "tavily": TavilySearchAdapter,
}


def _build_multi(**kwargs) -> MultiSearchAdapter:
    """Build a fusion adapter from every engine with available credentials.

    DDG always joins (no key needed); keyed engines join when their env var
    is set. Engines that fail to construct are logged and skipped - fusion
    with one engine is still valid (it just is that engine).
    """
    adapters: list[AbstractSearchAdapter] = []
    for name in ("ddg", "brave", "serper", "tavily"):
        try:
            adapters.append(_PROVIDERS[name]())
        except ValueError:
            logger.info("fusion: no credentials for %s - skipping", name)
    return MultiSearchAdapter(adapters, **kwargs)


def get_search_adapter(
    provider: str = DEFAULT_SEARCH_PROVIDER,
    **kwargs,
) -> AbstractSearchAdapter:
    """Return a search adapter instance by provider name.

    Args:
        provider: One of "ddg", "brave", "serper", "tavily", or "multi"
            (RRF fusion of every engine with available credentials).
        **kwargs: Passed through to the adapter constructor
            (e.g. api_key, region, country).

    Returns:
        A ready-to-use AbstractSearchAdapter instance.

    Raises:
        ValueError: If the provider name is unknown, or if the adapter
            requires an API key that is missing (raised by the adapter).
    """
    if provider == "multi":
        return _build_multi(**kwargs)
    try:
        adapter_cls = _PROVIDERS[provider]
    except KeyError:
        valid = ", ".join(sorted(_PROVIDERS) + ["multi"])
        raise ValueError(
            f"Unknown search provider {provider!r}. Valid providers: {valid}"
        ) from None
    return adapter_cls(**kwargs)


__all__ = [
    "AbstractSearchAdapter",
    "SearchResult",
    "DDGSearchAdapter",
    "BraveSearchAdapter",
    "SerperSearchAdapter",
    "TavilySearchAdapter",
    "MultiSearchAdapter",
    "get_search_adapter",
]
