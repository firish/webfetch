"""
Search package: pluggable search provider adapters.

Exposes a small factory, `get_search_adapter()`, so callers (pipeline, tool
handler) can pick a provider by name from config without importing concrete
adapter classes. Adding a new provider = new adapter file + one registry entry
below - pipeline.py never changes.
"""

from webfetch.config import DEFAULT_SEARCH_PROVIDER
from webfetch.search.base import AbstractSearchAdapter, SearchResult
from webfetch.search.brave import BraveSearchAdapter
from webfetch.search.ddg import DDGSearchAdapter
from webfetch.search.serper import SerperSearchAdapter

# Registry mapping provider names (as used in config/env) to adapter classes.
_PROVIDERS: dict[str, type[AbstractSearchAdapter]] = {
    "ddg": DDGSearchAdapter,
    "brave": BraveSearchAdapter,
    "serper": SerperSearchAdapter,
}


def get_search_adapter(
    provider: str = DEFAULT_SEARCH_PROVIDER,
    **kwargs,
) -> AbstractSearchAdapter:
    """Return a search adapter instance by provider name.

    Args:
        provider: One of "ddg", "brave", "serper".
        **kwargs: Passed through to the adapter constructor
            (e.g. api_key, region, country).

    Returns:
        A ready-to-use AbstractSearchAdapter instance.

    Raises:
        ValueError: If the provider name is unknown, or if the adapter
            requires an API key that is missing (raised by the adapter).
    """
    try:
        adapter_cls = _PROVIDERS[provider]
    except KeyError:
        valid = ", ".join(sorted(_PROVIDERS))
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
    "get_search_adapter",
]
