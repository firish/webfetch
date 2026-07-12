"""
Fetch stage public interface.

Use `fetch(url)` for all fetching - it routes to HTMLFetcher or PDFFetcher
automatically so pipeline.py never needs to know which fetcher to pick.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

from webfetch.config import DEFAULT_FETCH_WORKERS
from webfetch.fetch.html import HTMLFetcher
from webfetch.fetch.pdf import PDFFetcher, extract_pdf_links, extract_pdf_links_with_playwright

logger = logging.getLogger(__name__)

_html_fetcher = HTMLFetcher()
_pdf_fetcher = PDFFetcher()


def fetch(url: str) -> str | None:
    """Fetch a URL and return extracted plain text.

    Routes to PDFFetcher for direct .pdf URLs, HTMLFetcher for everything else.

    Args:
        url: The page URL to fetch.

    Returns:
        Extracted text, or None if the page could not be fetched.
    """
    if url.lower().split("?")[0].endswith(".pdf"):
        return _pdf_fetcher.fetch(url)
    return _html_fetcher.fetch(url)


def fetch_all(
    urls: list[str],
    max_workers: int = DEFAULT_FETCH_WORKERS,
) -> dict[str, str | None]:
    """Fetch multiple URLs concurrently and return extracted text per URL.

    Fetching is IO-bound (network waits dominate), so a thread pool cuts
    wall time roughly by the worker count vs sequential fetching. The
    module-level fetcher singletons hold only immutable config, so sharing
    them across threads is safe. The playwright fallback opens its own
    browser context per call - heavy, but it is a last-resort path.

    Args:
        urls: URLs to fetch. Duplicates are fetched once.
        max_workers: Thread pool size.

    Returns:
        Dict mapping each input URL to its extracted text, or None if the
        fetch failed. Keys follow the input order (dicts preserve insertion
        order).
    """
    unique_urls = list(dict.fromkeys(urls))
    if not unique_urls:
        return {}

    def _safe_fetch(url: str) -> str | None:
        # fetch() already returns None on known failures; this guard catches
        # anything unexpected so one bad URL never kills the whole batch.
        try:
            return fetch(url)
        except Exception:
            logger.warning("fetch_all: unexpected error fetching %s", url, exc_info=True)
            return None

    with ThreadPoolExecutor(max_workers=min(max_workers, len(unique_urls))) as pool:
        texts = list(pool.map(_safe_fetch, unique_urls))

    return dict(zip(unique_urls, texts))


__all__ = [
    "fetch",
    "fetch_all",
    "HTMLFetcher",
    "PDFFetcher",
    "extract_pdf_links",
    "extract_pdf_links_with_playwright",
]
