"""
Fetch stage public interface.

Use `fetch(url)` for all fetching - it routes to HTMLFetcher or PDFFetcher
automatically so pipeline.py never needs to know which fetcher to pick.
"""

from webfetch.fetch.html import HTMLFetcher
from webfetch.fetch.pdf import PDFFetcher, extract_pdf_links, extract_pdf_links_with_playwright

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


__all__ = ["fetch", "HTMLFetcher", "PDFFetcher", "extract_pdf_links", "extract_pdf_links_with_playwright"]
