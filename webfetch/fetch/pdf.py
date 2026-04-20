"""
PDF fetcher and PDF link scanner.

Three responsibilities:
  1. PDFFetcher: fetches a direct PDF URL and extracts text via pdfplumber.
  2. extract_pdf_links(): scans raw HTML for any href/src pointing to a .pdf
     file - Phase 1 of JS-gated PDF handling (catches most styled anchor tags
     without needing a browser).
  3. extract_pdf_links_with_playwright(): renders the page with a headless
     browser first, then scans the fully-rendered DOM for PDF links - Phase 2,
     catches links injected by JS. Does NOT click buttons or intercept downloads.

pdfplumber is an optional dep: pip install webfetch[pdf]
playwright is an optional dep: pip install webfetch[browser]
"""

from __future__ import annotations

import io
import logging
import re

import requests

from webfetch.config import FETCH_TIMEOUT_SECS
from webfetch.fetch.base import AbstractFetcher

logger = logging.getLogger(__name__)

# Matches href or src attributes ending in .pdf (case-insensitive, optional query string)
_PDF_LINK_RE = re.compile(
    r'(?:href|src)=["\']([^"\']*\.pdf(?:\?[^"\']*)?)["\']',
    re.IGNORECASE,
)


def extract_pdf_links(raw_html: str, base_url: str = "") -> list[str]:
    """Scan raw HTML and return all URLs that point to PDF files.

    Finds plain anchor hrefs - does not execute JS or click buttons.
    Handles relative URLs by prepending base_url when provided.

    Args:
        raw_html: The raw HTML string to scan.
        base_url: Base URL of the page, used to resolve relative PDF paths.
                  E.g. "https://example.com/products/".

    Returns:
        Deduplicated list of PDF URLs found in the HTML.
    """
    from urllib.parse import urljoin

    urls: list[str] = []
    for match in _PDF_LINK_RE.finditer(raw_html):
        href = match.group(1)
        if href.startswith("http"):
            urls.append(href)
        elif base_url:
            urls.append(urljoin(base_url, href))
        else:
            urls.append(href)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def extract_pdf_links_with_playwright(
    url: str,
    timeout: int = FETCH_TIMEOUT_SECS,
) -> list[str]:
    """Render a page with Playwright and return all PDF links from the DOM.

    Phase 2 of JS-gated PDF handling. Renders the full page (executing JS),
    then runs extract_pdf_links() on the rendered HTML. Catches PDF links
    injected by JavaScript that plain HTML scanning misses.

    Does NOT click buttons or intercept file downloads - for that see Phase 3
    which is currently out of scope.

    Requires: pip install webfetch[browser] && playwright install chromium

    Args:
        url: The page URL to render.
        timeout: Milliseconds before the Playwright navigation times out.

    Returns:
        Deduplicated list of PDF URLs found in the rendered DOM.
        Returns an empty list if Playwright is not installed or navigation fails.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.debug(
            "playwright not installed. Run: pip install 'webfetch[browser]' "
            "&& playwright install chromium"
        )
        return []

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=timeout * 1000)
            # wait_until="networkidle" ensures JS-triggered DOM mutations settle
            # before we scrape - important for lazy-loaded download links
            page.wait_for_load_state("networkidle", timeout=timeout * 1000)
            rendered_html = page.content()
            browser.close()

        return extract_pdf_links(rendered_html, base_url=url)
    except Exception as exc:
        logger.warning("playwright PDF link scan failed for %s: %s", url, exc)
        return []


class PDFFetcher(AbstractFetcher):
    """Fetches a PDF URL and extracts text using pdfplumber.

    pdfplumber handles embedded tables in PDFs better than PyPDF2/pypdf,
    which matters for datasheet PDFs that use table layouts for specs.

    Requires: pip install webfetch[pdf]

    Args:
        timeout: Seconds before the HTTP request times out.
        max_pages: Maximum number of pages to extract. None means all pages.
                   Datasheets are usually 1-4 pages so the default is generous.
    """

    def __init__(
        self,
        timeout: int = FETCH_TIMEOUT_SECS,
        max_pages: int | None = 20,
    ) -> None:
        self._timeout = timeout
        self._max_pages = max_pages

    def fetch(self, url: str) -> str | None:
        """Download a PDF and return its text content.

        Args:
            url: Direct URL to a PDF file.

        Returns:
            Extracted text joined across all pages, or None if pdfplumber
            is not installed or the PDF could not be read.
        """
        try:
            import pdfplumber
        except ImportError:
            logger.error(
                "pdfplumber not installed. Run: pip install 'webfetch[pdf]'"
            )
            return None

        try:
            resp = requests.get(
                url,
                timeout=self._timeout,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("Failed to download PDF %s: %s", url, exc)
            return None

        try:
            with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
                pages = pdf.pages
                if self._max_pages is not None:
                    pages = pages[: self._max_pages]

                parts: list[str] = []
                for page in pages:
                    page_text = page.extract_text()
                    if page_text:
                        parts.append(page_text)

            return "\n\n".join(parts) if parts else None
        except Exception as exc:
            logger.warning("pdfplumber failed to parse %s: %s", url, exc)
            return None
