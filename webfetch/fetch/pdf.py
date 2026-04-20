"""
PDF fetcher and PDF link scanner.

Two responsibilities:
  1. PDFFetcher: fetches a direct PDF URL and extracts text via pdfplumber.
  2. extract_pdf_links(): scans raw HTML for any href/src pointing to a .pdf
     file - Phase 1 of JS-gated PDF handling (catches most styled anchor tags
     without needing a browser).

pdfplumber is an optional dep: pip install webfetch[pdf]
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
