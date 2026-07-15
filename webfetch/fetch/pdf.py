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

# Matches href or src attributes ending in .pdf (case-insensitive,
# optional query string)
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


# Legibility gate thresholds. Garbled columnar extraction comes out as
# interleaved single characters ("q t t o s i u n r..."), so a high share
# of 1-char alphabetic tokens is the cleanest signal.
_MAX_SINGLE_CHAR_RATIO = 0.35
_MIN_TOKENS_FOR_CHECK = 20


def _is_legible(text: str) -> bool:
    """Heuristic: is this extracted page text real words, not char soup?

    Args:
        text: Extracted page text.

    Returns:
        True when the text looks like readable prose/tables. Short texts
        pass by default (not enough signal to condemn them).
    """
    tokens = [t for t in text.split() if t.isalpha()]
    if len(tokens) < _MIN_TOKENS_FOR_CHECK:
        return True
    single = sum(1 for t in tokens if len(t) == 1)
    return single / len(tokens) <= _MAX_SINGLE_CHAR_RATIO


def _table_to_markdown(table: list[list[str | None]]) -> str:
    """Render a pdfplumber table (list of rows) as a markdown table."""
    rows = [[(c or "").strip().replace("\n", " ") for c in row]
            for row in table if any(c and c.strip() for c in row)]
    if len(rows) < 2:
        return ""
    lines = ["| " + " | ".join(rows[0]) + " |",
             "|" + "|".join("---" for _ in rows[0]) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


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
        """Download a PDF and return its text content plus any tables.

        Per page: extract text; if it fails the legibility gate (columnar
        or overlapping layouts come out as interleaved single characters),
        retry with pdfplumber's layout-aware mode; if STILL garbled, drop
        the page - garbage chunks were observed winning ranking slots on
        keyword density and displacing legible content. Tables are
        extracted separately and appended as markdown, mirroring the HTML
        path.

        Args:
            url: Direct URL to a PDF file.

        Returns:
            Extracted text (with tables appended) joined across pages, or
            None if pdfplumber is missing or nothing legible was extracted.
        """
        try:
            import pdfplumber
        except ImportError:
            logger.error(
                "pdfplumber not installed. Run: pip install 'webfetch[pdf]'"
            )
            return None

        from webfetch.fetch.base import BROWSER_HEADERS
        try:
            resp = requests.get(url, timeout=self._timeout, headers=BROWSER_HEADERS)
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
                table_parts: list[str] = []
                dropped = 0
                for page in pages:
                    page_text = page.extract_text()
                    if page_text and not _is_legible(page_text):
                        page_text = page.extract_text(layout=True)
                    if page_text and _is_legible(page_text):
                        parts.append(page_text)
                    elif page_text:
                        dropped += 1
                    for table in page.extract_tables() or []:
                        md = _table_to_markdown(table)
                        if md:
                            table_parts.append(md)
                if dropped:
                    logger.info("%s: dropped %d garbled page(s)", url, dropped)

            text = "\n\n".join(parts)
            if table_parts:
                text = (text + "\n\n## Extracted Tables\n\n"
                        + "\n\n".join(table_parts)).strip()
            return text or None
        except Exception as exc:
            logger.warning("pdfplumber failed to parse %s: %s", url, exc)
            return None
