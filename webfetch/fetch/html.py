"""
HTML page fetcher with a multi-stage fallback chain.

Extraction order:
  1. trafilatura  - best general-purpose HTML -> clean text
  2. readability  - Mozilla reader-mode algorithm, good for product pages
  3. newspaper4k  - article/press-release optimized
  4. playwright   - JS-rendered pages
     (optional install: pip install webfetch-llm[browser])

After any successful text extraction, pandas.read_html() runs in parallel on
the raw HTML to capture spec tables that prose extractors tend to mangle.
The table markdown is appended to the extracted text before returning.
"""

from __future__ import annotations

import io
import logging

import requests
import trafilatura

from webfetch.config import FETCH_TIMEOUT_SECS, MIN_EXTRACTED_CHARS
from webfetch.fetch.base import BROWSER_HEADERS, AbstractFetcher

logger = logging.getLogger(__name__)


def _extract_tables_from_html(raw_html: str) -> str:
    """Parse all HTML tables and return them as markdown.

    Uses pandas.read_html() which handles complex/nested tables better
    than trafilatura's table serializer.

    Returns an empty string if pandas is not installed or no tables are found.
    """
    try:
        import pandas as pd
        tables = pd.read_html(io.StringIO(raw_html), flavor="lxml")
    except ImportError:
        logger.debug("pandas not installed - skipping table extraction")
        return ""
    except Exception:
        # read_html raises ValueError when no tables are found
        return ""

    parts: list[str] = []
    for df in tables:
        # Drop columns/rows that are entirely NaN (common in layout tables)
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            continue
        parts.append(df.to_markdown(index=False))

    return "\n\n".join(parts)


def _extract_metadata_block(raw_html: str) -> str:
    """Return "title - description" from the page head, or empty string.

    Prose extractors deliberately skip <head> content, but factoid answers
    frequently live in <title> and meta/og descriptions (measured: several
    eval recall misses had the answer ONLY there, e.g. album release dates
    in og:description). Prepending this block puts that text in front of
    the ranker like any other chunk.
    """
    try:
        meta = trafilatura.extract_metadata(raw_html)
    except Exception:
        return ""
    if meta is None:
        return ""
    parts = [p for p in (meta.title, meta.description) if p]
    return " - ".join(parts)


def _fetch_with_readability(raw_html: str, url: str) -> str | None:
    """Extract main content using the readability-lxml reader-mode algorithm."""
    try:
        from readability import Document
        doc = Document(raw_html)
        # summary() returns HTML - strip tags for plain text
        import re
        text = re.sub(r"<[^>]+>", " ", doc.summary())
        text = re.sub(r"\s+", " ", text).strip()
        return text if len(text) > 100 else None
    except ImportError:
        logger.debug("readability-lxml not installed - skipping fallback")
        return None
    except Exception as exc:
        logger.debug("readability failed for %s: %s", url, exc)
        return None


def _fetch_with_newspaper(url: str) -> str | None:
    """Extract article text using newspaper4k."""
    try:
        import newspaper
        article = newspaper.Article(url)
        article.download()
        article.parse()
        return article.text if article.text else None
    except ImportError:
        logger.debug("newspaper4k not installed - skipping fallback")
        return None
    except Exception as exc:
        logger.debug("newspaper4k failed for %s: %s", url, exc)
        return None


def _fetch_with_playwright(url: str) -> tuple[str, str] | None:
    """Fetch a JS-rendered page using playwright and return (raw_html, url).

    Returns None if playwright is not installed or the fetch fails.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.debug("playwright not installed - skipping JS fallback")
        return None

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            # Headless Chrome's default UA says "HeadlessChrome" - exactly
            # what bot walls key on. Present the normal browser UA instead.
            page = browser.new_page(
                user_agent=BROWSER_HEADERS["User-Agent"],
                locale="en-US",
            )
            resp = page.goto(url, timeout=FETCH_TIMEOUT_SECS * 1000)
            # goto() does not raise on HTTP errors - without this check,
            # 404/403 error pages would be returned as page "content".
            if resp is not None and resp.status >= 400:
                browser.close()
                logger.debug("playwright got HTTP %d for %s", resp.status, url)
                return None
            html = page.content()
            browser.close()
        return html, url
    except Exception as exc:
        logger.debug("playwright failed for %s: %s", url, exc)
        return None


class HTMLFetcher(AbstractFetcher):
    """Fetches HTML pages and extracts clean text via a multi-stage fallback chain.

    Also extracts HTML tables via pandas.read_html() and appends them to the
    extracted text so spec tables are not lost during prose extraction.

    Args:
        timeout: Seconds before an HTTP request is abandoned.
        include_tables: Whether to extract and append HTML tables.
    """

    def __init__(
        self,
        timeout: int = FETCH_TIMEOUT_SECS,
        include_tables: bool = True,
    ) -> None:
        self._timeout = timeout
        self._include_tables = include_tables

    def fetch(self, url: str) -> str | None:
        """Fetch a URL and return extracted plain text plus any spec tables.

        Args:
            url: The HTML page URL to fetch.

        Returns:
            Extracted text (with tables appended if found), or None if all
            extraction methods fail.
        """
        # Step 1: get raw HTML - trafilatura.fetch_url handles redirects/encoding
        raw_html = trafilatura.fetch_url(url)

        used_playwright = False
        if not raw_html:
            # trafilatura fetch failed - retry with full browser headers
            # (bot walls 403 the default UA but often pass a real header set)
            try:
                resp = requests.get(url, timeout=self._timeout,
                                    headers=BROWSER_HEADERS)
                resp.raise_for_status()
                raw_html = resp.text
            except Exception as exc:
                logger.warning("Failed to fetch %s: %s", url, exc)
                # Last resort: a real browser via playwright. Previously this
                # fallback only ran when extraction failed on FETCHED HTML,
                # so hard-blocked pages never reached it.
                result = _fetch_with_playwright(url)
                if result is None:
                    return None
                raw_html, _ = result
                used_playwright = True

        # Step 2: extract main text, trying each method in order
        text = trafilatura.extract(
            raw_html,
            include_tables=False,  # we handle tables separately below
            output_format="markdown",
            favor_recall=True,     # looser extraction threshold - better for spec pages
        )

        if not text:
            logger.debug("trafilatura empty for %s, trying readability", url)
            text = _fetch_with_readability(raw_html, url)

        if not text:
            logger.debug("readability empty for %s, trying newspaper4k", url)
            text = _fetch_with_newspaper(url)

        # Playwright render when extraction failed OR produced suspiciously
        # little text - JS-rendered SPAs return shell HTML that extracts to
        # a thin nav/footer remnant, which used to mask the real content.
        if (not text or len(text) < MIN_EXTRACTED_CHARS) and not used_playwright:
            logger.debug("thin/no text for %s, trying playwright render", url)
            result = _fetch_with_playwright(url)
            if result:
                playwright_html, _ = result
                rendered = trafilatura.extract(
                    playwright_html,
                    include_tables=False,
                    output_format="markdown",
                    favor_recall=True,
                )
                if rendered and len(rendered) > len(text or ""):
                    raw_html = playwright_html
                    text = rendered

        # Head metadata (title + description) - factoid answers often live
        # ONLY there. Prepend so it ranks like any other chunk; when every
        # prose extractor failed, metadata alone is still worth returning.
        meta_block = _extract_metadata_block(raw_html)
        if meta_block and text:
            text = meta_block + "\n\n" + text
        elif meta_block and not text:
            text = meta_block

        if not text:
            logger.warning("All extraction methods failed for %s", url)
            return None

        # Step 3: extract tables from the raw HTML and append to text
        if self._include_tables:
            table_md = _extract_tables_from_html(raw_html)
            if table_md:
                text = text + "\n\n## Extracted Tables\n\n" + table_md

        return text
