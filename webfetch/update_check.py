"""
Update notice: is a newer webfetch-llm on PyPI?

MCP has no way to push "your server is outdated" to users, and pip/uvx
installs are frozen at whatever version they resolved. This module makes
the server able to notice on its own: one fail-silent request to PyPI's
JSON API per process, surfaced as a single line appended to
savings_report output (the least noisy user-visible spot - never in
web_search results, which cost the model tokens).

Privacy: this is a network call to pypi.org disclosed in the README;
UPDATE_CHECK_ENABLED = False in config turns it off entirely.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_PYPI_URL = "https://pypi.org/pypi/webfetch-llm/json"

_lock = threading.Lock()
_checked = False
_notice: str | None = None


def _parse(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def available_update(timeout: float = 3.0) -> str | None:
    """Return a one-line update notice, or None.

    Checks PyPI at most once per process (cached afterwards, lock-guarded
    so concurrent tool calls do not race the first check). Any failure -
    offline, PyPI down, unparseable versions, package not installed via
    pip - returns None: an update check must never break anything.

    Args:
        timeout: Seconds for the PyPI request.

    Returns:
        Notice string when a newer release exists, else None.
    """
    global _checked, _notice
    from webfetch.config import UPDATE_CHECK_ENABLED
    if not UPDATE_CHECK_ENABLED:
        return None
    with _lock:
        if _checked:
            return _notice
        _checked = True
        try:
            from importlib.metadata import version

            import requests
            mine = version("webfetch-llm")
            resp = requests.get(_PYPI_URL, timeout=timeout)
            latest = resp.json()["info"]["version"]
            if _parse(latest) > _parse(mine):
                _notice = (f"webfetch-llm {mine} running; {latest} is "
                           "available: pip install -U webfetch-llm")
                logger.info(_notice)
        except Exception:
            pass
        return _notice


def _reset_for_tests() -> None:
    """Clear the cached result (test hook only)."""
    global _checked, _notice
    with _lock:
        _checked = False
        _notice = None


__all__ = ["available_update"]
