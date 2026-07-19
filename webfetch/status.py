"""
Setup status: which keys are set, which engines will serve, what degrades.

Answers the post-install questions ("is my Brave key being picked up? am I
on fusion or DDG-only? is the semantic cache actually on?") without making
anyone read server logs. Three surfaces share this module: the
`webfetch-status` CLI, the MCP `status` tool, and `webfetch.status_report()`.

Key VALUES are never printed - only whether each env var is set.
"""

from __future__ import annotations

import os

from webfetch.config import (
    COMPRESS_SCORER,
    COMPRESSION_ENABLED,
    DEFAULT_CACHE_DB,
    TTL_BY_FRESHNESS,
)

# Engine -> env var holding its key (None = no key needed). Mirrors the
# registry in webfetch/search/__init__.py.
ENGINE_KEY_ENVS: dict[str, str | None] = {
    "ddg": None,
    "brave": "BRAVE_API_KEY",
    "serper": "SERPER_API_KEY",
    "tavily": "TAVILY_API_KEY",
}


def _extra_available(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def status_report() -> str:
    """Render the setup status as a human-readable block.

    Returns:
        Multi-line string: version, engine keys, active provider chain,
        optional-extra availability with what each absence degrades, cache
        location/size, and the env-var configuration surface.
    """
    try:
        from importlib.metadata import version
        ver = version("webfetch-llm")
    except Exception:
        ver = "dev (not pip-installed)"

    lines = [f"webfetch {ver}", "", "search engines:"]
    active = []
    for engine, env in ENGINE_KEY_ENVS.items():
        if env is None:
            lines.append(f"  {engine:8s} ready (no key needed)")
            active.append(engine)
        elif os.environ.get(env):
            lines.append(f"  {engine:8s} ready ({env} is set)")
            active.append(engine)
        else:
            lines.append(f"  {engine:8s} off   ({env} not set)")

    provider = os.environ.get("WEBFETCH_PROVIDER", "multi")
    if provider == "multi":
        chain = f"multi({'+'.join(active)})" if len(active) > 1 else active[0]
        how = "RRF fusion" if len(active) > 1 else "single engine"
    elif provider == "fallback":
        chain = f"fallback({'>'.join(active)})"
        how = "priority failover"
    else:
        chain = provider
        how = "single engine"
    lines += ["", f"active provider: {chain} ({how})"
              + ("" if "WEBFETCH_PROVIDER" not in os.environ
                 else "  [from WEBFETCH_PROVIDER]")]

    rerank = _extra_available("sentence_transformers")
    lines += ["", "optional features:"]
    lines.append(
        "  semantic ranking/cache/compression: "
        + ("on" if rerank else
           "OFF - BM25 ranking, exact-match cache, lexical compression "
           "(pip install 'webfetch-llm[rerank]')"))
    lines.append("  JS-page rendering (playwright):     "
                 + ("on" if _extra_available("playwright") else
                    "OFF (pip install 'webfetch-llm[browser]' && "
                    "playwright install chromium)"))
    lines.append("  PDF extraction (pdfplumber):        "
                 + ("on" if _extra_available("pdfplumber") else
                    "OFF (pip install 'webfetch-llm[pdf]')"))
    lines.append("  HTML tables (pandas+tabulate):      "
                 + ("on" if _extra_available("pandas")
                    and _extra_available("tabulate") else
                    "OFF (pip install 'webfetch-llm[tables]')"))
    if COMPRESSION_ENABLED:
        scorer = COMPRESS_SCORER if rerank else "lexical (degraded)"
        lines.append(f"  result compression:                 on ({scorer})")
    else:
        lines.append("  result compression:                 off")

    db_path = os.path.expanduser(
        os.environ.get("WEBFETCH_CACHE_DB", DEFAULT_CACHE_DB))
    lines += ["", f"cache: {db_path}"]
    if os.path.exists(db_path):
        size_mb = os.path.getsize(db_path) / 1e6
        from webfetch.receipts import get_counters
        searches = int(get_counters(db_path).get("searches_total", 0))
        lines.append(f"  {size_mb:.1f} MB, {searches} lifetime searches "
                     "(webfetch-savings for the receipt)")
    else:
        lines.append("  not created yet (first search creates it)")

    ttls = ", ".join(f"{k}={v // 60}m" if v < 3600 else f"{k}={v // 86400}d"
                     for k, v in TTL_BY_FRESHNESS.items())
    lines += ["", "configuration (env vars, read at server start):",
              "  WEBFETCH_PROVIDER   multi | fallback | ddg | brave | "
              "serper | tavily (default multi)",
              "  WEBFETCH_CACHE_DB   cache file path "
              f"(default {DEFAULT_CACHE_DB})",
              "  <ENGINE>_API_KEY    per-engine keys, see above",
              f"  cache TTLs: {ttls} (library constants, "
              "webfetch/config.py)"]
    return "\n".join(lines)


def main() -> None:
    """Console entry point: `webfetch-status`."""
    print(status_report())


__all__ = ["status_report", "main"]
