"""
Cost receipts: what using webfetch has saved versus hosted web search.

Counters accumulate in the cache db's `stats` table as the pipeline runs
(see SqliteCache.bump_stats / Pipeline.bump_stats); this module turns them
into a human-readable receipt. The comparison baseline is a hosted
web_search tool: both Anthropic and OpenAI charge $10/1k searches, and the
hosted tool also injects far more retrieved-content tokens per call into
the paying context window (measured 2026-07-14, Layer 3 eval: ~17.4k
input tokens/query hosted vs ~3.5k with webfetch's compressed results).

The dollar figures are ESTIMATES and the assumptions are function
arguments: hosted per-search fee, hosted content tokens per call, and the
$/MTok price of the model reading the results. Counters themselves are
exact.
"""

from __future__ import annotations

import os
import sqlite3

from webfetch.config import (
    DEFAULT_CACHE_DB,
    RECEIPT_HOSTED_SEARCH_FEE,
    RECEIPT_HOSTED_TOKENS_PER_CALL,
    RECEIPT_TOKEN_PRICE_PER_MTOK,
)

CHARS_PER_TOKEN = 4


def get_counters(db_path: str = DEFAULT_CACHE_DB) -> dict[str, float]:
    """Read the lifetime usage counters from a cache db.

    Args:
        db_path: Path to the sqlite cache file.

    Returns:
        Counter name -> value. Empty dict if the db or table is missing.
    """
    path = os.path.expanduser(db_path)
    if not os.path.exists(path):
        return {}
    conn = sqlite3.connect(path)
    try:
        return dict(conn.execute("SELECT key, value FROM stats"))
    except sqlite3.OperationalError:  # pre-receipts cache db
        return {}
    finally:
        conn.close()


def savings_report(
    db_path: str = DEFAULT_CACHE_DB,
    hosted_fee: float = RECEIPT_HOSTED_SEARCH_FEE,
    hosted_tokens_per_call: int = RECEIPT_HOSTED_TOKENS_PER_CALL,
    token_price_per_mtok: float = RECEIPT_TOKEN_PRICE_PER_MTOK,
) -> str:
    """Format a savings receipt against hosted web-search pricing.

    Args:
        db_path: Path to the sqlite cache file holding the counters.
        hosted_fee: Hosted per-search fee ($10/1k at both major vendors).
        hosted_tokens_per_call: Content tokens a hosted search injects per
            call (measured mean; override to be more conservative).
        token_price_per_mtok: $/MTok of the model reading the results
            (default: Opus-class input pricing).

    Returns:
        A multi-line receipt string; a short notice if no counters exist.
    """
    c = get_counters(db_path)
    searches = int(c.get("searches_total", 0))
    if not searches:
        return "webfetch receipts: no searches recorded yet."
    exact = int(c.get("cache_hits_exact", 0))
    semantic = int(c.get("cache_hits_semantic", 0))
    fresh = int(c.get("fresh_searches", 0))
    cached = exact + semantic
    our_tokens = int(c.get("tool_chars_returned", 0)) // CHARS_PER_TOKEN

    fee_avoided = searches * hosted_fee
    hosted_tokens = searches * hosted_tokens_per_call
    token_savings = max(0, hosted_tokens - our_tokens) \
        * token_price_per_mtok / 1e6
    lines = [
        "webfetch savings receipt (lifetime of this cache)",
        f"  searches served:      {searches}",
        f"  from cache:           {cached} ({cached / searches:.0%})"
        f" - exact {exact}, semantic {semantic}"
        f" (zero marginal cost: no engine fees, no fetching)",
        f"  fresh pipeline runs:  {fresh}",
        f"  result tokens sent:   ~{our_tokens:,}"
        f" (hosted would inject ~{hosted_tokens:,})",
        "  ---",
        f"  hosted search fees avoided:  {searches} x ${hosted_fee:.3f}"
        f" = ${fee_avoided:.2f}",
        f"  content-token cost avoided:  ~${token_savings:.2f}"
        f" (at ${token_price_per_mtok:.2f}/MTok)",
        f"  ESTIMATED TOTAL AVOIDED:     ${fee_avoided + token_savings:.2f}",
    ]
    return "\n".join(lines)


def main() -> None:
    """Console entry point: `webfetch-savings [--db PATH]`."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Print the webfetch savings receipt.")
    parser.add_argument("--db", default=DEFAULT_CACHE_DB,
                        help="cache db path (default: %(default)s)")
    parser.add_argument("--token-price", type=float,
                        default=RECEIPT_TOKEN_PRICE_PER_MTOK,
                        help="$/MTok of the model reading results "
                             "(default: %(default)s, Opus-class)")
    args = parser.parse_args()
    print(savings_report(db_path=args.db,
                         token_price_per_mtok=args.token_price))
    from webfetch.update_check import available_update
    notice = available_update()
    if notice:
        print("\n" + notice)


__all__ = ["get_counters", "savings_report", "main"]
