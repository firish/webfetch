"""
MCP server: webfetch as a plugin for Claude Code, Claude Desktop, and any
other MCP client.

Local stdio server exposing two tools:
  - web_search: the full pipeline (search -> fetch -> rank -> compress)
    with cache provenance, force_fresh, and freshness hints - the same
    contract as the library's WEB_SEARCH_TOOL.
  - savings_report: the cost receipt for this machine's cache.

Run directly via the console script:  webfetch-mcp
Claude Code registration:             claude mcp add webfetch webfetch-mcp

One server per machine is the supported shape: the semantic cache assumes
a single process owns the cache file, and stdio gives exactly that.

Requires the optional extra:  pip install webfetch-llm[mcp]
"""

from __future__ import annotations

import sys
import threading


def main() -> None:
    """Start the stdio MCP server (console script entry point)."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        sys.exit("webfetch-mcp requires the mcp extra: "
                 "pip install 'webfetch-llm[mcp]'")

    from webfetch.receipts import savings_report as _savings_report
    from webfetch.tool import (
        SAVE_FINDING_TOOL,
        WEB_SEARCH_TOOL,
        get_default_pipeline,
        handle_save_finding,
        handle_web_search,
    )
    from webfetch.update_check import available_update

    server = FastMCP("webfetch")
    # One pipeline for the server's lifetime: encoder models stay warm and
    # the sqlite cache stays open across every tool call.
    pipeline = get_default_pipeline()

    @server.tool(description=WEB_SEARCH_TOOL["description"])
    def web_search(query: str, force_fresh: bool = False,
                   freshness: str | None = None,
                   full_results: bool = False) -> str:
        """Search the web and return ranked, source-labeled excerpts.

        Args:
            query: A focused web search query.
            force_fresh: Bypass the result cache for live data.
            freshness: "realtime" | "recent" | "stable" - how fast this
                query's answer changes; controls cache lifetime.
            full_results: Set true for lists/rankings/enumerations -
                returns uncompressed excerpts so items are not trimmed
                (still excerpts, not full pages).
        """
        return handle_web_search(
            {"query": query, "force_fresh": force_fresh,
             "freshness": freshness, "full_results": full_results},
            pipeline=pipeline,
        )

    @server.tool(description=SAVE_FINDING_TOOL["description"])
    def save_finding(query: str, content: str,
                     source_url: str | None = None) -> str:
        """Cache a fact learned outside web_search, marked unverified.

        Args:
            query: The search query this finding answers.
            content: The factual content, a few sentences.
            source_url: URL of the original source, if known.
        """
        return handle_save_finding(
            {"query": query, "content": content, "source_url": source_url},
            pipeline=pipeline,
        )

    @server.tool()
    def savings_report() -> str:
        """What webfetch has saved vs hosted web-search pricing (lifetime
        of this machine's cache)."""
        report = _savings_report()
        notice = available_update()
        return report + ("\n\n" + notice if notice else "")

    # Warm the update check off the handshake path so the first
    # savings_report call is instant either way.
    threading.Thread(target=available_update, daemon=True).start()
    server.run()  # stdio transport


if __name__ == "__main__":
    main()
