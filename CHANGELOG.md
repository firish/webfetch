# Changelog

## 0.1.0 - 2026-07-14

Initial release.

- `web_search` tool for LLM agent loops (Anthropic tool schema; OpenAI
  function-calling needs only a mechanical reshape) with a crash-proof
  handler: all failures return readable strings, never exceptions
- Pipeline: multi-engine search fusion (DDG/Brave/Serper/Tavily via
  reciprocal rank fusion), concurrent fetch with extraction fallback chain
  (trafilatura -> readability -> newspaper4k -> Playwright), hybrid
  BM25 + bi-encoder ranking with cross-encoder reranking
- Sentence-level compression of tool results: cross-encoder scored,
  measured 50% token reduction at zero recall loss
- Two-level sqlite cache (pages by URL, ranked chunks by query) with
  semantic paraphrase matching (bi-encoder shortlist + NLI verification)
  and volatility-aware TTLs (realtime/recent/stable)
- Search resilience: per-engine circuit breakers, silent-block detection,
  priority-failover "fallback" provider
- Cost receipts: lifetime counters in the cache db, `webfetch-savings` CLI
- MCP server (`webfetch-mcp`) for Claude Code/Desktop and other clients
- Three-layer eval harness with checked-in datasets, including a
  contamination-resistant fresh-events set; benchmark results in README
