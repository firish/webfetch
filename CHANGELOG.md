# Changelog

## Unreleased

- web_search gains a `full_results` flag for lists/rankings/enumerations:
  skips compression and raises the budget. Eval-backed: compression was
  trimming parallel list items (item recall 71.9% -> 76.5% uncompressed
  on the new list eval); extra chunks measured near-zero benefit.
- New save_finding tool (library + MCP): the model can cache facts
  learned outside webfetch, marked model-contributed; future reads show
  an explicit UNVERIFIED header with a force_fresh escape hatch.
  SAVE_FINDING_ENABLED=False disables it.

## 0.1.1 - 2026-07-15

- get_default_pipeline() (used by the MCP server and handle_web_search's
  default) now builds multi-engine fusion from whatever engine keys are in
  the environment, instead of silently running DDG-only. Zero-config
  behavior is unchanged (fusion of one engine is just that engine).
- mcp SDK moved from the [mcp] extra into core deps so
  `uvx --from webfetch-llm webfetch-mcp` works as a one-liner; [mcp]
  remains as an empty alias.

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
