# Changelog

## 0.1.2 - 2026-07-18

- Entry points (webfetch-mcp, webfetch-status, webfetch-savings) now load
  a .env from the working directory (cwd-upward search, never overriding
  real env vars) - "my keys are in .env but everything shows off" was the
  first trap every user hit. The library itself still never reads .env.

- New status surface: `webfetch-status` CLI, MCP `status` tool, and
  `webfetch.status_report()` - shows detected engine keys (names only),
  the active search configuration, degraded optional features, and cache
  location. New env config for the MCP server: WEBFETCH_PROVIDER and
  WEBFETCH_CACHE_DB.
- savings_report (MCP) now shows this-session and lifetime scopes.

- Semantic cache: NLI verification threshold 0.92 (was 0.97) - recovers
  question<->keyword-form paraphrases (React/TypeScript-style rewords)
  with zero trusted-negative false positives; the only additional matches
  on the eval set are audited QQP mislabels. Cross-form eval slice added.
- Freshness classifier: retrained centroids (FreshQA + new tech/listicle
  slice) - release-notes queries now classify stable and listicles recent
  instead of realtime (15-min TTL); "in <year>" only counts as historical
  for past years.

- New fetch_url tool (library + MCP): full extracted text of one page
  under a 24k-char budget with a truncation marker. Serves instantly from
  the pages cache for already-fetched pages; rejects non-public URLs
  (localhost, private ranges, metadata endpoints) since URLs are
  model-supplied.

- Update notice: the MCP server and webfetch-savings CLI check PyPI once
  per process (fail-silent, UPDATE_CHECK_ENABLED kill switch) and append
  a one-line notice to savings_report output when a newer release exists.

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
