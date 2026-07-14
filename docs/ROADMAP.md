# Roadmap

Ordered by priority. Every feature follows the project practice: eval first,
implementation second, measured acceptance gate third. Shipped work moves to
the bottom.

## 1. Robust PDF / table / datasheet extraction (important)

Essential for a usable webfetch tool - spec and datasheet content lives in
PDFs and HTML tables, and eval runs surfaced garbled PDF text entering the
ranker (e.g. columnar PDFs extracted as interleaved characters).

- Fix garbled pdfplumber output: detect multi-column layouts, extract
  per-column; drop chunks failing a legibility heuristic (dictionary-word
  ratio) so garbage never reaches the ranker
- PDF tables: pdfplumber extract_tables -> markdown, same as the HTML path
- JS-gated PDFs phase 2 (Playwright-rendered DOM link scrape) per the
  Known Gaps section in ARCHITECTURE.md
- Acceptance: extend Layer 2 with a PDF-heavy query slice; recall on that
  slice + a legibility metric on emitted chunks
- Lift: 1-2 days (layout detection is the unpredictable part)

## 2. Stale-while-revalidate

Serve expired-but-present cache rows instantly with honest provenance
("[cache: stale, 18m old, realtime - refreshing]") and refresh in a
background worker. Pairs with volatility TTLs - benefit concentrates on the
realtime class. Grace window ~2x class TTL; single refresh worker with an
in-flight set; keep stale row if refresh fails.

- Acceptance: TTL unit suite extension + latency-at-expiry measurement
- Lift: ~a day

## Cheap wins (from the 2026-07-13 competitor gap analysis)

Small lifts with clear value, orderable ahead of or between the numbered items:

- Freshness -> engine time filters: we already classify every query as
  realtime/recent/stable (model hint + fallback classifier); pass the class
  through to the engines' recency params (Brave `freshness`, Serper `tbs`,
  Tavily `time_range`) so realtime queries stop retrieving stale pages.
  Natural extension of the volatility system. Lift: hours.
- Domain include/exclude tool param: agents genuinely use this ("only
  search docs.python.org"); engines support it natively (site: operators /
  API params) + post-search URL filter as backstop. Lift: hours.
- Cache-only / lockdown mode: serve from cache, never hit the network
  (Firecrawl ships this as `lockdown`); we have the cache - it is one flag
  plus a miss message. Good for offline runs and deterministic tests.
  Lift: an hour.
- Expose structured extraction in tool mode: extraction mode (keys dict ->
  JSON via extractor adapters) is already built; surfacing it as an
  optional tool capability is packaging, not building. Lift: half a day.

## Backlog (unordered)

- MCP server wrapper (local stdio first; remote server needs a shared
  cache backend + eviction policy - see cache notes in ARCHITECTURE.md)
- OpenAI function-calling example loop
- OR-ensemble semantic-cache verifier (only if live hit rates disappoint)
- Cost receipts: cumulative $-saved telemetry vs hosted web search pricing
- Cache eviction policy (size cap / LRU) - required before any shared
  remote deployment
- Retry/backoff on search adapters; negative caching of failed fetches

## Deferred (valuable but heavy - from the competitor gap analysis)

Features common in the field that would take real engineering or conflict
with the library's design; revisit only with a concrete use case:

- News/images/video verticals: engines expose them, but plumbing separate
  result types through fetch/rank/extract is a real lift
- Site crawling / sitemap ingestion (Firecrawl's territory) - different
  product shape from query-driven search
- Hosted anti-bot muscle (proxy rotation, CAPTCHA solving) - an arms race;
  playwright rescue covers the measurable part today
- Async/batch endpoints - conflicts with the single-pipeline constraint;
  needs the shared-cache/eviction work first
- Geo/language localization params across engines
- Neural find-similar over a proprietary index (Exa's moat) - requires
  owning an index, out of scope by design
- Answer synthesis with citation formatting (Sonar-style) - in tool mode
  the calling model IS the synthesizer; a standalone answer endpoint is a
  different product

## Shipped

- 2026-07-13 (later): sentence-level compression (webfetch/compress.py) +
  tool context format (same-URL header merge, hostname-only headers).
  Eval-gated (evals/run_compression_eval.py: capture 50 live results once,
  sweep offline): tool results 665 -> 332 tokens mean (50%) with ZERO
  recall drop (29/29 answer survival). The first sweep FAILED the gate -
  bi-encoder/lexical scorers lose 3-5 answers at half tokens, and 26% of a
  result is header overhead compression can't touch; the cross-encoder
  scorer + header merging turned the negative result into a pass. Lexical
  fallback without [rerank]: 27/29 survival at 51%

- 2026-07-13: Layer 3 end-to-end eval (evals/run_e2e_eval.py): Claude Opus
  4.7 + our web_search tool vs Anthropic's hosted web_search, 50 SimpleQA
  questions, SimpleQA-style judging. RESULT: ours 48/50 (96%) at
  $0.025/query vs hosted 46/50 (92%) at $0.103/query - matches hosted
  accuracy (2-question edge is within noise at n=50) at 24% of the cost.
  Also hardened lazy model init with locks (concurrent tool serving raced
  torch construction) and serialized local pipeline execution in the eval
  (two concurrent pipelines deadlocked MPS/playwright)

- 2026-07-12: 4-engine fusion measured (ddg+brave+serper+tavily once real
  keys were added): Layer 2 recall 60% -> 64%, failed URLs 1.28 -> 0.8/query.
  With only 2 engines fusion was recall-flat; the gain came from Serper
  (Google index depth) and Tavily joining. Cost: fetch latency median
  10s -> 18s (more pages actually fetchable = more full downloads)

- 2026-07-12: extraction hardening (diagnosis-driven). PDF legibility gate +
  layout retry + table extraction; browser headers + playwright rescue for
  403-walled pages (with HTTP status check); head-metadata prepend (7/22
  misses had the answer only in title/og:description); thin-content render
  trigger for JS SPAs. Live Layer 2 recall 56% -> 60%, failed URLs
  1.68 -> 1.28/query. Negative result: headers alone recovered zero
  blocked pages - the playwright rescue is what works

- 2026-07-12: hybrid rank fusion (HybridRanker: full-list BM25 + bi-encoder
  RRF fusion -> cross-encoder, replacing the BM25-first gate). Live Layer 2
  recall 52% -> 56% at identical token cost; experiment on identical chunks
  measured 46% -> 54%. Fixed the "rank drops retrieved answers" gap
  (5/10 diagnosed misses)

- 2026-07-12: multi-engine RRF fusion (MultiSearchAdapter, Tavily adapter,
  "multi" factory provider). Measured: recall 52.0% vs 50% single-engine
  (flat - extraction is the recall bottleneck, see item 2), but suite
  errors 12 -> 0: transient engine failures are absorbed

- 2026-07-12: volatility-aware TTLs (realtime/recent/stable, model hint +
  hybrid classifier fallback, realtime recall 0.885)
- 2026-07-10: semantic query cache (bi 0.60 + NLI 0.97; paraphrase hits
  0/15 -> 11/11 runnable, zero wrong-target, offline prediction matched
  live 100%)
- 2026-07-09: two-layer eval harness (matcher sweep + live pipeline
  diagnostics; deterministic datasets)
- 2026-07-07: tool mode (pipeline orchestrator, sqlite cache, web_search
  tool schema + handler, concurrent fetch, agent loop example)
