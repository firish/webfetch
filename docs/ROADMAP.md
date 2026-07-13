# Roadmap

Ordered by priority. Every feature follows the project practice: eval first,
implementation second, measured acceptance gate third. Shipped work moves to
the bottom.

## 1. Sentence-level extractive compression

Within each top-ranked chunk, keep only the sentences relevant to the query
(bi-encoder scored, batched) with guards: preceding-sentence retention for
pronoun-initial sentences (anaphora), skip-compression for table-like lines
(digits/units dense). Runs at tool-result formatting time, AFTER the cache,
so cached chunks stay budget-agnostic.

- Expected: ~612 -> ~200-300 tokens/result (2-3x vs our own baseline;
  claims vs raw-snippet competitors are larger but must be measured)
- Acceptance: tokens-vs-recall frontier from a Layer 2 budget sweep;
  gate roughly "tokens halved, recall drop under 2-4 points"
- Lift: ~a day (compressor is small; the eval extension is the work)

## 2. Robust PDF / table / datasheet extraction (important)

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

## 3. Stale-while-revalidate

Serve expired-but-present cache rows instantly with honest provenance
("[cache: stale, 18m old, realtime - refreshing]") and refresh in a
background worker. Pairs with volatility TTLs - benefit concentrates on the
realtime class. Grace window ~2x class TTL; single refresh worker with an
in-flight set; keep stale row if refresh fails.

- Acceptance: TTL unit suite extension + latency-at-expiry measurement
- Lift: ~a day

## Backlog (unordered)

- Layer 3 eval: LLM-judged end-to-end answer accuracy (release gate)
- MCP server wrapper (local stdio first; remote server needs a shared
  cache backend + eviction policy - see cache notes in ARCHITECTURE.md)
- OpenAI function-calling example loop
- OR-ensemble semantic-cache verifier (only if live hit rates disappoint)
- Cost receipts: cumulative $-saved telemetry vs hosted web search pricing
- Cache eviction policy (size cap / LRU) - required before any shared
  remote deployment
- Retry/backoff on search adapters; negative caching of failed fetches

## Shipped

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
