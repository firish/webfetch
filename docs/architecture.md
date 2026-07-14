# Architecture

## Two Modes of Operation

1. **Tool mode (primary going forward):** webfetch serves a `web_search`
   custom tool inside an LLM agent loop. The model formulates queries and
   emits tool_use; webfetch returns ranked, source-labeled excerpts as the
   tool_result; the model reasons over them. The extract stage is NOT used -
   the calling model is the extractor.
2. **Structured extraction mode (original use case):** deterministic Python
   drives one query end-to-end and a cheap LLM call returns structured JSON.

## Tool Mode Data Flow

```
[user question]
      |
      v
+------------------+   tools=[WEB_SEARCH_TOOL]
|  Frontier model  | <--------------------------- examples/agent_loop.py
|  (Claude, etc.)  |
+--------+---------+
         | tool_use {"query": ...}
         v
+------------------+
| handle_web_search|  tool.py - catches ALL errors, never raises
+--------+---------+
         v
+------------------+
| Pipeline         |  pipeline.py
|  .search_chunks  |  search -> cache check -> fetch_all (parallel)
|                  |  -> chunk -> hybrid rank -> cache store
+--------+---------+
         | ranked chunks
         v
+------------------+
| build_context()  |  [Source: title | url] labeled excerpts under budget
+--------+---------+
         | tool_result (str)
         v
+------------------+
|  Frontier model  |  reasons over excerpts, cites URLs, may search again
+------------------+
```

## Extraction Mode Pipeline Overview

```
[query: str]
     │
     v
┌─────────────────┐
│  Search Adapter │  -> returns list of URLs + snippets
└────────┬────────┘
         │
         v
┌─────────────────┐
│   Cache Check   │  -> if mfr+model seen before, skip to LLM call
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Fetch + Extract │  -> trafilatura -> readability -> newspaper3k -> playwright
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Hybrid Ranker  │
│  1. BM25+bi RRF │  -> full-list lexical AND semantic rankings fused,
│     fusion      │     keep top 30 (either signal keeps a chunk alive)
│  2. Cross-enc.  │  -> pairwise rerank, keep top 5
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Token Trimmer  │  -> fit chunks within LLM context budget
└────────┬────────┘
         │
         v
┌─────────────────┐
│    LLM Call     │  -> extraction only, structured JSON output
└────────┬────────┘
         │
         v
   [JSON output]
```

## Module Structure

```
webfetch/
├── __init__.py          # package root - exports Pipeline, cache, tool API
├── pipeline.py          # Pipeline orchestrator: search_chunks() + run()
├── tool.py              # WEB_SEARCH_TOOL schema + handle_web_search handler
├── cache.py             # AbstractCache + CacheMatch + SqliteCache (exact match)
├── semcache.py          # SemanticSqliteCache: paraphrase-matching query cache
├── volatility.py        # freshness classifier (keywords -> centroid hybrid)
├── compress.py          # sentence-level extractive compression (tool results)
├── data/                # shipped artifacts (volatility centroids, ~25KB)
├── config.py            # constants, defaults, token budgets
│
├── search/
│   ├── __init__.py      # get_search_adapter() factory ("multi" = fusion) + exports
│   ├── base.py          # AbstractSearchAdapter interface
│   ├── ddg.py           # DuckDuckGo adapter (free, dev use)
│   ├── brave.py         # Brave Search API adapter (prod)
│   ├── serper.py        # Serper/Google adapter (prod, best for industrial)
│   ├── tavily.py        # Tavily adapter (LLM-focused API, free tier)
│   └── multi.py         # MultiSearchAdapter: parallel fan-out + URL-keyed RRF
│
├── fetch/
│   ├── __init__.py      # fetch() router + fetch_all() thread-pool fetching
│   ├── html.py          # trafilatura wrapper + fallback chain
│   └── pdf.py           # pdfplumber wrapper for datasheet PDFs
│
├── rank/
│   ├── __init__.py      # rank() cascade + shared ranker instances
│   ├── base.py          # Chunk dataclass + AbstractRanker interface
│   ├── chunker.py       # Text -> Chunk objects (char-based, whitespace-snapped)
│   ├── hybrid.py        # Stage 1 (default): BM25 + bi-encoder RRF fusion
│   ├── bm25.py          # BM25 keyword ranking (rank_bm25)
│   ├── biencoder.py     # bi-encoder semantic ranking
│   ├── crossencoder.py  # Stage 2: cross-encoder pairwise reranking
│   └── rrf.py           # Reciprocal Rank Fusion utility
│
└── extract/
    ├── base.py          # AbstractExtractor + context builder + JSON parser
    ├── claude.py        # Anthropic Claude adapter
    ├── gpt.py           # OpenAI adapter
    ├── gemini.py        # Google Gemini adapter (free tier)
    └── groq.py          # Groq adapter (free tier, fastest)

examples/
└── agent_loop.py        # manual Anthropic agentic loop using WEB_SEARCH_TOOL

evals/                   # standalone eval scripts - NOT part of the webfetch package
├── common.py            # answer normalization/variants, jaccard, jsonl io, tables
├── fetch_datasets.py    # downloads raw QQP/SimpleQA/FreshQA into datasets/raw/
├── build_datasets.py    # deterministic sampling (SEED=42) -> checked-in JSONL
├── run_matcher_eval.py  # Layer 1: offline semantic-matcher threshold sweep
├── run_pipeline_eval.py # Layer 2: live pipeline metrics + cache diagnostics
├── run_e2e_eval.py      # Layer 3: answer accuracy + cost vs hosted web_search
├── run_volatility_eval.py    # freshness classifier bake-off (FreshQA)
├── run_compression_eval.py   # compression frontier: capture + offline sweep
├── test_metrics.py      # plain-assert self-test of the metric primitives
├── datasets/            # checked-in JSONL samples (+ .meta.json provenance sidecars)
└── results/             # gitignored run outputs
```

## Component Interfaces

### Search Adapter (abstract base)
```python
class AbstractSearchAdapter:
    def search(self, query: str, n_results: int = 10) -> list[SearchResult]:
        ...

@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    rank: int
```

### Fetch + Extract
- Raw fetch chain: `trafilatura.fetch_url()` -> requests with full browser
  headers (bot walls 403 a bare UA) -> playwright real-browser rescue
  (checks HTTP status - error pages are NOT content)
- Extraction chain: `trafilatura.extract(markdown, favor_recall)` ->
  `readability-lxml` -> `newspaper4k` -> playwright render. The render also
  triggers on THIN output (< MIN_EXTRACTED_CHARS) - JS SPAs extract to a
  nav/footer remnant that used to mask the real content
- Head metadata (title + meta/og description) is prepended to every page -
  miss diagnosis found factoid answers living ONLY there (7 of 22 misses)
- HTML tables via pandas.read_html appended as markdown
- PDF: `pdfplumber` per page with a legibility gate - garbled columnar
  output retries layout-aware extraction, still-garbled pages are dropped
  (garbage chunks were winning ranking slots); PDF tables appended as
  markdown like the HTML path

### Chunker
`chunk_text(text, url, title, chunk_size=400, overlap_ratio=0.10) -> list[Chunk]`
- Character-based with whitespace-snapped boundaries (avoids splitting mid-word).
- 10% overlap between neighbors so a key phrase straddling a boundary still
  appears in one chunk whole.
- Zero extra deps - sentence splitters (nltk, spacy) are heavy and mishandle
  spec tables where "lines" are not sentences.

### Reranker Stages
Default cascade: HybridRanker (BM25 + bi-encoder full-list RRF fusion,
top 30) -> CrossEncoderRanker (top 5). `use_biencoder=False` falls back to
the plain BM25 gate.
Each stage implements `AbstractRanker.rank(query, chunks) -> list[Chunk]` and is
independently skippable via `rank()` flags.

| Stage | Library | Model | Runs on |
|---|---|---|---|
| Hybrid fusion (BM25 + bi-encoder via RRF) | rank_bm25 + sentence-transformers | all-MiniLM-L6-v2 | CPU, ~1-2s warm |
| Cross-encoder | sentence-transformers | cross-encoder/ms-marco-MiniLM-L-6-v2 | CPU |

### Extractor
```python
class AbstractExtractor:
    def extract(
        self,
        chunks: list[Chunk],
        keys: dict[str, str],       # field_name -> short description
        budget_chars: int = 6000,
    ) -> dict[str, str | None]:
        ...
```
- Shared base handles context trimming (respects `DEFAULT_TOKEN_BUDGET`),
  prompt assembly, and JSON parsing. Subclasses implement only `_call_llm()`.
- Adapters: `ClaudeExtractor`, `GPTExtractor`, `GeminiExtractor`.
- Deviation from initial architecture: `extract.py` + `trim.py` were merged
  into the `extract/` package so trimming lives with the LLM call that
  consumes it. No caller refactor needed - this is still the last pipeline
  stage.

### Pipeline (orchestrator)
```python
class Pipeline:
    def __init__(self, search=None, rankers=None, cache=None,
                 n_results=DEFAULT_N_RESULTS, max_workers=DEFAULT_FETCH_WORKERS,
                 use_biencoder=True, use_crossencoder=True) -> None: ...
    def search_chunks(self, query: str, n_results: int | None = None) -> SearchChunksResult: ...
    def run(self, query, keys, extractor, budget_chars=DEFAULT_TOKEN_BUDGET) -> dict[str, str | None]: ...

@dataclass
class SearchChunksResult:
    query: str
    chunks: list[Chunk]           # ranked, best first
    results: list[SearchResult]   # raw search hits
    failed_urls: list[str]        # fetches that returned no text
    from_cache: bool              # query-level cache hit
    elapsed_secs: float
```
- `search_chunks()` is tool mode; `run()` is extraction mode (search_chunks + extractor).
- All collaborators injected (composition) - swapping any stage touches zero
  pipeline code.
- Error boundary: search exceptions propagate from Pipeline (library layer);
  per-URL fetch failures are skipped into `failed_urls`, never raised.

### Tool layer (tool.py)
```python
WEB_SEARCH_TOOL: dict   # Anthropic custom-tool schema, single "query" param
def handle_web_search(tool_input: dict, pipeline: Pipeline | None = None,
                      budget_chars: int = DEFAULT_TOOL_RESULT_BUDGET) -> str: ...
def get_default_pipeline() -> Pipeline: ...  # lazy module-level singleton
```
- `handle_web_search` NEVER raises - all failures return readable error
  strings so an agent loop cannot crash mid-conversation.
- Result formatting: chunks are compressed (compress.py, below) then passed
  to `build_context()` from extract/base.py with the tool format - same-URL
  chunks merged under ONE `[Source: title | hostname]` header
  (TOOL_MERGE_SOURCES / TOOL_HEADER_STYLE in config).

### Sentence-level compression (compress.py)
```python
def compress_chunks(query: str, chunks: list[Chunk], ...) -> list[Chunk]: ...
```
- Runs at tool-result formatting time, AFTER the cache read - cached rows
  stay full-text, so retuning compression never invalidates the cache.
- Splits each chunk into sentences (regex, zero-dep; newlines are hard
  boundaries), scores them against the query with the ms-marco
  cross-encoder (one batched predict over all ~15-25 sentences of a
  result), keeps the best up to 50% of chunk chars, in document order.
- Guards: pronoun-initial keeps retain their predecessor (anaphora chains
  resolve); table-like lines (`|` or digit-dense) bypass scoring; sentences
  re-emitted by the chunker's 10% overlap are deduped across chunks.
- Never raises and never mutates inputs; degrades to a zero-dep lexical
  (IDF-weighted term coverage) scorer without webfetch[rerank].
- Config picked by evals/run_compression_eval.py: recall 29/50 vs 29/50
  uncompressed (29/29 answer survival) at 50% of baseline tokens
  (332 vs 665 mean, formats included).

### Concurrent fetching
- `fetch_all(urls, max_workers=DEFAULT_FETCH_WORKERS) -> dict[str, str | None]`
  in fetch/__init__.py runs `fetch()` in a ThreadPoolExecutor - fetching is
  IO-bound, so this cuts the slowest pipeline stage roughly by worker count.
- Fetcher singletons hold only immutable config, so they are thread-safe.

### Ranker instance reuse
- rank/__init__.py holds module-level shared ranker instances; encoder models
  lazy-load once per process and stay warm across queries. `default_rankers()`
  returns the cascade for Pipeline injection; `rank()` delegates to the same
  shared instances.

### Semantic query cache (semcache.py)
- `SemanticSqliteCache(SqliteCache)` adds a paraphrase fallback to query
  lookups. Two-stage match, mirroring the ranking cascade's
  cheap-then-precise shape:
  1. Bi-encoder shortlist: cosine of the incoming query's embedding vs all
     cached query embeddings (in-memory matrix, one dot product). Gate:
     `SEMCACHE_BI_THRESHOLD` (0.60) - a recall gate, NOT a match decision.
  2. NLI verification: `cross-encoder/nli-deberta-v3-base` scores the pair
     jointly in both directions; match requires min P(entailment) >=
     `SEMCACHE_CE_THRESHOLD` (0.97). Entailment natively rejects
     entity/number/time swaps ("Portland Oregon" vs "Portland Maine").
- Pipeline talks to caches only via `AbstractCache.lookup()/store()` which
  return/accept `CacheMatch` (chunks + kind + matched_query + age) - so
  exact and semantic caches are drop-in interchangeable.
- Degrades to exact-match without `webfetch[rerank]` (one logged warning),
  same pattern as the encoder ranking stages.
- New table: `query_embeddings(key PK, query, vector BLOB, created_at)`;
  embeddings of expired chunk rows are cleaned up lazily on lookup.
- Measured (evals, 247-pair set): precision 0.955 (~1.0 after label-noise
  audit), factoid paraphrase recall 11/15, zero trusted-negative false
  positives. Warm semantic lookup ~150-250ms (embed + NLI verify), only paid
  on an exact-cache miss with a shortlist candidate.

### Cache
- Two layers in one sqlite database (cache.py):
  - `pages(url PK, text, created_at)` - extracted text per URL, reusable
    across queries that hit the same page
  - `queries(key PK, chunks_json, created_at)` - final ranked chunks, keyed
    by `sha256(normalized_query + provider + n_results)`
- Backend: stdlib `sqlite3` (WAL mode, one lock) - no diskcache dependency
- TTL: volatility-aware per query row. Each row stores a freshness class
  ("realtime" 15m / "recent" 7d / "stable" 90d, `TTL_BY_FRESHNESS` in
  config); expiry is resolved at READ time so retuning applies to existing
  rows. Effective TTL = min(stored class, caller hint, `ttl_days` ceiling).
  A stricter hint produces a miss but does NOT delete a row still valid for
  its own class. Legacy rows (no class) count as "recent". Pages keep the
  flat ceiling (URL-keyed, shared across query classes). Expired rows are
  lazily deleted on read - no background sweeper
- Freshness classification: model hint (tool param) is authoritative; the
  library fallback is a hybrid classifier (volatility.py) - keyword cue
  rules, then nearest-centroid over a ~25KB shipped artifact derived from
  FreshQA. Measured (evals/run_volatility_eval.py): accuracy 0.627,
  realtime recall 0.885; degrades to keywords-only without [rerank]
- Failed fetches are NOT cached (a dead URL today may work tomorrow)
- Transparent: `Pipeline(cache=None)` behaves identically, just slower

## Dependency Groups

```toml
# Core (always installed) - cache uses stdlib sqlite3, no diskcache
dependencies = [
    "trafilatura",
    "rank-bm25",
    "requests",
    "ddgs",        # maintained successor of duckduckgo-search
    "anthropic",
]

# Optional: semantic reranking
[project.optional-dependencies]
rerank = [
    "sentence-transformers",
    "numpy",
]

# Optional: JS-rendered pages
browser = [
    "playwright",
]

# Optional: PDF datasheets
pdf = [
    "pdfplumber",
]

# Optional: all
all = ["webfetch[rerank,browser,pdf]"]
```

## Design Decisions

### Swappable adapters over a monolithic pipeline
Every stage (search, fetch, rank, extract) is an abstract interface. Swapping 
a search provider or reranker requires only a new adapter file — `pipeline.py` 
never changes. This follows the Open/Closed Principle and is how production 
systems like Cohere and Elastic structure their retrieval pipelines.

### Hybrid fusion first stage (replaced the BM25-first cascade gate)
The original cascade (BM25 top-20 -> bi-encoder top-10 -> cross-encoder
top-5) let BM25 act as a hard lexical gate: chunks that were semantically
relevant but shared no query keywords died before the encoders saw them.
Miss diagnosis measured this as 3 of 10 recall failures (e.g. gold
"405 x 480 cm" in a table chunk vs query "dimensions of the painting").
The default first stage is now HybridRanker: full-list BM25 AND full-list
bi-encoder rankings fused via RRF, top 30 to the cross-encoder. A chunk
survives if EITHER signal ranks it well. Measured (gap-1 experiment, 50
queries, identical chunks): recall 46% -> 54% at identical token cost,
+~2.2s ranking latency (fetch still dominates). Wider cascade gates alone
(50/15) bought only +2 points; final top-8 added 60% tokens for zero
recall gain - both rejected.

### RRF for signal fusion
Reciprocal Rank Fusion combines rankings without score normalization -
used twice: inside HybridRanker (BM25 + vector rankings) and in
MultiSearchAdapter (cross-engine URL fusion). Used in production at
Cohere, Elastic, and others.

### Cache before fetch, not after
Cache is checked after the search step (we have URLs) but before fetching 
page content. This avoids re-fetching pages for repeated queries on the same 
equipment model — the single biggest cost reduction lever for T&M use cases.

### LLM is extraction-only (extraction mode)
In extraction mode the LLM never formulates search queries, browses pages, or 
decides what to fetch. All of that is deterministic Python. The LLM only sees 
pre-ranked, pre-trimmed text and outputs JSON. This keeps the LLM call cheap 
and fast.

### Tool mode inverts the extraction-only principle (deliberately)
In tool mode the calling model DOES formulate queries and decide when to
search - that is the point: it replaces the hosted web-search tool (~$10/1k
searches plus content tokens) with a local pipeline whose marginal cost is
~2k tokens of ranked context per call. The extract stage is skipped entirely;
the calling model is the extractor. Both modes share every stage below the
entry point.

### Stdlib sqlite3 over diskcache
The access pattern is key lookups on small rows at low write volume. diskcache
would add a dependency for eviction policies we do not need. WAL mode plus a
single lock handles the concurrent fetch-pool writes. Alternative considered:
diskcache (original spec) - rejected to keep core deps lean per project rules.

### ThreadPoolExecutor over async rewrite
Fetching is IO-bound, so threads capture nearly all of the win without
converting the whole codebase to async and forcing async on library users.
Alternative considered: httpx + asyncio - rejected as a much larger refactor
for marginal gain at 10-URL batch sizes.

### NLI verifier over duplicate-question models (semantic cache)
Four cross-encoders were benchmarked on the factoid-enriched matcher set.
quora-distilroberta leaked a place-disambiguation trap (Portland Oregon/
Maine at 0.984) and was underconfident on factoid paraphrases (4/15).
nli-deberta-v3-base, scored as bidirectional entailment (min of
P(entailment) both ways), was the only model to clear the 0.95 precision
bar, with zero trusted-negative false positives and 11/15 factoid recall.
Alternatives considered: quora-roberta-large (best raw recall 0.76 but
below the precision bar), stsb-roberta-base (0.86 precision), OR-ensembles
(future work if live hit rates disappoint).

### No lexical veto stage (tested and rejected)
A conflict veto on numbers/entities was prototyped: it removed zero false
positives (NLI already rejects token conflicts) and killed true paraphrases
on unicode/capitalization edge cases. Kept out of the design; documented as
optional defense-in-depth only for weaker verifiers.

### Cache provenance to the model, thresholds to config
Tool results start with one provenance header line, e.g.
`[cache: semantic match to "X", 2d old]`, and the tool schema exposes a
single `force_fresh` boolean. The model gets facts and one lever; similarity
thresholds, verifier choice, and TTLs stay in config where the eval harness
can tune them. Rationale: showing the matched query lets the calling model
catch any residual semantic false positive for ~20 tokens, while per-call
threshold knobs would make behavior non-reproducible.

### Extraction fixes are diagnosis-driven
The extraction hardening (metadata prepend, thin-content render trigger,
PDF legibility gate, playwright fetch rescue) came from a per-miss
diagnostic that localized every Layer 2 recall failure to a pipeline stage
and inspected where the answer text actually lived in the raw HTML. Notable
negative result: browser-quality headers alone recovered ZERO of the
403-blocked pages (the walls check more than the UA) - the playwright
rescue is what gets through them.

### Multi-engine RRF fusion: resilience first, recall second (measured)
`MultiSearchAdapter` fans one query out to every engine with credentials and
fuses ranked URL lists via RRF (same formula as rank/rrf.py, keyed by
normalized URL). Engine failures are logged and skipped; it raises only when
ALL engines fail. Measured on Layer 2: with 2 engines (ddg+brave) recall was flat vs
brave-only but errors went 12 -> 0 (transient failures absorbed); with 4
engines (adding serper + tavily) recall rose 60% -> 64% and failed URLs
halved - index diversity is what pays, resilience comes free either way. Fusion runs cache
under their own provider_name ("multi(ddg+brave+...)") so cached results
never mix across provider configurations.

### Volatility classification: model hint first, measured fallback second
The calling model classifies its own query's volatility via the `freshness`
tool param - it has conversation context no library classifier can match.
The fallback for hint-less calls was picked by a bake-off on FreshQA labels:
a keywords-then-centroid hybrid (realtime recall 0.885) dominated keywords
alone (0.612), nearest-centroid alone (0.673), and kNN (0.750, and kNN would
have required shipping the dataset). Selection was cost-weighted: a
fast-changing query classified stable serves stale results for months, while
the reverse only costs an extra search.

### Compression: cross-encoder sentences, post-cache, formats count (measured)
Tool results are compressed by sentence selection, not LLM rewriting -
abstractive compression needs an API call per result, against the
zero-marginal-cost design. The first sweep FAILED the registered gate
(tokens halved, recall drop <= 2/50): bi-encoder and lexical scorers lose
3-5 answers of 29 at half tokens. Two findings turned it around: (1) the
ms-marco cross-encoder (already shipped for ranking, cheap at ~20 sentences
per result) selects strictly better - zero answers lost; (2) 26% of an
uncompressed result is fixed source-header overhead sentence selection
cannot touch, so formatting joined the frontier: same-URL header merging
(top-5 chunks span ~3.7 URLs) plus hostname-only headers are recall-free
token cuts. Titles stay in headers - the extraction-hardening diagnosis
showed answers living in them. Compression runs after the cache read so
cached rows stay budget-agnostic. Alternatives rejected by the sweep:
lead-position baseline (14/29 survival - scoring earns its keep), bi-encoder
(24-28/29), threshold policies (dominated by ratio at equal tokens).

### Tool handler as the error boundary
Pipeline (library layer) raises on search failure; handle_web_search (agent
boundary) catches everything and returns readable error strings. An exception
thrown mid-agent-loop kills the whole conversation, while an error string
lets the model retry or rephrase.

## Eval Harness

Three eval layers exist so feature claims ship with numbers:

- **Layer 1 (run_matcher_eval.py, offline):** labeled query pairs (QQP
  paraphrases, stratified hard negatives, hand-written adversarial traps
  like version/entity swaps) swept across similarity thresholds. Picks the
  operating point for the upcoming semantic query cache. Selection is
  precision-first (bar 0.98, fallback 0.95) because a semantic-cache false
  hit silently serves wrong results, while a miss only costs a re-search.
  Output: evals/results/matcher_recommendation.json, consumed by Layer 2.
- **Layer 2 (run_pipeline_eval.py, live DDG):** topic-stratified SimpleQA
  factoid queries through Pipeline.search_chunks. Measures retrieval recall
  (gold-answer variant substring match), tokens per tool result, latency,
  and cache diagnostics: exact-replay hit rate (must be 1.0), paraphrase
  hit rate under the exact cache (baseline ~0), the semantic-cache
  OPPORTUNITY rate (what a semantic cache would serve, measured before the
  feature exists), and false-hit risk across the suite.

- **Layer 3 (run_e2e_eval.py, live API):** the release gate - the same
  agent loop answers the same 50 questions with swappable arms:
  ours-multi (production 4-engine fusion), ours-ddg ("miser" - $0 engine
  fees), hosted Anthropic web_search, Tavily/Exa as tool backends, and
  Perplexity Sonar as a direct-answer alternative. SimpleQA-style judging;
  honest cost model (Opus tokens + per-search engine fees for ours, each
  competitor's published request fees). First two-arm run (Opus 4.7,
  2026-07-13): ours 96% at $0.025/query vs hosted 92% at $0.103/query -
  hosted accuracy matched at 24% of the cost (token cost only; engine
  fees add ~$0.013/fresh search).

Feature-scoped evals follow the same pattern: run_volatility_eval.py
(classifier bake-off) and run_compression_eval.py (capture 50 live results
once, sweep compression configs x context formats offline on the identical
chunks - controlled comparison, no repeated network runs).

Dataset provenance lives in the .meta.json sidecars; sampling is seeded so
rebuilds are byte-identical. FreshQA (volatility-labeled queries) is also
sampled and banked for the future volatility-TTL feature.

## Known Gaps (T&M Use Case)

These are content types the fetch stage does not handle well today, discovered
while planning for calibration equipment spec lookups.
> Status 2026-07-12: gap 1 (HTML tables) shipped via pandas.read_html; PDF
> text/tables hardened (legibility gate + layout retry + extract_tables);
> JS-rendered pages handled by the playwright render/rescue paths. Gap 3
> (image-only spec plates) remains out of scope.

### 1. HTML Table Extraction
**Problem:** trafilatura mangles complex or nested HTML tables. Most manufacturer
spec pages (Fluke, Keysight, Mitutoyo) present specs as HTML tables, not prose.

**Fix:** Add `pandas.read_html()` as a parallel table extractor that runs
alongside trafilatura. Tables get serialized to markdown and appended to the
page text before ranking.

**Lift:** Small - ~half a day. `pandas.read_html()` is one call; the work is
normalizing its DataFrame output into the same text format the ranker expects.
No new optional dep needed (pandas is already a transitive dep of
sentence-transformers).

### 2. JS-Gated PDF Downloads
**Problem:** Many manufacturer sites serve datasheets behind a JavaScript
download button (onclick handler or AJAX). The fetcher never sees the PDF URL
because it never executes JS.

**Two-phase fix:**
- Phase 1 (small lift, ~2 hours): After fetching a page, scan the raw HTML for
  any `href` or `src` attributes ending in `.pdf` and enqueue those URLs for
  the PDF fetcher. Catches most "Download Datasheet" links that are plain
  anchor tags.
- Phase 2 (medium lift, ~2 days): Use Playwright to render the page and scrape
  PDF links from the rendered DOM. Catches links injected by JS. Does NOT click
  buttons or intercept file downloads - that remains out of scope.

**Phase 1 alone covers the majority of real cases** - most "download" buttons
are just styled anchor tags with a direct PDF href.

### 3. Image-Only Spec Plates (Out of Scope)
Scanned PDFs or pages where specs appear only as images require OCR
(pytesseract) or a vision LLM. Not planned - the cost and complexity are not
justified for the current use case volume.

## Status
> Last updated: 2026-07-13 (later) - sentence-level compression
> (compress.py) + tool context format (same-URL header merge, hostname
> headers): tool results halved to ~332 tokens mean with ZERO measured
> recall drop (29/29 survival), cross-encoder scored, eval-gated
> (run_compression_eval.py). build_context() gained merge_sources /
> header_style params.
> Previous same day - Layer 3 e2e eval: ours 96% accuracy at $0.025/query
> vs hosted 92% at $0.103/query.
> Previous: 2026-07-12 - extraction hardening: PDF legibility
> gate/layout retry/tables, playwright fetch rescue + thin-content render,
> head-metadata prepend. Layer 2 recall 50% -> 64% across the day's five
> fixes (engine fusion, rank fusion, extraction, 4-engine fusion).
> Previous same day - volatility-aware TTLs: per-class expiry
> (realtime/recent/stable) resolved at read time, freshness tool param,
> hybrid classifier fallback (volatility.py + shipped centroids), and the
> volatility eval (evals/run_volatility_eval.py).
> Previous: 2026-07-10 semantic query cache; 2026-07-09 eval harness.
> Previous: 2026-07-07 - Pipeline orchestrator, sqlite cache, concurrent
> fetch_all, shared ranker instances, web_search tool layer, examples/.
> Update this file whenever: modules are added/renamed, interfaces change,
> new design decisions are made, or the pipeline data flow changes.