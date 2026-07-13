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
|                  |  -> chunk -> 3-stage rank -> cache store
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
│  3-Stage Ranker │
│  1. BM25        │  -> keyword filter, keep top 20 chunks
│  2. Bi-encoder  │  -> semantic similarity, keep top 10 chunks
│  3. Cross-enc.  │  -> pairwise rerank, keep top 3–5 chunks
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
│   ├── bm25.py          # Stage 1: BM25 keyword ranking (rank_bm25)
│   ├── biencoder.py     # Stage 2: bi-encoder semantic ranking
│   ├── crossencoder.py  # Stage 3: cross-encoder pairwise reranking
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
- Primary: `trafilatura.fetch_url()` + `trafilatura.extract(output_format='markdown')`
- Fallback 1: `readability-lxml` (reader-mode extraction)
- Fallback 2: `newspaper3k` (news/article optimized)
- Fallback 3: `playwright` (JS-rendered pages, optional install)
- PDF: `pdfplumber` triggered when URL ends in `.pdf` or content-type is PDF

### Chunker
`chunk_text(text, url, title, chunk_size=400, overlap_ratio=0.10) -> list[Chunk]`
- Character-based with whitespace-snapped boundaries (avoids splitting mid-word).
- 10% overlap between neighbors so a key phrase straddling a boundary still
  appears in one chunk whole.
- Zero extra deps - sentence splitters (nltk, spacy) are heavy and mishandle
  spec tables where "lines" are not sentences.

### Reranker Stages
Each stage implements `AbstractRanker.rank(query, chunks) -> list[Chunk]` and is
independently skippable via `rank()` flags.

| Stage | Library | Model | Runs on |
|---|---|---|---|
| BM25 | rank_bm25 | N/A | CPU, instant |
| Bi-encoder | sentence-transformers | all-MiniLM-L6-v2 or bge-small-en | CPU |
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
- Result formatting reuses `build_context()` from extract/base.py (the
  `[Source: title | url]` labeling under a char budget).

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

### 3-stage reranking mirrors production systems
BM25 alone misses semantic matches ("uncertainty" ≠ "accuracy tolerance"). 
Bi-encoder alone is slow over hundreds of chunks. Cross-encoder alone is too 
expensive at scale. The cascade (BM25 -> bi-encoder -> cross-encoder) mirrors 
what Perplexity and Bing AI use: cheap-fast-imprecise -> expensive-slow-precise.

### RRF for signal fusion
Reciprocal Rank Fusion combines BM25 and vector rankings without requiring 
score normalization. Used in production at Cohere, Elastic, and others.

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

### Multi-engine RRF fusion: resilience first, recall second (measured)
`MultiSearchAdapter` fans one query out to every engine with credentials and
fuses ranked URL lists via RRF (same formula as rank/rrf.py, keyed by
normalized URL). Engine failures are logged and skipped; it raises only when
ALL engines fail. Measured on Layer 2: retrieval recall 52.0% fused vs 50%
brave-only (within noise - search is not the recall bottleneck; extraction
is, see ROADMAP item 3), but errors went 12 -> 0 across a 65-query suite
because transient single-engine failures are absorbed. Fusion runs cache
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

### Tool handler as the error boundary
Pipeline (library layer) raises on search failure; handle_web_search (agent
boundary) catches everything and returns readable error strings. An exception
thrown mid-agent-loop kills the whole conversation, while an error string
lets the model retry or rephrase.

## Eval Harness

Two eval layers exist so feature claims ship with numbers (a third,
LLM-judged end-to-end layer is planned but not built):

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

Dataset provenance lives in the .meta.json sidecars; sampling is seeded so
rebuilds are byte-identical. FreshQA (volatility-labeled queries) is also
sampled and banked for the future volatility-TTL feature.

## Known Gaps (T&M Use Case)

These are content types the fetch stage does not handle well today, discovered
while planning for calibration equipment spec lookups.

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
> Last updated: 2026-07-12 - volatility-aware TTLs: per-class expiry
> (realtime/recent/stable) resolved at read time, freshness tool param,
> hybrid classifier fallback (volatility.py + shipped centroids), and the
> volatility eval (evals/run_volatility_eval.py).
> Previous: 2026-07-10 semantic query cache; 2026-07-09 eval harness.
> Previous: 2026-07-07 - Pipeline orchestrator, sqlite cache, concurrent
> fetch_all, shared ranker instances, web_search tool layer, examples/.
> Update this file whenever: modules are added/renamed, interfaces change,
> new design decisions are made, or the pipeline data flow changes.