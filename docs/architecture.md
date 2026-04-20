# Architecture

## Pipeline Overview

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
├── __init__.py
├── pipeline.py          # top-level orchestrator, ties all stages together
├── config.py            # constants, defaults, token budgets
│
├── search/
│   ├── base.py          # AbstractSearchAdapter interface
│   ├── ddg.py           # DuckDuckGo adapter (free, dev use)
│   ├── brave.py         # Brave Search API adapter (prod)
│   └── serper.py        # Serper/Google adapter (prod, best for industrial)
│
├── fetch/
│   ├── html.py          # trafilatura wrapper + fallback chain
│   └── pdf.py           # pdfplumber wrapper for datasheet PDFs
│
├── rank/
│   ├── bm25.py          # Stage 1: BM25 keyword ranking (rank_bm25)
│   ├── biencoder.py     # Stage 2: bi-encoder semantic ranking
│   ├── crossencoder.py  # Stage 3: cross-encoder pairwise reranking
│   └── rrf.py           # Reciprocal Rank Fusion utility
│
├── trim.py              # token budget management
├── extract.py           # LLM call -> structured JSON
└── cache.py             # diskcache/sqlite layer
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

### Reranker Stages
Each stage takes `(query: str, chunks: list[Chunk]) -> list[Chunk]` and is 
independently skippable via config.

| Stage | Library | Model | Runs on |
|---|---|---|---|
| BM25 | rank_bm25 | N/A | CPU, instant |
| Bi-encoder | sentence-transformers | all-MiniLM-L6-v2 or bge-small-en | CPU |
| Cross-encoder | sentence-transformers | cross-encoder/ms-marco-MiniLM-L-6-v2 | CPU |

### Cache
- Keyed by: `sha256(query + search_provider)`
- Stores: raw extracted text per URL, final ranked chunks, JSON output
- Backend: `diskcache` (default) or `sqlite3`
- TTL: configurable, default 30 days (specs rarely change)

## Dependency Groups

```toml
# Core (always installed)
dependencies = [
    "trafilatura",
    "rank-bm25",
    "requests",
    "diskcache",
    "duckduckgo-search",
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
all = ["specfetch[rerank,browser,pdf]"]
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

### LLM is extraction-only
The LLM never formulates search queries, browses pages, or decides what to 
fetch. All of that is deterministic Python. The LLM only sees pre-ranked, 
pre-trimmed text and outputs JSON. This keeps the LLM call cheap and fast.

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
> Last updated: 2026-04-19
> Update this file whenever: modules are added/renamed, interfaces change,
> new design decisions are made, or the pipeline data flow changes.