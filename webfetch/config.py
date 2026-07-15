"""
Central constants and defaults for the webfetch pipeline.

All tuneable knobs live here so nothing is hardcoded in module logic.
Change a value here and it propagates everywhere - no grep-and-replace needed.
"""

# --- Search ---
# How many URLs to retrieve from the search adapter per query.
DEFAULT_N_RESULTS: int = 10

# Which search provider to use when none is specified by the caller.
# "ddg" = DuckDuckGo (free, no key); see search/__init__.py for all.
DEFAULT_SEARCH_PROVIDER: str = "ddg"

# --- Fetch ---
# Seconds before an HTTP fetch is abandoned. Keeps slow pages from
# blocking the pipeline.
FETCH_TIMEOUT_SECS: int = 10

# Thread pool size for concurrent URL fetching. Fetching is IO-bound (network
# waits, not CPU), so more workers than cores is fine and cuts wall time a lot.
DEFAULT_FETCH_WORKERS: int = 8

# Extracted text shorter than this suggests a JS-rendered SPA whose real
# content never reached the prose extractor - triggers the playwright
# fallback even though extraction technically "succeeded".
MIN_EXTRACTED_CHARS: int = 300

# --- Ranking ---
# Default cascade: hybrid fusion (full-list BM25 + bi-encoder rankings fused
# via RRF) -> cross-encoder. Fusion replaced the BM25-first gate after the
# gap-1 experiment measured recall 46% -> 54% at identical token cost - the
# lexical gate was dropping semantically-relevant chunks it could not see.
HYBRID_FUSION_TOP_K: int = 30  # fused chunks passed to the cross-encoder

# Used by the degraded (no-embeddings) path and by callers composing the
# old-style cascade manually.
BM25_TOP_K: int = 20       # after BM25: keep best 20 chunks
BIENCODER_TOP_K: int = 10  # after bi-encoder: keep best 10
CROSSENCODER_TOP_K: int = 5  # after cross-encoder: keep best 5

# Default models for optional reranking stages.
BIENCODER_MODEL: str = "all-MiniLM-L6-v2"
CROSSENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Chunking ---
# Characters per text chunk fed into the ranker.
# 400 chars ~ 100 tokens - small enough for precise reranking, big
# enough for context.
DEFAULT_CHUNK_SIZE: int = 400

# Overlap between consecutive chunks to avoid splitting a key sentence
# at a boundary.
CHUNK_OVERLAP_RATIO: float = 0.10  # 10% overlap

# --- Token budget ---
# Maximum characters of ranked text sent to the LLM extraction call.
# ~6000 chars / 4 chars-per-token ~ 1500 tokens of context - cheap for
# haiku-class models.
DEFAULT_TOKEN_BUDGET: int = 6000

# --- Cache ---
# Default path for the sqlite3 cache database.
DEFAULT_CACHE_DB: str = "~/.webfetch/cache.db"

# How long cached results are kept before expiry. Acts as the hard ceiling
# for all rows (pages, legacy query rows without a freshness class).
DEFAULT_CACHE_TTL_DAYS: int = 90

# Volatility-aware TTLs, resolved at READ time so retuning these applies to
# already-cached rows. Effective TTL = min(TTL of the row's stored class,
# TTL of the caller's freshness hint, the ttl_days ceiling).
TTL_BY_FRESHNESS: dict[str, int] = {
    "realtime": 15 * 60,      # prices, scores, breaking news
    "recent": 7 * 86400,      # things that change over weeks/months
    "stable": 90 * 86400,     # historical/definitional facts, specs
}

# Class used when there is no model hint and the classifier is unsure,
# and for legacy rows cached before freshness existed.
DEFAULT_FRESHNESS: str = "recent"

# --- Search resilience ---
# Circuit breaker: an engine failing this many CONSECUTIVE times (errors,
# or silent-block empties) is benched for the cooldown, then given one
# half-open probe. Stops burning latency on a dead/blocking engine.
SEARCH_BREAKER_THRESHOLD: int = 3
SEARCH_BREAKER_COOLDOWN_SECS: float = 300.0

# In fusion, an engine's empty response only counts as a breaker failure
# when a peer engine returned at least this many results for the SAME
# query - distinguishes a silent block from a genuinely hard query.
FUSION_PEER_EMPTY_MIN: int = 3

# --- Cost receipts ---
# Baseline for savings estimates (webfetch/receipts.py). Both Anthropic and
# OpenAI charge $10/1k hosted searches (verified 2026-07); the hosted tools
# also inject ~17.4k content tokens per call vs our ~3.5k compressed
# (measured, Layer 3 eval 2026-07-14). Token price defaults to Opus-class
# input - override per call for cheaper models.
RECEIPT_HOSTED_SEARCH_FEE: float = 10.00 / 1000
RECEIPT_HOSTED_TOKENS_PER_CALL: int = 17400
RECEIPT_TOKEN_PRICE_PER_MTOK: float = 5.00

# --- LLM extraction ---
# Default model for the extraction step. Cheapest capable model by default.
DEFAULT_EXTRACT_MODEL: str = "claude-haiku-4-5-20251001"

# --- Semantic query cache ---
# Thresholds picked by the eval harness (evals/run_matcher_eval.py, 247-pair
# factoid-enriched set): bi >= 0.60 keeps paraphrase recall at 0.98 as a
# shortlist gate; NLI bidirectional entailment >= 0.97 gives precision 0.955
# with zero trusted-negative false positives. See matcher_recommendation.json.
SEMCACHE_BI_THRESHOLD: float = 0.60
SEMCACHE_CE_THRESHOLD: float = 0.97

# NLI verifier won the 4-model bake-off: only model to clear the 0.95
# precision bar, natively rejects entity/number swaps (Portland trap: 0.000).
SEMCACHE_CE_MODEL: str = "cross-encoder/nli-deberta-v3-base"

# Verify only the top cosine candidate - precision-first; Layer 2 measured
# zero wrong-target matches, so deeper candidate lists buy nothing.
SEMCACHE_MAX_CANDIDATES: int = 1

# --- Tool mode ---
# Max characters of ranked context returned per web_search tool call.
# Larger than DEFAULT_TOKEN_BUDGET because in tool mode the calling model is
# the extractor and benefits from a bit more surrounding context.
DEFAULT_TOOL_RESULT_BUDGET: int = 8000

# --- Sentence-level compression (tool-result formatting) ---
# Applied AFTER the cache read, so cached rows stay full-text and retuning
# these never invalidates the cache. Values picked by the eval sweep
# (evals/run_compression_eval.py, 50 captured production results): this
# config measured recall 29/50 vs 29/50 uncompressed (zero drop, 29/29
# answer survival) at 50% of baseline tokens (332 vs 665 mean). The
# cross-encoder scorer is what preserves recall - biencoder at the same
# tokens loses 3-4 answers. Without [rerank] it degrades to the lexical
# scorer (measured: 27/29 survival at 51%).
COMPRESSION_ENABLED: bool = True
COMPRESS_SCORER: str = "crossencoder"  # | "biencoder" | "lexical" | "lead"
COMPRESS_POLICY: str = "ratio"         # "ratio" | "topk" | "threshold"
COMPRESS_PARAM: float = 0.5     # keep best sentences, <= 50% of chunk chars
COMPRESS_ANAPHORA_GUARD: bool = True  # +14 tokens, keeps pronouns resolvable
COMPRESS_TABLE_GUARD: bool = True      # ablation: protects 1 answer of 29
COMPRESS_DEDUP: bool = True     # ablation: -13 tokens free (overlap dupes)

# Tool-result context format (build_context). Same eval: merging same-URL
# chunks under one header + hostname-only headers cut the fixed header
# overhead (26% of an uncompressed result) with zero measured recall cost -
# titles stay because answers live in them (extraction-hardening eval).
TOOL_MERGE_SOURCES: bool = True
TOOL_HEADER_STYLE: str = "domain"      # "full" | "domain"
