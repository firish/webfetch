"""
Central constants and defaults for the webfetch pipeline.

All tuneable knobs live here so nothing is hardcoded in module logic.
Change a value here and it propagates everywhere - no grep-and-replace needed.
"""

# --- Search ---
# How many URLs to retrieve from the search adapter per query.
DEFAULT_N_RESULTS: int = 10

# Which search provider to use when none is specified by the caller.
# "ddg" = DuckDuckGo (free, no key), "brave" = Brave Search, "serper" = Serper/Google
DEFAULT_SEARCH_PROVIDER: str = "ddg"

# --- Fetch ---
# Seconds before an HTTP fetch is abandoned. Keeps slow pages from blocking the pipeline.
FETCH_TIMEOUT_SECS: int = 10

# --- Ranking (cascade thresholds) ---
# BM25 is always-on. Bi-encoder and cross-encoder are opt-in via Pipeline flags.
# Each stage trims the candidate list before passing to the next (cheaper) stage.
BM25_TOP_K: int = 20       # after BM25: keep best 20 chunks
BIENCODER_TOP_K: int = 10  # after bi-encoder: keep best 10
CROSSENCODER_TOP_K: int = 5  # after cross-encoder: keep best 5

# Default models for optional reranking stages.
BIENCODER_MODEL: str = "all-MiniLM-L6-v2"
CROSSENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Chunking ---
# Characters per text chunk fed into the ranker.
# 400 chars ~ 100 tokens - small enough for precise reranking, big enough for context.
DEFAULT_CHUNK_SIZE: int = 400

# Overlap between consecutive chunks to avoid splitting a key sentence at a boundary.
CHUNK_OVERLAP_RATIO: float = 0.10  # 10% overlap

# --- Token budget ---
# Maximum characters of ranked text sent to the LLM extraction call.
# ~6000 chars / 4 chars-per-token ~ 1500 tokens of context - cheap for haiku-class models.
DEFAULT_TOKEN_BUDGET: int = 6000

# --- Cache ---
# Default path for the sqlite3 cache database.
DEFAULT_CACHE_DB: str = "~/.webfetch/cache.db"

# How long cached results are kept before expiry. Specs rarely change, 90 days is safe.
DEFAULT_CACHE_TTL_DAYS: int = 90

# --- LLM extraction ---
# Default model for the extraction step. Cheapest capable model by default.
DEFAULT_EXTRACT_MODEL: str = "claude-haiku-4-5-20251001"
