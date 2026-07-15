# Contributing

Dev setup:

```
git clone https://github.com/firish/webfetch && cd webfetch
pip install -e ".[all,dev]"
pytest tests/ -q          # offline, no keys needed, ~1s
ruff check webfetch/ tests/
```

The one rule this project actually enforces: **features ship eval-first**.
Every non-trivial change to retrieval, ranking, caching, or compression
needs a measured gate before it merges - build or extend an eval in
`evals/`, state the acceptance number up front, and report the result
honestly (negative results are documented in `docs/ROADMAP.md`, not
deleted). Look at `evals/run_compression_eval.py` for the shape: capture
once, sweep offline, gate, and the eval imports the shipped module so what
is measured is what ships.

Smaller conventions, mostly enforced by review:

- Every major stage (search, fetch, rank, extract, cache) is an abstract
  base with swappable adapters. A new search provider is a new file plus a
  registry entry in `webfetch/search/__init__.py` - `pipeline.py` never
  changes.
- Heavy dependencies stay optional. Code that needs them degrades with a
  logged warning, not an ImportError.
- No hardcoded values - tunables live in `webfetch/config.py`.
- Keep `docs/architecture.md` current when interfaces or data flow change.

Live evals cost real API money and take 20+ minutes; maintainers run
those. For PRs, the offline suite plus a clear description of what you
measured is enough.
