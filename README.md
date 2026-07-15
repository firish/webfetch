# webfetch

Web search for LLM agents that you run yourself.

Hosted web-search tools charge $10 per thousand searches (Anthropic and
OpenAI both, as of July 2026) and then bill you again for every token of
retrieved content they stuff into your context window - about 17,000 input
tokens per search in our measurements. webfetch replaces that with a local
pipeline: multi-engine search, page fetching and extraction, semantic
reranking, and sentence-level compression, exposed as a `web_search` tool
your model calls like any other. You pay for the tokens of a compressed,
ranked result (~2-3.5k) and, optionally, a fraction of a cent in search
engine fees. Repeated and paraphrased queries are served from a semantic
cache and cost nothing at all.

The pipeline was built eval-first: every stage shipped behind a measured
gate, and the numbers below come from a benchmark you can rerun from this
repo.

## The numbers

Same agent loop, same 50 SimpleQA questions, same judge. Only the search
tool changes. Cost per query includes model tokens and all search fees.

| tool backend | model | accuracy | cost/query | input tok/query |
|---|---|---|---|---|
| webfetch (4-engine fusion) | gpt-5.6-sol | 96% | $0.040 | 2,156 |
| OpenAI hosted web_search | gpt-5.6-sol | 100% | $0.066 | 10,027 |
| Anthropic hosted web_search | Opus 4.7 | 96% | $0.108 | 17,408 |
| webfetch (4-engine fusion) | Opus 4.7 | 92% | $0.035 | 3,467 |
| Exa (search + contents) | Opus 4.7 | 90% | $0.053 | 5,496 |
| Tavily | Opus 4.7 | 88% | $0.047 | 6,387 |
| webfetch (DDG only, $0 fees) | Opus 4.7 | 84% | $0.026 | 3,623 |
| webfetch (4-engine fusion) | Haiku 4.5 | 76%* | $0.031 | 3,021 |

\* the Haiku run's failures were mostly the model re-searching past the
turn cap, and its last few questions ran with a degraded engine set after
we exhausted a free tier mid-benchmark. Treat it as a floor.

On a second dataset of 27 questions about events from the two weeks before
the benchmark ran (hand-written, never published, so no vendor could have
tuned on them), webfetch scored 100% with fusion and 100% with DDG alone;
the hosted tools also scored 100%. Fresh events are not the hard part -
see the date-injection note below, which is the actual trap.

Two structural advantages don't show in a single-query benchmark:

- Caching. Identical queries hit an exact cache; paraphrased queries hit a
  semantic cache (embedding shortlist, then NLI verification - measured
  11/11 paraphrase hits with zero wrong matches). Cache hits skip engines
  and fetching entirely. No hosted tool or search API we surveyed offers
  any client-visible caching, let alone paraphrase-aware caching.
- Token efficiency compounds. Every result webfetch returns is roughly a
  fifth of what hosted search injects, and in a multi-search agent
  conversation that difference is paid on every subsequent turn.

Reproduce: `python evals/run_e2e_eval.py --arms ours-multi,hosted` (see
[evals/](evals/) for the harness, datasets, and per-question records).

## Install

```
pip install webfetch          # core: search, fetch, BM25 ranking, cache
pip install "webfetch[all]"   # + semantic rerank/cache/compression, JS
                              #   rendering, PDF, tables, MCP server
```

Needs pip >= 24 (`python -m pip install -U pip` first in a fresh venv -
older pips crash on a duplicated extra in our dependency tree).

The core install works with zero API keys - DuckDuckGo needs none. Add
keys to unlock more engines (all have free tiers) and better recall:

```
cp .env.example .env   # then fill in what you have
```

The benchmark numbers above use the 4-engine fusion config
(DDG + Brave + Serper + Tavily) with the `[rerank]` extra installed.

## Use it in an agent loop

```python
import time
import anthropic
from webfetch import WEB_SEARCH_TOOL, handle_web_search

client = anthropic.Anthropic()
messages = [{"role": "user", "content": "What did the FOMC decide this week?"}]
system = (f"Today's date is {time.strftime('%Y-%m-%d')}. "
          "Use web_search for recent facts.")

while True:
    response = client.messages.create(
        model="claude-opus-4-7", max_tokens=2000, system=system,
        tools=[WEB_SEARCH_TOOL], messages=messages,
    )
    if response.stop_reason != "tool_use":
        break
    messages.append({"role": "assistant", "content": response.content})
    results = [{"type": "tool_result", "tool_use_id": b.id,
                "content": handle_web_search(b.input)}
               for b in response.content if b.type == "tool_use"]
    messages.append({"role": "user", "content": results})
```

A complete version with prompt caching and adaptive thinking is in
[examples/agent_loop.py](examples/agent_loop.py).

**Put today's date in your system prompt.** This is not optional. Models
refuse to search for events they believe haven't happened yet: on our
fresh-events dataset, arms without the date declined to even call the tool
on up to 10 of 27 questions ("Wimbledon 2026 hasn't taken place yet").
One line fixes it. Hosted search tools do this server-side, which is part
of why nobody notices until they run their own tool.

`handle_web_search` never raises. Engine failures, empty results, and
malformed input all come back as readable strings the model can react to,
because an exception mid-conversation kills the whole agent loop.

## Use it from Claude Code / Claude Desktop

```
pip install "webfetch[all]"
claude mcp add webfetch webfetch-mcp
```

The MCP server exposes `web_search` and `savings_report`. Run one server
per machine - the semantic cache assumes a single process owns its file.

## Configurations

| config | engines | search fees | when |
|---|---|---|---|
| `multi` (default for benchmarks) | DDG+Brave+Serper+Tavily fused | ~$0.012/search | best recall |
| `fallback` | DDG first, keyed engines catch its blocks | ~$0 typical | cheap with a safety net |
| `ddg` | DuckDuckGo only | $0 | no keys at all |

```python
from webfetch import Pipeline, SemanticSqliteCache
from webfetch.search import get_search_adapter

pipeline = Pipeline(search=get_search_adapter("fallback"),
                    cache=SemanticSqliteCache())
```

DDG deserves a caveat: it fingerprint-blocks automated clients with silent
empty responses. webfetch detects that (empty-with-peers in fusion, any
empty in the fallback chain), benches the engine on a circuit breaker, and
routes around it. The TLS side is handled by the `ddgs` dependency.

## What's inside

Search results go through: multi-engine fusion (reciprocal rank fusion
keyed by URL) -> concurrent fetch with a fallback chain (trafilatura ->
readability -> newspaper4k -> Playwright rendering for JS pages and
403 walls) -> 400-char chunking -> hybrid ranking (BM25 and bi-encoder
fused, then a cross-encoder picks the top 5) -> sentence-level compression
(cross-encoder scored; halves tokens with zero measured recall loss) ->
source-labeled output.

Results are cached at two levels (page text by URL, ranked chunks by
query) in a single sqlite file at `~/.webfetch/cache.db`. Cache lifetimes
depend on how volatile the answer is: queries are classified as
realtime/recent/stable (15 minutes / 7 days / 90 days), either by a hint
from the calling model or by a small local classifier. The model sees
cache provenance in every result (`[cache: semantic match to "...", 2h
old, recent]`) and can send `force_fresh` when it disagrees.

Everything heavy is optional. Without `[rerank]` you get BM25 ranking and
exact-match caching; the library degrades with a logged warning rather
than an ImportError.

## Cost receipts

Counters accumulate in the cache file as you use the tool:

```
$ webfetch-savings
webfetch savings receipt (lifetime of this cache)
  searches served:      1240
  from cache:           472 (38%) - exact 310, semantic 162 (zero marginal cost...)
  fresh pipeline runs:  768
  result tokens sent:   ~4,340,000 (hosted would inject ~21,576,000)
  ---
  hosted search fees avoided:  1240 x $0.010 = $12.40
  content-token cost avoided:  ~$86.18 (at $5.00/MTok)
  ESTIMATED TOTAL AVOIDED:     $98.58
```

Counters are exact; the dollar lines are estimates with the assumptions
(hosted fee, hosted tokens per call, your model's token price) exposed as
arguments to `webfetch.savings_report()`.

## Caveats, honestly

- Latency. A fresh search takes 10-40 seconds (real pages get fetched and
  ranked locally). Hosted search returns in ~10s; Tavily-style snippet
  APIs in ~6s. Cache hits are instant. If you need sub-second search and
  don't care about cost, this is not your tool.
- The cache is single-process. Point two long-running processes at the
  same cache file and the semantic index of one goes stale. One agent
  loop, one MCP server, or one notebook at a time is the supported shape.
- Engine free tiers are real quotas. We exhausted Brave's monthly tier
  during benchmarking; the resilience layer degraded gracefully, but your
  recall degrades with it. Fees above are estimates from published prices.
- Answer quality depends on the model driving the tool. Weak models
  formulate worse queries and re-search instead of reading (see the Haiku
  row).

## Benchmarks and data

The eval harness has three layers: an offline matcher eval for the
semantic-cache thresholds, a live retrieval eval (recall, tokens, cache
diagnostics), and the end-to-end answer eval quoted above (SimpleQA
protocol: exact-match fast path, then an LLM judge). Datasets are built
deterministically (seeded) from SimpleQA (MIT), QQP/GLUE, and FreshQA
(CC-BY-SA); provenance sidecars sit next to each file in
[evals/datasets/](evals/datasets/). The fresh-events set was hand-written
against verified news sources days before the benchmark ran, specifically
so that no model or vendor pipeline could have seen it.

Design decisions and measured results are documented as they happened in
[docs/architecture.md](docs/architecture.md) and
[docs/ROADMAP.md](docs/ROADMAP.md).

## License

MIT.
