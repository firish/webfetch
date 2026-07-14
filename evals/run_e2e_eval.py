"""
Layer 3: end-to-end answer accuracy and cost - webfetch vs the field.

Arms (each answers the same questions, graded by the same judge):

Agent-loop arms (identical Opus loop; only the web_search tool backend
differs - THE apples-to-apples comparison for "webfetch as a tool"):
- ours-multi: webfetch tool, 4-engine RRF fusion ("balanced" - production)
- ours-ddg:   webfetch tool, DDG only ("miser" - $0 engine fees)
- hosted:     Anthropic's server-side web_search ($10/1k searches)
- tavily:     Tavily search API results as the tool backend ($8/1k basic)
- exa:        Exa neural search (+contents) as the tool backend ($7/1k)

Direct-answer arm (no frontier model - the "skip the agent" alternative):
- sonar:      Perplexity Sonar answers the question itself ($5/1k + tokens)

Zero-cost arm:
- broke:      free-tier model (Groq llama or Gemini Flash, OpenAI-compat
              tool loop) + webfetch on DDG only - $0.00/query serving cost

Cost model is honest: ours-* arms are charged ESTIMATED per-search engine
fees (see ENGINE_FEE_PER_SEARCH - fresh searches only; production cache
hits cost $0, which no competitor matches). Competitor arms are charged
their published per-request fees plus Opus tokens where applicable.

Grading mirrors SimpleQA's protocol: normalized exact/substring match as
the fast path, then an LLM judge (correct / incorrect / not_attempted).

Run: python evals/run_e2e_eval.py [--arms ours-multi,ours-ddg,tavily]
     [--limit N] [--model ...]
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import signal
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# On a hang: `kill -USR1 <pid>` dumps every thread's stack to stderr.
faulthandler.register(signal.SIGUSR1)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.common import answer_variants, markdown_table, normalize_text, read_jsonl

DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "live_queries.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

DEFAULT_MODEL = "claude-opus-4-7"
JUDGE_MODEL = "claude-haiku-4-5"
MAX_TURNS = 6
MAX_TOKENS = 2000
TOOL_RESULT_BUDGET_CHARS = 8000  # same budget for every tool backend
HOSTED_TOOL = {"type": "web_search_20260209", "name": "web_search"}
# Identical for both arms - the comparison is the search tool, nothing else.
SYSTEM_PROMPT = (
    "Answer the user's question using web search. Search as needed, then "
    "give a SHORT final answer - just the fact asked for, no preamble."
)

# Schema for competitor tool backends: same name and role as ours, minimal
# params (force_fresh/freshness are webfetch features competitors lack).
GENERIC_TOOL: dict = {
    "name": "web_search",
    "description": (
        "Search the web and return the most relevant results, labeled with "
        "source title and URL. Use this for facts past your training "
        "cutoff. Call again with a different query if results are not "
        "enough."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string",
                                 "description": "A focused web search query."}},
        "required": ["query"],
    },
}

# Opus 4.7 pricing per million tokens; hosted search per-request fee.
PRICE_IN = 5.00
PRICE_CACHE_READ = 0.50
PRICE_CACHE_WRITE = 6.25
PRICE_OUT = 25.00
HOSTED_SEARCH_FEE = 10.00 / 1000

# Estimated engine fees per FRESH search (published prices, 2026-07):
# brave base ~$3/1k + serper ~$1/1k + tavily basic $8/1k + ddg $0.
# Production cache hits skip all of these; the eval uses a fresh cache so
# every search is charged - a worst case for us.
ENGINE_FEE_PER_SEARCH = {"ours-multi": 0.012, "ours-ddg": 0.0}
TAVILY_FEE_PER_SEARCH = 8.00 / 1000    # basic search = 1 credit = $0.008
EXA_FEE_PER_SEARCH = 7.00 / 1000       # search incl. contents (Mar 2026 pricing)
SONAR_REQUEST_FEE = 5.00 / 1000        # per request, non-pro tier
SONAR_PRICE_PER_MTOK = 1.00            # $1/M input and output (base sonar)

# Local pipeline runs are SERIALIZED. Two concurrent pipelines (16 fetch
# threads, multiple playwright browsers, 2 threads sharing MPS encoders)
# deadlocked a full run; one-at-a-time matches the concurrency profile that
# five full Layer 2 runs have proven stable. API calls still parallelize.
PIPELINE_SEMAPHORE = threading.Semaphore(1)

JUDGE_PROMPT = """Grade whether the predicted answer is correct.

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}

Reply with exactly one word:
CORRECT - the predicted answer contains or equals the gold answer's meaning
INCORRECT - it names a different value/entity than the gold answer
NOT_ATTEMPTED - it does not actually answer the question"""


def _usage_add(total: dict, usage) -> None:
    total["input"] += usage.input_tokens
    total["output"] += usage.output_tokens
    total["cache_read"] += getattr(usage, "cache_read_input_tokens", 0) or 0
    total["cache_write"] += getattr(usage, "cache_creation_input_tokens", 0) or 0


def _token_cost(u: dict) -> float:
    return (u["input"] * PRICE_IN + u["cache_read"] * PRICE_CACHE_READ
            + u["cache_write"] * PRICE_CACHE_WRITE + u["output"] * PRICE_OUT) / 1e6


def _final_text(response) -> str:
    return " ".join(b.text for b in response.content if b.type == "text").strip()


# --- tool backends (query -> tool_result string; must never raise) ---


def tavily_backend(query: str) -> str:
    """Tavily search results formatted as a source-labeled tool result."""
    import requests
    resp = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": os.environ["TAVILY_API_KEY"], "query": query,
              "max_results": 8, "include_answer": True,
              "include_raw_content": False},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    parts = []
    if data.get("answer"):
        parts.append(f"[Tavily answer] {data['answer']}")
    for h in data.get("results", []):
        parts.append(f"[Source: {h.get('title', '')} | {h['url']}]\n"
                     f"{h.get('content', '')}")
    return "\n\n".join(parts)[:TOOL_RESULT_BUDGET_CHARS] or "No results."


def exa_backend(query: str) -> str:
    """Exa neural search + contents formatted as a tool result."""
    import requests
    resp = requests.post(
        "https://api.exa.ai/search",
        headers={"x-api-key": os.environ["EXA_API_KEY"]},
        json={"query": query, "numResults": 8,
              "contents": {"text": {"maxCharacters": 1500},
                           "highlights": True}},
        timeout=30,
    )
    resp.raise_for_status()
    parts = []
    for h in resp.json().get("results", []):
        body = h.get("text") or " ".join(h.get("highlights") or [])
        parts.append(f"[Source: {h.get('title') or ''} | {h['url']}]\n{body}")
    return "\n\n".join(parts)[:TOOL_RESULT_BUDGET_CHARS] or "No results."


def _webfetch_backend(pipeline):
    from webfetch import handle_web_search

    def backend(tool_input: dict) -> str:
        with PIPELINE_SEMAPHORE:
            return handle_web_search(tool_input, pipeline=pipeline)
    return backend


# --- arms ---


def answer_tool_loop(client, model: str, question: str, tools: list[dict],
                     handler, fee_per_search: float) -> dict:
    """Shared agent loop: Opus + a client-side web_search tool backend.

    handler receives the tool_use block's input dict and returns the
    tool_result string. Identical loop across arms - the backend is the
    only variable.
    """
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    messages = [{"role": "user", "content": question}]
    searches = 0
    t0 = time.perf_counter()
    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=model, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
            tools=tools, cache_control={"type": "ephemeral"},
            messages=messages,
        )
        _usage_add(usage, response.usage)
        if response.stop_reason != "tool_use":
            return {"answer": _final_text(response), "searches": searches,
                    "usage": usage, "secs": time.perf_counter() - t0,
                    "cost": _token_cost(usage) + searches * fee_per_search,
                    "error": None}
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            searches += 1
            try:
                out = handler(block.input)
            except Exception as exc:
                # The backend failing is data, not a crash - the model can
                # retry, same contract as handle_web_search.
                out = f"web_search error: {type(exc).__name__}: {exc}"
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": out})
        messages.append({"role": "user", "content": results})
    return {"answer": "", "searches": searches, "usage": usage,
            "secs": time.perf_counter() - t0,
            "cost": _token_cost(usage) + searches * fee_per_search,
            "error": "max turns exceeded"}


def answer_hosted(client, model: str, question: str) -> dict:
    """Same loop with Anthropic's server-side web_search tool."""
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    messages = [{"role": "user", "content": question}]
    searches = 0
    t0 = time.perf_counter()
    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=model, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
            tools=[HOSTED_TOOL], messages=messages,
        )
        _usage_add(usage, response.usage)
        stu = getattr(response.usage, "server_tool_use", None)
        if stu is not None:
            searches += getattr(stu, "web_search_requests", 0) or 0
        if response.stop_reason == "pause_turn":
            # Server-side loop hit its iteration limit - resend to resume.
            messages.append({"role": "assistant", "content": response.content})
            continue
        cost = _token_cost(usage) + searches * HOSTED_SEARCH_FEE
        return {"answer": _final_text(response), "searches": searches,
                "usage": usage, "secs": time.perf_counter() - t0,
                "cost": cost, "error": None}
    return {"answer": "", "searches": searches, "usage": usage,
            "secs": time.perf_counter() - t0,
            "cost": _token_cost(usage) + searches * HOSTED_SEARCH_FEE,
            "error": "max turns exceeded"}


def answer_sonar(question: str) -> dict:
    """Perplexity Sonar answers directly - no frontier model in the loop."""
    import requests
    t0 = time.perf_counter()
    resp = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['PERPLEXITY_API_KEY']}"},
        json={"model": "sonar",
              "messages": [{"role": "user", "content": (
                  question + " Give a SHORT final answer - just the fact "
                  "asked for, no preamble.")}]},
        timeout=90,
    )
    resp.raise_for_status()
    data = resp.json()
    u = data.get("usage", {})
    usage = {"input": u.get("prompt_tokens", 0),
             "output": u.get("completion_tokens", 0),
             "cache_read": 0, "cache_write": 0}
    cost = (SONAR_REQUEST_FEE
            + (usage["input"] + usage["output"]) * SONAR_PRICE_PER_MTOK / 1e6)
    return {"answer": data["choices"][0]["message"]["content"].strip(),
            "searches": 1, "usage": usage,
            "secs": time.perf_counter() - t0, "cost": cost, "error": None}


def answer_broke(question: str, handler, provider: tuple) -> dict:
    """OpenAI-compatible tool loop on a free-tier model. Serving cost: $0.

    Free tiers rate-limit hard (Groq ~30 RPM, Gemini ~10 RPM), so 429s are
    retried with a backoff instead of failing the question.
    """
    import requests
    name, url, key, model = provider
    tool = {"type": "function", "function": {
        "name": "web_search",
        "description": GENERIC_TOOL["description"],
        "parameters": GENERIC_TOOL["input_schema"]}}
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    searches = 0
    t0 = time.perf_counter()
    for _ in range(MAX_TURNS):
        for attempt in range(4):
            resp = requests.post(
                url, headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "messages": messages, "tools": [tool],
                      "max_tokens": MAX_TOKENS},
                timeout=120,
            )
            if resp.status_code != 429:
                break
            time.sleep(15 * (attempt + 1))
        resp.raise_for_status()
        data = resp.json()
        u = data.get("usage") or {}
        usage["input"] += u.get("prompt_tokens", 0) or 0
        usage["output"] += u.get("completion_tokens", 0) or 0
        msg = data["choices"][0]["message"]
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            return {"answer": (msg.get("content") or "").strip(),
                    "searches": searches, "usage": usage,
                    "secs": time.perf_counter() - t0, "cost": 0.0,
                    "error": None}
        messages.append(msg)
        for tc in tool_calls:
            searches += 1
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
                out = handler(args)
            except Exception as exc:
                out = f"web_search error: {type(exc).__name__}: {exc}"
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": out})
    return {"answer": "", "searches": searches, "usage": usage,
            "secs": time.perf_counter() - t0, "cost": 0.0,
            "error": "max turns exceeded"}


# Free-tier providers for the broke arm, tried in order. Groq first: higher
# free RPM and much faster inference than Gemini's free tier.
BROKE_PROVIDERS = [
    ("groq", "GROQ_API_KEY",
     "https://api.groq.com/openai/v1/chat/completions",
     "llama-3.3-70b-versatile"),
    ("gemini", "GOOGLE_API_KEY",
     "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
     "gemini-2.5-flash"),
]


def _broke_provider() -> tuple | None:
    for name, env, url, model in BROKE_PROVIDERS:
        key = os.environ.get(env, "")
        # Placeholder keys from .env templates ("your-key-here", "...") are
        # set-but-fake; skip them so the arm skips instead of 401-ing 50x.
        if key and "your" not in key.lower() and len(key) > 20:
            return name, url, key, model
    return None


ARM_KEY_ENV = {"tavily": "TAVILY_API_KEY", "exa": "EXA_API_KEY",
               "sonar": "PERPLEXITY_API_KEY"}
ALL_ARMS = ["ours-multi", "ours-ddg", "hosted", "tavily", "exa", "sonar",
            "broke"]


def grade(client, question: str, gold: list[str], answer_type: str | None,
          predicted: str) -> str:
    """SimpleQA-style grade: exact fast path, then LLM judge."""
    if not predicted:
        return "not_attempted"
    pred_norm = normalize_text(predicted)
    for g in gold:
        for v in answer_variants(g, answer_type):
            if v in pred_norm:
                return "correct"
    response = client.messages.create(
        model=JUDGE_MODEL, max_tokens=10,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, gold="; ".join(gold), predicted=predicted)}],
    )
    verdict = _final_text(response).strip().upper()
    return {"CORRECT": "correct", "INCORRECT": "incorrect"}.get(
        verdict, "not_attempted")


def _make_pipeline_backend(provider: str, arm: str):
    import tempfile
    from webfetch import Pipeline, SemanticSqliteCache
    from webfetch.search import get_search_adapter
    cache_db = str(Path(tempfile.mkdtemp(prefix=f"webfetch_e2e_{provider}_"))
                   / "cache.db")
    pipeline = Pipeline(search=get_search_adapter(provider),
                        cache=SemanticSqliteCache(db_path=cache_db))
    print(f"[{arm}] cache: {cache_db}", flush=True)
    return _webfetch_backend(pipeline)


def make_answer_fn(arm: str, client, model: str):
    """Build the per-question answer function for an arm (lazy resources)."""
    if arm.startswith("ours-"):
        provider = "multi" if arm == "ours-multi" else "ddg"
        backend = _make_pipeline_backend(provider, arm)
        from webfetch import WEB_SEARCH_TOOL
        fee = ENGINE_FEE_PER_SEARCH[arm]
        return lambda q: answer_tool_loop(client, model, q,
                                          [WEB_SEARCH_TOOL], backend, fee)
    if arm == "broke":
        provider = _broke_provider()
        print(f"[broke] free-tier model: {provider[0]}/{provider[3]}",
              flush=True)
        backend = _make_pipeline_backend("ddg", arm)
        return lambda q: answer_broke(q, backend, provider)
    if arm == "hosted":
        return lambda q: answer_hosted(client, model, q)
    if arm == "tavily":
        return lambda q: answer_tool_loop(
            client, model, q, [GENERIC_TOOL],
            lambda ti: tavily_backend(str(ti.get("query", ""))),
            TAVILY_FEE_PER_SEARCH)
    if arm == "exa":
        return lambda q: answer_tool_loop(
            client, model, q, [GENERIC_TOOL],
            lambda ti: exa_backend(str(ti.get("query", ""))),
            EXA_FEE_PER_SEARCH)
    if arm == "sonar":
        return lambda q: answer_sonar(q)
    raise ValueError(f"unknown arm: {arm}")


def run_arm(arm: str, questions: list[dict], client, model: str,
            workers: int) -> list[dict]:
    answer = make_answer_fn(arm, client, model)

    def one(q: dict) -> dict:
        try:
            r = answer(q["query"])
        except Exception as exc:
            r = {"answer": "", "searches": 0, "usage": {}, "secs": 0.0,
                 "cost": 0.0, "error": f"{type(exc).__name__}: {exc}"}
        r["id"] = q["id"]
        r["grade"] = grade(client, q["query"], q["answers"],
                           q.get("answer_type"), r["answer"]) \
            if not r["error"] else "error"
        print(f"  [{arm}] {q['id']}: {r['grade']} "
              f"(searches={r['searches']}, {r['secs']:.0f}s, ${r['cost']:.3f})"
              + (f" ERROR {r['error']}" if r["error"] else ""), flush=True)
        return r

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, questions))


def summarize(arm: str, records: list[dict]) -> dict:
    ok = [r for r in records if not r["error"]]
    correct = sum(1 for r in records if r["grade"] == "correct")
    return {
        "arm": arm, "n": len(records), "correct": correct,
        "accuracy": round(correct / len(records), 3) if records else None,
        "errors": sum(1 for r in records if r["error"]),
        "total_cost": round(sum(r["cost"] for r in records), 3),
        "cost_per_q": round(sum(r["cost"] for r in records) / len(records), 4),
        "searches_per_q": round(statistics.mean(r["searches"] for r in ok), 2) if ok else None,
        "median_secs": round(statistics.median(r["secs"] for r in ok), 1) if ok else None,
    }


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    import anthropic

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="ours-multi,ours-ddg,hosted",
                        help=f"comma-separated from: {','.join(ALL_ARMS)}")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers-ours", type=int, default=4)
    parser.add_argument("--workers-api", type=int, default=6)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ALL_ARMS]
    if unknown:
        sys.exit(f"unknown arms: {unknown} (choose from {ALL_ARMS})")
    for arm in list(arms):
        env = ARM_KEY_ENV.get(arm)
        if env and not os.environ.get(env):
            print(f"SKIPPING arm {arm}: {env} not set", flush=True)
            arms.remove(arm)
    if "broke" in arms and _broke_provider() is None:
        print("SKIPPING arm broke: no real GROQ_API_KEY or GOOGLE_API_KEY",
              flush=True)
        arms.remove("broke")

    questions = [q for q in read_jsonl(args.dataset)
                 if q.get("paraphrase_of") is None]
    if args.limit:
        questions = questions[: args.limit]
    client = anthropic.Anthropic()
    print(f"{len(questions)} questions | model {args.model} | arms: {arms}",
          flush=True)

    all_records: dict[str, list[dict]] = {}
    summaries = []
    for arm in arms:
        print(f"\n## arm: {arm}", flush=True)
        # Free tiers rate-limit: keep the broke arm at 2 concurrent calls.
        workers = (2 if arm == "broke"
                   else args.workers_ours if arm.startswith("ours-")
                   else args.workers_api)
        records = run_arm(arm, questions, client, args.model, workers)
        all_records[arm] = records
        summaries.append(summarize(arm, records))

    print()
    print(markdown_table(
        ["arm", "accuracy", "errors", "cost/query", "total cost",
         "searches/q", "median secs"],
        [[s["arm"], f"{s['correct']}/{s['n']} ({(s['accuracy'] or 0)*100:.0f}%)",
          s["errors"], f"${s['cost_per_q']:.4f}", f"${s['total_cost']:.2f}",
          s["searches_per_q"], s["median_secs"]] for s in summaries],
    ))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"e2e_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": {"model": args.model, "n": len(questions),
                              "arms": arms, "dataset": str(args.dataset),
                              "engine_fees": ENGINE_FEE_PER_SEARCH},
                   "summaries": summaries, "records": all_records}, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
