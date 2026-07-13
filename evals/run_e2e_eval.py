"""
Layer 3: end-to-end answer accuracy - our web_search tool vs hosted search.

Two arms on the same questions, same model, same judge:
- ours:   manual agent loop with webfetch's web_search tool (full production
          config: 4-engine fusion, semantic cache on a fresh db)
- hosted: the same loop with Anthropic's server-side web_search tool
          (the $10/1k-searches product this library replaces)

Grading mirrors SimpleQA's protocol: normalized exact/substring match as
the fast path, then an LLM judge (correct / incorrect / not_attempted).

Outputs per arm: answer accuracy, measured cost/query (from usage fields +
search counts), searches/question, latency - and the headline ratio.

Run: python evals/run_e2e_eval.py [--arm both] [--limit N] [--model ...]
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
HOSTED_TOOL = {"type": "web_search_20260209", "name": "web_search"}
# Identical for both arms - the comparison is the search tool, nothing else.
SYSTEM_PROMPT = (
    "Answer the user's question using web search. Search as needed, then "
    "give a SHORT final answer - just the fact asked for, no preamble."
)

# Opus 4.7 pricing per million tokens; hosted search per-request fee.
PRICE_IN = 5.00
PRICE_CACHE_READ = 0.50
PRICE_CACHE_WRITE = 6.25
PRICE_OUT = 25.00
HOSTED_SEARCH_FEE = 10.00 / 1000

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


def answer_ours(client, model: str, question: str, pipeline) -> dict:
    """Agent loop with webfetch's client-side web_search tool."""
    from webfetch import WEB_SEARCH_TOOL, handle_web_search

    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    messages = [{"role": "user", "content": question}]
    searches = 0
    t0 = time.perf_counter()
    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=model, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
            tools=[WEB_SEARCH_TOOL], cache_control={"type": "ephemeral"},
            messages=messages,
        )
        _usage_add(usage, response.usage)
        if response.stop_reason != "tool_use":
            return {"answer": _final_text(response), "searches": searches,
                    "usage": usage, "secs": time.perf_counter() - t0,
                    "cost": _token_cost(usage), "error": None}
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            searches += 1
            with PIPELINE_SEMAPHORE:
                out = handle_web_search(block.input, pipeline=pipeline)
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": out})
        messages.append({"role": "user", "content": results})
    return {"answer": "", "searches": searches, "usage": usage,
            "secs": time.perf_counter() - t0, "cost": _token_cost(usage),
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


def run_arm(arm: str, questions: list[dict], client, model: str,
            workers: int) -> list[dict]:
    pipeline = None
    if arm == "ours":
        import tempfile
        from webfetch import Pipeline, SemanticSqliteCache
        from webfetch.search import get_search_adapter
        cache_db = str(Path(tempfile.mkdtemp(prefix="webfetch_e2e_")) / "cache.db")
        pipeline = Pipeline(search=get_search_adapter("multi"),
                            cache=SemanticSqliteCache(db_path=cache_db))
        print(f"[ours] cache: {cache_db}", flush=True)

    def one(q: dict) -> dict:
        try:
            if arm == "ours":
                r = answer_ours(client, model, q["query"], pipeline)
            else:
                r = answer_hosted(client, model, q["query"])
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
    parser.add_argument("--arm", choices=["ours", "hosted", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers-ours", type=int, default=4)
    parser.add_argument("--workers-hosted", type=int, default=6)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    questions = [q for q in read_jsonl(DATASET_PATH)
                 if q.get("paraphrase_of") is None]
    if args.limit:
        questions = questions[: args.limit]
    client = anthropic.Anthropic()
    print(f"{len(questions)} questions | model {args.model}", flush=True)

    all_records: dict[str, list[dict]] = {}
    summaries = []
    arms = ["ours", "hosted"] if args.arm == "both" else [args.arm]
    for arm in arms:
        print(f"\n## arm: {arm}", flush=True)
        workers = args.workers_ours if arm == "ours" else args.workers_hosted
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
    if len(summaries) == 2:
        ours, hosted = summaries
        if ours["accuracy"] and hosted["accuracy"] and hosted["cost_per_q"]:
            print(f"\nHEADLINE: {ours['accuracy']/hosted['accuracy']*100:.0f}% "
                  f"of hosted accuracy at "
                  f"{ours['cost_per_q']/hosted['cost_per_q']*100:.0f}% of the cost")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"e2e_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": {"model": args.model, "n": len(questions)},
                   "summaries": summaries, "records": all_records}, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
