# Eval results - the evidence behind the README claims

Every number in the main README's benchmark tables comes from a JSON file
in this directory. Each file records the exact config, per-arm summaries,
and per-question records (question, expected answer, model output, judge
verdict, token counts, cost) so grading can be audited without re-running
anything.

## Which file backs which claim

| claim | file | arms |
|---|---|---|
| Headline table: 50 SimpleQA questions, all tools, same-day run (2026-07-14) | `e2e_eval_20260714_005531.json` | ours-multi 92%, ours-ddg 84%, ours-gpt 96%, hosted (Anthropic) 96%, openai-hosted 100%, tavily 88%, exa 90% |
| Haiku 4.5 arm (76%, degraded engines mid-run - see README footnote) | `e2e_eval_20260714_173905.json` | ours-haiku |
| Fresh-events set: 27 hand-written questions about the two weeks before the run | `e2e_eval_20260714_024356.json` | every arm 96-100% |
| Compression: 50% fewer tokens at zero recall loss | `compression_eval_*.json` | - |
| Semantic cache matcher: zero wrong-target matches | `matcher_eval_*.json` | - |
| Retrieval pipeline component evals | `pipeline_eval_*.json` | - |

Earlier-dated e2e files are development runs (smoke tests with n=2-3,
single-arm reruns, harness debugging). The `broke` arm is a free-stack
experiment (Groq/Gemini free tiers) that did not survive contact with rate
limits - kept because negative results are results.

## Judging protocol

SimpleQA-style three-way grading (correct / incorrect / not_attempted):
an exact-match fast path, then an LLM judge (`claude-haiku-4-5`) with the
prompt in `evals/run_e2e_eval.py`. Same judge for every arm. Don't trust
our judge? The per-question records include the raw model outputs - grade
them yourself.

## Reproducing

```
python evals/fetch_datasets.py            # public SimpleQA CSV
python evals/build_datasets.py            # seeded sample (seed 42) -> the exact 50 questions
python evals/run_e2e_eval.py --arms ours-multi,hosted
```

Two caveats, honestly:

1. The live web moves between runs, so expect point estimates to wobble
   within confidence bounds - at n=50, 96% vs 92% is 48 vs 46 correct,
   which is parity within noise, not a real gap. The token and cost
   columns are the mechanically stable numbers.
2. The hosted arms call each vendor's own API at their current defaults;
   vendor-side changes move those numbers too.

## What is not in the public copies

- `*.log` files (console output from runs) are not committed - the JSONs
  are the record.
- A Perplexity Sonar arm ran in four of these files and is stripped from
  the public copies (marked in-file under `"stripped_arms"`). Sonar is a
  direct answer API, not a search tool inside your own agent loop, so it
  is not the comparison webfetch is positioned against. Nothing stops you
  measuring it yourself: `python evals/run_e2e_eval.py --arms sonar` with
  a `PERPLEXITY_API_KEY`.
