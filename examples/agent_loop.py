"""
Example: a manual agentic loop where Claude uses webfetch as its web search.

This demonstrates the core value proposition: instead of paying for the
hosted web-search tool (~$10/1k searches plus retrieved-content tokens),
the model calls a locally-served `web_search` custom tool. Claude decides
when to search and formulates the queries; webfetch does search -> fetch ->
rank and returns compact, source-labeled excerpts.

Usage:
    python examples/agent_loop.py "What did the S&P 500 close at today?"

Requires ANTHROPIC_API_KEY in the environment or a .env file.
"""

from __future__ import annotations

import sys

import anthropic
from dotenv import load_dotenv

from webfetch import Pipeline, SqliteCache, WEB_SEARCH_TOOL, handle_web_search

MODEL = "claude-opus-4-7"
MAX_TOKENS = 16000
# Bounds the tool-use loop so a confused model cannot search forever.
MAX_TURNS = 10

# Frozen string: a stable prompt prefix is what makes prompt caching hit.
SYSTEM_PROMPT = (
    "You are a helpful research assistant. Use the web_search tool for any "
    "facts you are not certain about, especially current events, prices, and "
    "product specs. Issue focused queries; search again with a different "
    "query if results are insufficient. Cite source URLs in your answer."
)


def main() -> None:
    load_dotenv()
    question = sys.argv[1] if len(sys.argv) > 1 else "What is the DC voltage accuracy of the Fluke 87V?"

    client = anthropic.Anthropic()
    # Build the pipeline once, outside the loop: encoder models load once,
    # the sqlite cache persists, and every tool call reuses both.
    pipeline = Pipeline(cache=SqliteCache())

    messages: list[dict] = [{"role": "user", "content": question}]
    print(f"QUESTION: {question}\n")

    for turn in range(1, MAX_TURNS + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=[WEB_SEARCH_TOOL],
            # Adaptive thinking: the model decides when reasoning over tool
            # results needs deeper thought. Thinking blocks flow back into
            # history via response.content below, as the API requires.
            thinking={"type": "adaptive"},
            # Top-level cache_control: automatically caches the longest
            # reusable prefix (tools + system + conversation so far). Note:
            # Opus 4.7 needs a ~4096-token prefix before caching activates,
            # so cache_read stays 0 until the history grows past that.
            cache_control={"type": "ephemeral"},
            messages=messages,
        )

        usage = response.usage
        print(
            f"-- turn {turn}: stop_reason={response.stop_reason}, "
            f"in={usage.input_tokens}, out={usage.output_tokens}, "
            f"cache_read={getattr(usage, 'cache_read_input_tokens', 0)}"
        )

        if response.stop_reason != "tool_use":
            final_text = "".join(b.text for b in response.content if b.type == "text")
            print(f"\nANSWER:\n{final_text}")
            return

        # The assistant turn (including its tool_use blocks) must go back
        # into history before the tool results.
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"   web_search: {block.input.get('query', '')!r}")
            # handle_web_search never raises - errors come back as text the
            # model can read and react to.
            result_text = handle_web_search(block.input, pipeline=pipeline)
            print(f"   -> {len(result_text)} chars of ranked context")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                }
            )
        messages.append({"role": "user", "content": tool_results})

    print(f"\nStopped after {MAX_TURNS} turns without a final answer.")


if __name__ == "__main__":
    main()
