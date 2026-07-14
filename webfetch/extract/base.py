"""
Abstract interface for LLM extraction adapters.

Every concrete adapter (Claude, GPT, Gemini) implements AbstractExtractor.
The pipeline calls only this interface so swapping providers requires no
changes elsewhere.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from urllib.parse import urlparse

from webfetch.config import DEFAULT_TOKEN_BUDGET
from webfetch.rank.base import Chunk

# System prompt shared across all adapters. Kept provider-agnostic so switching
# models does not change prompt semantics - each adapter just wraps this.
EXTRACTION_SYSTEM_PROMPT: str = (
    "You extract structured data from web search results. "
    "You respond with a single JSON object and no other text. "
    "Use null for any field you cannot determine from the provided sources. "
    "Never guess or fabricate values - if the sources do not say, return null."
)


def _header(chunk: Chunk, style: str) -> str:
    """Source label for a chunk group.

    Styles (measured by evals/run_compression_eval.py):
        full: title + full URL - maximum attribution.
        domain: title + hostname - URL paths are long and rarely carry
            information the title does not (titles DO carry answers - the
            extraction-hardening eval found answers living only in metadata,
            so the title always stays).
    """
    if style == "domain":
        host = urlparse(chunk.url).netloc or chunk.url
        return f"[Source: {chunk.title} | {host}]"
    return f"[Source: {chunk.title} | {chunk.url}]"


def build_context(
    chunks: list[Chunk],
    budget_chars: int = DEFAULT_TOKEN_BUDGET,
    merge_sources: bool = False,
    header_style: str = "full",
) -> str:
    """Concatenate the top chunks into a single context string under a budget.

    Chunks are appended in order (assumed best-first from the ranker) until
    the next one would exceed the character budget. Each chunk is labeled
    with its source title/url so the LLM can attribute or prefer one source
    over another if they disagree.

    Args:
        chunks: Ranked chunks from the rank stage (best first).
        budget_chars: Max characters of context to send to the LLM.
        merge_sources: Group same-URL chunks under ONE header (first-
            occurrence order; top-5 results typically span ~3.7 distinct
            URLs, so duplicate headers are pure overhead). Within a group,
            chunks keep rank order.
        header_style: "full" or "domain" - see _header().

    Returns:
        A single string of labeled, budget-trimmed chunks.
    """
    if merge_sources:
        groups: dict[str, list[Chunk]] = {}
        for c in chunks:
            groups.setdefault(c.url, []).append(c)
        pieces = [
            _header(cs[0], header_style) + "\n"
            + "\n".join(c.text for c in cs) + "\n"
            for cs in groups.values()
        ]
    else:
        pieces = [f"{_header(c, header_style)}\n{c.text}\n" for c in chunks]

    parts: list[str] = []
    used = 0
    for piece in pieces:
        if used + len(piece) > budget_chars and parts:
            # Stop before overshooting the budget - but always include at
            # least one chunk even if it exceeds the budget alone.
            break
        parts.append(piece)
        used += len(piece)
    return "\n".join(parts)


def build_user_prompt(keys: dict[str, str], context: str) -> str:
    """Build the user-facing extraction prompt from keys and context.

    Args:
        keys: Mapping of field_name -> short description of what to extract.
        context: Concatenated source text from build_context().

    Returns:
        A single user-message prompt.
    """
    field_lines = "\n".join(f"- {k}: {desc}" for k, desc in keys.items())
    return (
        "Extract the following fields from the SOURCES below.\n\n"
        f"FIELDS:\n{field_lines}\n\n"
        f"SOURCES:\n{context}\n\n"
        "Respond with a single JSON object using the exact field names above. "
        "Use null where the sources do not contain the value."
    )


def parse_json_response(text: str) -> dict[str, str | None]:
    """Extract a JSON object from an LLM response, tolerating common noise.

    Handles:
      - JSON wrapped in ``` or ```json code fences
      - Leading/trailing prose around the JSON object
      - Stray whitespace

    Raises:
        ValueError: if no JSON object can be found or parsed.
    """
    # Strip ```json ... ``` or ``` ... ``` code fences first.
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1)
    else:
        # Fall back to the first balanced-looking {...} block in the response.
        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if not brace_match:
            raise ValueError(f"No JSON object found in response: {text!r}")
        candidate = brace_match.group(1)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in response: {candidate!r}") from exc


class AbstractExtractor(ABC):
    """Base class for all LLM extraction adapters.

    Subclasses implement `_call_llm()` - the rest (context building, prompt
    assembly, JSON parsing) is shared via this base so each adapter stays
    focused on the provider-specific API call.
    """

    @abstractmethod
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Send a single system+user pair to the provider and return raw text."""
        ...

    def extract(
        self,
        chunks: list[Chunk],
        keys: dict[str, str],
        budget_chars: int = DEFAULT_TOKEN_BUDGET,
    ) -> dict[str, str | None]:
        """Extract values for each key from the ranked chunks.

        Args:
            chunks: Ranked chunks from the rank stage (best first).
            keys: Mapping of field_name -> short description.
                  E.g. `{"accuracy": "measurement accuracy with units"}`.
            budget_chars: Max characters of context to send to the LLM.

        Returns:
            Dict mapping each requested key to its extracted value (or None).
            Only keys present in the LLM response are guaranteed - callers
            should default-fill missing keys if strict coverage is needed.
        """
        if not chunks:
            return {k: None for k in keys}

        context = build_context(chunks, budget_chars=budget_chars)
        user_prompt = build_user_prompt(keys, context)
        raw = self._call_llm(EXTRACTION_SYSTEM_PROMPT, user_prompt)
        return parse_json_response(raw)
