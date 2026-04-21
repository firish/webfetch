"""
Anthropic Claude extractor.

Uses the Anthropic SDK. Default model is the cheapest capable Claude
(haiku 4.5) since this is an extraction-only call - the LLM is not reasoning
or composing, just pulling values from pre-ranked text.

Requires ANTHROPIC_API_KEY env var, or pass api_key= to the constructor.
"""

from __future__ import annotations

import os

from webfetch.config import DEFAULT_EXTRACT_MODEL
from webfetch.extract.base import AbstractExtractor


class ClaudeExtractor(AbstractExtractor):
    """Extractor backed by Anthropic's Claude.

    Args:
        api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
        model: Claude model ID. Defaults to config.DEFAULT_EXTRACT_MODEL.
        max_tokens: Max tokens in the response. Extraction fits easily in 1024.
        temperature: 0 for deterministic extraction - we do not want creative
                     interpretation when pulling numeric specs from text.

    Raises:
        ValueError: If no API key is provided or found in the environment.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_EXTRACT_MODEL,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Anthropic API key required. "
                "Pass api_key= or set ANTHROPIC_API_KEY env var."
            )
        self._api_key = resolved_key
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        # Lazy-imported on first call - keeps `anthropic` an optional dep
        # for users who only want the OpenAI or Gemini adapter.
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_client()
        response = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # Anthropic SDK returns a list of content blocks - we prompted for a
        # single JSON object so we expect a single text block.
        return response.content[0].text
