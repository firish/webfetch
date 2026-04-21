"""
OpenAI (GPT) extractor.

Uses the OpenAI SDK. Default model is gpt-4o-mini - capable, cheap, and
supports native JSON response formatting which reduces parse failures vs
free-form prompting.

Requires OPENAI_API_KEY env var, or pass api_key= to the constructor.
"""

from __future__ import annotations

import os

from webfetch.extract.base import AbstractExtractor


class GPTExtractor(AbstractExtractor):
    """Extractor backed by OpenAI's GPT.

    Args:
        api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
        model: OpenAI model ID. gpt-4o-mini is the cheapest capable tier.
        max_tokens: Max tokens in the response.
        temperature: 0 for deterministic extraction.

    Raises:
        ValueError: If no API key is provided or found in the environment.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key required. "
                "Pass api_key= or set OPENAI_API_KEY env var."
            )
        self._api_key = resolved_key
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            # response_format forces the model to return valid JSON - removes
            # the "wrapped in markdown" failure mode entirely.
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
