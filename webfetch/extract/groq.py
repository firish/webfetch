"""
Groq extractor - another strong FREE adapter.

Groq hosts open-weight models (Llama 3.3 70B, Mixtral, etc.) on custom
inference hardware that delivers ~500 tokens/sec - roughly 10x faster than
a typical OpenAI/Anthropic call. The free tier is generous: 30 RPM and
14,400 RPD for Llama 3.3 70B.

When to pick Groq over Gemini:
  - You want significantly lower latency per extraction.
  - You prefer open-weight models for reproducibility.
  - You have already hit Gemini's 15 RPM cap and want more throughput.

The API is OpenAI-compatible - supports response_format=json_object, same
chat.completions shape, etc.

Get a free API key at https://console.groq.com/keys
Requires GROQ_API_KEY env var, or pass api_key= to the constructor.
"""

from __future__ import annotations

import os

from webfetch.extract.base import AbstractExtractor


class GroqExtractor(AbstractExtractor):
    """Extractor backed by Groq's hosted open-weight models.

    Args:
        api_key: Groq API key. Defaults to GROQ_API_KEY env var.
        model: Groq model ID. llama-3.3-70b-versatile is the best quality/free
               tier tradeoff. For lower latency (and smaller context) use
               llama-3.1-8b-instant.
        max_tokens: Max tokens in the response.
        temperature: 0 for deterministic extraction.

    Raises:
        ValueError: If no API key is provided or found in the environment.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "llama-3.3-70b-versatile",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Groq API key required. "
                "Pass api_key= or set GROQ_API_KEY env var. "
                "Get a free key at https://console.groq.com/keys."
            )
        self._api_key = resolved_key
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self._api_key)
        return self._client

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            # Groq supports OpenAI's json_object response_format - eliminates
            # markdown-wrapped-JSON failures.
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
