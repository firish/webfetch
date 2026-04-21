"""
Google Gemini extractor - the recommended FREE adapter.

Gemini 2.0 Flash has a generous free tier (15 RPM, 1500 RPD) that covers
typical low-to-medium volume extraction workloads end-to-end at zero cost.
It also has a `response_mime_type="application/json"` mode that enforces
JSON output, same as OpenAI's json_object mode.

Get a free API key at https://aistudio.google.com/apikey

Requires GOOGLE_API_KEY env var, or pass api_key= to the constructor.
"""

from __future__ import annotations

import os

from webfetch.extract.base import AbstractExtractor


class GeminiExtractor(AbstractExtractor):
    """Extractor backed by Google Gemini (free tier recommended).

    Args:
        api_key: Google AI API key. Defaults to GOOGLE_API_KEY env var.
        model: Gemini model ID. gemini-2.0-flash has the best free tier
               quota/quality tradeoff. For higher accuracy (paid) try
               gemini-2.0-flash-exp or gemini-1.5-pro.
        max_tokens: Max tokens in the response.
        temperature: 0 for deterministic extraction.

    Raises:
        ValueError: If no API key is provided or found in the environment.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Google API key required. "
                "Pass api_key= or set GOOGLE_API_KEY env var. "
                "Get a free key at https://aistudio.google.com/apikey."
            )
        self._api_key = resolved_key
        self._model_name = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._model = None

    def _get_model(self):
        if self._model is None:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            self._model = genai.GenerativeModel(
                model_name=self._model_name,
                # Gemini does not accept a separate `system` message like
                # Claude/OpenAI - system instructions are passed in the model
                # config instead.
                system_instruction=None,
            )
        return self._model

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        import google.generativeai as genai
        genai.configure(api_key=self._api_key)

        # Gemini wants the system prompt as system_instruction on the model,
        # so we rebuild the model here to carry it. Cheap - just local state.
        model = genai.GenerativeModel(
            model_name=self._model_name,
            system_instruction=system_prompt,
        )

        response = model.generate_content(
            user_prompt,
            generation_config=genai.GenerationConfig(
                temperature=self._temperature,
                max_output_tokens=self._max_tokens,
                # Force JSON output - removes the markdown-wrapped-JSON failure mode.
                response_mime_type="application/json",
            ),
        )
        return response.text
