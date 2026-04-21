"""
Extract stage public interface.

Takes the top chunks from the ranking stage, concatenates them into a
token-budgeted context, calls an LLM, and returns structured values for the
requested extraction keys.

Pick an adapter explicitly:

    from webfetch.extract import ClaudeExtractor
    extractor = ClaudeExtractor()
    values = extractor.extract(
        chunks=top_chunks,
        keys={
            "accuracy": "measurement accuracy including units and conditions",
            "range": "full measurement range with units",
            "resolution": "smallest measurable increment with units",
        },
    )
"""

from webfetch.extract.base import AbstractExtractor, build_context
from webfetch.extract.claude import ClaudeExtractor
from webfetch.extract.gemini import GeminiExtractor
from webfetch.extract.gpt import GPTExtractor
from webfetch.extract.groq import GroqExtractor

__all__ = [
    "AbstractExtractor",
    "ClaudeExtractor",
    "GPTExtractor",
    "GeminiExtractor",
    "GroqExtractor",
    "build_context",
]
