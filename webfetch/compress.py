"""
Sentence-level extractive compression for tool results.

Within each top-ranked chunk, keep only the sentences relevant to the query.
Runs at tool-result FORMATTING time, after the cache read, so cached chunks
stay full-text and budget-agnostic - retuning compression never invalidates
the cache.

Why extractive (select sentences) rather than abstractive (rewrite):
abstractive needs an LLM call per result, which is against the library's
zero-marginal-cost design; extractive is a pure function of (query, chunks).

Parameters (scorer, selection policy, guards) are chosen by the eval harness
(evals/run_compression_eval.py) which imports THIS module - the shipped code
is exactly what was measured. Config defaults in webfetch/config.py hold the
winning configuration: cross-encoder scored, ratio 0.5, all guards on -
measured ZERO recall drop (29/29 answer survival) at 50% of baseline tokens
on 50 captured production results. The cross-encoder is what makes lossless
compression possible here: bi-encoder selection at the same token level
loses 3-4 answers of 29.

Guards against the known failure modes of naive sentence selection:
  - anaphora: a kept sentence starting with a pronoun/connective would be
    unreadable alone, so its preceding sentence is retained with it
  - table-like lines (markdown rows, digit-dense spec lines) are not prose -
    sentence scoring misjudges them, so they bypass scoring and are kept
  - cross-chunk dedup: the chunker's 10% overlap re-emits boundary sentences
    in the next chunk; duplicates waste budget

Requires the optional rerank extra for the bi-encoder scorer; without it the
scorer degrades to the zero-dep lexical scorer (never crashes the tool path).
"""

from __future__ import annotations

import logging
import math
import re
import threading

from webfetch.config import (
    BIENCODER_MODEL,
    COMPRESS_ANAPHORA_GUARD,
    COMPRESS_DEDUP,
    COMPRESS_PARAM,
    COMPRESS_POLICY,
    COMPRESS_SCORER,
    COMPRESS_TABLE_GUARD,
    CROSSENCODER_MODEL,
)
from webfetch.rank.base import Chunk

logger = logging.getLogger(__name__)

# Sentence boundary: terminal punctuation, whitespace, then something that
# looks like a sentence start. Deliberately regex-based and zero-dep (same
# philosophy as the chunker) - abbreviation mistakes cost a split, and the
# eval measures the net effect rather than assuming splitter perfection.
_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

# First words that signal the sentence leans on its predecessor.
_ANAPHORIC_STARTS = frozenset({
    "it", "its", "this", "that", "these", "those", "they", "them", "their",
    "he", "she", "his", "her", "him", "however", "but", "also", "moreover",
    "additionally", "furthermore", "meanwhile", "instead", "still", "yet",
    "so", "such", "both", "neither", "otherwise", "consequently", "thus",
    "therefore", "here", "there",
})

_FIRST_WORD_RE = re.compile(r"[a-zA-Z]+")

# Digit share of non-space chars above which a line is treated as data,
# not prose (spec lines, score lines, table rows).
_TABLE_DIGIT_RATIO = 0.20

# Sentences shorter than this are never deduped - short strings ("Yes.",
# a bare number) can legitimately repeat with different meanings.
_DEDUP_MIN_CHARS = 30

_bi_model = None
_ce_model = None
_models_unavailable = False
# Concurrent tool calls racing lazy model construction crash inside torch -
# same locking pattern as rank/biencoder.py.
_init_lock = threading.Lock()


def split_sentences(text: str) -> list[tuple[int, str]]:
    """Split text into (line_index, sentence) pairs.

    Newlines are hard boundaries (extracted pages use them structurally:
    headings, list items, table rows); prose lines are further split on
    sentence punctuation. Chunk-edge fragments come through as ordinary
    sentences and are scored like any other.

    Args:
        text: Chunk text.

    Returns:
        List of (line_index, sentence) pairs in document order.
    """
    out: list[tuple[int, str]] = []
    for line_idx, line in enumerate(text.split("\n")):
        line = line.strip()
        if not line:
            continue
        for sent in _SENT_BOUNDARY_RE.split(line):
            sent = sent.strip()
            if sent:
                out.append((line_idx, sent))
    return out


def is_table_like(sentence: str) -> bool:
    """True for lines that are data rather than prose (skip scoring)."""
    if "|" in sentence:
        return True
    compact = sentence.replace(" ", "")
    if not compact:
        return False
    digits = sum(ch.isdigit() for ch in compact)
    return digits / len(compact) >= _TABLE_DIGIT_RATIO


def starts_anaphoric(sentence: str) -> bool:
    """True if the sentence's first word leans on the preceding sentence."""
    m = _FIRST_WORD_RE.search(sentence[:24])
    return bool(m) and m.group(0).lower() in _ANAPHORIC_STARTS


def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def score_lexical(query: str, sentences: list[str]) -> list[float]:
    """IDF-weighted query-term coverage per sentence, in [0, 1].

    IDF is computed over the sentences of THIS call (the per-result corpus):
    a query term appearing in every sentence carries no selection signal,
    while a rare one dominates. Zero-dep fallback scorer.
    """
    q_terms = list(dict.fromkeys(_tokenize(query)))
    if not q_terms or not sentences:
        return [0.0] * len(sentences)
    sent_tokens = [set(_tokenize(s)) for s in sentences]
    n = len(sentences)
    idf = {t: math.log(1 + n / (1 + sum(t in st for st in sent_tokens)))
           for t in q_terms}
    total = sum(idf.values()) or 1.0
    return [sum(idf[t] for t in q_terms if t in st) / total
            for st in sent_tokens]


def _deps_available() -> bool:
    global _models_unavailable
    if _models_unavailable:
        return False
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        logger.warning(
            "sentence-transformers not installed - compression degrades to "
            "the lexical scorer. Install webfetch[rerank]."
        )
        _models_unavailable = True
        return False
    return True


def score_biencoder(query: str, sentences: list[str]) -> list[float] | None:
    """Cosine similarity of each sentence to the query; None without [rerank]."""
    global _bi_model
    if not _deps_available():
        return None
    from sentence_transformers import SentenceTransformer
    with _init_lock:
        if _bi_model is None:
            _bi_model = SentenceTransformer(BIENCODER_MODEL)
    q_vec = _bi_model.encode([query], normalize_embeddings=True,
                             show_progress_bar=False)[0]
    s_vecs = _bi_model.encode(sentences, normalize_embeddings=True,
                              show_progress_bar=False)
    return [float(v @ q_vec) for v in s_vecs]


def score_crossencoder(query: str, sentences: list[str]) -> list[float] | None:
    """Joint (query, sentence) relevance scores; None without [rerank].

    Same ms-marco model as the ranking cascade's stage 3. O(N) forward
    passes, but N here is the ~15-25 sentences of one tool result - tens of
    milliseconds, not the cost profile that keeps cross-encoders out of
    full-list ranking.
    """
    global _ce_model
    if not _deps_available():
        return None
    from sentence_transformers import CrossEncoder
    with _init_lock:
        if _ce_model is None:
            _ce_model = CrossEncoder(CROSSENCODER_MODEL)
    scores = _ce_model.predict([(query, s) for s in sentences],
                               show_progress_bar=False)
    return [float(s) for s in scores]


def _select_indices(scores: list[float], sentences: list[str],
                    policy: str, param: float) -> set[int]:
    """Pick sentence indices for one chunk under the selection policy.

    Policies:
        ratio: best-scoring sentences until ~param of the chunk's chars kept.
        topk: best int(param) sentences.
        threshold: every sentence scoring >= param.

    Always returns at least one index (the best-scoring sentence) so a chunk
    can never compress to nothing.
    """
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep: set[int] = set()
    if policy == "ratio":
        total = sum(len(s) for s in sentences)
        used = 0
        for i in order:
            if keep and used + len(sentences[i]) > param * total:
                continue  # try smaller later sentences - greedy knapsack
            keep.add(i)
            used += len(sentences[i])
    elif policy == "topk":
        keep = set(order[: max(1, int(param))])
    elif policy == "threshold":
        keep = {i for i in order if scores[i] >= param}
    else:
        raise ValueError(f"unknown compression policy: {policy!r}")
    if not keep:
        keep = {order[0]}
    return keep


def compress_chunks(
    query: str,
    chunks: list[Chunk],
    scorer: str = COMPRESS_SCORER,
    policy: str = COMPRESS_POLICY,
    param: float = COMPRESS_PARAM,
    anaphora_guard: bool = COMPRESS_ANAPHORA_GUARD,
    table_guard: bool = COMPRESS_TABLE_GUARD,
    dedup: bool = COMPRESS_DEDUP,
) -> list[Chunk]:
    """Compress each chunk to its query-relevant sentences.

    Returns NEW Chunk objects - inputs are never mutated, because cached
    chunk instances may be shared with other readers. Kept sentences stay in
    document order; sentences from the same source line rejoin with a space,
    different lines with a newline (preserves table/list structure).

    On any unexpected failure the original chunks are returned unchanged -
    this runs on the tool path, where a formatting bug must never turn a
    good search into an error.

    Args:
        query: The search query driving relevance.
        chunks: Ranked chunks (best first) from the pipeline.
        scorer: "biencoder", "crossencoder", "lexical", or "lead"
            (position baseline).
        policy: Selection policy - "ratio", "topk", or "threshold".
        param: Policy parameter (ratio fraction, k, or score cutoff).
        anaphora_guard: Retain the predecessor of pronoun-initial keeps.
        table_guard: Keep table-like lines unconditionally.
        dedup: Drop sentences already emitted by an earlier chunk.

    Returns:
        Compressed chunks, same order; chunks fully deduplicated away are
        dropped. Empty input returns [].
    """
    if not chunks:
        return []
    try:
        return _compress(query, chunks, scorer, policy, param,
                         anaphora_guard, table_guard, dedup)
    except Exception:
        logger.warning("compression failed - returning chunks uncompressed",
                       exc_info=True)
        return chunks


def _compress(query: str, chunks: list[Chunk], scorer: str, policy: str,
              param: float, anaphora_guard: bool, table_guard: bool,
              dedup: bool) -> list[Chunk]:
    per_chunk = [split_sentences(c.text) for c in chunks]

    # Score every sentence across all chunks in ONE batch (matters for the
    # bi-encoder: one encode call instead of one per chunk).
    flat = [s for sents in per_chunk for (_, s) in sents]
    if scorer == "biencoder":
        flat_scores = score_biencoder(query, flat)
        if flat_scores is None:
            flat_scores = score_lexical(query, flat)
    elif scorer == "crossencoder":
        flat_scores = score_crossencoder(query, flat)
        if flat_scores is None:
            flat_scores = score_lexical(query, flat)
    elif scorer == "lexical":
        flat_scores = score_lexical(query, flat)
    elif scorer == "lead":
        # Position baseline: earlier in chunk = higher. Eval control arm.
        flat_scores = []
        for sents in per_chunk:
            flat_scores.extend(1.0 / (1 + i) for i in range(len(sents)))
    else:
        raise ValueError(f"unknown compression scorer: {scorer!r}")

    out: list[Chunk] = []
    seen_norm: set[str] = set()
    pos = 0
    for chunk, sents in zip(chunks, per_chunk):
        scores = flat_scores[pos: pos + len(sents)]
        pos += len(sents)
        if not sents:
            continue
        if len(sents) == 1:
            keep = {0}
        else:
            texts = [s for (_, s) in sents]
            keep = _select_indices(scores, texts, policy, param)
            if table_guard:
                keep |= {i for i, t in enumerate(texts) if is_table_like(t)}
            if anaphora_guard:
                # Walk each kept sentence's predecessors so chains resolve
                # ("It ... " preceded by "This ..." pulls in both).
                for i in sorted(keep):
                    j = i
                    while (j > 0 and starts_anaphoric(texts[j])
                           and (j - 1) not in keep):
                        keep.add(j - 1)
                        j -= 1

        kept = [sents[i] for i in sorted(keep)]
        if dedup:
            deduped = []
            for line_idx, s in kept:
                norm = " ".join(_tokenize(s))
                if len(s) >= _DEDUP_MIN_CHARS and norm in seen_norm:
                    continue
                seen_norm.add(norm)
                deduped.append((line_idx, s))
            kept = deduped
        if not kept:
            continue  # chunk was pure overlap-duplicate content

        parts = [kept[0][1]]
        for (prev_line, _), (line_idx, s) in zip(kept, kept[1:]):
            parts.append(("\n" if line_idx != prev_line else " ") + s)
        out.append(Chunk(text="".join(parts), url=chunk.url,
                         title=chunk.title, score=chunk.score))
    return out


__all__ = ["compress_chunks", "split_sentences", "score_lexical",
           "score_biencoder", "score_crossencoder", "is_table_like",
           "starts_anaphoric"]
