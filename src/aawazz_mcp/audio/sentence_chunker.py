"""Sentence-boundary chunker for streaming LLM output → TTS.

Phase 2 of v1.4 (SPEC §4.3). Reads the LLM token stream as it arrives,
accumulates until it hits a sentence boundary, flushes the chunk
downstream. Final chunk on EOS regardless of boundary.

Two modes:

* **Regex fallback** (default; no extra deps) — splits on ``.?!``
  followed by whitespace, plus blank-line breaks. Handles English and
  most Latin-script languages adequately for v1.4.0.
* **pysbd** (``[chunking]`` extra) — multi-language sentence
  segmentation, more robust on abbreviations and numerals. Activate by
  installing ``pysbd>=0.3``.

A min-chunk-size guard accumulates short sentences (<5 words) into the
next sentence to avoid synthesizing tiny clips that produce artifacts —
the SPEC §15 Q8 spike result said ~5 words is the floor where synthesis
stays stable.
"""

from __future__ import annotations

import logging
import re
from typing import AsyncIterator

log = logging.getLogger("aawazz_mcp.audio.sentence_chunker")

# Match a sentence boundary: ``.``, ``?``, ``!``, ``।`` (Devanagari danda),
# ``。`` (Chinese/Japanese), or ``\n\n`` (paragraph break). Followed by
# whitespace OR end-of-buffer.
_SENTENCE_END = re.compile(r"([.!?।。]|\n\n+)(\s+|$)")

_DEFAULT_MIN_WORDS = 5


def _probe_pysbd() -> bool:
    try:
        import pysbd  # noqa: F401, PLC0415
        return True
    except Exception:
        return False


def _split_with_pysbd(text: str, language: str) -> list[str]:
    import pysbd  # noqa: PLC0415
    seg = pysbd.Segmenter(language=language[:2], clean=False)
    return [s for s in seg.segment(text) if s.strip()]


def _split_with_regex(text: str) -> tuple[list[str], str]:
    """Return (complete_sentences, remainder)."""
    sentences: list[str] = []
    pos = 0
    for m in _SENTENCE_END.finditer(text):
        end = m.end()
        chunk = text[pos:end].strip()
        if chunk:
            sentences.append(chunk)
        pos = end
    return sentences, text[pos:]


def _word_count(text: str) -> int:
    return len(text.split())


async def chunk_stream(
    text_stream: AsyncIterator[str],
    *,
    language: str = "en",
    min_words: int = _DEFAULT_MIN_WORDS,
    use_pysbd: bool | None = None,
) -> AsyncIterator[str]:
    """Async generator: consume LLM deltas, yield complete sentences.

    Args:
        text_stream: Async iterator yielding incremental text deltas
            (``LlmChunk.text`` values, NOT cumulative).
        language: ISO 639-1 hint for pysbd. Ignored by the regex fallback.
        min_words: Sentences shorter than this get accumulated into the
            next one. Set to 1 to disable the floor.
        use_pysbd: ``None`` (default) auto-detects. ``True`` requires the
            ``[chunking]`` extra. ``False`` always uses the regex fallback.

    Yields:
        Complete sentence strings, in order. The final yield is whatever
        text remains in the buffer when ``text_stream`` is exhausted —
        flushed even if it doesn't terminate with a boundary punctuation.
    """
    use_pysbd_actual = (
        _probe_pysbd() if use_pysbd is None else use_pysbd
    )
    if use_pysbd is True and not _probe_pysbd():
        log.warning(
            "pysbd requested but not installed; falling back to regex"
        )
        use_pysbd_actual = False

    buffer = ""
    pending = ""  # accumulated short sentence waiting for the next one

    async for delta in text_stream:
        if not delta:
            continue
        buffer += delta

        # Cheap inner loop: split on punctuation boundaries until none
        # remain in the buffer. pysbd only kicks in if installed.
        if use_pysbd_actual:
            try:
                pieces = _split_with_pysbd(buffer, language)
            except Exception:  # noqa: BLE001
                log.exception("pysbd failed; falling back to regex")
                use_pysbd_actual = False
                pieces = []

            # pysbd returns ALL sentences including a possibly-incomplete
            # tail. Heuristic: if the buffer doesn't end in a sentence
            # terminator, the last piece is incomplete — keep it as the
            # new buffer and yield the rest.
            if pieces:
                ends_clean = bool(
                    re.search(r"[.!?।。]\s*$", buffer)
                    or buffer.endswith("\n\n")
                )
                if ends_clean:
                    complete, buffer = pieces, ""
                else:
                    complete, buffer = pieces[:-1], pieces[-1]

                for s in complete:
                    s = s.strip()
                    if not s:
                        continue
                    candidate = (pending + " " + s).strip() if pending else s
                    if _word_count(candidate) < min_words:
                        pending = candidate
                    else:
                        pending = ""
                        yield candidate
                continue

        # Regex fallback path.
        sentences, buffer = _split_with_regex(buffer)
        for s in sentences:
            candidate = (pending + " " + s).strip() if pending else s
            if _word_count(candidate) < min_words:
                pending = candidate
            else:
                pending = ""
                yield candidate

    # Stream exhausted — flush whatever's left.
    tail = (pending + " " + buffer).strip() if pending else buffer.strip()
    if tail:
        yield tail
