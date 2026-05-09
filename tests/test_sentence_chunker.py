"""Coverage for :mod:`aawazz_mcp.audio.sentence_chunker`."""

from __future__ import annotations

import pytest

from aawazz_mcp.audio.sentence_chunker import chunk_stream


async def _stream(deltas: list[str]):
    for d in deltas:
        yield d


@pytest.mark.asyncio
async def test_emits_complete_sentences_at_boundaries() -> None:
    deltas = [
        "Hello ", "world. ", "This is the second ", "sentence. ", "And ", "third!",
    ]
    out = []
    async for s in chunk_stream(_stream(deltas), use_pysbd=False, min_words=1):
        out.append(s)
    assert out == ["Hello world.", "This is the second sentence.", "And third!"]


@pytest.mark.asyncio
async def test_min_words_floor_accumulates_short_sentences() -> None:
    """A short fragment <min_words gets bundled into the next sentence."""
    deltas = ["Hi. ", "This is a much longer sentence about things.", " Done."]
    out = []
    async for s in chunk_stream(_stream(deltas), use_pysbd=False, min_words=5):
        out.append(s)
    # "Hi." (1 word) bundles into the next; "Done." (1 word) tails out.
    assert out[0] == "Hi. This is a much longer sentence about things."
    assert "Done" in out[-1]


@pytest.mark.asyncio
async def test_flushes_remainder_when_stream_ends_without_punctuation() -> None:
    deltas = ["A complete sentence. ", "An incomplete tail with no period"]
    out = []
    async for s in chunk_stream(_stream(deltas), use_pysbd=False, min_words=1):
        out.append(s)
    assert out == ["A complete sentence.", "An incomplete tail with no period"]


@pytest.mark.asyncio
async def test_handles_empty_deltas() -> None:
    """Empty / whitespace-only deltas don't break the stream."""
    deltas = ["", "Hello.", " ", " World."]
    out = []
    async for s in chunk_stream(_stream(deltas), use_pysbd=False, min_words=1):
        out.append(s)
    assert out == ["Hello.", "World."]


@pytest.mark.asyncio
async def test_handles_paragraph_break() -> None:
    """A blank-line break terminates the current sentence."""
    deltas = ["First paragraph", "\n\n", "Second paragraph"]
    out = []
    async for s in chunk_stream(_stream(deltas), use_pysbd=False, min_words=1):
        out.append(s)
    # First paragraph ends at \n\n; second has no terminator and tails out.
    assert any("First paragraph" in s for s in out)
    assert any("Second paragraph" in s for s in out)


@pytest.mark.asyncio
async def test_devanagari_danda_terminator() -> None:
    """``।`` is the Hindi/Nepali sentence terminator — splits like ``.``."""
    deltas = ["यह पहला वाक्य है। ", "यह दूसरा है।"]
    out = []
    async for s in chunk_stream(_stream(deltas), use_pysbd=False, min_words=1):
        out.append(s)
    assert len(out) == 2
    assert out[0].endswith("।")


@pytest.mark.asyncio
async def test_empty_stream_yields_nothing() -> None:
    out = []
    async for s in chunk_stream(_stream([]), use_pysbd=False, min_words=1):
        out.append(s)
    assert out == []
