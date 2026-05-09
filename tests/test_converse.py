"""Coverage for :mod:`aawazz_mcp.converse` — the v1.4 phase-4 console script.

The loop is driven by ``LocalBackend.listen()`` and ``.respond()``. We
inject a stub backend so the tests never touch the real mic, LLM, or
TTS — and so the suite stays fast.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from aawazz_mcp.converse import (
    _load_persona,
    _matches_exit_phrase,
    run_loop,
)


# ── Pure helpers ────────────────────────────────────────────────────────────


def test_exit_phrase_match_case_insensitive() -> None:
    assert _matches_exit_phrase("Goodbye", ["exit", "goodbye"])
    assert _matches_exit_phrase("Goodbye!", ["exit", "goodbye"])
    assert _matches_exit_phrase("please stop listening.", ["stop listening"])
    assert not _matches_exit_phrase("good bye now", ["goodbye"])
    assert not _matches_exit_phrase("Hello there", ["exit", "goodbye"])


def test_load_persona_from_path(tmp_path: Path) -> None:
    p = tmp_path / "test_persona.md"
    p.write_text("You are a terse assistant. Reply in one sentence.\n")
    assert _load_persona(str(p)) == "You are a terse assistant. Reply in one sentence."


def test_load_persona_squadron_name(monkeypatch, tmp_path: Path) -> None:
    """Bare name resolves under the squadron personas dir."""
    fake_root = tmp_path / "personas"
    (fake_root / "myrole").mkdir(parents=True)
    (fake_root / "myrole" / "persona.md").write_text("# Myrole\n\nBe terse.")
    monkeypatch.setattr(
        "aawazz_mcp.converse._DEFAULT_SQUADRON_PERSONAS", fake_root
    )
    assert _load_persona("myrole") == "# Myrole\n\nBe terse."


def test_load_persona_missing_returns_none(tmp_path: Path) -> None:
    assert _load_persona(str(tmp_path / "does_not_exist.md")) is None


def test_load_persona_none_input_is_none() -> None:
    assert _load_persona(None) is None
    assert _load_persona("") is None


# ── Loop with stub backend ──────────────────────────────────────────────────


class _StubBackend:
    """Mock LocalBackend driving the converse loop deterministically."""

    def __init__(
        self,
        listen_results: list[dict],
        respond_results: list[dict] | None = None,
    ) -> None:
        self._listen_q = list(listen_results)
        self._respond_q = list(respond_results or [])
        self.listen_calls: list[dict] = []
        self.respond_calls: list[dict] = []

    async def listen(self, **kwargs):
        self.listen_calls.append(kwargs)
        if not self._listen_q:
            return {"error": "stub listen exhausted", "backend": "local"}
        return self._listen_q.pop(0)

    async def respond(self, **kwargs):
        self.respond_calls.append(kwargs)
        if not self._respond_q:
            return {"text": "stub reply", "backend": "local"}
        return self._respond_q.pop(0)


def _args(**overrides):
    """Build a minimal argparse-style namespace for run_loop."""
    import argparse
    base = dict(
        persona=None, system_prompt=None,
        llm_provider=None, llm_model=None,
        max_tokens=64, temperature=0.5,
        tts_provider=None, voice="MALE",
        language="en",
        listen_duration=1.0, no_vad=False, model_arch="tiny_streaming",
        max_turns=5, exit_phrase=None, max_empty_turns=2,
        initial_prompt=None, save_transcript=None,
        no_stream=True, lang_mismatch="off",
        log_level="WARNING",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.mark.asyncio
async def test_loop_exits_on_exit_phrase() -> None:
    backend = _StubBackend(listen_results=[
        {"text": "Hello there"},
        {"text": "Goodbye"},
    ], respond_results=[
        {"text": "Hi back"},
    ])
    rc = await run_loop(_args(max_turns=10), backend=backend)
    assert rc == 0
    # Two listens (the second triggered exit before respond).
    assert len(backend.listen_calls) == 2
    # Only one respond — the exit phrase short-circuited the second.
    assert len(backend.respond_calls) == 1


@pytest.mark.asyncio
async def test_loop_exits_on_max_turns() -> None:
    backend = _StubBackend(listen_results=[
        {"text": "one"}, {"text": "two"}, {"text": "three"},
    ])
    rc = await run_loop(_args(max_turns=2), backend=backend)
    assert rc == 0
    # max_turns=2 → exactly 2 listen+respond pairs.
    assert len(backend.listen_calls) == 2
    assert len(backend.respond_calls) == 2


@pytest.mark.asyncio
async def test_loop_exits_on_consecutive_empties() -> None:
    backend = _StubBackend(listen_results=[
        {"text": ""}, {"text": ""}, {"text": "should not reach"},
    ])
    rc = await run_loop(
        _args(max_turns=10, max_empty_turns=2), backend=backend
    )
    assert rc == 0
    # 2 empties → exit; the third listen never runs.
    assert len(backend.listen_calls) == 2
    assert len(backend.respond_calls) == 0


@pytest.mark.asyncio
async def test_loop_listen_error_returns_nonzero() -> None:
    backend = _StubBackend(listen_results=[
        {"error": "mic capture failed", "hint": "OS mute / UEFI mute / routing"},
    ])
    rc = await run_loop(_args(max_turns=5), backend=backend)
    assert rc == 1
    assert len(backend.respond_calls) == 0


@pytest.mark.asyncio
async def test_loop_initial_prompt_bootstraps(tmp_path: Path) -> None:
    """--initial-prompt sends a kicker user message; the LLM reply plays
    BEFORE the first listen."""
    transcript = tmp_path / "transcript.jsonl"
    backend = _StubBackend(
        listen_results=[{"text": "exit"}],  # exit immediately after bootstrap
        respond_results=[{"text": "Hello, captain."}],
    )
    rc = await run_loop(
        _args(
            initial_prompt="Greet me",
            save_transcript=str(transcript),
            max_turns=5,
        ),
        backend=backend,
    )
    assert rc == 0
    # Bootstrap respond ran first; then one listen returned "exit".
    assert len(backend.respond_calls) == 1
    assert len(backend.listen_calls) == 1
    # Transcript captured both turns of the bootstrap.
    lines = transcript.read_text().strip().split("\n")
    assert len(lines) == 2  # user + assistant
    import json
    user, asst = json.loads(lines[0]), json.loads(lines[1])
    assert user["role"] == "user" and user["content"] == "Greet me"
    assert asst["role"] == "assistant" and asst["content"] == "Hello, captain."


@pytest.mark.asyncio
async def test_loop_passes_persona_as_system_prompt(tmp_path: Path) -> None:
    persona_file = tmp_path / "persona.md"
    persona_file.write_text("You are Foreman. Be terse.")

    backend = _StubBackend(
        listen_results=[{"text": "exit"}],
    )
    await run_loop(
        _args(persona=str(persona_file), max_turns=2), backend=backend
    )
    # The single listen returned 'exit' — but no respond. So persona didn't
    # land. Re-run with a non-exit transcript.
    backend = _StubBackend(
        listen_results=[{"text": "Hello"}, {"text": "exit"}],
        respond_results=[{"text": "Hi"}],
    )
    await run_loop(
        _args(persona=str(persona_file), max_turns=5), backend=backend
    )
    assert backend.respond_calls[0]["system_prompt"] == "You are Foreman. Be terse."


@pytest.mark.asyncio
async def test_loop_history_grows_across_turns() -> None:
    backend = _StubBackend(
        listen_results=[
            {"text": "first user message"},
            {"text": "second user message"},
            {"text": "exit"},
        ],
        respond_results=[
            {"text": "first reply"},
            {"text": "second reply"},
        ],
    )
    await run_loop(_args(max_turns=10), backend=backend)
    # Second respond call sees both prior turns in messages history.
    second_call = backend.respond_calls[1]
    msgs = second_call["messages"]
    assert len(msgs) == 3  # user1, assistant1, user2
    assert msgs[0]["content"] == "first user message"
    assert msgs[1]["content"] == "first reply"
    assert msgs[2]["content"] == "second user message"


@pytest.mark.asyncio
async def test_loop_passes_pre_process_unless_no_vad() -> None:
    backend = _StubBackend(listen_results=[{"text": "exit"}])
    await run_loop(_args(no_vad=False, max_turns=2), backend=backend)
    assert backend.listen_calls[0]["pre_process"] == ["vad:webrtc"]

    backend = _StubBackend(listen_results=[{"text": "exit"}])
    await run_loop(_args(no_vad=True, max_turns=2), backend=backend)
    assert backend.listen_calls[0]["pre_process"] is None


@pytest.mark.asyncio
async def test_loop_save_transcript_writes_jsonl(tmp_path: Path) -> None:
    transcript = tmp_path / "out.jsonl"
    backend = _StubBackend(
        listen_results=[{"text": "Hello"}, {"text": "exit"}],
        respond_results=[{"text": "Hi"}],
    )
    await run_loop(
        _args(save_transcript=str(transcript), max_turns=5), backend=backend
    )
    assert transcript.exists()
    lines = transcript.read_text().strip().split("\n")
    import json
    parsed = [json.loads(l) for l in lines]
    assert len(parsed) == 2
    assert parsed[0]["role"] == "user"
    assert parsed[1]["role"] == "assistant"
    assert parsed[0]["content"] == "Hello"
    assert parsed[1]["content"] == "Hi"
