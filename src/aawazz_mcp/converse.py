"""aawazz-converse — listen → respond → speak conversational loop.

Phase 4 of v1.4 (SPEC §7). Local-only console script (NOT an MCP tool —
round-trip overhead per turn is wrong shape for a tool surface). Pairs
captain's mic, the routed LLM (default ``pipefish``), and a TTS provider
into a hands-free conversational session.

Flow:

    [optional initial prompt, played as audio]
      ↓
    listen (vad:webrtc pre_process trims silence)
      ↓
    transcript checked against exit phrases
      ↓
    respond(messages=history, system_prompt=persona, play=True)
      ↓
    history += [user, assistant]
      ↓
    repeat until exit phrase / max_turns / Ctrl-C / N consecutive empties

Multi-turn history lives in memory; ``--save-transcript path.jsonl`` writes
each turn (one ``{role, content, ts}`` per line). VAD interrupts during
playback are deferred to a follow-up — v1.4.1's converse-toggle wrapper
will bind Super+V to interrupt-and-record mid-response.

Persona system prompts come from squadron persona files: ``--persona
foreman`` reads ``projects/squadron/personas/foreman/persona.md``. Pass a
path instead (``--persona /tmp/x.md``) to point at any file. ``--system-prompt
TEXT`` overrides everything.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

log = logging.getLogger("aawazz_mcp.converse")


_DEFAULT_SQUADRON_PERSONAS = Path(
    os.environ.get(
        "AAWAZZ_SQUADRON_PERSONAS",
        "/home/nixp/WORKSPACE/projects/squadron/personas",
    )
)


def _load_persona(spec: str | None) -> str | None:
    """Resolve a persona spec to a system_prompt string.

    - ``None`` → no persona.
    - ``"/abs/path"`` or anything containing ``/`` → read that file.
    - bare name (e.g. ``"foreman"``) → squadron persona at
      ``$AAWAZZ_SQUADRON_PERSONAS/<name>/persona.md``.
    """
    if not spec:
        return None
    p: Path
    if "/" in spec or spec.endswith(".md"):
        p = Path(spec).expanduser()
    else:
        p = _DEFAULT_SQUADRON_PERSONAS / spec / "persona.md"

    if not p.exists():
        log.warning("persona file not found: %s", p)
        return None
    return p.read_text().strip()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aawazz-converse",
        description=(
            "Listen → LLM → speak conversational loop. Local-only; uses the "
            "configured pipefish (or override via --llm-provider) for "
            "generation and the routed TTS provider for synthesis."
        ),
    )
    # Persona / system prompt
    p.add_argument(
        "--persona", default=None,
        help="Squadron persona name (e.g. 'foreman') OR path to a markdown file.",
    )
    p.add_argument(
        "--system-prompt", default=None,
        help="Direct system prompt text. Overrides --persona when both given.",
    )
    # LLM
    p.add_argument("--llm-provider", default=None,
                   help="Override LLM routing chain (default: pipefish).")
    p.add_argument("--llm-model", default=None,
                   help="Model name; defaults to first available on the LLM provider.")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    # TTS
    p.add_argument("--tts-provider", default=None)
    p.add_argument("--voice", default="MALE")
    p.add_argument("--language", default="en",
                   help="ISO 639-1; drives both STT and the lang_mismatch baseline.")
    # STT
    p.add_argument("--listen-duration", type=float, default=8.0,
                   help="Mic capture duration per turn (seconds).")
    p.add_argument("--no-vad", action="store_true",
                   help="Disable vad:webrtc pre_process on listen.")
    p.add_argument("--model-arch", default="tiny_streaming",
                   help="Moonshine arch for non-Whisper STT.")
    # Loop control
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument(
        "--exit-phrase", action="append", default=None,
        help="Phrase that ends the session (case-insensitive). Repeatable. "
             "Default: 'exit', 'goodbye', 'stop listening'.",
    )
    p.add_argument(
        "--max-empty-turns", type=int, default=3,
        help="Exit after this many consecutive empty STT results.",
    )
    p.add_argument(
        "--initial-prompt", default=None,
        help="Send this prompt to the LLM first (bootstraps the conversation).",
    )
    # Output
    p.add_argument(
        "--save-transcript", default=None,
        help="JSONL path; appends one {role,content,ts} object per turn.",
    )
    p.add_argument("--no-stream", action="store_true",
                   help="Disable streaming respond (use batch).")
    p.add_argument("--lang-mismatch", default="route",
                   choices=("route", "warn", "error", "off"))
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p


def _matches_exit_phrase(text: str, phrases: list[str]) -> bool:
    needle = text.lower().strip().rstrip(".!?")
    for phrase in phrases:
        if phrase.lower() in needle:
            return True
    return False


def _append_transcript(path: Path | None, entry: dict) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        log.exception("failed to write transcript to %s", path)


async def run_loop(args, *, backend=None) -> int:
    """Main converse loop. ``backend`` injection point makes this testable."""
    if backend is None:
        from aawazz_mcp.backends.local import LocalBackend  # noqa: PLC0415
        from aawazz_mcp.config import AawazzConfig  # noqa: PLC0415
        import aawazz_mcp.providers  # noqa: F401, PLC0415  - register
        import aawazz_mcp.post_processors  # noqa: F401, PLC0415  - register

        cfg_args = argparse.Namespace(
            remote=None, transport="stdio", host="127.0.0.1", port=7860,
            warm=False, log_level=args.log_level,
            routing_config=None, tts_default=None, stt_default=None,
            llm_default=None,
        )
        cfg = AawazzConfig.from_args(cfg_args)
        backend = LocalBackend(cfg)

    # Resolve system prompt: --system-prompt > --persona > none.
    system_prompt = args.system_prompt or _load_persona(args.persona)
    if system_prompt:
        log.info("system prompt loaded (%d chars)", len(system_prompt))

    exit_phrases = args.exit_phrase or ["exit", "goodbye", "stop listening"]
    transcript_path = (
        Path(args.save_transcript).expanduser()
        if args.save_transcript
        else None
    )

    pre_process: list[str] | None = (
        None if args.no_vad else ["vad:webrtc"]
    )
    stream = not args.no_stream

    messages: list[dict] = []
    turn = 0
    consecutive_empty = 0

    print(
        f"\n[converse] persona={args.persona or 'none'} "
        f"language={args.language} stream={stream} "
        f"vad={'on' if pre_process else 'off'} "
        f"max_turns={args.max_turns}",
        flush=True,
    )

    # Optional initial prompt to bootstrap.
    if args.initial_prompt:
        print(f"\n[bootstrap] {args.initial_prompt}", flush=True)
        messages.append({"role": "user", "content": args.initial_prompt})
        _append_transcript(
            transcript_path,
            {"role": "user", "content": args.initial_prompt, "ts": time.time()},
        )
        result = await backend.respond(
            messages=list(messages),
            system_prompt=system_prompt,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            tts_provider=args.tts_provider,
            language=args.language,
            voice=args.voice,
            play=True,
            stream=stream,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            lang_mismatch=args.lang_mismatch,
        )
        if "error" in result:
            print(f"[error] {result['error']}", flush=True)
        else:
            text = result.get("text", "")
            print(f"[assistant] {text}", flush=True)
            messages.append({"role": "assistant", "content": text})
            _append_transcript(
                transcript_path,
                {"role": "assistant", "content": text, "ts": time.time()},
            )
        turn += 1

    # Main loop.
    try:
        while turn < args.max_turns:
            print(f"\n[listen turn {turn + 1}] ({args.listen_duration:.1f}s)", flush=True)
            listen_result = await backend.listen(
                duration_s=args.listen_duration,
                language=args.language,
                model_arch=args.model_arch,
                save_audio=False,
                pre_process=pre_process,
            )
            if "error" in listen_result:
                print(
                    f"[listen error] {listen_result['error']}",
                    flush=True,
                )
                hint = listen_result.get("hint")
                if hint:
                    print(f"  hint: {hint}", flush=True)
                return 1

            user_text = (listen_result.get("text") or "").strip()
            if not user_text:
                consecutive_empty += 1
                print(
                    f"[empty] no transcript ({consecutive_empty}/"
                    f"{args.max_empty_turns})",
                    flush=True,
                )
                if consecutive_empty >= args.max_empty_turns:
                    print("[exit] consecutive empties reached.", flush=True)
                    return 0
                continue

            consecutive_empty = 0
            print(f"[user] {user_text}", flush=True)

            if _matches_exit_phrase(user_text, exit_phrases):
                print("[exit] exit phrase detected.", flush=True)
                return 0

            messages.append({"role": "user", "content": user_text})
            _append_transcript(
                transcript_path,
                {"role": "user", "content": user_text, "ts": time.time()},
            )

            result = await backend.respond(
                messages=list(messages),
                system_prompt=system_prompt,
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
                tts_provider=args.tts_provider,
                language=args.language,
                voice=args.voice,
                play=True,
                stream=stream,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                lang_mismatch=args.lang_mismatch,
            )
            if "error" in result:
                print(f"[error] {result['error']}", flush=True)
                hint = result.get("hint")
                if hint:
                    print(f"  hint: {hint}", flush=True)
                # Don't append assistant message on errors; let caller try again.
                turn += 1
                continue

            assistant_text = result.get("text", "")
            print(f"[assistant] {assistant_text}", flush=True)
            if result.get("language_mismatch"):
                lm = result["language_mismatch"]
                print(
                    f"  (language_mismatch: requested={lm['requested']} "
                    f"detected={lm['detected']}, tts={result.get('tts_provider')})",
                    flush=True,
                )

            messages.append({"role": "assistant", "content": assistant_text})
            _append_transcript(
                transcript_path,
                {"role": "assistant", "content": assistant_text, "ts": time.time()},
            )
            turn += 1

        print("[exit] max_turns reached.", flush=True)
        return 0
    except KeyboardInterrupt:
        print("\n[exit] interrupted.", flush=True)
        return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s aawazz-converse %(message)s",
    )
    return asyncio.run(run_loop(args))


if __name__ == "__main__":
    raise SystemExit(main())
