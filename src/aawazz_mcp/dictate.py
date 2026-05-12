"""aawazz-dictate — push-to-talk dictation CLI.

Standalone tool, NOT an MCP tool. Records mic for N seconds, transcribes via
Moonshine, then dispatches the transcript via:

  - ``type``      keystroke injection into the focused window
                  (xdotool on X11, wtype/ydotool on Wayland, osascript on macOS)
  - ``clipboard`` paste into the system clipboard
                  (xclip/xsel on X11, wl-copy on Wayland, pbcopy on macOS,
                  termux-clipboard-set on Android/Termux)
  - ``stdout``    print transcript only

Designed for hotkey binding when typing is inconvenient (wet hands, cooking,
walking). Pairs naturally with the agent-side ``listen`` tool but lives entirely
on the operator's machine — no MCP runtime is involved on this path.

The ``auto`` mode (default) picks the best output for the detected session:
type → clipboard → stdout, in that order. ``--verbose`` prints the chosen mode
and timings to stderr for hotkey-script debugging.

Exit codes:
  0  ok — transcript dispatched
  1  no input device (mic missing, OS-muted, sandboxed)
  2  transcribe returned empty / failed
  3  output dispatch failed (typer/clipboarder errored)
  4  no typer or clipboarder available for the session

CRITICAL — stdout safety: Moonshine and tiny-tts both have stdout-print
landmines (see audio/playback.py + models/tts_loader.stdout_to_stderr); this
script does NOT load tiny-tts (TTS is not used) so the only concern is
Moonshine. The shared SttLoader handles that wrapping internally.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_NO_INPUT = 1
EXIT_TRANSCRIBE_FAILED = 2
EXIT_DISPATCH_FAILED = 3
EXIT_NO_DISPATCHER = 4

DEFAULT_DURATION_S = 8.0
DEFAULT_LANGUAGE = "en"
DEFAULT_ARCH = "tiny_streaming"


# ---------------------------------------------------------------- session/host


_TERMUX_PREFIX: str = "/data/data/com.termux/files/usr"


def _detect_session_type() -> str:
    """Return ``"wayland" | "x11" | "darwin" | "termux" | "unknown"``.

    macOS is detected via ``sys.platform``; Linux via ``XDG_SESSION_TYPE`` then
    ``WAYLAND_DISPLAY`` / ``DISPLAY`` fallbacks. Termux/Android is detected
    only after display checks — Termux:X11 users keep the X11 clipboard path
    when they have a display set up, and headless Termux falls into the
    Android-clipboard branch. Headless non-Termux boxes return ``"unknown"``
    and the only viable mode is ``stdout``.
    """
    if sys.platform == "darwin":
        return "darwin"
    s = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if "wayland" in s:
        return "wayland"
    if "x11" in s:
        return "x11"
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    # Termux on Android — native shell sets $PREFIX, proot-distro doesn't
    # but the Termux PREFIX dir is bind-mounted in. Check both.
    prefix = os.environ.get("PREFIX", "")
    if prefix.startswith(_TERMUX_PREFIX) or os.path.isdir(_TERMUX_PREFIX):
        return "termux"
    return "unknown"


def _resolve_typer(session: str) -> tuple[list[str], str] | None:
    """Pick a keystroke-injection command for the session, or None.

    Returns ``(argv_template, name)`` where ``argv_template`` contains a
    ``"{TEXT}"`` placeholder slotted at call time. None means no typer is
    on PATH for this session.
    """
    table = {
        "wayland": [
            (["wtype", "{TEXT}"], "wtype"),
            (["ydotool", "type", "{TEXT}"], "ydotool"),
        ],
        "x11": [
            # `--` prevents text starting with `-` from being parsed as flags.
            (["xdotool", "type", "--", "{TEXT}"], "xdotool"),
        ],
        "darwin": [
            # AppleScript keystroke. Note: text containing `"` will break this
            # naive interpolation; clipboard mode is more robust on macOS.
            (
                ["osascript", "-e", 'tell application "System Events" to keystroke "{TEXT}"'],
                "osascript",
            ),
        ],
    }
    for cmd, name in table.get(session, []):
        if shutil.which(cmd[0]):
            return cmd, name
    return None


def _resolve_clipboarder(session: str) -> tuple[list[str], str] | None:
    """Pick a clipboard-write command for the session, or None.

    The returned argv is invoked with the text on stdin (no placeholder
    substitution; safer than argv interpolation for arbitrary transcript
    content).
    """
    table = {
        "wayland": [
            (["wl-copy"], "wl-copy"),
            # X11 fallback can work under XWayland on some compositors.
            (["xclip", "-selection", "clipboard"], "xclip-via-xwayland"),
        ],
        "x11": [
            (["xclip", "-selection", "clipboard"], "xclip"),
            (["xsel", "--clipboard", "--input"], "xsel"),
        ],
        "darwin": [
            (["pbcopy"], "pbcopy"),
        ],
        "termux": [
            # Termux:API addon. Writes to Android's system clipboard
            # (visible to all apps), not just to a per-app buffer.
            (["termux-clipboard-set"], "termux-clipboard-set"),
        ],
    }
    for cmd, name in table.get(session, []):
        if shutil.which(cmd[0]):
            return cmd, name
    return None


def _resolve_default_mode(session: str) -> str:
    """Auto-pick: ``type`` if a typer is on PATH; else ``clipboard``; else ``stdout``."""
    if _resolve_typer(session) is not None:
        return "type"
    if _resolve_clipboarder(session) is not None:
        return "clipboard"
    return "stdout"


# --------------------------------------------------------------------- beeps


def _beep(start: bool) -> None:
    """Audible cue. start=True → high tone; start=False → low tone.

    Best-effort: any failure (sounddevice missing, audio server down) is
    swallowed so a missing speaker doesn't break the dictate flow. The
    ``--no-beep`` flag bypasses this entirely.
    """
    try:
        import numpy as np
        import sounddevice as sd
    except Exception:
        return
    freq = 880.0 if start else 540.0
    dur = 0.10
    sr = 16000
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    tone = (np.sin(2 * np.pi * freq * t) * 0.25).astype(np.float32)
    try:
        sd.play(tone, samplerate=sr, blocking=True)
    except Exception:
        pass


# -------------------------------------------------------- capture + transcribe


async def _capture_and_transcribe(
    duration_s: float,
    language: str,
    arch: str,
    save_audio: Path | None,
) -> dict[str, Any]:
    """Record mic → transcribe via local Moonshine. v0 is local-only."""
    from aawazz_mcp.audio.capture import has_input_device

    if not has_input_device():
        return {
            "error": "no input device available",
            "hint": (
                "check sounddevice query_devices output; mic muted at OS / UEFI? "
                "headless boxes / sandboxed runtimes may not see a mic."
            ),
        }

    # Capture path — caller-provided save_audio sticks; otherwise tempfile.
    if save_audio is not None:
        save_audio.parent.mkdir(parents=True, exist_ok=True)
        capture_path = save_audio
        delete_after = False
    else:
        tmp = tempfile.NamedTemporaryFile(prefix="aawazz-dictate-", suffix=".wav", delete=False)
        tmp.close()
        capture_path = Path(tmp.name)
        delete_after = True

    # sd.wait() can wedge inside PortAudio when the device enumerates but never
    # yields samples. The audio.capture helper isolates the recording in a child
    # process so this CLI gets a real hard timeout. Same helper now wires the
    # MCP listen tool too, so dictate and listen share one fix surface.
    from aawazz_mcp.audio.capture import record_to_wav_hard_timeout

    rec = await asyncio.to_thread(
        record_to_wav_hard_timeout,
        duration_s,
        str(capture_path),
        duration_s + 5.0,
        "/tmp/aawazz-dictate.pid",
    )
    if rec.get("error") or rec.get("audio_path") is None:
        if delete_after:
            capture_path.unlink(missing_ok=True)
        return rec

    try:
        from aawazz_mcp.models.stt_loader import SttLoader

        stt = SttLoader()
        meta = await stt.transcribe(
            audio_path=str(capture_path),
            language=language,
            model_arch=arch,
        )
    except Exception as exc:  # noqa: BLE001 — surface anything as a structured error
        if delete_after:
            capture_path.unlink(missing_ok=True)
        return {
            "error": f"transcribe failed: {exc}",
            "hint": "Moonshine model load or inference error; --verbose may help",
        }

    if delete_after:
        capture_path.unlink(missing_ok=True)
    else:
        meta = dict(meta)
        meta["audio_path"] = str(capture_path)

    return meta


# ----------------------------------------------------------------- dispatch


def _dispatch(text: str, mode: str, session: str) -> tuple[bool, str]:
    """Send transcript to chosen sink. Returns (ok, dispatcher-name-or-mode)."""
    if mode == "stdout":
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
        return True, "stdout"

    if mode == "type":
        typer = _resolve_typer(session)
        if typer is None:
            return False, "type"
        cmd_template, name = typer
        cmd = [c.replace("{TEXT}", text) for c in cmd_template]
        try:
            subprocess.run(cmd, check=True)
            return True, name
        except subprocess.CalledProcessError:
            return False, name

    if mode == "clipboard":
        cb = _resolve_clipboarder(session)
        if cb is None:
            return False, "clipboard"
        cmd, name = cb
        try:
            subprocess.run(cmd, input=text, text=True, check=True)
            return True, name
        except subprocess.CalledProcessError:
            return False, name

    raise ValueError(f"unknown mode: {mode!r}")


# --------------------------------------------------------------------- main


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aawazz-dictate",
        description=(
            "Push-to-talk dictation: record mic → Moonshine → output text. "
            "Bind to a hotkey for hands-on-keyboard-but-no-typing flow."
        ),
        epilog=(
            "Examples:\n"
            "  aawazz-dictate                   # 8s capture, auto-pick output\n"
            "  aawazz-dictate -d 4 -m stdout    # 4s capture, print only (safe smoke)\n"
            "  aawazz-dictate -m clipboard      # paste into clipboard\n"
            "  aawazz-dictate -v --save-audio /tmp/note.wav  # debug + keep WAV"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "-d", "--duration", type=float, default=DEFAULT_DURATION_S,
        help=f"recording duration in seconds, clamped to [0.5, 30] (default: {DEFAULT_DURATION_S})",
    )
    p.add_argument(
        "-m", "--mode",
        choices=["type", "clipboard", "stdout", "auto"], default="auto",
        help="output sink (default: auto — type if a typer is on PATH, else clipboard, else stdout)",
    )
    p.add_argument(
        "-l", "--language", default=DEFAULT_LANGUAGE,
        help=f"STT language ISO 639-1 (default: {DEFAULT_LANGUAGE})",
    )
    p.add_argument(
        "-a", "--arch", default=DEFAULT_ARCH,
        help=(
            f"Moonshine model arch (default: {DEFAULT_ARCH}; "
            "valid: tiny, tiny_streaming, base, base_streaming, small_streaming, medium_streaming)"
        ),
    )
    p.add_argument(
        "--no-beep", action="store_true",
        help="disable start/stop audio cues",
    )
    p.add_argument(
        "--save-audio", type=Path, default=None, metavar="PATH",
        help="save captured WAV to PATH (default: discard after transcribe)",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="print mode / latency / transcript / session info to stderr",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    session = _detect_session_type()
    mode = args.mode if args.mode != "auto" else _resolve_default_mode(session)

    # Validate dispatch availability before tying up the mic.
    if mode == "type" and _resolve_typer(session) is None:
        sys.stderr.write(
            f"error: no typer available for session={session!r}.\n"
            "  install: xdotool (X11), wtype/ydotool (Wayland), or use --mode clipboard / --mode stdout\n"
        )
        return EXIT_NO_DISPATCHER
    if mode == "clipboard" and _resolve_clipboarder(session) is None:
        sys.stderr.write(
            f"error: no clipboarder available for session={session!r}.\n"
            "  install: xclip/xsel (X11), wl-copy (Wayland), pbcopy (macOS),\n"
            "    termux-clipboard-set via Termux:API (Android), or use --mode stdout\n"
        )
        return EXIT_NO_DISPATCHER

    if not args.no_beep:
        _beep(start=True)

    sys.stderr.write(f"\U0001f534 recording {args.duration:.1f}s... ")
    sys.stderr.flush()

    t0 = time.monotonic()
    # Drive the event loop manually rather than via asyncio.run(): the latter
    # calls shutdown_default_executor() at exit, which BLOCKS forever waiting
    # for the leaked sd.wait() worker thread when the mic produces no samples.
    # By skipping that cleanup, we can os._exit cleanly on the timeout path.
    # Any leaked threads are reaped by the kernel at process exit.
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            _capture_and_transcribe(
                duration_s=args.duration,
                language=args.language,
                arch=args.arch,
                save_audio=args.save_audio,
            )
        )
    finally:
        # loop.close() doesn't wait for the executor — that part is asyncio.run's
        # convenience. Closing the loop here is safe.
        loop.close()
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    if not args.no_beep:
        _beep(start=False)

    sys.stderr.write(f"done ({elapsed_ms}ms)\n")

    if result.get("error"):
        sys.stderr.write(f"error: {result['error']}\n")
        if "hint" in result:
            sys.stderr.write(f"hint:  {result['hint']}\n")
        # On capture timeout the underlying sd.wait() thread is leaked and
        # Python's threadpool shutdown at process exit will hang on it forever.
        # Bypass via os._exit — no Python cleanup but the OS reclaims everything.
        if "timed out" in result["error"].lower():
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(EXIT_NO_INPUT)
        # Distinguish "no mic" from other failures for hotkey scripts.
        if "no input device" in result["error"].lower():
            return EXIT_NO_INPUT
        return EXIT_TRANSCRIBE_FAILED

    text = (result.get("text") or "").strip()
    if not text:
        sys.stderr.write("transcript empty (no speech detected? try a longer --duration)\n")
        return EXIT_TRANSCRIBE_FAILED

    if args.verbose:
        sys.stderr.write(
            f"  transcript:       {text!r}\n"
            f"  audio_duration_s: {result.get('audio_duration_s')}\n"
            f"  stt_latency_ms:   {result.get('latency_ms')}\n"
            f"  arch:             {result.get('model_arch')}\n"
            f"  language:         {result.get('language')}\n"
            f"  session:          {session}\n"
            f"  mode:             {mode}\n"
        )
        if args.save_audio is not None:
            sys.stderr.write(f"  audio_path:       {result.get('audio_path')}\n")

    ok, used = _dispatch(text, mode, session)
    if not ok:
        sys.stderr.write(f"error: dispatch via {used!r} failed\n")
        return EXIT_DISPATCH_FAILED

    if args.verbose:
        sys.stderr.write(f"  dispatched via:   {used}\n")

    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
