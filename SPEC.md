# aawazz-mcp v1.0 — design spec

> Authoritative design doc for `aawazz-mcp` v1.0. Tracks tool surface, dispatcher policy, model lifecycle, and the design rationale behind the non-obvious choices.

## 0. What it is

`aawazz` (आवाज़ — *voice / sound* in Hindi, Urdu, and Nepali) is a portable Python MCP server providing local-CPU TTS + STT to any MCP runtime: Claude Code, Claude Desktop, Codex, Cursor, Zed, Cline, Continue, Goose, Gemini CLI.

It bundles two MIT-licensed models:
- [`tiny-tts`](https://github.com/backtracking/tiny-tts) (~3.4 MB ONNX) — text-to-speech.
- [Moonshine](https://github.com/usefulsensors/moonshine) (Useful Sensors, ~80 MB ONNX) — speech-to-text.

A single `pip install` works on a fresh machine; no FastAPI, no systemd, no API keys. An optional `--remote http://host:port` flag delegates to a separately-running `aawazz-mouth` / `aawazz-ears` FastAPI pair when one is available.

Linux + macOS only (Moonshine ships only `.so` and `.dylib`). Windows: WSL.

---

## 1. Tool surface — 4 tools + 1 resource

Tool names are bare verbs. Namespacing is the runtime's job (server name in client config); double-prefixing makes prompts ugly (`aawazz_mcp_speak`).

### 1.1 `speak`

```python
speak(
    text: str,                            # required, 1..4000 chars
    voice: str = "MALE",                  # tiny-tts: "MALE" only (see §6.2)
    speed: float = 1.0,                   # 0.5..2.0
    output_path: str | None = None,       # absolute; default $AAWAZZ_HOME/mouth/<ts>-<hash>.wav
    play: bool = False,                   # autoplay via aplay/paplay/afplay if available
) -> {
    "audio_path": str,
    "duration_s": float,
    "sample_rate": int,                   # tiny-tts emits 44100 (resampled internally from native 22050)
    "latency_ms": int,
    "voice": str,
    "speed": float,
    "text_hash": str,                     # sha1[:8] of input — caller can dedupe
    "played": bool,
    "backend": "local" | "remote",
}
```

### 1.2 `transcribe`

```python
transcribe(
    audio_path: str,                      # absolute path or http(s):// URL
    language: str = "en",
    model_arch: str = "tiny_streaming",   # tiny | tiny_streaming | base | base_streaming | small_streaming | medium_streaming
) -> {
    "text": str,
    "audio_duration_s": float,
    "sample_rate": int,
    "latency_ms": int,
    "model_arch": str,
    "language": str,
    "audio_path": str,
    "backend": "local" | "remote",
}
```

If `audio_path` is an http(s) URL, the server downloads to `${TMPDIR}/aawazz-stt-<sha8>.wav` and unlinks after.

### 1.3 `listen` — bounded mic capture → transcribe

```python
listen(
    duration_s: float = 5.0,              # 0.5..30.0 hard cap
    language: str = "en",
    model_arch: str = "tiny_streaming",
    save_audio: bool = False,
) -> {
    "text": str,
    "audio_duration_s": float,
    "sample_rate": int,
    "latency_ms": int,
    "model_arch": str,
    "language": str,
    "audio_path": str | None,             # only if save_audio=true
    "backend": "local",                   # always — mic lives on MCP server's host
}
```

`listen` is locked to local; remote-mode mic-tunneling is out of scope (no clean way to pipe mic audio over HTTP without breaking the request/response model, and no MCP runtime in scope today consumes streaming partials).

### 1.4 `voices_list`

```python
voices_list() -> {
    "tts": {
        "backend": "tiny-tts",
        "voices": [{"id": "MALE", "language": "en", "default": True}],
    },
    "stt": {
        "backend": "moonshine",
        "languages": ["en", ...],
        "model_archs": ["tiny", "tiny_streaming", "base", "base_streaming",
                        "small_streaming", "medium_streaming"],
    },
    "capabilities": {
        "listen": bool,                   # sounddevice + default input device present
        "play": bool,                     # aplay/paplay/afplay on PATH
        "backend_mode": "local" | "remote",
        "remote_url": {"mouth": str | None, "ears": str | None},
    },
}
```

`voices_list` does NOT trigger model load — metadata only. Use as a cheap probe.

### 1.5 Resource: `aawazz://health`

```json
{
  "version": "1.0.0",
  "mode": "local" | "remote",
  "remote_url": {"mouth": str | null, "ears": str | null},
  "models_loaded": {"tts": bool, "stt_archs": [...]},
  "capabilities": {"listen": bool, "play": bool}
}
```

### Tools considered and cut

- `tts_voices` / `stt_languages` separate — folded into `voices_list`.
- `stream_speak` / `stream_listen` — defer (no shipping runtime in scope consumes streaming).
- `save_voice` / `clone_voice` — out of scope (tiny-tts can't clone).

---

## 2. Directory layout

```
aawazz-mcp/
├── README.md
├── SPEC.md                              # this file
├── LICENSE                              # MIT
├── pyproject.toml
├── .gitignore
├── .python-version                      # 3.10
├── assets/                              # logo + branding
├── examples/clients/                    # 9 runtime config snippets
├── src/aawazz_mcp/
│   ├── __init__.py                      # __version__
│   ├── __main__.py                      # python -m aawazz_mcp
│   ├── cli.py                           # argparse: --remote, --transport, --port, --warm, --log-level
│   ├── server.py                        # FastMCP, 4 tools, lifespan hook
│   ├── config.py                        # AawazzConfig dataclass
│   ├── dispatcher.py                    # local|remote selection, fail-loud
│   ├── resources.py                     # aawazz://health spec
│   ├── backends/
│   │   ├── base.py                      # Backend ABC
│   │   ├── local.py                     # bundled tiny-tts + Moonshine + sounddevice
│   │   └── remote.py                    # httpx → /tts, /transcribe
│   ├── models/
│   │   ├── tts_loader.py                # lazy TinyTTS singleton, NLTK preload, stdout-redirect
│   │   └── stt_loader.py                # lazy Transcriber, keyed (lang, arch)
│   └── audio/
│       ├── capture.py                   # sounddevice → bounded WAV
│       ├── playback.py                  # aplay/paplay/afplay shellout
│       └── paths.py                     # default_output_dir, hash naming
├── tests/                               # pytest; markers: slow, mic, remote
└── scripts/
    └── prefetch_models.py               # eager warm for offline boxes
```

`src/`-layout (not flat) so editable installs don't shadow tests.

---

## 3. MCP SDK choice — FastMCP via `mcp.server.fastmcp`

**Use `from mcp.server.fastmcp import FastMCP` from the official `mcp` Python SDK.** Pin `mcp >= 1.24, < 2.0`.

**Not Prefect's `fastmcp` v3.x** — heavier dep tree (`authlib`, `cyclopts`, `griffelib`, `jsonschema-path`, `opentelemetry-api`, `openapi-pydantic`); features we don't need (OAuth, OpenAPI, install-CLI). Stock FastMCP has the same ergonomics with a smaller surface. Switching is a one-import change if `fastmcp install <runtime>` autoinstall sugar matters in v1.1.

Tool **docstrings become MCP tool descriptions verbatim** — review the prose like code.

Default transport: `stdio` (`mcp.run(transport="stdio")`). Opt-in `--transport streamable-http --host 127.0.0.1 --port 7860` for multi-client multiplexing. Note FastMCP 1.24+ reads host/port from `mcp.settings` rather than `run()` kwargs.

---

## 4. Hybrid mode dispatcher

Resolution priority at startup:

1. CLI flag: `--remote http://host:port` (joint mouth+ears base; per-service env overrides).
2. Env vars: `AAWAZZ_REMOTE_URL` (joint), `AAWAZZ_MOUTH_URL` / `AAWAZZ_EARS_URL` (per-service).
3. Default: `local` (bundled).

### Per-tool routing

`speak` and `transcribe` honor the resolved mode. `listen` is locked to local — the mic is on the host running this MCP server, not on the host of any remote FastAPI service.

```python
class Dispatcher:
    async def speak(self, *args, **kwargs):
        return await self._pick("mouth").speak(*args, **kwargs)

    async def listen(self, *args, **kwargs):
        # listen ALWAYS local — mic lives on this host
        if self._local is None:
            self._local = LocalBackend(self.cfg)
        return await self._local.listen(*args, **kwargs)
```

### Failure policy: **fail loud, no silent fallback**

When `--remote` (or `AAWAZZ_*_URL`) is set but the FastAPI server is unreachable, the tool returns:

```json
{
  "error": "remote aawazz-mouth at http://127.0.0.1:7861/tts unreachable: connection refused",
  "hint": "is aawazz-mouth running? Check the service. Or unset AAWAZZ_MOUTH_URL.",
  "backend": "remote",
  "url": "http://127.0.0.1:7861/tts"
}
```

Silent fallback would mask misconfiguration; the user explicitly opted into remote. v1.1 may add `--remote-fallback=local` if there's demand.

Hang prevention: every remote call uses `httpx.AsyncClient(timeout=httpx.Timeout(connect=2.0, read=120.0))`.

---

## 5. Model lifecycle — lazy first-call + opt-in `--warm`

Eager warm is the wrong default for an MCP server (subprocess spawned per session; some runtimes' `list_tools` calls time out at 5–10 s). Lazy first-call has ~1–3 s one-time cost, observable via `latency_ms`.

- `--warm` CLI flag (or `AAWAZZ_WARM=1`): triggers `dispatcher.warm()` in FastMCP `lifespan` async context manager before stdio loop starts.
- `voices_list` does NOT load models — metadata only.
- `aawazz://health` reports `models_loaded: {tts: bool, stt_archs: [...]}`.

### Weight locations (reuse upstream caches; no `~/.cache/aawazz/` invented)

| Asset | Default location | Override env | First-run? |
|---|---|---|---|
| tiny-tts G.pth | `~/.cache/huggingface/hub/models--backtracking--tiny-tts/` | `HF_HOME` | yes |
| Moonshine ONNX | `~/.cache/moonshine_voice/` | `MOONSHINE_VOICE_CACHE` | yes |
| NLTK (`g2p_en`) | `~/nltk_data/` | `NLTK_DATA` | yes |

`AAWAZZ_HOME` (default `~/.local/share/aawazz/`) is for runtime artifacts only — output WAVs.

`scripts/prefetch_models.py` runs the loaders once with stderr progress for offline boxes:

```bash
python -m aawazz_mcp.scripts.prefetch_models
```

---

## 6. Risks and gotchas

1. **`tiny_tts.TinyTTS.speak()` does `print()` on stdout.** Corrupts FastMCP stdio transport. The `stdout_to_stderr` context manager in `models/tts_loader.py` wraps every call site. Not optional — without the wrap, stdio runtimes hang on the first `speak()`.

2. **Tiny-tts has only ONE voice** (`SPK2ID = {"MALE": 0}`). We reject anything else with a structured error including `available_voices: ["MALE"]`. v1.1 may add Moonshine TTS as an alternative backend with a real voice catalog.

3. **`listen` mic access in MCP runtimes.** Many runtimes spawn the server in a sandboxed subprocess. PulseAudio/PipeWire socket inheritance "just works" for desktop sessions; breaks for headless / SSH / Codespaces. `voices_list().capabilities.listen` returns false when `sounddevice.query_devices(kind="input")` raises or returns empty. `listen` itself returns a clean error rather than crashing.

4. **ONNX runtime cross-platform.** `moonshine_voice` ships only `libmoonshine.so` + `libonnxruntime.*.dylib` — Linux + macOS only. Windows: clean `RuntimeError` with WSL hint at server start.

5. **`tiny-tts` pulls torch (~600 MB).** Document in README. v1.1 may carve a `[lite]` extra.

6. **First-run network requirement.** No way around it — document `prefetch_models.py`.

7. **No PyPI in v1.0.** Install via `pip install git+https://github.com/nixpt/aawazz-mcp.git`. PyPI deferred to v1.1.

8. **Stdio binary safety.** Default `logging.basicConfig(stream=sys.stderr)`. Lint rule banning `print()` in source.

9. **Concurrent calls to same model.** `TinyTTS` and `moonshine_voice.Transcriber` are not thread-safe and FastMCP serves tools concurrently. `asyncio.Lock` per loader.

10. **Sandboxed `~/.local/share/aawazz/` writability.** `paths.default_output_dir()` falls back to `tempfile.gettempdir()` if the default isn't writable. `audio_path` returned is always absolute.

11. **NLTK download flakiness.** Wrap `nltk.download(quiet=True)` with try/except; don't let captive-portal or cert errors crash the server.

12. **Sample rate is 44100.** tiny-tts upsamples internally from its native 22050. Documented for callers comparing against alternative TTS engines that expose the original 22050.

---

## 7. Wire compatibility

The remote backend's HTTP wire format matches a pre-existing `aawazz-mouth` + `aawazz-ears` FastAPI pair so users already running those services can swap freely:

**TTS** — `POST <mouth_url>/tts` body `{"text": str, "voice": str, "speed": float}` → `{"audio_path", "duration_s", "sample_rate", "latency_ms", "voice", "speed", "text_hash"}`.

**STT** — `POST <ears_url>/transcribe` body `{"audio_path": str, "language": str, "model_arch": str}` → `{"text", "audio_duration_s", "sample_rate", "latency_ms", "model_arch", "language", "audio_path"}`.

URLs without a path component get `/tts` or `/transcribe` appended automatically.
