# aawazz-mcp v1.0 — SPEC

> Authoritative design doc. Horse units (Wave 1A/1B/1C/1D + Wave 2/3) code against this contract.

## 0. Context

`aawazz` (आवाज़, "voice / sound") is the captain's voice/sound subsystem. Session 144 shipped two FastAPI services running as systemd-user units:

- `aawazz-mouth` — TTS, `tiny-tts`, port 7861.
- `aawazz-ears` — STT, `moonshine-voice tiny_streaming_en`, port 7862.

`aawazz-mcp` v1.0 is the **portable, distribution-friendly** form: one pip-installable Python MCP server that any MCP runtime (Claude Code, Codex, Cursor, Zed, Cline, Continue, Goose) can spawn over stdio.

It bundles `tiny-tts` + `moonshine-voice` so a fresh-machine `pip install` works without FastAPI / systemd. An optional `--remote http://host:port` flag delegates to the existing FastAPI servers when running on a captain's machine that already has them up.

Repo: `github.com/nixpt/aawazz-mcp`. PyPI publish deferred (v1.1).

---

## 1. Tool surface — 4 tools + 1 resource

Tool names are bare verbs. Namespacing is the runtime's job (server name in client config); double-prefixing makes prompts ugly (`aawazz_mcp_speak`).

### 1.1 `speak`

```python
speak(
    text: str,                            # required, 1..4000 chars
    voice: str = "MALE",                  # tiny-tts: "MALE" only (see §6.1)
    speed: float = 1.0,                   # 0.5..2.0
    output_path: str | None = None,       # absolute; default ~/.local/share/aawazz/mouth/<ts>-<hash>.wav
    play: bool = False,                   # autoplay via aplay/paplay/afplay if available
) -> {
    "audio_path": str,
    "duration_s": float,
    "sample_rate": int,                   # tiny-tts emits 44100 (verified live; resampled from native 22050 internally)
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

If `audio_path` is an http(s) URL, the server downloads to `${TMPDIR}/aawazz-stt-<sha8>.wav` and unlinks after. (Mirrors the existing Rust arm.)

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

`listen` is locked to local; remote-mode mic-tunneling is out of scope (no clean way to pipe mic audio over HTTP without breaking the request/response model, and no MCP runtime in scope consumes streaming partials).

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

Returns JSON: `{models_loaded: {tts: bool, stt_archs: [...]}, mode: "local"|"remote", remote_url: {...}, version: "1.0.0"}`.

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
├── examples/
│   └── clients/                         # copy-paste configs, one per runtime
│       ├── claude_code.json
│       ├── claude_desktop.json
│       ├── codex.toml
│       ├── cursor.json
│       ├── zed.json
│       ├── cline.json
│       ├── continue.yaml
│       └── goose.yaml
├── src/aawazz_mcp/
│   ├── __init__.py                      # __version__
│   ├── __main__.py                      # python -m aawazz_mcp
│   ├── cli.py                           # argparse: --remote, --transport, --port, --warm, --log-level
│   ├── server.py                        # FastMCP, 4 tools, lifespan hook
│   ├── config.py                        # AawazzConfig dataclass
│   ├── dispatcher.py                    # local|remote selection, fail-loud
│   ├── resources.py                     # aawazz://health
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py                      # Backend ABC
│   │   ├── local.py                     # bundled tiny-tts + moonshine + sounddevice
│   │   └── remote.py                    # httpx → 7861/tts, 7862/transcribe
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tts_loader.py                # lazy TinyTTS singleton, NLTK preload, stdout-redirect
│   │   └── stt_loader.py                # lazy Transcriber, keyed (lang, arch)
│   └── audio/
│       ├── __init__.py
│       ├── capture.py                   # sounddevice → bounded WAV
│       ├── playback.py                  # aplay/paplay/afplay shellout
│       └── paths.py                     # default_output_dir, hash naming
├── tests/
│   ├── __init__.py
│   ├── conftest.py                      # tmp_path fixture, env isolation
│   ├── test_smoke_local.py              # round-trip speak → transcribe
│   ├── test_smoke_remote.py             # @pytest.mark.remote
│   ├── test_dispatcher.py
│   ├── test_voices_list.py
│   ├── test_listen.py                   # @pytest.mark.mic
│   └── test_paths.py
└── scripts/
    └── prefetch_models.py               # eager warm for offline boxes
```

`src/`-layout (not flat) so editable installs don't shadow tests; matches FastMCP examples.

---

## 3. MCP SDK choice — FastMCP via `mcp.server.fastmcp`

**Use `from mcp.server.fastmcp import FastMCP` from the official `mcp` Python SDK.** Pin `mcp >= 1.24, < 2.0`.

**Not Prefect's `fastmcp` v3.x** — heavier dep tree (`authlib`, `cyclopts`, `griffelib`, `jsonschema-path`, `opentelemetry-api`, `openapi-pydantic`); features we don't need (OAuth, OpenAPI, install-CLI). Stock FastMCP is the same ergonomics with a smaller surface. v1.1 can switch if we want `fastmcp install <runtime>` autoinstall sugar.

### Wire shape

```python
# src/aawazz_mcp/server.py
from mcp.server.fastmcp import FastMCP
from aawazz_mcp.config import AawazzConfig
from aawazz_mcp.dispatcher import Dispatcher

def build_server(cfg: AawazzConfig) -> FastMCP:
    mcp = FastMCP("aawazz", instructions=INSTRUCTIONS_MD)
    dispatch = Dispatcher(cfg)

    @mcp.tool()
    async def speak(...): ...

    @mcp.tool()
    async def transcribe(...): ...

    @mcp.tool()
    async def listen(...): ...

    @mcp.tool()
    async def voices_list(): ...

    @mcp.resource("aawazz://health")
    async def health() -> str: ...

    return mcp
```

Tool **docstrings become MCP tool descriptions verbatim** — review the prose like code.

Default transport: `stdio` (`mcp.run(transport="stdio")`). Opt-in `--transport streamable-http --host 127.0.0.1 --port 7860` for multi-client multiplexing.

---

## 4. Hybrid mode dispatcher

Resolution priority at startup:

1. CLI flag: `--remote http://host:port` (joint mouth+ears base; per-service env overrides).
2. Env vars: `AAWAZZ_REMOTE_URL` (joint), `AAWAZZ_MOUTH_URL` / `AAWAZZ_EARS_URL` (per-service — matches the existing Rust arm verbatim).
3. Default: `local` (bundled).

### Per-tool routing

```python
class Dispatcher:
    def __init__(self, cfg):
        self.cfg = cfg
        self._local: LocalBackend | None = None
        self._remote: RemoteBackend | None = None
        if cfg.mode == "remote":
            self._remote = RemoteBackend(cfg.remote_mouth_url, cfg.remote_ears_url)
        elif cfg.mode == "local":
            self._local = LocalBackend(cfg)

    async def speak(self, *args, **kwargs):
        return await self._pick("mouth").speak(*args, **kwargs)

    async def listen(self, *args, **kwargs):
        # listen ALWAYS local — mic is on this host
        if self._local is None:
            self._local = LocalBackend(self.cfg)
        return await self._local.listen(*args, **kwargs)
```

### Failure policy: **fail loud, no silent fallback**

When `--remote` (or `AAWAZZ_*_URL`) is set but the FastAPI server is unreachable, the tool returns:

```json
{
  "error": "remote aawazz-mouth at http://127.0.0.1:7861/tts unreachable: connection refused",
  "hint": "is aawazz-mouth running? `systemctl --user status aawazz-mouth`. Or pass --no-remote.",
  "backend": "remote",
  "url": "http://127.0.0.1:7861/tts"
}
```

Silent fallback would mask misconfig; captain explicitly opted into remote. v1.1 may add `--remote-fallback=local` if there's demand.

Hang prevention: every remote call uses `httpx.AsyncClient(timeout=httpx.Timeout(connect=2.0, read=120.0))`.

---

## 5. Model lifecycle — lazy first-call + opt-in `--warm`

Eager warm is the wrong default for an MCP server (subprocess spawned per session; runtime list_tools timeouts at 5-10s for some clients). Lazy first-call has ~1-3s one-time cost, observable via `latency_ms`.

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

1. **`tiny_tts.TinyTTS.speak()` does `print(f"Synthesizing: ...")` on stdout.** Corrupts FastMCP stdio transport. **MUST** wrap in `contextlib.redirect_stdout(sys.stderr)` in `models/tts_loader.py`. Not optional.

2. **Tiny-tts has only ONE voice** (`SPK2ID = {"MALE": 0}`). Reject anything else with structured error including `available_voices: ["MALE"]`. The s144 server silently downgrades — we don't.

3. **`listen` mic access in MCP runtimes.** Many MCP runtimes spawn the server in a sandboxed subprocess. PulseAudio/PipeWire socket inheritance "just works" for desktop sessions; breaks for headless / SSH / Codespaces. `voices_list().capabilities.listen` should return false when `sounddevice.query_devices(kind="input")` raises or returns empty. `listen` itself returns a clean error rather than crashing.

4. **ONNX runtime cross-platform.** `moonshine_voice` ships only `libmoonshine.so` + `libonnxruntime.*.dylib` — Linux + macOS only. Windows: clean `RuntimeError` with WSL hint at server start.

5. **`tiny-tts` pulls torch (~600MB).** Document weight in README.

6. **First-run network requirement.** No way around it — document `prefetch_models.py`.

7. **No PyPI in v1.0.** Install via `pip install git+https://github.com/nixpt/aawazz-mcp.git`. Document precisely.

8. **Stdio binary safety.** Default `logging.basicConfig(stream=sys.stderr)`. Lint rule banning `print()` in source.

9. **Concurrent calls to same model.** `TinyTTS` / `Transcriber` aren't necessarily thread-safe and FastMCP serves tools concurrently. `asyncio.Lock` per backend instance.

10. **Tool-name collision with joker-mcp.** None — `joker-mcp` exposes `joker_text_to_speech` / `joker_transcribe_audio`; aawazz-mcp's `speak` / `transcribe` are flatter and more discoverable. Namespacing per-server.

11. **Sandboxed `~/.local/share/aawazz/` writability.** `paths.default_output_dir()` falls back to `tempfile.gettempdir()` if the default isn't writable. `audio_path` returned is always absolute.

12. **NLTK download flakiness.** Wrap `nltk.download(quiet=True)` in 15s timeout + try/except.

---

## 7. Wave plan

### Wave 0 — scaffold (foreman, sequential prereq, ~10 min)
Create dir, `pyproject.toml`, `LICENSE`, `.gitignore`, `.python-version`, `README.md` skeleton, `SPEC.md`, all `src/aawazz_mcp/` shells with typed signatures + `NotImplementedError`, `tests/conftest.py`. Init git, `gh repo create nixpt/aawazz-mcp --public --source .`, push. Defines the contract every other unit codes against.

### Wave 1A — local backend (sonnet, ~10 turns, parallel)
`backends/local.py`, `models/tts_loader.py`, `models/stt_loader.py`. Lifts the `_load_model` pattern from `~/.local/aawazz/{mouth,ears}/server.py`, wraps with `asyncio.Lock` + stdout-redirect contextmanager. Implements `speak()` and `transcribe()` against bundled `tiny_tts` / `moonshine_voice`.

### Wave 1B — remote backend + dispatcher (codex, ~6-8 turns, parallel)
`backends/remote.py` (httpx async → 7861/7862, identical wire format to FastAPI), `dispatcher.py` (env+CLI resolution, fail-loud, per-tool routing), `config.py` (full).

### Wave 1C — audio I/O (kimi, ~5-6 turns, parallel)
`audio/capture.py` (sounddevice bounded record → WAV), `audio/playback.py` (subprocess shellout: `paplay` → `aplay` → `afplay`, matches existing joker-mcp `maybe_autoplay`).

### Wave 1D — README + 7 client configs (opencode, ~4-6 turns, parallel)
Polished README replacing the Wave 0 skeleton, all 8 files in `examples/clients/` with verified syntax.

### Wave 2 — server wiring + 4 tools (sonnet, ~6-8 turns, sequential after 1A+1B+1C)
Replace stub `server.py` with real FastMCP tool registrations. End-to-end `speak`, `transcribe`, `listen` (1C capture → 1A STT loader), `voices_list`. `aawazz://health` resource. Lifespan hook for `--warm`.

### Wave 3 — smoke + tag (foreman, ~3-4 turns, sequential closer)
Round-trip smoke (bundle + `--remote`), polish, `git tag v1.0.0`, `git push --tags`.

### Sequencing

```
Wave 0 (scaffold + repo)
   ↓
   ├──→ Wave 1A (local)
   ├──→ Wave 1B (remote + dispatcher)
   ├──→ Wave 1C (audio I/O)
   └──→ Wave 1D (docs + configs)
       ↓ (1A+1B+1C must land for Wave 2; 1D can land late)
   Wave 2 (server wiring)
       ↓
   Wave 3 (smoke + tag v1.0.0)
```

Critical path: Wave 0 → 2 → 3. 1D is independent of Wave 2.

---

## 8. Reference files (for horse units)

Already on captain's disk. **Consult before implementing.**

- `~/.local/aawazz/mouth/server.py` — TTS contract, `tts.speak()` invocation, NLTK preload, output-WAV naming convention.
- `~/.local/aawazz/ears/server.py` — STT contract, `_load_model(language, arch)` keying, `Transcriber.transcribe_without_streaming` clean-text extraction from `transcript.lines`.
- `~/WORKSPACE/projects/exosphere/crates/ai/services/joker-mcp/src/modalities.rs` (~lines 361-440 STT, ~625-705 TTS) — exact wire format used by the existing Rust arm, env-var names `AAWAZZ_MOUTH_URL` / `AAWAZZ_EARS_URL`, error-message hint format for unreachable services.
- `~/.config/systemd/user/aawazz-mouth.service` — env defaults: `AAWAZZ_MOUTH_PORT=7861`, `AAWAZZ_MOUTH_OUT=~/.local/share/aawazz/mouth`, `HF_HOME=~/.cache/huggingface`.
- `/build/venv-aawazz/lib/python3.12/site-packages/tiny_tts/__init__.py` — confirms `SPK2ID = {"MALE": 0}` (only voice), `print()` polluting stdout (must redirect).
- `/build/venv-aawazz/lib/python3.12/site-packages/moonshine_voice/__init__.py` — `Transcriber`, `get_model_for_language`, `load_wav_file`, `MicTranscriber`, voice catalog helpers.
