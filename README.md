<p align="center">
  <img src="assets/banner.png" alt="aawazz-mcp banner" width="100%">
</p>

# aawazz-mcp

> **आवाज़** — Hindi/Urdu/Nepali for *voice / sound*.

A portable, local-CPU **TTS + STT MCP server** for any agent runtime that speaks the Model Context Protocol — Claude Code, Claude Desktop, Codex, Cursor, Zed, Cline, Continue, Goose, Gemini CLI. One `pip install` and four tools (`speak`, `transcribe`, `listen`, `voices_list`) light up across every runtime simultaneously. Bundles [tiny-tts](https://github.com/backtracking/tiny-tts) (~3.4 MB ONNX) and [Useful Sensors / Moonshine](https://github.com/usefulsensors/moonshine) (~80 MB ONNX) so it runs offline once weights are cached. **v1.3 adds a pluggable backend layer** so you can swap in [Piper](https://github.com/rhasspy/piper), [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), [Coqui XTTS-v2](https://huggingface.co/coqui/XTTS-v2), and post-processors (DSP profiles, VAD, gain) without forking. An optional `--remote` mode delegates to a separately-running FastAPI mouth/ears so model load doesn't double up on machines that already have those services.

---

## Status & support matrix

| | |
|---|---|
| **Version** | v1.3 (pluggable backends; default routing matches v1.2) |
| **OS** | Linux, macOS — Moonshine ships only `.so` and `.dylib` |
| **Python** | 3.10–3.14 |
| **Distribution** | `pip install git+...` (PyPI release queued) |
| **Transport** | stdio default; opt-in `streamable-http` |

**Windows users**: install [WSL2](https://learn.microsoft.com/windows/wsl/install) and follow the Linux instructions inside it. Native Windows support is gated on Moonshine shipping `.dll` artifacts — see [usefulsensors/moonshine#TBD](https://github.com/usefulsensors/moonshine).

---

## Install

```bash
pip install git+https://github.com/nixpt/aawazz-mcp.git
```

That's it. The `aawazz-mcp` console script lands on your `PATH`. PyPI publication is queued; until then the git URL is the canonical install path.

---

## Quickstart

1. **Verify the install.** The CLI prints its flags:

   ```bash
   aawazz-mcp --help
   ```

2. **Wire it into one runtime.** Claude Code is the canonical example; drop this into `~/.claude.json` and restart Claude Code:

   ```json
   {
     "mcpServers": {
       "aawazz": {
         "command": "aawazz-mcp",
         "args": []
       }
     }
   }
   ```

3. **(Optional) Pre-warm models.** First-run downloads tiny-tts, Moonshine, and a couple of NLTK datasets totalling ~130 MB plus a torch dep weighing ~600 MB on disk. On offline / air-gapped boxes, run the prefetch script while you still have a network:

   ```bash
   python -m aawazz_mcp.scripts.prefetch_models
   ```

   Models live in standard upstream caches (`~/.cache/huggingface/hub/`, `~/.cache/moonshine_voice/`, `~/nltk_data/`) — no `~/.cache/aawazz/` directory is invented.

Once Claude Code reloads, ask the model to *"say hello using aawazz"* and you should get a `.wav` back.

---

## Multi-runtime config grid

Every runtime below talks to `aawazz-mcp` over stdio. The bare-minimum invocation is identical (`command: aawazz-mcp`, no args); the differences live in **where the config file lives** and **how the runtime picks up changes**. Copy-paste blocks below match exactly the files in [`examples/clients/`](examples/clients/) — pull from there if your editor mangles JSON.

The optional `env:` table on each block enables [hybrid mode](#hybrid-mode-advanced) — uncomment if you have an `aawazz-mouth` / `aawazz-ears` FastAPI pair already running on this host.

### Claude Code

| | |
|---|---|
| **Config file** | `~/.claude.json` (user-level) or `.mcp.json` (per-project, repo root) |
| **Restart** | Manual — quit & relaunch the Claude Code CLI |
| **Example** | [`examples/clients/claude_code.json`](examples/clients/claude_code.json) |

```json
{
  "mcpServers": {
    "aawazz": {
      "command": "aawazz-mcp",
      "args": []
      // "env": {
      //   "AAWAZZ_MOUTH_URL": "http://127.0.0.1:7861/tts",
      //   "AAWAZZ_EARS_URL": "http://127.0.0.1:7862/transcribe"
      // }
    }
  }
}
```

**Gotchas**
- `~/.claude.json` is shared with other Claude Code state — merge into the existing `mcpServers` table, don't overwrite the file.
- Per-project `.mcp.json` overrides user-level for the cwd; useful for pinning hybrid-mode env to a workspace.

### Claude Desktop

| | |
|---|---|
| **Config file** | Linux: `~/.config/Claude/claude_desktop_config.json` · macOS: `~/Library/Application Support/Claude/claude_desktop_config.json` |
| **Restart** | Manual — fully quit Claude Desktop and relaunch (menu bar quit on macOS, not just the window close) |
| **Example** | [`examples/clients/claude_desktop.json`](examples/clients/claude_desktop.json) |

```json
{
  "mcpServers": {
    "aawazz": {
      "command": "aawazz-mcp",
      "args": []
    }
  }
}
```

**Gotchas**
- If `aawazz-mcp` isn't on the GUI app's `PATH` (common on macOS), use the absolute path: `which aawazz-mcp` from your shell, then paste the result into `command`.
- Claude Desktop on macOS sandboxes the subprocess; the bundled mic-capture path (`listen`) may not have access. Check `voices_list().capabilities.listen`.

### Codex (OpenAI Codex CLI)

| | |
|---|---|
| **Config file** | `~/.codex/config.toml` |
| **Restart** | None — Codex auto-detects on the next `codex` invocation |
| **Example** | [`examples/clients/codex.toml`](examples/clients/codex.toml) |
| **Sandbox** | bwrap (audio-blocking) — the example defaults to **hybrid mode**; see [Sandboxed runners](#sandboxed-runners) |

```toml
[mcp_servers.aawazz]
command = "aawazz-mcp"
args = []

# Default-on for sandboxed Codex — routes through host FastAPI so audio
# actually plays. Comment out if Codex is running outside its sandbox.
[mcp_servers.aawazz.env]
AAWAZZ_MOUTH_URL = "http://127.0.0.1:7861/tts"
AAWAZZ_EARS_URL = "http://127.0.0.1:7862/transcribe"
```

**Gotchas**
- TOML — table form `[mcp_servers.<name>]`, not the JSON `mcpServers: { ... }` shape used elsewhere.
- `args = []` is required even when empty; Codex won't infer it.

### Cursor

| | |
|---|---|
| **Config file** | `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (per-project) |
| **Restart** | Auto-reload on save — Cursor watches the file |
| **Example** | [`examples/clients/cursor.json`](examples/clients/cursor.json) |

```json
{
  "mcpServers": {
    "aawazz": {
      "command": "aawazz-mcp",
      "args": []
    }
  }
}
```

**Gotchas**
- Cursor supports `${env:VAR}` and `${workspaceFolder}` interpolation in command/args/env values — useful for per-machine paths.
- On some setups the GUI launcher's `PATH` doesn't include user-installed scripts; if `aawazz-mcp` isn't found, swap to the absolute path from `which aawazz-mcp`.

### Zed

| | |
|---|---|
| **Config file** | `~/.config/zed/settings.json` |
| **Restart** | Auto-applied — Zed reloads MCP servers on settings save |
| **Example** | [`examples/clients/zed.json`](examples/clients/zed.json) |

```json
{
  "context_servers": {
    "aawazz": {
      "command": "aawazz-mcp",
      "args": []
    }
  }
}
```

**Gotchas**
- The top-level key is `context_servers`, **not** `mcpServers` — Zed uses its own naming. The inner shape (command / args / env) matches.
- Zed's settings are merge-friendly; you can drop `context_servers` next to your existing `theme`/`buffer_font_size`/etc. without losing them.

### Cline (VS Code)

| | |
|---|---|
| **Config file** | UI-driven — Cline panel → MCP Servers → Add. Persisted in VS Code's MCP storage. |
| **Restart** | Click the per-server **Restart Server** button in the Cline UI |
| **Example** | [`examples/clients/cline.json`](examples/clients/cline.json) |

```json
{
  "command": "aawazz-mcp",
  "args": [],
  "alwaysAllow": [],
  "disabled": false
}
```

**Gotchas**
- Cline is primarily UI-driven; the JSON above is what the dialog produces under the hood. Easiest path is to add via the panel rather than hand-editing.
- `alwaysAllow` is a Cline-specific allowlist of tool names the agent can call without per-call approval — leaving it `[]` keeps the human-in-the-loop prompt.

### Continue

| | |
|---|---|
| **Config file** | `~/.continue/config.yaml` (user-level) or `.continue/config.yaml` (per-project) |
| **Restart** | Hot-reload on save |
| **Example** | [`examples/clients/continue.yaml`](examples/clients/continue.yaml) |

```yaml
modelContextProtocolServers:
  - name: aawazz
    type: stdio
    command: aawazz-mcp
    # env:
    #   AAWAZZ_MOUTH_URL: http://127.0.0.1:7861/tts
    #   AAWAZZ_EARS_URL: http://127.0.0.1:7862/transcribe
```

**Gotchas**
- YAML, not JSON — top-level key is `modelContextProtocolServers` (a list of mappings), not `mcpServers`.
- Continue supports `${{ secrets.X }}` interpolation; if you already keep your remote URLs in the secrets store, prefer that over inline values.

### Goose

| | |
|---|---|
| **Config file** | Linux/macOS: `~/.config/goose/config.yaml` · Windows-WSL: `%APPDATA%\Block\goose\config\config.yaml` |
| **Restart** | Manual — restart the `goose` CLI after editing |
| **Example** | [`examples/clients/goose.yaml`](examples/clients/goose.yaml) |

```yaml
extensions:
  aawazz:
    cmd: aawazz-mcp
    args: []
    enabled: true
    type: stdio
    timeout: 300
    # envs:
    #   AAWAZZ_MOUTH_URL: http://127.0.0.1:7861/tts
    #   AAWAZZ_EARS_URL: http://127.0.0.1:7862/transcribe
```

**Gotchas**
- Goose uses `cmd` (not `command`) and `envs` (not `env`) — copy-paste from another runtime's block won't work, use the exact keys above.
- `timeout` is in seconds and applies to tool calls — bump it (e.g. `timeout: 600`) if first-run model download bites you.
- `enabled: false` keeps the extension registered but inactive — handy when toggling between hybrid and bundled mode without deleting the block.

### Gemini CLI

| | |
|---|---|
| **Config file** | `~/.gemini/settings.json` (user-level) or `./.gemini/settings.json` (per-project) |
| **Restart** | Auto-detected on next `gemini` invocation |
| **Example** | [`examples/clients/gemini_cli.json`](examples/clients/gemini_cli.json) |

```json
{
  "mcpServers": {
    "aawazz": {
      "command": "aawazz-mcp",
      "args": []
    }
  }
}
```

**Gotchas**
- Same `mcpServers` shape as Claude Code — copy-paste interchangeable.
- Gemini CLI's folder-trust prompt may block stdio spawn on first run; accept the trust prompt or pass `--skip-trust` (see [`agent-launch`](https://github.com/nixpt/squadron) for an example wrapper).

---

## Hybrid mode (advanced)

If you have `aawazz-mouth` and `aawazz-ears` running as separate FastAPI services on this host (typical setup: systemd-user units on `:7861` and `:7862`), you don't want this MCP server to load its own copies of tiny-tts and Moonshine on every runtime spawn. Hybrid mode delegates to those FastAPI services instead.

**Why bother**
- Existing FastAPI services stay useful — other clients on the same host can keep hitting them in parallel.
- Model load happens once, in the long-running service — not per MCP-runtime subprocess.
- ~600 MB of torch state is held in one place, not duplicated across every Claude Code / Cursor / Zed window.

**How**

CLI flag form (joint base for both services, overrideable per-service via env):

```bash
aawazz-mcp --remote http://127.0.0.1:7861,http://127.0.0.1:7862
```

Per-service env-var form (independent overrides for mouth and ears):

```bash
AAWAZZ_MOUTH_URL=http://127.0.0.1:7861/tts \
AAWAZZ_EARS_URL=http://127.0.0.1:7862/transcribe \
  aawazz-mcp
```

In an MCP-runtime config, drop the env table into the server entry — see the commented block in any of the [`examples/clients/*`](examples/clients/) files.

**`listen` is always local.** The mic lives on the MCP server's host; tunneling raw audio over the FastAPI request/response cycle isn't a clean fit and no shipping runtime in scope consumes streaming partials. Use `transcribe` against a pre-recorded WAV if you need remote STT.

**Failure mode is fail-loud.** If `--remote` is set and the FastAPI server is unreachable, the tool returns a structured error instead of silently falling back to local:

```json
{
  "error": "remote aawazz-mouth at http://127.0.0.1:7861/tts unreachable: connection refused",
  "hint": "is aawazz-mouth running? `systemctl --user status aawazz-mouth`. Or pass --no-remote.",
  "backend": "remote",
  "url": "http://127.0.0.1:7861/tts"
}
```

Silent fallback would mask misconfig; you explicitly opted into remote. A future `--remote-fallback=local` flag is on the table if there's demand.

---

## Sandboxed runners

Some MCP-aware runtimes execute their server subprocesses inside a sandbox (bwrap, container, app sandbox) that blocks access to host audio devices. The classic symptoms:

- `voices_list().capabilities.play == true` — the `paplay` / `aplay` binary IS on PATH.
- but `speak(play=true).played == false` — the actual PortAudio / ALSA call is denied.
- `listen()` returns immediately with empty text — mic enumeration sees no device.

This is the **expected** behavior. aawazz-mcp surfaces the sandbox boundary cleanly rather than silently dropping audio; the sandbox is doing its job.

**Fix: route through host-side FastAPI mouth/ears.** Set the env vars on the MCP server block so the sandboxed agent makes HTTP calls to `localhost:7861` / `:7862` instead of trying to talk to the host audio device directly. The audio actually plays on the host (where the sandbox boundary doesn't apply); the sandboxed agent just makes network calls. See [Hybrid mode](#hybrid-mode-advanced) for the protocol.

Known-sandboxed runtimes that need this:

| Runner | Sandbox class | Recommendation |
|---|---|---|
| **Codex** (OpenAI Codex CLI) | bwrap | Hybrid mode by default — see [`examples/clients/codex.toml`](examples/clients/codex.toml) |
| **Gemini CLI** | folder-trust + path allowlist | Hybrid mode — even with `--skip-trust` + `--include-directories`, audio devices are unreliable. See [`examples/clients/gemini_cli.json`](examples/clients/gemini_cli.json) |
| **Opencode** | rejects `external_directory` paths | Hybrid mode if it loads aawazz-mcp at all (per-project `permission` rules may also be required) |
| **Claude Desktop** (macOS) | App sandbox | Hybrid mode if `listen` returns empty |
| Most others (Claude Code CLI, Cursor, Zed, Cline, Continue, Goose) | None | Bundled mode works — hybrid optional for resource savings |

**Concurrency caveat (s147):** if multiple sandboxed agents call `listen()` simultaneously, they race for the host's single mic. The current `aawazz-ears` FastAPI server on `:7862` does not serialize requests across clients — first to arrive wins, others get garbage. Plan dispatch so only one agent "has the mic" at a time, or wait for v1.4-class server-side serialization.

---

## Dictation (push-to-talk)

`pip install` also lands an **`aawazz-dictate`** console script — a standalone
push-to-talk dictation CLI. Not an MCP tool: it lives entirely on the
operator's machine, captures mic audio, runs Moonshine, and dispatches the
transcript via:

- **`type`** — keystroke injection into the focused window (`xdotool` on X11, `wtype`/`ydotool` on Wayland, `osascript` on macOS).
- **`clipboard`** — paste into the system clipboard (`xclip`/`xsel` on X11, `wl-copy` on Wayland, `pbcopy` on macOS).
- **`stdout`** — print the transcript only (safe smoke / pipeline use).

Designed for hotkey binding when typing is inconvenient (wet hands, cooking,
walking). Bind any key in your window manager / DE settings to:

```bash
aawazz-dictate                 # 8s capture, auto-pick output mode
```

Auto-mode prefers `type` → `clipboard` → `stdout` based on what's installed
for the detected session type.

**Common invocations**

```bash
aawazz-dictate                          # default: 8s, auto output
aawazz-dictate -d 4                     # shorter capture
aawazz-dictate -m clipboard             # force clipboard (paste with Ctrl-V)
aawazz-dictate -m stdout                # safest mode — pipe-friendly
aawazz-dictate -v --save-audio /tmp/note.wav  # debug + keep WAV
aawazz-dictate --no-beep                # silence the start/stop tones
```

**Exit codes** (useful for hotkey wrapper scripts)

| Code | Meaning |
|---|---|
| 0 | Transcript dispatched successfully |
| 1 | No input device (mic missing, OS-muted, sandboxed) |
| 2 | Transcribe returned empty / failed |
| 3 | Output dispatch failed (typer/clipboarder errored) |
| 4 | No typer or clipboarder available for the detected session |

**Hotkey-binding examples**

Hyprland (`~/.config/hypr/hyprland.conf`):
```
bind = SUPER, V, exec, aawazz-dictate -m clipboard
```

i3 (`~/.config/i3/config`):
```
bindsym $mod+v exec --no-startup-id aawazz-dictate -m type
```

GNOME / KDE: bind via Settings → Keyboard → Custom Shortcuts → command `aawazz-dictate`.

**Caveats**

- **Wayland typer**: `wtype` is not installed by default on most distros. `sudo apt install wtype` (Debian/Ubuntu) / `sudo pacman -S wtype` (Arch). Without a typer, auto-mode falls back to clipboard.
- **macOS `type` mode**: AppleScript keystroke is fragile with text containing `"`. Prefer `-m clipboard`.
- **First-run latency**: Moonshine cold-load is ~10–30 s on a fresh install. Subsequent calls are sub-second. Run `python -m aawazz_mcp.scripts.prefetch_models` to pre-warm.
- **Mic muted at OS / UEFI**: `aawazz-dictate` exits 1 with a structured stderr message — your hotkey script can branch on the exit code to surface a notification.

The dictation flow uses the same Moonshine STT as the MCP `transcribe` tool —
share the cache, share the install. v0 ships local-only; future versions may
add `--remote` to delegate to an `aawazz-ears` FastAPI service.

---

## Pluggable providers (v1.3)

The audio pipeline has five swappable stages — TTS, STT, post-processors, mic capture, playback. The base install ships with sensible defaults that match v1.2 behavior exactly; opt-in extras add more providers and the routing layer picks between them.

### TTS providers

| Name | Languages | Voices | Install | Notes |
|------|-----------|--------|---------|-------|
| `tiny-tts` | en | 1 (`MALE`) | bundled | ~3.4 MB ONNX, default for English |
| `gtts` | 69 | 1 per lang | `[multilingual]` | Google Translate TTS, requires internet |
| `piper` | 44 | hundreds | `[piper]` | ~100 MB ONNX/voice, voices auto-download from rhasspy/piper-voices |
| `kokoro` | 8 | 54 | `[kokoro]` | ~330 MB model + 25 MB voices.bin, no torch dep |
| `xtts` | 17 | voice cloning | `[xtts]` | Coqui XTTS-v2, ~2 GB, clone voice from a 3–30 s reference WAV |

### STT providers

| Name | Languages | Install | Notes |
|------|-----------|---------|-------|
| `moonshine` | en, es, zh, ja, ko, ar, vi, uk | bundled | default; ONNX, ~80 MB |
| `whisper` | ne | `[multilingual]` | HF transformers; model auto-downloads on first call |

### Post-processors

`speak(post_process=[...])` runs effects on synthesized audio. `transcribe`/`listen(pre_process=[...])` runs effects on STT input.

| Name | Direction | Install | Effect |
|------|-----------|---------|--------|
| `dsp:DEEP` / `BRIGHT` / `SOFT` / `GRAVEL` / `ROBOT` / `ECHO` / `WIDE` | tts | bundled | 7 numpy DSP voice profiles applied to any TTS output (provider must opt in via `accepts_dsp_profiles`) |
| `gain:auto` | both | bundled | Peak-normalize to −0.5 dBFS |
| `vad:webrtc` | both | `[vad]` | Silence trim front + back; cuts STT latency on quiet inputs |

### Routing chain

Resolution order, highest first:

1. **Per-call** `tts_provider=…` / `stt_provider=…` on the tool call (hard-fails if missing or language-incompatible — no silent fallback).
2. **CLI** `--tts-default <name>` / `--stt-default <name>` — replaces the `default` chain only.
3. **Env** `AAWAZZ_TTS_PROVIDER` / `AAWAZZ_STT_PROVIDER` — same shape as CLI.
4. **Config file** `~/.config/aawazz/aawazz.toml` (or `$AAWAZZ_ROUTING_FILE`).
5. **Built-in default** — `en → tiny-tts`, `default → gtts`, `ne → whisper`, `default → moonshine`.

Example config:

```toml
[tts.routing]
en      = ["piper", "tiny-tts"]
es      = ["piper", "gtts"]
ne      = ["xtts", "gtts"]
default = ["gtts"]

[stt.routing]
en      = ["moonshine"]
ne      = ["whisper"]
default = ["moonshine"]
```

A chain fails over to the next entry when a provider is unregistered or doesn't support the requested language; the override path (per-call) hard-fails instead.

### Voice IDs

Voice IDs are namespaced as `<provider>:<voice>`:

```
tiny-tts:MALE
piper:en_US-amy-medium
piper:en_GB-jenny-medium
kokoro:af_bella
kokoro:zf_xiaoxiao
xtts:cloned-from-/abs/path/to/reference.wav
```

Legacy unprefixed `voice="DEEP"` / `voice="MALE"` still works — DSP names auto-rewrite to `voice="MALE"` + `post_process=["dsp:DEEP"]`.

### Optional extras at a glance

```bash
pip install "aawazz-mcp[multilingual]"   # gtts + Whisper STT (transformers, torch)
pip install "aawazz-mcp[piper]"          # piper-tts (Rhasspy)
pip install "aawazz-mcp[kokoro]"         # kokoro-onnx (Kokoro-82M)
pip install "aawazz-mcp[xtts]"           # coqui-tts (XTTS-v2 voice cloning)
pip install "aawazz-mcp[vad]"            # webrtcvad-wheels (silence trim)
```

Each extra is independent — install only what you need. `voices_list()` reflects what's currently registered.

---

## Tools

Four tools and one resource. Tool docstrings become MCP tool descriptions verbatim — what your agent sees in `tools/list` mirrors the contracts below.

### `speak(text, voice="MALE", speed=1.0, output_path=None, play=False, language="en", tts_provider=None, post_process=None, playback_provider=None)`

Render text to speech and write a `.wav`. Returns `{audio_path, duration_s, sample_rate, latency_ms, voice, speed, text_hash, played, backend, provider, post_process_chain}`.

- `text` — required, 1–4000 chars.
- `voice` — provider-specific ID (e.g. `"piper:en_US-amy-medium"`). Legacy unprefixed DSP names (`"DEEP"`, `"BRIGHT"`, …) auto-rewrite to `voice="MALE"` + `post_process=["dsp:<NAME>"]`.
- `speed` — 0.5–2.0 multiplier (provider must support it; XTTS is fixed-speed).
- `output_path` — absolute path; default `~/.local/share/aawazz/mouth/<ts>-<hash>.wav`.
- `play` — autoplay via the registered playback provider (default: `paplay` / `aplay` / `afplay` on `PATH`).
- `language` — ISO 639-1 code routes the chain (default `"en"`).
- `tts_provider` — override the chain (hard-fails if missing or language-incompatible). See [Pluggable providers](#pluggable-providers-v13).
- `post_process` — ordered list of post-processor names (`["dsp:DEEP", "gain:auto"]`).
- `playback_provider` — override the playback provider (default `"shell"`).

Example (mcp-inspector):
```
speak({"text": "Hello world", "play": true})
speak({"text": "Hola mundo", "language": "es"})
speak({"text": "Hi there", "tts_provider": "piper", "voice": "piper:en_US-amy-medium"})
speak({"text": "Robot voice", "post_process": ["dsp:ROBOT", "gain:auto"]})
```

### `transcribe(audio_path, language="en", model_arch="tiny_streaming", stt_provider=None, pre_process=None)`

Transcribe a WAV file (local path or `http(s)://` URL). Returns `{text, audio_duration_s, sample_rate, latency_ms, model_arch, language, audio_path, backend, provider, pre_process_chain}`.

- `audio_path` — absolute path or `http(s)://` URL. URL inputs are downloaded to `${TMPDIR}/aawazz-stt-<sha8>.wav` and unlinked after.
- `model_arch` — Moonshine arch (`tiny | tiny_streaming | base | base_streaming | small_streaming | medium_streaming`). Ignored when the resolved provider isn't Moonshine.
- `stt_provider` — override the chain.
- `pre_process` — ordered list of post-processors with `direction="stt"` or `"both"` (e.g. `["vad:webrtc"]`). The original `audio_path` is never modified — pre-processing runs on a tempfile copy.

Example:
```
transcribe({"audio_path": "/tmp/note.wav", "language": "en"})
transcribe({"audio_path": "/tmp/quiet.wav", "pre_process": ["vad:webrtc"]})
```

### `listen(duration_s=5.0, language="en", model_arch="tiny_streaming", save_audio=False, stt_provider=None, pre_process=None, capture_provider=None)`

Bounded mic capture, then transcribe. Returns same shape as `transcribe` plus `audio_path: str | None` (only set when `save_audio=true`). `backend` is always `"local"`.

- `duration_s` — 0.5–30.0 hard cap.
- `save_audio` — keep the captured WAV at `~/.local/share/aawazz/ears/<ts>.wav`. When `true` and `pre_process` is set, the saved WAV reflects the pre-processed audio.
- `capture_provider` — override the capture provider (default `"sounddevice"`).
- `pre_process` — see `transcribe` above.

Example:
```
listen({"duration_s": 4.0, "save_audio": true})
listen({"duration_s": 10.0, "pre_process": ["vad:webrtc"]})
```

### `voices_list()`

Cheap probe — does **not** load models. v1.3 response shape:

```jsonc
{
  "providers": {
    "tts": [{"name": "tiny-tts", "version": "0.3.2", "languages": ["en"], "voices": [...], "accepts_dsp_profiles": true, ...}, ...],
    "stt": [{"name": "moonshine", "languages": ["en", "es", ...], "model_archs": {...}}, ...],
    "post_processors": [{"name": "dsp:DEEP", "direction": "tts"}, ...],
    "capture": [{"name": "sounddevice", "has_input_device": true}],
    "playback": [{"name": "shell", "has_player": true}]
  },
  "routing": {"tts": {"en": ["tiny-tts"], "default": ["gtts"]}, "stt": {...}},
  "capabilities": {"listen": true, "play": true, "backend_mode": "local", ...},
  "tts": {...},  // v1.0/v1.2 alias view (preserved until v2.0)
  "stt": {...}
}
```

Use it to discover what providers / voices / post-processors are available, what the resolved routing chain is, and whether `listen` will work on the current host.

### Resource: `aawazz://health`

Returns JSON `{models_loaded: {tts, stt_archs}, mode, remote_url, version}`. Read it from the resource panel in your MCP runtime, or via `mcp-inspector` `resources/read aawazz://health`.

Full contract: [`SPEC.md` §1](SPEC.md).

---

## Caveats

- **Linux + macOS only.** Moonshine ships only `libmoonshine.so` and `libonnxruntime.*.dylib`; native Windows is gated upstream. WSL2 works.
- **First-run network requirement.** Three caches get populated on first call: `~/.cache/huggingface/hub/` (tiny-tts G.pth, ~50 MB via HF hub), `~/.cache/moonshine_voice/` (~80 MB ONNX), `~/nltk_data/` (`g2p_en`, `cmudict`, `averaged_perceptron_tagger_eng`). Run `python -m aawazz_mcp.scripts.prefetch_models` ahead of time on offline / air-gapped boxes.
- **Default-install TTS is single-voice English.** `tiny-tts` ships exactly one voice (`MALE`); other tiny-tts voice names return a structured error. Multi-voice and multi-language synthesis arrives via the v1.3 optional extras (`[multilingual]` for gTTS, `[piper]` for Piper, `[kokoro]` for Kokoro, `[xtts]` for XTTS-v2 voice cloning). Default routing keeps tiny-tts for English so existing v1.2 callers see no change.
- **`listen` needs a mic + audio server.** `sounddevice` requires PulseAudio / PipeWire / CoreAudio access. Headless boxes, SSH sessions without `--enable-audio`, and most Docker containers will see `voices_list().capabilities.listen: false`. The tool itself returns a clean error rather than crashing.
- **torch dependency.** `tiny-tts` pulls PyTorch — ~600 MB on disk. A `[lite]` extras carving for STT-only users is on the future-work list.
- **stdio-safe by construction.** All logging goes to stderr; tiny-tts's stdout `print()` is wrapped in a `redirect_stdout` context manager. Don't add `print()` calls to source — see the lint rule in `pyproject.toml`.
- **Sandboxed runtimes can break audio at runtime even though probes succeed.** `voices_list().capabilities.play == true` only checks that a player binary (`paplay` / `aplay` / `afplay`) is on `PATH`; it can't predict whether the binary will be allowed to reach the host audio device. Confirmed v1.1.0 (Codex CLI sandbox): `paplay` returned `Connection refused / Operation not permitted`, `aplay` returned `audio open error: Operation not permitted`. The aawazz response is correct (`speak(play=true).played == false`), but the operator should know to either run `aawazz-mcp` outside the restricted sandbox or wire `--remote` to a host-side `aawazz-mouth` that has PulseAudio / PipeWire / ALSA access. Same trap exists for `listen` — a probe-true but routing-blocked mic.
- **Mic capture has a hard timeout (since v1.1.1).** `record_to_wav_hard_timeout` runs the capture in a child process and force-kills after `duration_s + 5s` if `sd.wait()` wedges (mic enumerates but produces no samples — OS mute, UEFI mute, routing wrong source). Both the MCP `listen` tool and `aawazz-dictate` now share this surface. In v1.1.0 only dictate had it; `listen` could hang the runtime indefinitely.

---

## Development

```bash
git clone https://github.com/nixpt/aawazz-mcp
cd aawazz-mcp
pip install -e ".[dev]"
pytest
```

Test markers (configured in `pyproject.toml`):

- `@pytest.mark.slow` — model-load round-trips. Run with `pytest -m slow`; skip in CI default.
- `@pytest.mark.remote` — requires the FastAPI mouth/ears running on `:7861` / `:7862`. Auto-skip when unreachable.
- `@pytest.mark.mic` — requires a working audio input device. Auto-skip when `sounddevice.query_devices(kind="input")` is empty.

Lint: `ruff check src tests`. Format: `ruff format src tests`. The ruff config bans `print()` in `src/` to keep stdio safe.

---

## License & credits

MIT — see [`LICENSE`](LICENSE).

Bundled / linked work, each under its own license:

- **[Useful Sensors / Moonshine](https://github.com/usefulsensors/moonshine)** — MIT — STT model + ONNX runtime wrapper.
- **[backtracking / tiny-tts](https://github.com/backtracking/tiny-tts)** — TTS model + inference.
- **[Anthropic / `mcp` Python SDK](https://github.com/modelcontextprotocol/python-sdk)** — MIT — `FastMCP` server framework, the protocol layer.
