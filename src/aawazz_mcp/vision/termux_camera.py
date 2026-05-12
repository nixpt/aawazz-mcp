"""Camera capture via Termux:API's ``termux-camera-photo``.

Talks to Android's Camera2 service via Termux:API. Returns a path to
the saved JPEG plus dimensions and size — no in-memory image handling,
no pixel-level processing. Callers feed the path to whichever
vision-LLM provider they want (issue #15).

**proot-distro path trap (same as termux-mic):** the binary runs in
Termux's namespace and can't write to proot's ``/tmp``. The default
output directory lives under ``$AAWAZZ_HOME/eyes`` if that path is
under ``/sdcard`` or ``/data/data/com.termux/...``; otherwise we fall
back to ``/sdcard/aawazz-eyes/`` (overridable via
``$AAWAZZ_TERMUX_CAMERA_DIR``).

Contract:
    has_camera() -> bool
    available_cameras() -> list[dict]      # [{id, facing, jpeg_output_sizes}, ...]
    capture(camera_id, output_path) -> dict
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

_LOG = logging.getLogger("aawazz.vision.termux_camera")

_CAMERA_BIN: str = "termux-camera-photo"
_INFO_BIN: str = "termux-camera-info"

# termux-camera-photo blocks until the photo is captured + written.
# Most devices return in 1-3 seconds; cap conservatively to avoid the
# rare wedged-Camera2-service case.
_CAPTURE_TIMEOUT_S: float = 30.0
_INFO_TIMEOUT_S: float = 5.0

# Directory hint for cross-namespace writes — proot's /tmp isn't visible
# to the Termux:API service. ``/sdcard/aawazz-eyes/`` is reachable from
# both sides on stock Android.
_DEFAULT_TMP_DIR: str = "/sdcard/aawazz-eyes"


def _output_dir() -> Path:
    """Resolve the default output directory.

    Honours ``$AAWAZZ_TERMUX_CAMERA_DIR`` first; otherwise picks
    ``/sdcard/aawazz-eyes/`` so the Termux:API service can write to it.
    proot's ``$AAWAZZ_HOME`` is intentionally NOT used as the default
    because it usually points at ``~/.local/share/aawazz`` inside proot
    — invisible to the Termux service.
    """
    return Path(os.environ.get("AAWAZZ_TERMUX_CAMERA_DIR", _DEFAULT_TMP_DIR))


def _default_output_path(camera_id: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rand = uuid.uuid4().hex[:8]
    return str(_output_dir() / f"{ts}-cam{camera_id}-{rand}.jpg")


def has_camera() -> bool:
    """True iff ``termux-camera-photo`` is on PATH (Termux:API addon installed)."""
    return shutil.which(_CAMERA_BIN) is not None


def available_cameras() -> list[dict]:
    """Enumerate cameras via ``termux-camera-info``.

    Returns the JSON list verbatim: ``[{id, facing, jpeg_output_sizes,
    auto_focus_modes, ...}, ...]``. Empty list when the binary is
    missing or fails — caller decides whether to error or fall back.
    """
    bin_path = shutil.which(_INFO_BIN)
    if bin_path is None:
        return []
    try:
        result = subprocess.run(
            [bin_path], capture_output=True, timeout=_INFO_TIMEOUT_S
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        _LOG.warning("%s probe failed: %s", _INFO_BIN, exc)
        return []
    if result.returncode != 0:
        return []
    try:
        data = json.loads(result.stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def capture(camera_id: int = 0, output_path: str | None = None) -> dict:
    """Capture a JPEG from the given camera. Returns a result dict.

    Returns ``{image_path, width, height, size_bytes, camera_id,
    latency_ms}`` on success, ``{error, hint?, stderr?}`` on failure.
    Never raises.
    """
    bin_path = shutil.which(_CAMERA_BIN)
    if bin_path is None:
        return {
            "error": f"{_CAMERA_BIN} not on PATH",
            "hint": "install the Termux:API addon (Android/Termux only)",
        }

    if output_path is None:
        out_dir = _output_dir()
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return {
                "error": f"output directory {out_dir} unwritable: {exc}",
                "hint": "override via $AAWAZZ_TERMUX_CAMERA_DIR",
            }
        output_path = _default_output_path(camera_id)
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        result = subprocess.run(
            [bin_path, "-c", str(int(camera_id)), output_path],
            capture_output=True,
            timeout=_CAPTURE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return {
            "error": f"{_CAMERA_BIN} timed out after {_CAPTURE_TIMEOUT_S:.0f}s",
            "hint": "Camera2 service may be wedged; check if another app holds the camera",
        }
    except OSError as exc:
        return {"error": f"{_CAMERA_BIN} failed to spawn: {exc}"}

    latency_ms = int((time.time() - t0) * 1000)

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        return {
            "error": f"{_CAMERA_BIN} exited {result.returncode}",
            "stderr": stderr,
            "latency_ms": latency_ms,
        }

    out_path = Path(output_path)
    if not out_path.exists() or out_path.stat().st_size == 0:
        return {
            "error": f"output file {output_path} missing or empty",
            "hint": "Termux:API camera permission may be denied — check Android Settings",
            "latency_ms": latency_ms,
        }

    width, height = _probe_jpeg_dimensions(out_path)
    return {
        "image_path": str(out_path),
        "width": width,
        "height": height,
        "size_bytes": out_path.stat().st_size,
        "camera_id": int(camera_id),
        "latency_ms": latency_ms,
    }


def _probe_jpeg_dimensions(path: Path) -> tuple[int | None, int | None]:
    """Read JPEG SOF marker to get width/height without loading the full image.

    Avoids a Pillow dependency for what's essentially a 4-byte header
    parse. Returns ``(None, None)`` on parse failure so the caller can
    still ship the path even if metadata is unreadable.
    """
    try:
        with open(path, "rb") as f:
            data = f.read(2)
            if data != b"\xff\xd8":  # JPEG SOI
                return None, None
            while True:
                marker = f.read(2)
                if len(marker) < 2:
                    return None, None
                if marker[0] != 0xFF:
                    return None, None
                # SOFn markers contain width/height. 0xC0 baseline, 0xC2 progressive.
                if marker[1] in {0xC0, 0xC1, 0xC2, 0xC3}:
                    f.read(3)  # length(2) + precision(1)
                    h_bytes = f.read(2)
                    w_bytes = f.read(2)
                    height = int.from_bytes(h_bytes, "big")
                    width = int.from_bytes(w_bytes, "big")
                    return width, height
                # Skip this segment using its length field.
                seg_len = int.from_bytes(f.read(2), "big")
                if seg_len < 2:
                    return None, None
                f.seek(seg_len - 2, 1)
    except OSError:
        return None, None
