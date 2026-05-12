"""Coverage for the termux-camera helper (issue #15, camera half)."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from aawazz_mcp.vision import termux_camera


def _which_fake(binaries: set[str]):
    return lambda name: f"/bin/{name}" if name in binaries else None


def _write_min_jpeg(path, width: int = 640, height: int = 480) -> None:
    """Write a minimal valid JPEG with SOF0 reporting the requested dims.

    Just enough header for :func:`_probe_jpeg_dimensions` to parse —
    no actual scan data needed for the tests we run.
    """
    soi = b"\xff\xd8"
    # SOF0 marker, length=17, precision=8, height, width, components=3
    sof = (
        b"\xff\xc0\x00\x11\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    )
    eoi = b"\xff\xd9"
    with open(path, "wb") as f:
        f.write(soi + sof + eoi)


@pytest.fixture(autouse=True)
def _redirect_termux_camera_dir(tmp_path, monkeypatch) -> None:
    """Redirect /sdcard/aawazz-eyes/ to a tmp_path for tests."""
    monkeypatch.setenv("AAWAZZ_TERMUX_CAMERA_DIR", str(tmp_path))


# ── Capability probes ───────────────────────────────────────────────────────


def test_has_camera_false_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.vision.termux_camera.shutil.which", lambda name: None
    )
    assert termux_camera.has_camera() is False


def test_has_camera_true_when_present(monkeypatch) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.vision.termux_camera.shutil.which",
        _which_fake({"termux-camera-photo"}),
    )
    assert termux_camera.has_camera() is True


def test_available_cameras_parses_json(monkeypatch) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.vision.termux_camera.shutil.which",
        _which_fake({"termux-camera-info"}),
    )
    payload = json.dumps(
        [
            {"id": "0", "facing": "back", "jpeg_output_sizes": [{"width": 4032, "height": 3024}]},
            {"id": "1", "facing": "front", "jpeg_output_sizes": [{"width": 1920, "height": 1080}]},
        ]
    ).encode()
    monkeypatch.setattr(
        "aawazz_mcp.vision.termux_camera.subprocess.run",
        lambda argv, **kw: SimpleNamespace(returncode=0, stdout=payload, stderr=b""),
    )
    cams = termux_camera.available_cameras()
    assert len(cams) == 2
    assert cams[0]["facing"] == "back"
    assert cams[1]["facing"] == "front"


def test_available_cameras_empty_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.vision.termux_camera.shutil.which", lambda name: None
    )
    assert termux_camera.available_cameras() == []


def test_available_cameras_empty_on_garbage(monkeypatch) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.vision.termux_camera.shutil.which",
        _which_fake({"termux-camera-info"}),
    )
    monkeypatch.setattr(
        "aawazz_mcp.vision.termux_camera.subprocess.run",
        lambda argv, **kw: SimpleNamespace(returncode=0, stdout=b"not json", stderr=b""),
    )
    assert termux_camera.available_cameras() == []


# ── capture() error paths ───────────────────────────────────────────────────


def test_capture_errors_when_binary_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.vision.termux_camera.shutil.which", lambda name: None
    )
    out = termux_camera.capture()
    assert "error" in out
    assert "termux-camera-photo" in out["error"]
    assert "Termux:API" in out["hint"]


def test_capture_errors_on_nonzero_exit(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.vision.termux_camera.shutil.which",
        _which_fake({"termux-camera-photo"}),
    )
    monkeypatch.setattr(
        "aawazz_mcp.vision.termux_camera.subprocess.run",
        lambda argv, **kw: SimpleNamespace(
            returncode=1, stdout=b"", stderr=b"Camera service died"
        ),
    )
    out = termux_camera.capture(output_path=str(tmp_path / "x.jpg"))
    assert "error" in out
    assert "exited 1" in out["error"]
    assert out["stderr"] == "Camera service died"


def test_capture_errors_on_timeout(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.vision.termux_camera.shutil.which",
        _which_fake({"termux-camera-photo"}),
    )

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="termux-camera-photo", timeout=30.0)

    monkeypatch.setattr(
        "aawazz_mcp.vision.termux_camera.subprocess.run", raise_timeout
    )
    out = termux_camera.capture(output_path=str(tmp_path / "x.jpg"))
    assert "error" in out
    assert "timed out" in out["error"]
    assert "Camera2" in out["hint"]


def test_capture_errors_when_file_missing(monkeypatch, tmp_path) -> None:
    """Exit 0 but no output file — permission denied is the usual cause."""
    monkeypatch.setattr(
        "aawazz_mcp.vision.termux_camera.shutil.which",
        _which_fake({"termux-camera-photo"}),
    )
    monkeypatch.setattr(
        "aawazz_mcp.vision.termux_camera.subprocess.run",
        lambda argv, **kw: SimpleNamespace(returncode=0, stdout=b"", stderr=b""),
    )
    out = termux_camera.capture(output_path=str(tmp_path / "x.jpg"))
    assert "error" in out
    assert "missing or empty" in out["error"]
    assert "permission" in out["hint"]


# ── capture() success path ──────────────────────────────────────────────────


def test_capture_returns_metadata_on_success(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.vision.termux_camera.shutil.which",
        _which_fake({"termux-camera-photo"}),
    )

    def fake_run(argv, **kw):
        # Pretend the camera wrote a JPEG to the path passed as the last arg.
        _write_min_jpeg(argv[-1], width=4032, height=3024)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("aawazz_mcp.vision.termux_camera.subprocess.run", fake_run)

    out_path = tmp_path / "snap.jpg"
    out = termux_camera.capture(camera_id=0, output_path=str(out_path))
    assert "error" not in out
    assert out["image_path"] == str(out_path)
    assert out["camera_id"] == 0
    assert out["width"] == 4032
    assert out["height"] == 3024
    assert out["size_bytes"] > 0
    assert isinstance(out["latency_ms"], int)


def test_capture_uses_default_output_when_none(monkeypatch, tmp_path) -> None:
    """No output_path → file lands under $AAWAZZ_TERMUX_CAMERA_DIR."""
    monkeypatch.setattr(
        "aawazz_mcp.vision.termux_camera.shutil.which",
        _which_fake({"termux-camera-photo"}),
    )

    seen: dict = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        _write_min_jpeg(argv[-1], width=1920, height=1080)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("aawazz_mcp.vision.termux_camera.subprocess.run", fake_run)

    out = termux_camera.capture(camera_id=1)
    assert "error" not in out
    assert out["image_path"].startswith(str(tmp_path))
    assert out["camera_id"] == 1
    # argv shape: [bin, -c, "1", path]
    assert seen["argv"][1] == "-c"
    assert seen["argv"][2] == "1"
    assert seen["argv"][3].endswith(".jpg")


def test_capture_passes_camera_id_to_binary(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.vision.termux_camera.shutil.which",
        _which_fake({"termux-camera-photo"}),
    )
    seen: dict = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        _write_min_jpeg(argv[-1])
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("aawazz_mcp.vision.termux_camera.subprocess.run", fake_run)
    termux_camera.capture(camera_id=42, output_path=str(tmp_path / "x.jpg"))
    assert "-c" in seen["argv"]
    c_idx = seen["argv"].index("-c")
    assert seen["argv"][c_idx + 1] == "42"


# ── JPEG dimension parser ───────────────────────────────────────────────────


def test_probe_jpeg_dimensions_reads_sof(tmp_path) -> None:
    p = tmp_path / "x.jpg"
    _write_min_jpeg(p, width=1280, height=720)
    w, h = termux_camera._probe_jpeg_dimensions(p)
    assert w == 1280
    assert h == 720


def test_probe_jpeg_dimensions_returns_none_on_non_jpeg(tmp_path) -> None:
    p = tmp_path / "x.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    w, h = termux_camera._probe_jpeg_dimensions(p)
    assert w is None
    assert h is None
