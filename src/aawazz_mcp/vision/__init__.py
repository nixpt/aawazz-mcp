"""Vision primitives — camera capture today, vision LLMs / OCR later.

Sibling to ``audio/`` (TTS, STT, playback, capture). This package's
contract is intentionally narrower for now: capture a photo, return a
path. Vision-LLM integration (issue #15) goes through the existing
``LlmProvider`` abstraction rather than a parallel ``VisionProvider``,
so the only thing this package needs to own is the camera side.
"""
