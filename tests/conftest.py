"""Shared fixtures. CLIP stays off unless a test turns it on."""

from __future__ import annotations

import shutil

import pytest


@pytest.fixture(autouse=True)
def _clip_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLIP_INDEX_CLIP", raising=False)
    monkeypatch.delenv("CLIP_INDEX_CLIP_CHECKPOINT", raising=False)


@pytest.fixture
def needs_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required")
