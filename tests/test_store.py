from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from clip_index.store import (
    IndexCorrupt,
    IndexStore,
    ScanError,
    file_unchanged,
    merge_record,
    parse_jsonl,
    thumb_key,
)


def _incoming(path: str = "December/a.mov", **social_extra) -> dict:
    social = {
        "project": "December",
        "graded": False,
        "subject": "scenery",
        "face_visible": {"visible": False, "fraction": 0.0},
        "lighting": "day",
        "brightness": 0.5,
        "camera_motion": "static",
        "calm_moments": [[0.0, 2.5]],
        "tags": ["day", "static", "scenery", "calm"],
    }
    social.update(social_extra)
    return {
        "path": path,
        "size": 10,
        "mtime": 20.0,
        "duration": 4.0,
        "fps": 24.0,
        "resolution": "160x90",
        "social": social,
        "autolog": {"scene": 99},
    }


def test_thumb_key_is_sha1_of_relative_path():
    rel = "Décembre/prise.mkv"
    assert thumb_key(rel) == hashlib.sha1(rel.encode("utf-8")).hexdigest()


def test_merge_keeps_foreign_namespaces_and_drops_incoming_ones():
    existing = {
        "path": "December/a.mov",
        "autolog": {"scene": "1", "note": "keep me"},
        "custom_ns": {"owner": "um-autolog"},
    }
    merged = merge_record(existing, _incoming())
    assert merged["autolog"] == {"scene": "1", "note": "keep me"}
    assert merged["custom_ns"] == {"owner": "um-autolog"}
    assert merged["social"]["project"] == "December"
    assert "scene" not in merged["social"]

    fresh = merge_record(None, _incoming())
    assert "autolog" not in fresh
    assert set(fresh) == {
        "path",
        "size",
        "mtime",
        "duration",
        "fps",
        "resolution",
        "social",
    }


def test_round_trip_preserves_autolog(tmp_path: Path):
    store = IndexStore(tmp_path)
    store.ensure()
    existing = {
        "path": "December/a.mov",
        "size": 1,
        "mtime": 1.0,
        "autolog": {"scene": "1"},
    }
    store.path.write_text(json.dumps(existing) + "\n", encoding="utf-8")
    store.update_path(_incoming())
    records = store.load_existing()
    assert len(records) == 1
    assert records[0]["autolog"] == {"scene": "1"}
    assert records[0]["size"] == 10
    assert records[0]["social"]["lighting"] == "day"
    assert list(tmp_path.glob(".index.jsonl.*.tmp")) == []


def test_failed_render_does_not_touch_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = IndexStore(tmp_path)
    store.ensure()
    store.update_path(_incoming())
    before = store.path.read_bytes()

    def explode(_records):
        raise RuntimeError("disk full")

    monkeypatch.setattr(store, "_render", explode)
    with pytest.raises(RuntimeError, match="disk full"):
        store.update_path(_incoming("December/b.mov"))
    assert store.path.read_bytes() == before
    assert list(tmp_path.glob(".index.jsonl.*.tmp")) == []


def test_corrupt_index_is_not_rewritten(tmp_path: Path):
    store = IndexStore(tmp_path)
    store.ensure()
    store.path.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(IndexCorrupt):
        store.load_existing()
    assert store.path.read_text(encoding="utf-8") == "{not json\n"
    with pytest.raises(IndexCorrupt):
        parse_jsonl('{"nope": 1}\n')


def test_refuses_index_inside_scan_root(tmp_path: Path):
    root = tmp_path / "drive"
    root.mkdir()
    store = IndexStore(root / "clip-index", scan_root=root)
    with pytest.raises(ScanError, match="read-only"):
        store.ensure()
    assert not (root / "clip-index").exists()


def test_file_unchanged_tolerance():
    record = {"size": 5, "mtime": 10.0}
    assert file_unchanged(record, 5, 10.0004) is True
    assert file_unchanged(record, 5, 10.01) is False
    assert file_unchanged(record, 6, 10.0) is False


def test_lock_blocks_another_process(tmp_path: Path):
    store = IndexStore(tmp_path)
    store.ensure()
    script = """
import fcntl, os, sys, time
fd = os.open(sys.argv[1], os.O_CREAT | os.O_RDWR, 0o644)
fcntl.flock(fd, fcntl.LOCK_EX)
print("locked", flush=True)
time.sleep(30)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(store.lock_path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    try:
        assert proc.stdout.readline().strip() == "locked"
        fd = os.open(store.lock_path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
    finally:
        proc.kill()
        proc.wait(timeout=5)

    started = time.monotonic()
    with store._lock(fcntl.LOCK_EX):
        pass
    assert time.monotonic() - started < 2
