from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from clip_index.cli import default_index_dir, main
from clip_index.media import ClipReadError, iter_sampled_frames, probe_video
from clip_index.schema import social_complete
from clip_index.scan import scan
from clip_index.store import IndexStore, ScanError, thumb_key
from tests.clips import write_solid_clip


def _social() -> dict:
    return {
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


def _analyzer(calls: list[str]):
    def analyze(video: Path, **kwargs):
        rel = kwargs["rel_path"]
        calls.append(rel)
        if kwargs.get("fail"):
            raise ClipReadError("boom")
        social = _social()
        social["project"] = rel.split("/", 1)[0] if "/" in rel else "root"
        return {
            "path": rel,
            "size": kwargs["size"],
            "mtime": kwargs["mtime"],
            "duration": 4.0,
            "fps": 24.0,
            "resolution": "160x90",
            "social": social,
        }

    return analyze


def _snapshot(root: Path) -> dict[str, tuple[int, int, int]]:
    found: dict[str, tuple[int, int, int]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            st = path.lstat()
            found[str(path.relative_to(root))] = (st.st_size, st.st_mtime_ns, st.st_mode)
    return found


def _freeze(root: Path) -> None:
    for dirpath, _dirnames, filenames in os.walk(root, topdown=False):
        for name in filenames:
            os.chmod(Path(dirpath) / name, 0o444)
        os.chmod(dirpath, 0o555)


def _thaw(root: Path) -> None:
    for dirpath, _dirnames, filenames in os.walk(root):
        os.chmod(dirpath, 0o755)
        for name in filenames:
            os.chmod(Path(dirpath) / name, 0o644)


def test_default_index_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CLIP_INDEX_DIR", raising=False)
    assert default_index_dir() == (tmp_path / "Documents" / "clip-index").resolve()
    custom = tmp_path / "custom"
    monkeypatch.setenv("CLIP_INDEX_DIR", str(custom))
    assert default_index_dir() == custom.resolve()


def test_scan_skips_personal_is_incremental_and_preserves_namespaces(tmp_path: Path):
    root = tmp_path / "TOSHIBA EXT"
    (root / "December").mkdir(parents=True)
    (root / "personal").mkdir()
    keep = root / "December" / "keep.mov"
    secret = root / "personal" / "secret.mov"
    other = root / "December" / "other.mov"
    keep.write_bytes(b"keep-bytes")
    secret.write_bytes(b"secret")
    other.write_bytes(b"other")
    keep_stat = keep.stat()

    index = tmp_path / "index"
    store = IndexStore(index, scan_root=root)
    store.ensure()
    seeded = {
        "path": "December/keep.mov",
        "size": keep_stat.st_size,
        "mtime": keep_stat.st_mtime,
        "autolog": {"scene": "1", "note": "keep me"},
        "custom_ns": {"owner": "um-autolog"},
    }
    store.path.write_text(json.dumps(seeded, ensure_ascii=False) + "\n", encoding="utf-8")

    calls: list[str] = []
    _freeze(root)
    try:
        before = _snapshot(root)
        stats = scan(root, index, analyzer=_analyzer(calls))
        assert _snapshot(root) == before
    finally:
        _thaw(root)

    assert calls == ["December/keep.mov", "December/other.mov"]
    assert stats.analyzed == 2
    assert stats.personal_dirs == ["personal"]
    records = {row["path"]: row for row in store.load_existing()}
    assert set(records) == {"December/keep.mov", "December/other.mov"}
    assert records["December/keep.mov"]["autolog"] == {"scene": "1", "note": "keep me"}
    assert records["December/keep.mov"]["custom_ns"] == {"owner": "um-autolog"}
    assert social_complete(records["December/keep.mov"]["social"])
    assert records["December/keep.mov"]["social"]["project"] == "December"

    calls.clear()
    again = scan(root, index, analyzer=_analyzer(calls))
    assert calls == []
    assert again.unchanged == 2
    assert again.analyzed == 0
    kept = store.path.read_bytes()
    kept_mtime = store.path.stat().st_mtime_ns
    third = scan(root, index, analyzer=_analyzer(calls))
    assert third.unchanged == 2
    assert store.path.read_bytes() == kept
    assert store.path.stat().st_mtime_ns == kept_mtime

    forced = scan(root, index, force=True, analyzer=_analyzer(calls))
    assert forced.analyzed == 2
    assert calls == ["December/keep.mov", "December/other.mov"]
    reloaded = {row["path"]: row for row in store.load_existing()}
    assert reloaded["December/keep.mov"]["autolog"]["note"] == "keep me"


def test_failed_clip_is_retried_and_root_named_personal_is_skipped(tmp_path: Path):
    root = tmp_path / "drive"
    (root / "December").mkdir(parents=True)
    good = root / "December" / "good.mov"
    bad = root / "December" / "bad.mov"
    good.write_bytes(b"good-clip")
    bad.write_bytes(b"bad-clip")
    index = tmp_path / "index"
    calls: list[str] = []

    def analyze(video: Path, **kwargs):
        rel = kwargs["rel_path"]
        calls.append(rel)
        if rel.endswith("bad.mov") and calls.count(rel) == 1:
            raise ClipReadError("decoder blew up")
        return _analyzer([])(video, **kwargs)

    first = scan(root, index, analyzer=analyze)
    assert first.failed == 1
    assert first.analyzed == 1
    paths = {row["path"] for row in IndexStore(index).load_existing()}
    assert paths == {"December/good.mov"}

    second = scan(root, index, analyzer=analyze)
    assert second.analyzed == 1
    assert second.unchanged == 1
    paths = {row["path"] for row in IndexStore(index).load_existing()}
    assert paths == {"December/good.mov", "December/bad.mov"}

    personal = tmp_path / "Personal"
    personal.mkdir()
    (personal / "secret.mov").write_bytes(b"nope")
    exploded: list[str] = []

    def boom(*_args, **_kwargs):
        exploded.append("called")
        raise AssertionError("personal root must not be read for classification")

    skipped = scan(personal, tmp_path / "personal-index", analyzer=boom)
    assert exploded == []
    assert skipped.analyzed == 0
    assert skipped.personal_dirs == ["."]


def test_index_inside_source_and_missing_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "drive"
    root.mkdir()
    (root / "a.mov").write_bytes(b"abc")
    before = _snapshot(root)
    with pytest.raises(ScanError, match="read-only"):
        scan(root, root / "clip-index")
    assert _snapshot(root) == before
    assert not (root / "clip-index").exists()

    monkeypatch.setattr("clip_index.scan.shutil.which", lambda _name: None)
    with pytest.raises(ScanError, match="ffmpeg"):
        scan(root, tmp_path / "index")
    assert not (tmp_path / "index").exists()


def test_cli_scan_of_synthetic_clips(tmp_path: Path, needs_ffmpeg: None, capsys: pytest.CaptureFixture[str]):
    root = tmp_path / "TOSHIBA EXT"
    colors = {
        "December/camera/day.mkv": (70, 140, 210),
        "December/05_renders_proxies/warm.mkv": (220, 150, 60),
        "December/camera/night.mkv": (6, 6, 8),
        "December/camera/snow.mkv": (245, 245, 248),
        "December/camera/flat.mkv": (150, 150, 155),
        "personal/secret.mkv": (20, 200, 20),
        "December/Personal/hidden.mkv": (20, 200, 20),
    }
    for rel, rgb in colors.items():
        write_solid_clip(root / rel, rgb, seconds=4.0, fps=10)

    day = root / "December/camera/day.mkv"
    day_stat = day.stat()
    index = tmp_path / "index"
    store = IndexStore(index, scan_root=root)
    store.ensure()
    store.path.write_text(
        json.dumps(
            {
                "path": "December/camera/day.mkv",
                "size": day_stat.st_size,
                "mtime": day_stat.st_mtime,
                "autolog": {"scene": "4A", "note": "do not drop"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _freeze(root)
    try:
        before = _snapshot(root)
        code = main(["scan", str(root), "--index", str(index), "--sample-fps", "2"])
        after = _snapshot(root)
    finally:
        _thaw(root)
    assert code == 0
    assert after == before
    printed = capsys.readouterr().out
    assert "analyzed=" in printed
    assert "personal/secret.mkv" not in printed
    assert "source drive will not be modified" in printed

    rows = {row["path"]: row for row in store.load_existing()}
    assert "personal/secret.mkv" not in rows
    assert "December/Personal/hidden.mkv" not in rows
    assert set(rows) == {
        "December/camera/day.mkv",
        "December/05_renders_proxies/warm.mkv",
        "December/camera/night.mkv",
        "December/camera/snow.mkv",
        "December/camera/flat.mkv",
    }
    day_row = rows["December/camera/day.mkv"]
    assert day_row["autolog"] == {"scene": "4A", "note": "do not drop"}
    assert day_row["resolution"] == "160x90"
    assert day_row["fps"] == pytest.approx(10.0)
    assert day_row["duration"] == pytest.approx(4.0, abs=0.15)
    assert day_row["social"]["project"] == "December"
    assert day_row["social"]["graded"] is False
    assert day_row["social"]["lighting"] == "day"
    assert day_row["social"]["camera_motion"] == "static"
    assert day_row["social"]["face_visible"]["visible"] is False
    assert day_row["social"]["calm_moments"]

    warm = rows["December/05_renders_proxies/warm.mkv"]["social"]
    assert warm["graded"] is True
    assert warm["project"] == "December"
    assert warm["lighting"] == "golden"
    assert "graded" in warm["tags"]

    assert rows["December/camera/night.mkv"]["social"]["lighting"] == "night"
    assert rows["December/camera/night.mkv"]["social"]["brightness"] < 0.2
    assert rows["December/camera/snow.mkv"]["social"]["lighting"] == "snow"
    assert rows["December/camera/flat.mkv"]["social"]["lighting"] == "overcast"

    thumb = index / "thumbs" / thumb_key("December/camera/day.mkv")
    jpegs = list(thumb.glob("*.jpg"))
    assert jpegs
    assert not any(path.name == "thumbs" for path in root.rglob("*"))

    saved = store.path.read_bytes()
    saved_mtime = store.path.stat().st_mtime_ns
    code = main(["scan", str(root), "--index", str(index)])
    assert code == 0
    assert "unchanged=5" in capsys.readouterr().out
    assert store.path.read_bytes() == saved
    assert store.path.stat().st_mtime_ns == saved_mtime

    code = main(
        [
            "query",
            "--index",
            str(index),
            "--lighting",
            "golden",
            "--motion",
            "static",
            "--min-brightness",
            "0.4",
            "--project",
            "december",
            "--graded",
        ]
    )
    assert code == 0
    queried = capsys.readouterr().out
    assert "December/05_renders_proxies/warm.mkv" in queried
    assert "calm:" in queried
    assert "day.mkv" not in queried


def test_probe_and_scaled_frames(tmp_path: Path, needs_ffmpeg: None):
    path = tmp_path / "wide.mkv"
    write_solid_clip(path, (70, 140, 210), seconds=0.6, fps=10, size="800x450")
    probed = probe_video(path)
    assert (probed.width, probed.height) == (800, 450)
    assert probed.fps == pytest.approx(10.0)
    frames = list(iter_sampled_frames(path, 2, probed.duration))
    assert frames
    assert frames[0].shape == (360, 640, 3)
    # Blue daylight survives the sample pipe.
    assert frames[0][..., 2].mean() > frames[0][..., 0].mean()


def test_entrypoint_version():
    script = Path(sys.executable).with_name("clip-index")
    if not script.exists():
        pytest.skip("console script is created by an editable install")
    import subprocess

    result = subprocess.run([str(script), "--version"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "clip-index" in result.stdout
