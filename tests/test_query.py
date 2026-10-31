from __future__ import annotations

import json
from pathlib import Path

import pytest

from clip_index.cli import main
from clip_index.query import Query, format_text, matches
from clip_index.store import IndexStore


def _record(
    path: str,
    *,
    lighting: str = "day",
    motion: str = "static",
    brightness: float = 0.4,
    face: bool = False,
    fraction: float = 0.0,
    project: str = "January",
    graded: bool = False,
    calm: list | None = None,
    tags: list[str] | None = None,
) -> dict:
    social = {
        "project": project,
        "graded": graded,
        "subject": "person" if face else "scenery",
        "face_visible": {"visible": face, "fraction": fraction},
        "lighting": lighting,
        "brightness": brightness,
        "camera_motion": motion,
        "calm_moments": [[1.0, 3.5], [6.0, 8.5]] if calm is None else calm,
        "tags": tags
        if tags is not None
        else [lighting, motion, "person" if face else "scenery"],
    }
    return {
        "path": path,
        "size": 100,
        "mtime": 1.0,
        "duration": 12.5,
        "fps": 24.0,
        "resolution": "1920x1080",
        "social": social,
        "autolog": {"scene": path},
    }


def _library() -> list[dict]:
    return [
        _record(
            "December/05_renders_proxies/hero.mov",
            lighting="golden",
            motion="static",
            brightness=0.62,
            face=True,
            fraction=0.4,
            project="December",
            graded=True,
            tags=["golden", "static", "person", "graded", "face", "calm"],
        ),
        _record(
            "December/camera/night.mov",
            lighting="night",
            motion="handheld",
            brightness=0.08,
            project="December",
        ),
        _record(
            "January/day.mov",
            lighting="day",
            motion="pan",
            brightness=0.7,
            project="January",
        ),
        {"path": "notes/only-autolog.mov", "autolog": {"scene": "x"}},
    ]


def test_filters_combine_and_ignore_records_without_social():
    rows = _library()
    hero = rows[0]
    query = Query(
        face=True,
        lighting="golden",
        motion="static",
        min_brightness=0.5,
        project="december",
        graded=True,
    )
    assert [row["path"] for row in rows if matches(row, query)] == [
        "December/05_renders_proxies/hero.mov"
    ]
    assert matches(hero, Query(min_brightness=0.62)) is True
    assert matches(hero, Query(min_brightness=0.63)) is False
    assert matches(hero, Query(tag="face")) is True
    assert matches(hero, Query(tag="night")) is False
    assert matches(hero, Query(project="emb")) is True
    assert matches(rows[2], Query(project="december")) is False
    text = format_text([hero])
    assert "December/05_renders_proxies/hero.mov" in text
    assert "1.00-3.50s" in text
    assert "6.00-8.50s" in text
    assert "calm:" in text


def test_query_cli_text_and_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    store = IndexStore(tmp_path)
    store.ensure()
    lines = [json.dumps(row, ensure_ascii=False) for row in _library()]
    store.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    code = main(
        [
            "query",
            "--index",
            str(tmp_path),
            "--face",
            "--lighting",
            "golden",
            "--motion",
            "static",
            "--min-brightness",
            "0.5",
            "--project",
            "december",
            "--graded",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "hero.mov" in printed
    assert "1.00-3.50s" in printed
    assert "night.mov" not in printed
    assert "only-autolog" not in printed

    code = main(["query", "--index", str(tmp_path), "--lighting", "golden", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [row["path"] for row in payload] == ["December/05_renders_proxies/hero.mov"]
    assert payload[0]["autolog"]["scene"] == "December/05_renders_proxies/hero.mov"
    assert payload[0]["social"]["calm_moments"] == [[1.0, 3.5], [6.0, 8.5]]


def test_query_missing_index_does_not_create_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    missing = tmp_path / "nope"
    code = main(["query", "--index", str(missing), "--json"])
    assert code == 0
    assert json.loads(capsys.readouterr().out) == []
    assert not missing.exists()


def test_query_corrupt_index_exits_2(tmp_path: Path):
    (tmp_path / "index.jsonl").write_text("{not json\n", encoding="utf-8")
    code = main(["query", "--index", str(tmp_path), "--json"])
    assert code == 2
    assert (tmp_path / "index.jsonl").read_text(encoding="utf-8") == "{not json\n"


def test_help_and_version():
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0


def test_min_brightness_range(tmp_path: Path):
    code = main(["query", "--index", str(tmp_path), "--min-brightness", "2"])
    assert code == 1
