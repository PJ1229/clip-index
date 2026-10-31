from __future__ import annotations

from pathlib import Path

from clip_index.paths import graded_from_path, is_video_filename, project_name
from clip_index.scan import discover


def test_project_is_top_level_folder():
    assert project_name("December/camera/A001.mov", "External Drive") == "December"
    assert project_name("clip.mov", "External Drive") == "External Drive"


def test_graded_matches_folder_names_only():
    assert graded_from_path("December/05_renders_proxies/A001.mov") is True
    assert graded_from_path("December/02_graded/A001.mov") is True
    assert graded_from_path("December/Graded/A001.mov") is True
    assert graded_from_path("December/camera/A001.mov") is False
    assert graded_from_path("December/graded_shot.mov") is False
    assert graded_from_path("my_grades/A001.mov") is False
    assert graded_from_path("December/proxies/A001.mov") is False


def test_video_names_skip_appledouble_and_odd_types():
    assert is_video_filename("A001.MOV") is True
    assert is_video_filename("reel.mxf") is True
    assert is_video_filename("._A001.mov") is False
    assert is_video_filename(".DS_Store") is False
    assert is_video_filename("notes.txt") is False


def test_discover_skips_personal_hidden_and_symlinks(tmp_path: Path):
    root = tmp_path / "External Drive"
    (root / "personal").mkdir(parents=True)
    (root / "personal" / "secret.mov").write_bytes(b"x")
    (root / "December" / "Personal").mkdir(parents=True)
    (root / "December" / "Personal" / "nope.MOV").write_bytes(b"x")
    (root / "December" / "personal_notes").mkdir(parents=True)
    (root / "December" / "personal_notes" / "keep.mp4").write_bytes(b"x")
    (root / "Décembre").mkdir()
    (root / "Décembre" / "prise.mkv").write_bytes(b"x")
    (root / "December" / "._junk.mov").write_bytes(b"x")
    (root / ".Spotlight-V100").mkdir()
    (root / ".Spotlight-V100" / "index.mov").write_bytes(b"x")
    (root / "December" / "shot.MXF").write_bytes(b"x")
    (root / "read.me").write_bytes(b"hi")
    link = root / "December" / "link.mov"
    link.symlink_to(root / "December" / "shot.MXF")

    videos, personal = discover(root)
    rels = [rel for _, rel in videos]
    assert rels == [
        "December/personal_notes/keep.mp4",
        "December/shot.MXF",
        "Décembre/prise.mkv",
    ]
    assert personal == ["December/Personal", "personal"]
