from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from clip_index.classify import (
    FACE_SUBJECT_MIN,
    HaarFaceDetector,
    build_tags,
    camera_motion_from_vectors,
    classify_frames,
    find_calm_moments,
    frame_features,
    label_lighting,
    label_subject_frame,
    majority,
    measure_flow,
)
from clip_index.clip_model import (
    LIGHTING_PROMPTS,
    SUBJECT_PROMPTS,
    ClipZeroShot,
    find_checkpoint,
    pick_best,
    pick_device,
)
from clip_index.schema import LIGHTING, SUBJECTS


class NoFace:
    def frame_has_face(self, _rgb) -> bool:
        return False


class AlwaysFace:
    def frame_has_face(self, _rgb) -> bool:
        return True


def _solid(color: tuple[int, int, int], shape: tuple[int, int] = (90, 160)) -> np.ndarray:
    frame = np.zeros((*shape, 3), dtype=np.uint8)
    frame[..., 0] = color[0]
    frame[..., 1] = color[1]
    frame[..., 2] = color[2]
    return frame


def test_lighting_labels_cover_the_five_conditions():
    assert label_lighting(frame_features(_solid((8, 8, 10)))) == "night"
    assert label_lighting(frame_features(_solid((245, 245, 248)))) == "snow"
    assert label_lighting(frame_features(_solid((220, 150, 60)))) == "golden"
    assert label_lighting(frame_features(_solid((150, 150, 155)))) == "overcast"
    assert label_lighting(frame_features(_solid((70, 140, 210)))) == "day"


def test_lighting_tie_prefers_the_more_specific_label():
    assert majority(["day", "golden"]) == "golden"
    assert majority(["day", "day", "golden"]) == "day"


def test_subject_from_pixels_and_from_faces():
    horizon = _solid((40, 90, 210))
    horizon[30:] = (50, 130, 60)
    assert label_subject_frame(horizon) == "scenery"
    assert label_subject_frame(_solid((220, 150, 60))) == "scenery"

    rng = np.random.default_rng(1)
    obj = np.full((90, 160, 3), 160, dtype=np.uint8)
    obj[30:60, 55:105] = rng.integers(0, 255, size=(30, 50, 3), dtype=np.uint8)
    assert label_subject_frame(obj) == "object"

    dark = _solid((5, 5, 6), (48, 64))
    out = classify_frames(
        [dark, dark, dark, dark],
        sample_fps=2,
        duration=4,
        face_detector=AlwaysFace(),
    )
    assert out["subject"] == "person"
    assert out["face_visible"] == {"visible": True, "fraction": 1.0}
    assert out["lighting"] == "night"
    assert out["brightness"] < 0.1

    # One hit in twenty is under the person threshold but still visible.
    hits = {"n": 0}

    def once(_frame):
        hits["n"] += 1
        return hits["n"] == 1

    rare = classify_frames(
        [dark] * 20,
        sample_fps=2,
        duration=10,
        face_detector=type("Det", (), {"frame_has_face": staticmethod(once)})(),
    )
    assert rare["face_visible"]["visible"] is True
    assert rare["face_visible"]["fraction"] == 0.05
    assert rare["face_visible"]["fraction"] < FACE_SUBJECT_MIN
    assert rare["subject"] == "scenery"


def test_motion_labels_from_flow_vectors():
    assert camera_motion_from_vectors([(0.1, 0.0)] * 5) == "static"
    assert camera_motion_from_vectors([(4.0, 0.2)] * 6) == "pan"
    handheld = [(5, 0), (-4, 1), (0, 6), (-2, -5), (3, -4), (-6, 2), (1, 5), (-3, -3)]
    assert camera_motion_from_vectors(handheld) == "handheld"


def test_optical_flow_on_shifted_frames():
    rng = np.random.default_rng(4)
    canvas = rng.integers(0, 255, size=(140, 260), dtype=np.uint8)

    def frame(dy: int, dx: int) -> np.ndarray:
        gray = canvas[20 + dy : 110 + dy, 40 + dx : 200 + dx]
        return np.dstack([gray, gray, gray])

    static = [frame(0, 0) for _ in range(6)]
    pan = [frame(0, dx) for dx in (0, 4, 8, 12, 16, 20)]
    shifts = [(0, 0), (6, -4), (-6, 5), (5, 6), (-5, -6), (7, -2), (-3, 7)]
    shaky = [frame(dy, dx) for dy, dx in shifts]

    static_flow = measure_flow(static)
    pan_flow = measure_flow(pan)
    shaky_flow = measure_flow(shaky)
    assert camera_motion_from_vectors(static_flow) == "static", static_flow
    assert camera_motion_from_vectors(pan_flow) == "pan", pan_flow
    assert camera_motion_from_vectors(shaky_flow) == "handheld", shaky_flow


def test_calm_moments_are_still_and_avoid_pans():
    samples = [(t, 0.0, 0.0) for t in [i * 0.5 for i in range(1, 121)]]
    moments = find_calm_moments(samples, 60)
    assert len(moments) == 3
    for start, end in moments:
        assert 2.0 <= end - start <= 3.0 + 1e-6
    for earlier, later in zip(moments, moments[1:]):
        assert earlier[1] <= later[0] + 1e-6

    assert find_calm_moments([(0.5, 5.0, 0.0), (1.0, 5.0, 0.0)], 4) == []
    assert find_calm_moments([(0.25, 0.0, 0.0)], 1.5) == []

    mixed = []
    t = 0.25
    while t <= 9:
        if 3.25 <= t <= 5.75:
            flip = 5.0 if int(t * 4) % 2 == 0 else -4.0
            mixed.append((round(t, 2), flip, 1.0))
        else:
            mixed.append((round(t, 2), 0.0, 0.0))
        t += 0.25
    calm = find_calm_moments(mixed, 9)
    assert calm
    for start, end in calm:
        assert end <= 3.26 or start >= 5.74
        assert 2.0 <= end - start <= 3.0 + 1e-6


def test_classify_static_golden_clip_reports_calm_stretches():
    frame = _solid((220, 150, 60))
    out = classify_frames(
        [frame] * 8,
        sample_fps=2,
        duration=4,
        face_detector=NoFace(),
    )
    assert out["lighting"] == "golden"
    assert out["camera_motion"] == "static"
    assert out["subject"] == "scenery"
    assert out["face_visible"]["visible"] is False
    assert out["calm_moments"]
    for start, end in out["calm_moments"]:
        assert 2.0 <= end - start <= 3.0 + 1e-6


def test_tags_follow_the_classification():
    social = {
        "lighting": "golden",
        "camera_motion": "static",
        "subject": "scenery",
        "graded": True,
        "face_visible": {"visible": False, "fraction": 0.0},
        "calm_moments": [[0.0, 2.5]],
    }
    assert build_tags(social) == ["golden", "static", "scenery", "graded", "calm"]


def test_haar_loads_and_ignores_a_blank_frame():
    detector = HaarFaceDetector()
    blank = np.zeros((80, 80, 3), dtype=np.uint8)
    assert detector.frame_has_face(blank) is False


def test_clip_prompts_match_the_schema_and_stay_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    assert set(LIGHTING_PROMPTS) == set(LIGHTING)
    assert set(SUBJECT_PROMPTS) == set(SUBJECTS)
    assert pick_best({"golden": 0.1, "day": 0.9}) == "day"
    assert pick_device(True, True) == "mps"
    assert pick_device(False, True) == "cuda"
    assert pick_device(False, False) == "cpu"

    monkeypatch.delenv("CLIP_INDEX_CLIP_CHECKPOINT", raising=False)
    assert find_checkpoint(tmp_path) is None
    def refuse_load(*_args, **_kwargs):
        raise AssertionError("checkpoint load must not run when no weights exist")

    clip = ClipZeroShot(tmp_path, enabled=True, loader=refuse_load)
    assert clip.available is False
    assert "download" in clip.reason.lower() or "checkpoint" in clip.reason.lower()

    weights = tmp_path / "models" / "ViT-B-32.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"not really weights")
    loaded: list[Path] = []

    def loader(path: Path, model_name: str = "ViT-B-32"):
        loaded.append(path)
        raise RuntimeError("fake weights")

    broken = ClipZeroShot(tmp_path, enabled=True, loader=loader)
    assert broken.available is False
    assert loaded == [weights]
    assert "heuristics" in broken.reason.lower()


def test_clip_labels_override_heuristics_when_valid():
    class SnowObject:
        available = True

        def lighting(self, _rgb):
            return "snow"

        def subject(self, _rgb):
            return "object"

    orange = _solid((220, 150, 60))
    out = classify_frames(
        [orange, orange],
        sample_fps=2,
        duration=2,
        face_detector=NoFace(),
        clip=SnowObject(),
    )
    assert out["lighting"] == "snow"
    assert out["subject"] == "object"

    class Garbage:
        available = True

        def lighting(self, _rgb):
            return "moonlight"

        def subject(self, _rgb):
            return "cat"

    fallback = classify_frames(
        [orange],
        sample_fps=2,
        duration=2,
        face_detector=NoFace(),
        clip=Garbage(),
    )
    assert fallback["lighting"] == "golden"
    assert fallback["subject"] == "scenery"


def test_faces_override_clip_subject():
    class Scenery:
        available = True

        def lighting(self, _rgb):
            return "day"

        def subject(self, _rgb):
            return "scenery"

    frame = _solid((70, 140, 210), (48, 64))
    out = classify_frames(
        [frame] * 4,
        sample_fps=2,
        duration=2,
        face_detector=AlwaysFace(),
        clip=Scenery(),
    )
    assert out["subject"] == "person"
    assert out["lighting"] == "day"
