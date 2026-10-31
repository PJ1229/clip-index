"""Turn one clip into core fields plus a social object, and pick thumbnails."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from clip_index.classify import build_tags, classify_frames
from clip_index.media import ClipReadError, iter_sampled_frames, probe_video
from clip_index.paths import graded_from_path, project_name
from clip_index.store import IndexStore


def analyze_clip(
    video: Path,
    *,
    rel_path: str,
    root: Path,
    store: IndexStore,
    sample_fps: float,
    size: int,
    mtime: float,
    clip=None,
    face_detector=None,
) -> dict:
    """Read ``video`` and return a record of core fields plus ``social``.

    Thumbnails are written under the index directory only after classification
    succeeds. The source path is opened for reading by ffmpeg and never for
    writing.
    """
    probe = probe_video(video)
    thumb_slots = _thumb_targets(probe.duration, sample_fps)
    chosen: dict[int, tuple[float, np.ndarray]] = {}

    def frames():
        for index, frame in enumerate(iter_sampled_frames(video, sample_fps, probe.duration)):
            timestamp = index / sample_fps
            for slot, target in enumerate(thumb_slots):
                if slot in chosen:
                    continue
                if timestamp + 1e-6 >= target:
                    chosen[slot] = (timestamp, frame.copy())
            yield frame

    try:
        visual = classify_frames(
            frames(),
            sample_fps=sample_fps,
            duration=probe.duration,
            face_detector=face_detector,
            clip=clip,
        )
    except ValueError as exc:
        raise ClipReadError(f"no frames sampled from {rel_path}") from exc
    social = {
        "project": project_name(rel_path, root.name),
        "graded": graded_from_path(rel_path),
        "subject": visual["subject"],
        "face_visible": visual["face_visible"],
        "lighting": visual["lighting"],
        "brightness": visual["brightness"],
        "camera_motion": visual["camera_motion"],
        "calm_moments": visual["calm_moments"],
    }
    social["tags"] = build_tags(social)
    _write_thumbs(store, rel_path, [chosen[k] for k in sorted(chosen)])
    return {
        "path": rel_path,
        "size": int(size),
        "mtime": float(mtime),
        "duration": float(probe.duration),
        "fps": float(probe.fps),
        "resolution": f"{probe.width}x{probe.height}",
        "social": social,
    }


def _thumb_targets(duration: float, sample_fps: float) -> list[float]:
    if duration <= 0:
        return [0.0]
    count = 5 if duration >= 2 else 1
    last = max(0.0, duration - (1.0 / sample_fps))
    if count == 1:
        return [0.0]
    return [last * i / (count - 1) for i in range(count)]


def _write_thumbs(store: IndexStore, rel_path: str, thumbs: list[tuple[float, np.ndarray]]) -> None:
    dest = store.thumb_dir(rel_path)
    dest.mkdir(parents=True, exist_ok=True)
    keep: list[str] = []
    for timestamp, frame in thumbs:
        name = f"t_{timestamp:.2f}.jpg"
        target = dest / name
        store._check_safe(target)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(target), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85]):
            raise ClipReadError(f"could not write thumbnail {target}")
        keep.append(name)
    for old in dest.iterdir():
        if old.is_file() and old.name not in keep:
            old.unlink()
