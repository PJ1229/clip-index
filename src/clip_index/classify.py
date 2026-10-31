"""Lighting, subject, faces, and camera motion from sampled RGB frames.

Motion magnitudes are pixels of optical flow on a grayscale proxy at most
320px wide, between consecutive samples (not between original camera frames).
No frame leaves the machine.
"""

from __future__ import annotations

import bisect
import math
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence

import cv2
import numpy as np

from clip_index.schema import LIGHTING, SUBJECTS

# Tuned for the 320px-wide proxy. A tripod shot sits under STATIC_MAX;
# a coherent translation above that is a pan; anything busier is handheld.
STATIC_MAX = 0.6
PAN_COHERENCE = 0.75
CALM_WINDOW = 2.5
CALM_MIN = 2.0
CALM_MAX_SPAN = 3.0
CALM_KEEP = 3
CALM_STEP = 0.25
FACE_SUBJECT_MIN = 0.08

_LIGHT_PRIORITY = ("night", "snow", "golden", "overcast", "day")


class HaarFaceDetector:
    """OpenCV's bundled frontal-face cascade. No extra download."""

    def __init__(self) -> None:
        cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._clf = cv2.CascadeClassifier(cascade)
        if self._clf.empty():
            raise RuntimeError(f"could not load face cascade at {cascade}")

    def frame_has_face(self, rgb: np.ndarray) -> bool:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        found = self._clf.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(30, 30),
        )
        return len(found) > 0


_HAAR: HaarFaceDetector | None = None


def default_face_detector() -> HaarFaceDetector:
    global _HAAR
    if _HAAR is None:
        _HAAR = HaarFaceDetector()
    return _HAAR


def frame_features(rgb: np.ndarray) -> dict[str, float]:
    pixels = rgb.astype(np.float32)
    red = float(pixels[..., 0].mean())
    green = float(pixels[..., 1].mean())
    blue = float(pixels[..., 2].mean())
    brightness = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255.0
    peak = max(red, green, blue)
    floor = min(red, green, blue)
    saturation = 0.0 if peak < 1.0 else (peak - floor) / peak
    warmth = (red - blue) / 255.0
    return {
        "brightness": brightness,
        "saturation": saturation,
        "warmth": warmth,
        "r": red,
        "g": green,
        "b": blue,
    }


def label_lighting(features: dict[str, float]) -> str:
    brightness = features["brightness"]
    saturation = features["saturation"]
    warmth = features["warmth"]
    if brightness < 0.18:
        return "night"
    if brightness > 0.75 and saturation < 0.12:
        return "snow"
    if (
        warmth > 0.12
        and features["r"] > features["g"] > features["b"] * 0.95
        and brightness < 0.9
    ):
        return "golden"
    if saturation < 0.15 and 0.2 <= brightness <= 0.75:
        return "overcast"
    return "day"


def label_subject_frame(rgb: np.ndarray) -> str:
    """Scenery for horizons and quiet fields, otherwise a compact object.

    A face-based person decision is applied later, across the whole clip.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    height, width = gray.shape[:2]
    y0, y1 = height // 3, 2 * height // 3
    x0, x1 = width // 3, 2 * width // 3
    if y1 <= y0 or x1 <= x0:
        center = float(lap.mean())
        border = center
    else:
        center = float(lap[y0:y1, x0:x1].mean())
        mask = np.ones(lap.shape, dtype=bool)
        mask[y0:y1, x0:x1] = False
        border = float(lap[mask].mean()) if mask.any() else center
    upper = rgb[: max(height // 3, 1)].astype(np.float32).mean(axis=(0, 1))
    lower = rgb[min(2 * height // 3, height - 1) :].astype(np.float32).mean(axis=(0, 1))
    band_delta = float(np.abs(upper - lower).mean())
    texture = float(lap.mean())
    if center > 12.0 and center > border * 1.8 and band_delta < 35.0:
        return "object"
    if band_delta >= 20.0 or texture < 6.0:
        return "scenery"
    return "object"


def majority(labels: Sequence[str], priority: Sequence[str] = _LIGHT_PRIORITY) -> str:
    if not labels:
        raise ValueError("majority of an empty vote")
    counts = Counter(labels)
    best = max(counts.values())
    tied = {label for label, count in counts.items() if count == best}
    for label in priority:
        if label in tied:
            return label
    return sorted(tied)[0]


def to_flow_gray(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape[:2]
    if width > 320:
        new_h = max(2, int(round(height * (320 / width))))
        gray = cv2.resize(gray, (320, new_h), interpolation=cv2.INTER_AREA)
    return gray


def mean_flow_vector(prev_gray: np.ndarray, gray: np.ndarray) -> tuple[float, float]:
    levels = 3 if min(prev_gray.shape) >= 80 else 2
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        0.5,
        levels,
        21,
        3,
        5,
        1.1,
        0,
    )
    return float(flow[..., 0].mean()), float(flow[..., 1].mean())


def measure_flow(frames: Sequence[np.ndarray]) -> list[tuple[float, float]]:
    """Mean flow vector between each consecutive pair."""
    vectors: list[tuple[float, float]] = []
    prev = None
    for frame in frames:
        gray = to_flow_gray(frame)
        if prev is not None and prev.shape == gray.shape:
            vectors.append(mean_flow_vector(prev, gray))
        prev = gray
    return vectors


def coherence(vectors: Sequence[tuple[float, float]]) -> float:
    if not vectors:
        return 0.0
    magnitudes = [math.hypot(x, y) for x, y in vectors]
    mean_mag = sum(magnitudes) / len(magnitudes)
    if mean_mag < 1e-6:
        return 0.0
    mean_x = sum(x for x, _ in vectors) / len(vectors)
    mean_y = sum(y for _, y in vectors) / len(vectors)
    return math.hypot(mean_x, mean_y) / mean_mag


def camera_motion_from_vectors(vectors: Sequence[tuple[float, float]]) -> str:
    if not vectors:
        return "static"
    magnitudes = [math.hypot(x, y) for x, y in vectors]
    median = statistics.median(magnitudes)
    if median < STATIC_MAX:
        return "static"
    if coherence(vectors) >= PAN_COHERENCE:
        return "pan"
    return "handheld"


def find_calm_moments(
    samples: Sequence[tuple[float, float, float]],
    duration: float,
) -> list[list[float]]:
    """Steadiest non-overlapping 2–3s windows that are not pans.

    ``samples`` are ``(time, flow_x, flow_y)`` at the end of each interval.
    A window qualifies only when its median flow is in the static range,
    so a shaky clip does not get a "least bad" stretch labeled calm.
    """
    if duration < CALM_MIN or not samples:
        return []
    window = CALM_WINDOW if duration > CALM_MAX_SPAN else duration
    times = [sample[0] for sample in samples]
    starts: list[float] = []
    cursor = 0.0
    while cursor + CALM_MIN <= duration + 1e-6:
        starts.append(cursor)
        cursor += CALM_STEP
    tail = max(0.0, duration - window)
    if not starts or abs(starts[-1] - tail) > 1e-6:
        starts.append(tail)

    candidates: list[tuple[float, float, float]] = []
    for start in starts:
        end = min(start + window, duration)
        if end - start > CALM_MAX_SPAN:
            end = start + CALM_MAX_SPAN
        if end - start < CALM_MIN - 1e-6:
            continue
        left = bisect.bisect_left(times, start - 1e-6)
        right = bisect.bisect_right(times, end + 1e-6)
        chunk = samples[left:right]
        if not chunk:
            continue
        # One pan or shake interval disqualifies the window. Median would
        # hide a short move inside an otherwise still stretch.
        magnitudes = [math.hypot(fx, fy) for _, fx, fy in chunk]
        if any(mag >= STATIC_MAX for mag in magnitudes):
            continue
        candidates.append((statistics.median(magnitudes), start, end))

    candidates.sort(key=lambda item: (item[0], item[1]))
    chosen: list[tuple[float, float]] = []
    for _median, start, end in candidates:
        if len(chosen) >= CALM_KEEP:
            break
        overlaps = any(not (end <= prev_s + 1e-6 or start >= prev_e - 1e-6) for prev_s, prev_e in chosen)
        if overlaps:
            continue
        chosen.append((start, end))
    chosen.sort()
    return [[round(start, 3), round(end, 3)] for start, end in chosen]


def build_tags(social: dict) -> list[str]:
    tags: list[str] = []
    for value in (social.get("lighting"), social.get("camera_motion"), social.get("subject")):
        if isinstance(value, str) and value and value not in tags:
            tags.append(value)
    if social.get("graded") and "graded" not in tags:
        tags.append("graded")
    face = social.get("face_visible") or {}
    if face.get("visible") and "face" not in tags:
        tags.append("face")
    if social.get("calm_moments") and "calm" not in tags:
        tags.append("calm")
    return tags


def classify_frames(
    frames: Iterable[np.ndarray],
    *,
    sample_fps: float,
    duration: float,
    face_detector=None,
    clip=None,
) -> dict:
    """Visual half of a social record. Project, graded, and tags are added later."""
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    detector = face_detector if face_detector is not None else default_face_detector()
    use_clip = clip is not None and bool(getattr(clip, "available", False))
    total_est = max(1, int(round(duration * sample_fps)))
    clip_stride = max(1, total_est // 8)

    brightnesses: list[float] = []
    light_labels: list[str] = []
    subject_labels: list[str] = []
    face_hits = 0
    flow_samples: list[tuple[float, float, float]] = []
    prev_gray = None
    count = 0
    clip_used = 0

    for index, frame in enumerate(frames):
        count += 1
        timestamp = index / sample_fps
        features = frame_features(frame)
        brightnesses.append(features["brightness"])

        light = None
        subject = None
        if use_clip and index % clip_stride == 0 and clip_used < 8:
            light = _allowed(clip.lighting(frame), LIGHTING)
            subject = _allowed(clip.subject(frame), SUBJECTS)
            clip_used += 1
        light_labels.append(light or label_lighting(features))
        subject_labels.append(subject or label_subject_frame(frame))
        if detector.frame_has_face(frame):
            face_hits += 1

        gray = to_flow_gray(frame)
        if prev_gray is not None and prev_gray.shape == gray.shape:
            fx, fy = mean_flow_vector(prev_gray, gray)
            flow_samples.append((timestamp, fx, fy))
        prev_gray = gray

    if count == 0:
        raise ValueError("no frames to classify")

    fraction = face_hits / count
    face_visible = {"visible": face_hits > 0, "fraction": round(fraction, 4)}
    subject = majority(subject_labels, SUBJECTS)
    if fraction >= FACE_SUBJECT_MIN:
        subject = "person"
    vectors = [(fx, fy) for _, fx, fy in flow_samples]
    return {
        "subject": subject,
        "face_visible": face_visible,
        "lighting": majority(light_labels),
        "brightness": round(sum(brightnesses) / len(brightnesses), 4),
        "camera_motion": camera_motion_from_vectors(vectors),
        "calm_moments": find_calm_moments(flow_samples, duration),
    }


def _allowed(label: object, choices: tuple[str, ...]) -> str | None:
    if isinstance(label, str) and label in choices:
        return label
    return None
