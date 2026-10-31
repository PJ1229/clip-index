"""Shared record shape for clip-index and um-autolog.

See SCHEMA.md. This module is the code-side list of field names; it does
not import OpenCV or touch the filesystem.
"""

from __future__ import annotations

SCHEMA_VERSION = 1

CORE_FIELDS = (
    "path",
    "size",
    "mtime",
    "duration",
    "fps",
    "resolution",
)

SOCIAL_FIELDS = (
    "project",
    "graded",
    "subject",
    "face_visible",
    "lighting",
    "brightness",
    "camera_motion",
    "calm_moments",
    "tags",
)

LIGHTING = ("golden", "day", "night", "snow", "overcast")
MOTION = ("static", "handheld", "pan")
SUBJECTS = ("person", "scenery", "object")

# A file is unchanged when size matches and mtime is within this many seconds.
MTIME_TOLERANCE = 0.001

VIDEO_EXTENSIONS = frozenset(
    {
        ".mov",
        ".mp4",
        ".m4v",
        ".mxf",
        ".mkv",
        ".avi",
        ".mts",
        ".m2ts",
        ".webm",
        ".mpg",
        ".mpeg",
    }
)


def social_complete(social: object) -> bool:
    """True when clip-index can skip reanalysis of an existing social block."""
    if not isinstance(social, dict):
        return False
    if any(key not in social for key in SOCIAL_FIELDS):
        return False
    face = social["face_visible"]
    if not isinstance(face, dict):
        return False
    if "visible" not in face or "fraction" not in face:
        return False
    if social["lighting"] not in LIGHTING:
        return False
    if social["camera_motion"] not in MOTION:
        return False
    if social["subject"] not in SUBJECTS:
        return False
    if not isinstance(social["graded"], bool):
        return False
    if not isinstance(social["calm_moments"], list):
        return False
    if not isinstance(social["tags"], list):
        return False
    return True
