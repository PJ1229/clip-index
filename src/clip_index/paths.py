"""Path rules shared by scan and the social.project / social.graded fields."""

from __future__ import annotations

from pathlib import PurePosixPath

from clip_index.schema import VIDEO_EXTENSIONS


def is_personal_segment(name: str) -> bool:
    """A folder segment named personal, in any case, is never indexed."""
    return name.lower() == "personal"


def is_skipped_dir(name: str) -> bool:
    """Dot directories and zip leftovers are drive metadata, not footage."""
    return name.startswith(".") or name == "__MACOSX" or is_personal_segment(name)


def is_video_filename(name: str) -> bool:
    if not name or name.startswith("."):
        return False
    return PurePosixPath(name).suffix.lower() in VIDEO_EXTENSIONS


def project_name(rel_posix: str, root_name: str) -> str:
    """Top-level folder under the scan root, or the root's own name."""
    parts = PurePosixPath(rel_posix).parts
    if len(parts) >= 2:
        return parts[0]
    return root_name


def graded_from_path(rel_posix: str) -> bool:
    """True when a parent folder looks like a graded or renders-proxy bin.

    The filename itself does not count. Matching is case-insensitive and
    uses substrings so names like ``05_renders_proxies`` and ``02_graded``
    both count. ``personal_notes`` is not personal, and ``grades`` is not
    graded.
    """
    parents = PurePosixPath(rel_posix).parts[:-1]
    for part in parents:
        low = part.lower()
        if "graded" in low or "renders_proxies" in low:
            return True
    return False
