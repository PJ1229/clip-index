"""Walk a footage folder and refresh the shared index.

The scan root is only ever stat'd and opened for read. Personal folders
are not descended into. A clip whose size and mtime are unchanged, and
whose social block is complete, is left alone so a re-run resumes.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from clip_index.analyze import analyze_clip
from clip_index.clip_model import ClipZeroShot, clip_requested
from clip_index.media import ClipReadError
from clip_index.paths import is_personal_segment, is_skipped_dir, is_video_filename
from clip_index.schema import MTIME_TOLERANCE, social_complete
from clip_index.store import IndexStore, ScanError, file_unchanged


@dataclass
class ScanStats:
    analyzed: int = 0
    unchanged: int = 0
    failed: int = 0
    personal_dirs: list[str] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)
    analyzed_paths: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def discover(root: Path) -> tuple[list[tuple[Path, str]], list[str]]:
    """Return ``(videos, personal_dirs)`` as paths relative to ``root``.

    Symlinks are skipped so a link cannot pull the walk off the drive.
    """
    videos: list[tuple[Path, str]] = []
    personal_dirs: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        kept: list[str] = []
        for name in dirnames:
            if is_personal_segment(name):
                rel = Path(dirpath, name).relative_to(root).as_posix()
                personal_dirs.append(rel)
                continue
            if is_skipped_dir(name):
                continue
            kept.append(name)
        dirnames[:] = kept
        for name in filenames:
            if not is_video_filename(name):
                continue
            full = Path(dirpath) / name
            try:
                if full.is_symlink() or not full.is_file():
                    continue
            except OSError:
                continue
            rel = full.relative_to(root).as_posix()
            if any(is_personal_segment(part) for part in rel.split("/")):
                continue
            videos.append((full, rel))
    videos.sort(key=lambda item: item[1])
    personal_dirs.sort()
    return videos, personal_dirs


def scan(
    root: Path,
    index_dir: Path,
    *,
    sample_fps: float = 2.0,
    force: bool = False,
    use_clip: bool = False,
    analyzer=analyze_clip,
) -> ScanStats:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ScanError(f"Scan root is not a directory: {root}")
    if sample_fps <= 0 or sample_fps > 30:
        raise ScanError("sample fps must be between 0 and 30")

    store = IndexStore(index_dir, scan_root=root)
    # Refuse before creating directories or reading footage past the check.
    store.assert_outside_root()
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise ScanError("ffmpeg and ffprobe must be on PATH. On a Mac: brew install ffmpeg")
    store.ensure()

    previous = store.read_meta()
    if not previous or previous.get("root") != str(root) or previous.get("schema_version") != 1:
        store.write_meta(root)
    stats = ScanStats()
    note_root = previous.get("root") if previous else None
    if note_root and note_root != str(root):
        stats.notes.append(
            f"index was last scanned from {note_root}; this root is {root}. "
            "Relative paths are still the record keys."
        )

    if is_personal_segment(root.name):
        stats.personal_dirs.append(".")
        stats.notes.append("scan root is named personal; nothing was indexed")
        return stats

    videos, personal_dirs = discover(root)
    stats.personal_dirs = personal_dirs

    clip = None
    if clip_requested(use_clip):
        clip = ClipZeroShot(store.dir, enabled=True)
        if not clip.available and clip.reason:
            stats.notes.append(clip.reason)

    for full, rel in videos:
        try:
            st = full.stat()
        except OSError as exc:
            stats.failed += 1
            stats.failures.append((rel, str(exc)))
            continue
        if st.st_size <= 0:
            stats.failed += 1
            stats.failures.append((rel, "empty file"))
            continue
        existing = store.get(rel)
        if (
            not force
            and existing is not None
            and file_unchanged(existing, st.st_size, st.st_mtime)
            and social_complete(existing.get("social"))
        ):
            stats.unchanged += 1
            continue
        try:
            incoming = analyzer(
                full,
                rel_path=rel,
                root=root,
                store=store,
                sample_fps=sample_fps,
                size=st.st_size,
                mtime=st.st_mtime,
                clip=clip,
            )
        except (ClipReadError, OSError) as exc:
            stats.failed += 1
            stats.failures.append((rel, str(exc)))
            continue
        try:
            st2 = full.stat()
        except OSError as exc:
            stats.failed += 1
            stats.failures.append((rel, str(exc)))
            continue
        if st2.st_size != st.st_size or abs(st2.st_mtime - st.st_mtime) > MTIME_TOLERANCE:
            stats.failed += 1
            stats.failures.append((rel, "file changed during read; will retry next scan"))
            continue
        store.update_path(incoming)
        stats.analyzed += 1
        stats.analyzed_paths.append(rel)
    return stats
