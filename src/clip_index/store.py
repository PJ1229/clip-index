"""Atomic, locked index.jsonl shared with um-autolog.

clip-index writes core media fields and the ``social`` object only.
Every other top-level key (``autolog``, and anything we do not know)
is copied through unchanged.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from clip_index.schema import CORE_FIELDS, MTIME_TOLERANCE, SOCIAL_FIELDS


class IndexCorrupt(RuntimeError):
    """index.jsonl cannot be parsed. The file is left untouched."""


class ScanError(RuntimeError):
    """The scan cannot start, or a write would touch the source drive."""


def thumb_key(rel_posix: str) -> str:
    """Hex SHA-1 of the UTF-8 relative path. Stable id for thumbs/<key>/."""
    return hashlib.sha1(rel_posix.encode("utf-8")).hexdigest()


def merge_record(existing: dict | None, incoming: dict) -> dict:
    """Overlay core fields and social onto existing. Ignore other incoming keys.

    Foreign namespaces already stored on ``existing`` stay as they are, even
    when ``incoming`` carries an ``autolog`` block by mistake. A brand-new
    record receives only core fields and ``social``.
    """
    for key in CORE_FIELDS:
        if key not in incoming:
            raise ValueError(f"incoming record is missing {key}")
    if "social" not in incoming or not isinstance(incoming["social"], dict):
        raise ValueError("incoming record is missing social")

    merged: dict = dict(existing or {})
    for key in CORE_FIELDS:
        merged[key] = incoming[key]
    social = dict(incoming["social"])
    ordered_social: dict = {}
    for key in SOCIAL_FIELDS:
        if key in social:
            ordered_social[key] = social[key]
    for key, value in social.items():
        if key not in ordered_social:
            ordered_social[key] = value
    merged["social"] = ordered_social
    return _ordered_record(merged)


def _ordered_record(record: dict) -> dict:
    ordered: dict = {}
    for key in CORE_FIELDS:
        if key in record:
            ordered[key] = record[key]
    if "social" in record:
        ordered["social"] = record["social"]
    for key, value in record.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def file_unchanged(record: dict, size: int, mtime: float) -> bool:
    if record.get("size") != size:
        return False
    try:
        stored = float(record["mtime"])
    except (KeyError, TypeError, ValueError):
        return False
    return abs(stored - mtime) <= MTIME_TOLERANCE


def render_jsonl(records: list[dict]) -> str:
    lines = [
        json.dumps(_ordered_record(rec), ensure_ascii=False, separators=(",", ":"))
        for rec in records
    ]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def parse_jsonl(text: str) -> list[dict]:
    records: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise IndexCorrupt(f"index.jsonl line {lineno}: {exc}") from exc
        if not isinstance(obj, dict) or not isinstance(obj.get("path"), str):
            raise IndexCorrupt(f"index.jsonl line {lineno}: record needs a string path")
        records.append(obj)
    return _collapse(records)


def _collapse(records: list[dict]) -> list[dict]:
    """Keep the last copy of a duplicated path, in the original first-seen order."""
    order: list[str] = []
    by_path: dict[str, dict] = {}
    for rec in records:
        key = rec["path"]
        if key not in by_path:
            order.append(key)
        by_path[key] = rec
    return [by_path[key] for key in order]


class IndexStore:
    """The index directory. Never points at a path inside the scan root."""

    def __init__(self, index_dir: Path, scan_root: Path | None = None) -> None:
        self.dir = index_dir.expanduser().resolve()
        self.scan_root = scan_root.expanduser().resolve() if scan_root is not None else None
        self.path = self.dir / "index.jsonl"
        self.lock_path = self.dir / "index.lock"
        self.meta_path = self.dir / "meta.json"

    def assert_outside_root(self) -> None:
        self._check_safe(self.dir)

    def ensure(self) -> None:
        """Create the index directory. Refuses to create it on the scan root."""
        self.assert_outside_root()
        self._check_safe(self.dir / "thumbs")
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "thumbs").mkdir(exist_ok=True)

    def thumb_dir(self, rel_posix: str) -> Path:
        dest = self.dir / "thumbs" / thumb_key(rel_posix)
        self._check_safe(dest)
        return dest

    def load_existing(self) -> list[dict]:
        """Read the index. A missing file is an empty library, not an error.

        Does not create the index directory.
        """
        if not self.path.is_file():
            return []
        with self._lock(fcntl.LOCK_SH):
            return self._read()

    def get(self, rel_posix: str) -> dict | None:
        if not self.path.is_file():
            return None
        with self._lock(fcntl.LOCK_SH):
            for rec in self._read():
                if rec.get("path") == rel_posix:
                    return rec
        return None

    def update_path(self, incoming: dict) -> None:
        """Reload, merge this clip, and atomically replace index.jsonl."""
        path = incoming["path"]
        with self._lock(fcntl.LOCK_EX):
            records = self._read()
            found = False
            for index, rec in enumerate(records):
                if rec.get("path") == path:
                    records[index] = merge_record(rec, incoming)
                    found = True
                    break
            if not found:
                records.append(merge_record(None, incoming))
            self._write(records)

    def read_meta(self) -> dict | None:
        if not self.meta_path.is_file():
            return None
        try:
            data = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def write_meta(self, root: Path) -> None:
        payload = {
            "schema_version": 1,
            "tool": "clip-index",
            "root": str(root),
        }
        text = json.dumps(payload, indent=2) + "\n"
        with self._lock(fcntl.LOCK_EX):
            self._atomic_write(self.meta_path, text)

    def _read(self) -> list[dict]:
        if not self.path.is_file():
            return []
        return parse_jsonl(self.path.read_text(encoding="utf-8"))

    def _render(self, records: list[dict]) -> str:
        return render_jsonl(records)

    def _write(self, records: list[dict]) -> None:
        self._atomic_write(self.path, self._render(records))

    def _atomic_write(self, path: Path, text: str) -> None:
        self._check_safe(path)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        self._check_safe(tmp)
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

    @contextmanager
    def _lock(self, mode: int) -> Iterator[None]:
        """Advisory flock on index.lock. The directory must already exist."""
        self._check_safe(self.lock_path)
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, mode)
            if mode == fcntl.LOCK_EX:
                os.ftruncate(fd, 0)
                os.write(fd, f"{os.getpid()}\n".encode())
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _check_safe(self, path: Path) -> None:
        if self.scan_root is None:
            return
        root = self.scan_root
        try:
            target = path.resolve()
        except OSError:
            target = path
        if target == root or root in target.parents:
            raise ScanError(
                f"Refusing to write {target} because it is inside the scan root {root}. "
                "The source drive is read-only; choose an index directory on the internal disk."
            )
