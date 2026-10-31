"""Filter the shared index and print the clips an editor asked for."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from clip_index.store import IndexStore


@dataclass
class Query:
    face: bool = False
    lighting: str | None = None
    motion: str | None = None
    min_brightness: float | None = None
    project: str | None = None
    graded: bool = False
    tag: str | None = None


def load_records(index_dir: Path) -> list[dict]:
    store = IndexStore(index_dir, scan_root=None)
    return store.load_existing()


def matches(record: dict, query: Query) -> bool:
    social = record.get("social")
    if not isinstance(social, dict):
        return False
    if query.face:
        face = social.get("face_visible")
        if not isinstance(face, dict) or not face.get("visible"):
            return False
    if query.lighting and social.get("lighting") != query.lighting:
        return False
    if query.motion and social.get("camera_motion") != query.motion:
        return False
    if query.min_brightness is not None:
        try:
            brightness = float(social.get("brightness"))
        except (TypeError, ValueError):
            return False
        if brightness < query.min_brightness:
            return False
    if query.project:
        project = str(social.get("project") or "")
        if query.project.lower() not in project.lower():
            return False
    if query.graded and social.get("graded") is not True:
        return False
    if query.tag:
        tags = social.get("tags") or []
        if not isinstance(tags, list):
            return False
        wanted = query.tag.lower()
        if wanted not in {str(tag).lower() for tag in tags}:
            return False
    return True


def run_query(index_dir: Path, query: Query) -> list[dict]:
    found = [rec for rec in load_records(index_dir) if matches(rec, query)]
    found.sort(key=lambda rec: rec.get("path") or "")
    return found


def format_text(records: list[dict]) -> str:
    if not records:
        return "No matching clips."
    blocks = [_format_one(rec) for rec in records]
    return "\n\n".join(blocks)


def format_json(records: list[dict]) -> str:
    return json.dumps(records, indent=2, ensure_ascii=False)


def _format_one(record: dict) -> str:
    social = record.get("social") or {}
    face = social.get("face_visible") or {}
    visible = bool(face.get("visible"))
    try:
        fraction = float(face.get("fraction") or 0)
    except (TypeError, ValueError):
        fraction = 0.0
    try:
        brightness = float(social.get("brightness") or 0)
    except (TypeError, ValueError):
        brightness = 0.0
    try:
        duration = float(record.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    calm = social.get("calm_moments") or []
    calm_text = ", ".join(_fmt_span(span) for span in calm) or "none"
    face_text = f"yes ({fraction:.0%})" if visible else f"no ({fraction:.0%})"
    graded = "yes" if social.get("graded") else "no"
    header = record.get("path") or ""
    detail = (
        f"  project={social.get('project') or ''}  graded={graded}  "
        f"subject={social.get('subject') or ''}  lighting={social.get('lighting') or ''}  "
        f"motion={social.get('camera_motion') or ''}  brightness={brightness:.2f}  "
        f"face={face_text}  {duration:.1f}s"
    )
    return f"{header}\n{detail}\n  calm: {calm_text}"


def _fmt_span(span: object) -> str:
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return "?"
    try:
        start = float(span[0])
        end = float(span[1])
    except (TypeError, ValueError):
        return "?"
    return f"{start:.2f}-{end:.2f}s"
