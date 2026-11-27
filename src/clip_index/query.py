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
    # autolog namespace (written by um-autolog `autolog clip-index`)
    sport: str | None = None
    scene: str | None = None
    text: str | None = None


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
    if query.sport or query.scene or query.text:
        return _matches_autolog(record.get("autolog"), query)
    return True


def _matches_autolog(autolog: object, query: Query) -> bool:
    """Filters on um-autolog's namespace. Clips AutoLog has not seen never match."""
    if not isinstance(autolog, dict):
        return False
    if query.sport and str(autolog.get("sport") or "").upper() != query.sport.upper():
        return False
    if query.scene and str(autolog.get("scene_type") or "").lower() != query.scene.lower():
        return False
    if query.text and query.text.lower() not in autolog_text(autolog):
        return False
    return True


def autolog_text(autolog: dict) -> str:
    """Lower-cased searchable text: summary, tags, and event tags/text."""
    parts: list[str] = [str(autolog.get("summary") or "")]
    tags = autolog.get("tags")
    if isinstance(tags, list):
        parts.extend(str(tag) for tag in tags)
    events = autolog.get("events")
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict):
                parts.append(str(event.get("tag") or ""))
                parts.append(str(event.get("text") or ""))
    return " ".join(part for part in parts if part).lower()


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
    text = f"{header}\n{detail}\n  calm: {calm_text}"
    autolog = record.get("autolog")
    if isinstance(autolog, dict):
        bits = [f"sport={autolog.get('sport') or ''}", f"scene={autolog.get('scene_type') or ''}"]
        summary = str(autolog.get("summary") or "").strip()
        if summary:
            bits.append(summary)
        text += "\n  autolog: " + "  ".join(bits)
    return text


def _fmt_span(span: object) -> str:
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return "?"
    try:
        start = float(span[0])
        end = float(span[1])
    except (TypeError, ValueError):
        return "?"
    return f"{start:.2f}-{end:.2f}s"
