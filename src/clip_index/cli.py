"""Command line for clip-index scan and clip-index query."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from clip_index import __version__
from clip_index.query import Query, format_json, format_text, run_query
from clip_index.scan import ScanStats, scan
from clip_index.schema import LIGHTING, MOTION
from clip_index.store import IndexCorrupt, ScanError


def default_index_dir() -> Path:
    env = os.environ.get("CLIP_INDEX_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / "Documents" / "clip-index").resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clip-index",
        description=(
            "Classify a filmmaker's clips on a local disk so an editor can find "
            "the usable shots. The scan root is never written to."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            '  clip-index scan "/Volumes/TOSHIBA EXT"\n'
            "  clip-index query --face --lighting golden --motion static "
            "--min-brightness 0.5 --project december --graded\n"
            "  clip-index query --lighting night --json\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"clip-index {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    scan_p = sub.add_parser("scan", help="index a folder of clips")
    scan_p.add_argument("root", type=Path, help="footage folder, read-only")
    scan_p.add_argument(
        "--index",
        type=Path,
        default=None,
        help="index directory (default ~/Documents/clip-index or $CLIP_INDEX_DIR)",
    )
    scan_p.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="frames per second to sample for analysis (default 2)",
    )
    scan_p.add_argument(
        "--force",
        action="store_true",
        help="reanalyze clips even when size and mtime are unchanged",
    )
    scan_p.add_argument(
        "--clip",
        action="store_true",
        help="use a local CLIP checkpoint for lighting and subject, if one is on disk",
    )

    query_p = sub.add_parser("query", help="print clips that match social filters")
    query_p.add_argument("--index", type=Path, default=None, help="index directory")
    query_p.add_argument("--face", action="store_true", help="require a visible face")
    query_p.add_argument("--lighting", choices=LIGHTING, help="golden, day, night, snow, or overcast")
    query_p.add_argument("--motion", choices=MOTION, help="static, handheld, or pan")
    query_p.add_argument("--min-brightness", type=float, dest="min_brightness", help="0 to 1")
    query_p.add_argument(
        "--project",
        help="case-insensitive substring of the top-level project folder",
    )
    query_p.add_argument("--graded", action="store_true", help="only graded or renders-proxy bins")
    query_p.add_argument("--tag", help="exact social tag, case-insensitive")
    query_p.add_argument("--json", action="store_true", help="print matching records as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.cmd == "scan":
            return _cmd_scan(args)
        if args.cmd == "query":
            return _cmd_query(args)
    except IndexCorrupt as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ScanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"error: unknown command {args.cmd}", file=sys.stderr)
    return 2


def _cmd_scan(args: argparse.Namespace) -> int:
    index = args.index if args.index is not None else default_index_dir()
    stats = scan(
        args.root,
        index,
        sample_fps=args.sample_fps,
        force=args.force,
        use_clip=args.clip,
    )
    print(format_scan(stats, index, args.root.expanduser().resolve()))
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    if args.min_brightness is not None and not 0.0 <= args.min_brightness <= 1.0:
        print("error: --min-brightness must be between 0 and 1", file=sys.stderr)
        return 1
    index = args.index if args.index is not None else default_index_dir()
    query = Query(
        face=args.face,
        lighting=args.lighting,
        motion=args.motion,
        min_brightness=args.min_brightness,
        project=args.project,
        graded=args.graded,
        tag=args.tag,
    )
    records = run_query(index, query)
    if args.json:
        print(format_json(records))
    else:
        print(format_text(records))
    return 0


def format_scan(stats: ScanStats, index: Path, root: Path) -> str:
    lines = [
        f"index: {index}",
        f"root: {root}",
        "source drive will not be modified",
    ]
    lines.extend(stats.notes)
    for rel in stats.analyzed_paths:
        lines.append(f"analyzed  {rel}")
    for rel, reason in stats.failures:
        lines.append(f"failed    {rel}  {reason}")
    for directory in stats.personal_dirs:
        lines.append(f"skipped   personal directory {directory}")
    lines.append(
        f"done analyzed={stats.analyzed} unchanged={stats.unchanged} "
        f"failed={stats.failed} personal_dirs={len(stats.personal_dirs)}"
    )
    return "\n".join(lines)
