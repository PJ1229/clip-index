"""Tiny synthetic clips for the end-to-end scan tests."""

from __future__ import annotations

import subprocess
from pathlib import Path


def write_solid_clip(
    path: Path,
    rgb: tuple[int, int, int],
    *,
    seconds: float = 4.0,
    fps: int = 10,
    size: str = "160x90",
) -> None:
    """Lossless RGB clip so classification sees the color we asked for."""
    color = "0x{:02X}{:02X}{:02X}".format(*rgb)
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={size}:r={fps}:d={seconds}",
        "-c:v",
        "ffv1",
        "-pix_fmt",
        "rgb24",
        str(path),
    ]
    subprocess.run(cmd, check=True)
