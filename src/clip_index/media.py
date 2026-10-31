"""Read clips with ffmpeg and ffprobe. This module never writes a file."""

from __future__ import annotations

import json
import os
import select
import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class ClipReadError(RuntimeError):
    """The clip could not be decoded. The source file was not modified."""


@dataclass(frozen=True)
class Probe:
    duration: float
    fps: float
    width: int
    height: int


def probe_video(path: Path, *, timeout: float = 30) -> Probe:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-select_streams",
        "v:0",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            check=False,
            cwd="/tmp",
        )
    except subprocess.TimeoutExpired as exc:
        raise ClipReadError(f"ffprobe timed out for {path.name}") from exc
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", "replace").strip()
        raise ClipReadError(err or f"ffprobe failed for {path.name}")
    try:
        payload = json.loads(result.stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ClipReadError(f"ffprobe returned no JSON for {path.name}") from exc
    streams = payload.get("streams") or []
    if not streams:
        raise ClipReadError(f"no video stream in {path.name}")
    stream = streams[0]
    try:
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ClipReadError(f"could not read resolution for {path.name}") from exc
    rotate = _rotate_degrees(stream)
    if abs(rotate) % 180 == 90:
        width, height = height, width
    fps = _rate(stream.get("avg_frame_rate")) or _rate(stream.get("r_frame_rate")) or 0.0
    duration = _num(stream.get("duration"))
    if duration is None:
        duration = _num((payload.get("format") or {}).get("duration"))
    if duration is None or duration <= 0 or width <= 0 or height <= 0:
        raise ClipReadError(f"could not read duration for {path.name}")
    return Probe(duration=float(duration), fps=float(fps), width=width, height=height)


def iter_sampled_frames(
    path: Path,
    sample_fps: float,
    duration: float,
) -> Iterator[np.ndarray]:
    """Yield RGB frames at ``sample_fps`` without writing beside the clip.

    ffmpeg's stdout is an MJPEG pipe. The process cwd is ``/tmp``, so a
    stray log cannot land on the source drive.
    """
    if sample_fps <= 0:
        raise ClipReadError("sample fps must be positive")
    vf = f"fps={sample_fps},scale='min(640,iw)':-2"
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        vf,
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-q:v",
        "2",
        "pipe:1",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd="/tmp",
        bufsize=0,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None
    err_box: list[bytes] = []

    def _drain() -> None:
        err_box.append(proc.stderr.read())

    drain = threading.Thread(target=_drain, daemon=True)
    drain.start()
    try:
        yield from _iter_rgb(proc, duration)
        rc = proc.wait(timeout=15)
        drain.join(timeout=5)
        if rc != 0:
            err = (err_box[0] if err_box else b"").decode("utf-8", "replace").strip()
            raise ClipReadError(err[:500] or f"ffmpeg exited {rc}")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        drain.join(timeout=5)


def _iter_rgb(proc: subprocess.Popen, duration: float) -> Iterator[np.ndarray]:
    assert proc.stdout is not None
    fd = proc.stdout.fileno()
    buf = b""
    deadline = time.monotonic() + max(180.0, float(duration) * 8.0)
    last_data = time.monotonic()
    while True:
        now = time.monotonic()
        if now > deadline:
            proc.kill()
            raise ClipReadError("timed out sampling frames")
        frame, buf = _pop_jpeg(buf)
        if frame is not None:
            rgb = _decode_jpeg(frame)
            if rgb is not None:
                yield rgb
            continue
        if len(buf) > 20_000_000:
            proc.kill()
            raise ClipReadError("jpeg frame exceeded 20MB")
        try:
            ready, _, _ = select.select([fd], [], [], 1.0)
        except (ValueError, OSError):
            break
        if not ready:
            if proc.poll() is not None:
                try:
                    rest = os.read(fd, 1 << 20)
                except OSError:
                    rest = b""
                buf += rest
                if not rest:
                    break
            elif now - last_data > 60:
                proc.kill()
                raise ClipReadError("ffmpeg produced no data")
            continue
        try:
            data = os.read(fd, 1 << 16)
        except OSError:
            break
        if not data:
            break
        last_data = time.monotonic()
        buf += data
    while True:
        frame, buf = _pop_jpeg(buf)
        if frame is None:
            break
        rgb = _decode_jpeg(frame)
        if rgb is not None:
            yield rgb


def _pop_jpeg(buf: bytes) -> tuple[bytes | None, bytes]:
    start = buf.find(b"\xff\xd8")
    if start < 0:
        return None, buf[-1:]
    end = buf.find(b"\xff\xd9", start + 2)
    if end < 0:
        return None, buf[start:]
    return buf[start : end + 2], buf[end + 2 :]


def _decode_jpeg(blob: bytes) -> np.ndarray | None:
    arr = np.frombuffer(blob, dtype=np.uint8)
    bgr = cv2.imdecode(arr.copy(), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _rate(value: object) -> float | None:
    if value is None:
        return None
    text = str(value)
    if text in {"", "0/0", "N/A"}:
        return None
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            den_f = float(den)
            if den_f == 0:
                return None
            return float(num) / den_f
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _num(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _rotate_degrees(stream: dict) -> int:
    tags = stream.get("tags") or {}
    try:
        return int(float(tags.get("rotate", 0)))
    except (TypeError, ValueError):
        return 0
