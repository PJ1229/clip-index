# clip-index

`clip-index` classifies a filmmaker's video clips on the machine that has the drive plugged in, so an editor can ask for the still, bright, golden-hour shots (and the calm stretches inside them) without scrubbing the whole disk.

It is a Python 3.12 command-line tool. Decoding, face detection, optical flow, and the optional CLIP pass all run locally. Frames are not uploaded anywhere, and the footage folder is never written to.

The index is a folder another tool, `um-autolog`, can read and write at the same time. This tool only fills in the `social` namespace. `autolog` and any other namespace are left as they were. The record layout, the lock, and the thumbnail paths are specified in [SCHEMA.md](SCHEMA.md).

## What you need

- Python 3.12
- `ffmpeg` and `ffprobe` on `PATH` (`brew install ffmpeg` on a Mac)
- An index directory on the internal disk. The default is `~/Documents/clip-index`. Do not point it at the footage drive.

OpenCV 4 comes with the package and supplies optical flow plus the frontal-face cascade used for `face_visible`. A CLIP zero-shot model is optional and stays off unless you pass `--clip` and already have weights on disk.

Apple Silicon is the machine this is meant to run on. The optional CLIP path uses PyTorch's MPS device when it is available. Motion, faces, brightness, and the default lighting and subject labels use OpenCV on CPU, which is enough at a couple of sampled frames per second. The same code runs the test suite on Linux.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Check the install:

```bash
clip-index --help
ffmpeg -version
```

Optional local CLIP (lighting and subject only; motion and faces stay on OpenCV):

```bash
python -m pip install -e ".[clip,dev]"
mkdir -p ~/Documents/clip-index/models
# Put an OpenCLIP ViT-B-32 weights file there yourself, or set the path:
# export CLIP_INDEX_CLIP_CHECKPOINT=/path/to/ViT-B-32.pt
```

`clip-index` will not download those weights and will not send frames to a hub. If `--clip` is set and the file is missing or will not load, the scan says so and keeps the color and texture heuristics.

## Scan

```bash
clip-index scan "/Volumes/TOSHIBA EXT"
```

That walks the drive, samples about 2 frames per second, and writes `~/Documents/clip-index/index.jsonl` plus JPEGs under `thumbs/<sha1(path)>/`. The drive is only opened for reading. Nothing is created, renamed, or updated on it, including sidecars and AppleDouble files.

Useful flags:

```bash
clip-index scan "/Volumes/TOSHIBA EXT" --index ~/Documents/clip-index
clip-index scan "/Volumes/TOSHIBA EXT" --sample-fps 2
clip-index scan "/Volumes/TOSHIBA EXT" --force          # ignore size/mtime and redo social.*
clip-index scan "/Volumes/TOSHIBA EXT" --clip           # local CLIP, if weights are present
```

The index directory can also be set with `CLIP_INDEX_DIR`.

What the walk skips:

- Any folder whose name is `personal`, in any case (`personal`, `Personal`, `PERSONAL`). The walk does not descend into it. A folder named `personal_notes` is not skipped.
- Dotfiles and dot directories (`.Spotlight-V100`, `.Trashes`, `._clip.mov`, …) and `__MACOSX`.
- Symlinks.
- Anything that is not a video (`mov`, `mp4`, `m4v`, `mxf`, `mkv`, `avi`, `mts`, `m2ts`, `webm`, `mpg`, `mpeg`).

If the scan root itself is named `personal`, the command indexes nothing.

### Incremental and resumable

A clip is skipped when its byte size matches the record, its mtime is within 1 millisecond, and `social` already has every field in the schema. A second run of the same command does not rewrite `index.jsonl` if nothing changed.

Each clip is committed on its own, under a lock, by writing a temp file and renaming it onto `index.jsonl`. A crash loses at most the clip that was in progress. Run the same command again and that clip is retried. Clips that already committed are left alone, including whatever `um-autolog` stored next to them.

Failed clips are listed and are not given a `social` block, so the next scan tries them again. The process still exits 0 when the walk itself finished; the summary line shows `failed=`.

Scan does not delete records for files that disappeared. Unplug the drive and the index stays as it was.

The index directory must sit outside the scan root. `clip-index scan /Volumes/TOSHIBA\ EXT --index "/Volumes/TOSHIBA EXT/clip-index"` is refused before any directory is created.

## Query

Filters are AND. Omit a flag to leave that field open. `--project` is a case-insensitive substring of the top-level folder (`december` matches `December` and `02_december_trip`).

```bash
clip-index query --face --lighting golden --motion static --min-brightness 0.5 --project december --graded
```

Human output is one block per clip, including the calm stretches:

```text
December/05_renders_proxies/hero.mov
  project=December  graded=yes  subject=person  lighting=golden  motion=static  brightness=0.62  face=yes (40%)  12.5s
  calm: 1.00-3.50s, 6.00-8.50s
```

`--json` prints the matching records as a JSON array, including any namespace this tool does not own. With no index yet, the command prints `[]` and does not create the folder.

```bash
clip-index query --lighting night --json
clip-index query --tag calm --project december
```

`--tag` matches one entry of `social.tags` (case-insensitive). Lighting is `golden`, `day`, `night`, `snow`, or `overcast`. Motion is `static`, `handheld`, or `pan`.

## How a shot is labeled

| Field | Meaning |
| --- | --- |
| `project` | Top-level folder under the scan root. A clip that sits directly in the root uses the root folder's name. |
| `graded` | True when a parent folder name contains `graded` or `renders_proxies` (so `05_renders_proxies` and `02_graded` count). The filename does not count. |
| `subject` | `person` when a face shows up in at least 8% of the sampled frames, otherwise `scenery` (horizon or a quiet field) or `object` (a compact subject on a calmer background). |
| `face_visible` | `visible` is true if any sampled frame has a frontal face. `fraction` is how many of the sampled frames did. The detector is OpenCV's built-in Haar cascade. Profile and tiny faces are often missed. |
| `lighting` | `night` (dark), `snow` (very bright and flat), `golden` (warm, red above green above blue), `overcast` (flat mid tones), otherwise `day`. |
| `brightness` | Mean Rec. 709 luma of the sampled frames, from 0 to 1. |
| `camera_motion` | Optical flow on a grayscale proxy at most 320px wide. Low flow is `static`. A steady translation is `pan`. Busy, directionless flow is `handheld`. |
| `calm_moments` | Up to three non-overlapping 2–3 second windows where every sampled interval stays in the static range, so a pan or a shake is not included. Empty when the clip never settles. Short clips under 2 seconds have none. Times are seconds from the start of the file, on the sample clock, not a burned-in timecode. |
| `tags` | Generated from those labels (`golden`, `static`, `face`, `graded`, `calm`, …). Replaced each time the clip is reanalyzed. Put notes that must survive a rescan in another namespace, not in `social.tags`. |

Pass `--clip` when a local ViT-B-32 checkpoint is installed and CLIP replaces the lighting and subject votes. Faces, brightness, motion, and calm moments still come from OpenCV. A frequent face still forces `subject` to `person`.

## Tests

The tests build a few seconds of synthetic color with ffmpeg and never need a camera, a GPU, or a network connection.

```bash
source .venv/bin/activate
pytest
```

## Layout

```text
src/clip_index/    the command and the library
tests/             unit tests and synthetic-clip scans
SCHEMA.md          the shared index contract
```
