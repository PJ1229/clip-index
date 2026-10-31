# clip-index

Find the usable shots in a folder of footage without scrubbing the drive.

`clip-index` is a Python 3.12 command-line tool for editors. It samples each clip locally, labels what is in the frame, and writes a small index you can query: face, golden-hour light, a locked-off camera, a graded bin, a project folder. Calm stretches come back as timestamps.

## What it does

Point it at a folder of camera media. It reads each video with `ffmpeg`, samples a few frames a second, and records:

- project (the top-level folder) and whether the path looks graded (`graded`, `05_renders_proxies`)
- subject (`person`, `scenery`, or `object`) and whether a face is visible
- lighting (`golden`, `day`, `night`, `snow`, `overcast`) and brightness
- camera motion (`static`, `handheld`, `pan`) from optical flow
- up to three calm 2–3 second windows with no pan

The index lives on the internal disk, default `~/Documents/clip-index`. Records are one JSON object per line in `index.jsonl`. This tool writes the `social` fields and leaves every other namespace untouched, so another local tool can share the same file. Thumbnails, when written, stay in that index directory. The field list, lock, and thumbnail names are in [SCHEMA.md](SCHEMA.md).

A second scan skips clips whose size and mtime have not changed, and a crash loses at most the clip in progress. Run the same command again to resume.

## Example

```bash
clip-index scan "/Volumes/Footage"

clip-index query --face --lighting golden --motion static \
  --min-brightness 0.5 --project december --graded
```

```text
December/05_renders_proxies/hero.mov
  project=December  graded=yes  subject=person  lighting=golden  motion=static  brightness=0.62  face=yes (40%)  12.5s
  calm: 1.00-3.50s, 6.00-8.50s
```

`--json` prints the same matches as a JSON array. Lighting is `golden`, `day`, `night`, `snow`, or `overcast`. Motion is `static`, `handheld`, or `pan`. `--project` is a case-insensitive substring of the top-level folder.

## Setup

Python 3.12, plus `ffmpeg` and `ffprobe` on `PATH` (`brew install ffmpeg` on a Mac).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
pytest
```

The package installs OpenCV 4, which supplies optical flow and the frontal-face cascade. Analysis runs on CPU at the sample rate (default 2 fps) and is aimed at Apple Silicon. The tests generate a few seconds of synthetic color and do not need a camera or a GPU.

Optional: a local OpenCLIP ViT-B-32 checkpoint can replace the lighting and subject guesses. Pass `--clip` only after the weights are already on disk (`CLIP_INDEX_CLIP_CHECKPOINT` or `~/Documents/clip-index/models/ViT-B-32.pt`). The tool does not download them. Faces, brightness, and motion still come from OpenCV.

```bash
clip-index scan "/Volumes/Footage" --index ~/Documents/clip-index
clip-index scan "/Volumes/Footage" --sample-fps 2
clip-index scan "/Volumes/Footage" --force
```

`CLIP_INDEX_DIR` overrides the index location. The index directory has to sit outside the footage folder.

Folders named `personal` (any case) are skipped entirely. So are dotfiles, AppleDouble files, `__MACOSX`, symlinks, and non-video extensions.

## Privacy

- The footage folder is read-only. The scan never creates, renames, or updates anything on that drive, including sidecars.
- Any path with a `personal` folder segment is skipped and is not indexed.
- Frames stay on the machine. Nothing is uploaded. There is no account, no API key, and no network call on the default path.
