# Shared clip index

`clip-index` and `um-autolog` share one index directory. The default path is `~/Documents/clip-index/`. A scan or query can point somewhere else with `--index` or with the `CLIP_INDEX_DIR` environment variable.

`clip-index` is the only writer of the `social` namespace. `um-autolog` is the expected writer of `autolog`. Neither tool deletes or rewrites a namespace it does not own.

The footage drive is not part of the index. `clip-index scan` never creates, renames, or writes a file under the scan root. No sidecars, no AppleDouble files, no thumbnails next to the media. Put the index on the internal disk. The command refuses to create an index directory inside the scan root.

## Files

| Path | Who writes it | What it is |
| --- | --- | --- |
| `index.jsonl` | both tools, under the lock | one JSON object per line, one line per clip |
| `index.lock` | both tools | advisory lock file |
| `thumbs/<sha1>/` | clip-index | JPEG previews for that relative path |
| `meta.json` | clip-index | bookkeeping, not a clip record |

`meta.json` looks like this. Readers of `index.jsonl` can ignore it.

```json
{
  "schema_version": 1,
  "tool": "clip-index",
  "root": "/Volumes/TOSHIBA EXT"
}
```

`root` is the last scan root. Records are still keyed by the relative path, so a drive that remounts somewhere else does not by itself invalidate the file.

## index.jsonl

UTF-8 text. One JSON object per line. Every record ends with a newline, including the last line. The file is not a JSON array and it is not pretty-printed.

The identity of a clip is `path`: the path relative to the scan root, forward slashes, no leading slash, no `.` or `..` segments. Matching is exact and case-sensitive. The string that is hashed for thumbnails is this same UTF-8 string.

### Core fields

These describe the file. A tool that has just looked at the media may refresh them. Refreshing them must not drop unknown top-level keys.

| Field | Type | Meaning |
| --- | --- | --- |
| `path` | string | relative path, primary key |
| `size` | integer | size in bytes (`st_size`) |
| `mtime` | number | POSIX mtime in seconds (`st_mtime`) |
| `duration` | number | seconds |
| `fps` | number | video frame rate. `0` if the container did not say |
| `resolution` | string | `{width}x{height}`, for example `1920x1080` |

`clip-index` swaps width and height when the video stream tag `rotate` is 90 or 270 degrees. `fps` is `avg_frame_rate`, then `r_frame_rate`. `duration` is the stream duration, then the format duration.

A clip is treated as unchanged by `clip-index` when `size` equals `st_size`, `mtime` is within 0.001 seconds of `st_mtime`, and `social` contains every field listed below with a legal enum value. Unchanged clips are not decoded again and their lines are not rewritten.

If size and mtime match but `social` is missing or incomplete, `clip-index` fills `social` and leaves every other namespace as it found it.

### Namespaces

Any other top-level key is a namespace object owned by one tool.

| Key | Owner |
| --- | --- |
| `social` | clip-index |
| `autolog` | um-autolog |

Rules for every writer:

1. Copy through every top-level key you do not own. Unknown keys stay, byte-for-value.
2. Replace your own namespace as a whole. Do not merge field-by-field into someone else's object.
3. Do not invent keys inside a namespace you do not own. A `clip-index` record that is new has no `autolog` key.
4. Do not drop a clip's line just because you do not understand it.
5. `path` is the key. Moving a clip adds a new line; it does not rewrite the old path in place.
6. Duplicate paths are a broken file. Readers keep the last copy. Writers should not emit duplicates. The next locked rewrite collapses them.

`social.tags` is rewritten whenever `clip-index` reanalyzes that clip. Notes that must survive belong in another namespace.

### social

| Field | Type | Meaning |
| --- | --- | --- |
| `project` | string | Top-level folder under the scan root. If the clip is directly in the root, the root folder's name. |
| `graded` | bool | True when any parent folder name contains `graded` or `renders_proxies`, case-insensitive. `05_renders_proxies` and `02_graded` match. The filename does not. `grades` and `proxies` do not. |
| `subject` | string | `person`, `scenery`, or `object` |
| `face_visible` | object | `{"visible": bool, "fraction": number}`. `fraction` is in `[0, 1]`, the share of sampled frames with at least one detected frontal face. `visible` is true when that count is at least one. |
| `lighting` | string | `golden`, `day`, `night`, `snow`, or `overcast` |
| `brightness` | number | mean Rec. 709 luma of the sampled frames, 0 to 1. `Y = (0.2126 R + 0.7152 G + 0.0722 B) / 255` |
| `camera_motion` | string | `static`, `handheld`, or `pan` |
| `calm_moments` | array | `[start, end]` pairs, seconds from the start of the file. Each pair is 2 to 3 seconds long. At most three, they do not overlap, and they are ordered by start time. Empty when none qualify, including when the clip is shorter than 2 seconds. |
| `tags` | array of strings | labels produced with the classification. Free-form for readers. This writer replaces the array on each analysis. |

`subject` is `person` when `fraction` is at least 0.08, even if a scenery model disagrees. Below that, a face can still be `visible`.

A calm window is one where every sampled optical-flow interval stays in the static range. A pan or a shake inside the window drops it. Times follow the sample clock (frame index divided by the sample rate, default 2 Hz), not a timecode track.

### Example

```json
{"path":"December/05_renders_proxies/A001_C003.mov","size":184332211,"mtime":1735689600.0,"duration":12.5,"fps":23.976,"resolution":"1920x1080","social":{"project":"December","graded":true,"subject":"person","face_visible":{"visible":true,"fraction":0.37},"lighting":"golden","brightness":0.54,"camera_motion":"static","calm_moments":[[1.0,3.5],[6.0,8.5]],"tags":["golden","static","person","graded","face","calm"]},"autolog":{"scene":"3A","take":2}}
```

## Locking and atomic replace

The lock file is `<index>/index.lock`.

1. Open it with `O_CREAT` and mode `0644`. The index directory must already exist (the scan creates it; a query of a missing index does not).
2. Take `flock(fd, LOCK_EX)` for a read-modify-write. A reader that wants a stable snapshot may take `LOCK_SH`.
3. Read `index.jsonl` to the end. A missing file means an empty library. If any non-empty line is not a JSON object with a string `path`, stop and do not rewrite the file.
4. Write the full next version to `.<name>.<pid>.tmp` in the same directory. `flush` and `fsync` that file.
5. `os.replace` the temp file onto the real name. On the same filesystem that replace is atomic: readers see the old file or the new file, never a half line.
6. Unlock by closing the descriptor, or `LOCK_UN` then close.

Do not truncate `index.jsonl` in place. Do not hold the lock while decoding video. `clip-index` holds it only around the reload-merge-rename, and around writing `meta.json`.

A crash during decode leaves the previous `index.jsonl` in place, so the clip is retried. A crash while the temp file is still being written also leaves the previous file in place. The temp file is removed if the write raises.

The index should live on a local disk. `flock` on the footage volume is the wrong place for it, and that volume is not writable from this tool anyway.

## Thumbnails

Directory name: hex SHA-1 of the UTF-8 relative `path` (the exact `path` string).

```text
thumbs/<sha1 of the UTF-8 path>/t_0.00.jpg
thumbs/<sha1 of the UTF-8 path>/t_1.50.jpg
```

Files are JPEGs named `t_<seconds>.jpg`, up to five, spread across the clip. Reanalysis replaces the files in that directory. `um-autolog` should not delete or rewrite `thumbs/`. Thumbnails are never written beside the source media.

## What clip-index will not ingest

- A path with a directory segment equal to `personal` in any case. The scan does not descend into that directory.
- The whole tree, when the scan root itself is named `personal`.
- Dot directories, dotfiles, AppleDouble `._*` names, and `__MACOSX`.
- Symlinks.
- Extensions outside `.mov`, `.mp4`, `.m4v`, `.mxf`, `.mkv`, `.avi`, `.mts`, `.m2ts`, `.webm`, `.mpg`, `.mpeg`.

## Deleted media

Scan does not remove a line whose file is gone. A later tool may prune. Until then a query can still return that path.

## Writer checklist

Before saving `index.jsonl`:

- You hold `LOCK_EX` on `index.lock`.
- You loaded the file after taking the lock, not before.
- Every namespace you do not own is still on the record, unchanged.
- You only replaced your own namespace, plus core fields if you actually re-probed the file.
- The output is one JSON object per line, UTF-8, trailing newline, then an atomic replace.
- You did not write anything under the footage root.
