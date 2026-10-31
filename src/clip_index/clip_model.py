"""Optional local CLIP-style zero-shot. Never downloads and never uploads.

The default scan does not import torch. Pass ``--clip`` (or set
``CLIP_INDEX_CLIP=1``) and point ``CLIP_INDEX_CLIP_CHECKPOINT`` at a
weights file that is already on disk. On Apple Silicon the model runs
on MPS when PyTorch provides it.
"""

from __future__ import annotations

import os
from pathlib import Path

from clip_index.schema import LIGHTING, SUBJECTS

# Prompts are fixed so a local checkpoint is scored the same way every run.
LIGHTING_PROMPTS: dict[str, str] = {
    "golden": "a video frame lit by golden hour sunlight",
    "day": "a video frame in bright daylight",
    "night": "a video frame at night in the dark",
    "snow": "a video frame of a snowy scene",
    "overcast": "a video frame under an overcast sky",
}
SUBJECT_PROMPTS: dict[str, str] = {
    "person": "a video frame of a person",
    "scenery": "a video frame of scenery or a landscape",
    "object": "a video frame of a physical object",
}


def clip_requested(flag: bool) -> bool:
    if flag:
        return True
    return os.environ.get("CLIP_INDEX_CLIP", "").strip().lower() in {"1", "true", "yes"}


def pick_device(has_mps: bool, has_cuda: bool) -> str:
    """Prefer the Apple Silicon GPU, then CUDA, then CPU."""
    if has_mps:
        return "mps"
    if has_cuda:
        return "cuda"
    return "cpu"


def pick_best(scores: dict[str, float]) -> str | None:
    if not scores:
        return None
    return max(scores, key=lambda label: scores[label])


def find_checkpoint(index_dir: Path) -> Path | None:
    candidates: list[Path] = []
    env = os.environ.get("CLIP_INDEX_CLIP_CHECKPOINT")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(index_dir / "models" / "ViT-B-32.pt")
    candidates.append(Path.home() / ".cache" / "clip" / "ViT-B-32.pt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_open_clip(checkpoint: Path, model_name: str = "ViT-B-32"):
    """Load a local OpenCLIP checkpoint. Raises if torch or the file is unusable.

    Hugging Face and the OpenCLIP downloader are forced offline before import
    so a missing config cannot turn into a network fetch of frames or weights.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import open_clip
    import torch

    has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    has_cuda = bool(torch.cuda.is_available())
    device = pick_device(has_mps, has_cuda)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=str(checkpoint),
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return {
        "model": model,
        "preprocess": preprocess,
        "tokenizer": tokenizer,
        "device": device,
        "torch": torch,
    }


class ClipZeroShot:
    """Scores a few frames against the lighting and subject prompts.

    ``available`` is False when CLIP was not requested, no local checkpoint
    exists, or the weights failed to load. Callers then keep the heuristic.
    """

    def __init__(self, index_dir: Path, enabled: bool, loader=load_open_clip) -> None:
        self.index_dir = index_dir
        self.enabled = enabled
        self._loader = loader
        self._bundle = None
        self._failed = False
        self.reason = ""

    @property
    def available(self) -> bool:
        if not self.enabled or self._failed:
            return False
        if self._bundle is not None:
            return True
        checkpoint = find_checkpoint(self.index_dir)
        if checkpoint is None:
            self._failed = True
            self.reason = (
                "No local CLIP checkpoint. Place ViT-B-32 weights at "
                f"{self.index_dir / 'models' / 'ViT-B-32.pt'} "
                "or set CLIP_INDEX_CLIP_CHECKPOINT. clip-index will not download them."
            )
            return False
        try:
            self._bundle = self._loader(checkpoint)
        except Exception as exc:  # noqa: BLE001 — fall back to heuristics
            self._failed = True
            self.reason = f"CLIP weights did not load ({exc}). Using color heuristics."
            return False
        return True

    def lighting(self, rgb) -> str | None:
        return self._rank(rgb, LIGHTING_PROMPTS, LIGHTING)

    def subject(self, rgb) -> str | None:
        return self._rank(rgb, SUBJECT_PROMPTS, SUBJECTS)

    def _rank(self, rgb, prompts: dict[str, str], allowed: tuple[str, ...]) -> str | None:
        if not self.available or self._bundle is None:
            return None
        try:
            scores = _similarities(self._bundle, rgb, prompts)
        except Exception:  # noqa: BLE001
            return None
        label = pick_best(scores)
        if label not in allowed:
            return None
        return label


def _similarities(bundle: dict, rgb, prompts: dict[str, str]) -> dict[str, float]:
    torch = bundle["torch"]
    from PIL import Image

    image = bundle["preprocess"](Image.fromarray(rgb)).unsqueeze(0)
    image = image.to(bundle["device"])
    tokens = bundle["tokenizer"](list(prompts.values())).to(bundle["device"])
    with torch.no_grad():
        image_features = bundle["model"].encode_image(image)
        text_features = bundle["model"].encode_text(tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        sims = (image_features @ text_features.T).squeeze(0)
    labels = list(prompts)
    return {label: float(sims[i]) for i, label in enumerate(labels)}
