"""
CLIP feature extraction. The frozen half of the system.

Design note that matters more than it looks: CLIP resizes everything to
224x224 before the patch embedding. That throws away the high-frequency
generator fingerprints that pixel-space detectors (CNNSpot, NPR, frequency
methods) rely on. It is exactly why CLIP features survive JPEG and blur so
well, and exactly why they cannot see subtle resampling traces.

`--preproc nativecrop` gives you the other side of that trade: crop 224x224
windows at native resolution so high-frequency content reaches the encoder
intact. Run both, report both. The comparison is a real finding.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import augment as A

Image.MAX_IMAGE_PIXELS = None

CLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
RES = 224

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------
# data scanning
# --------------------------------------------------------------------------


@dataclass
class Sample:
    path: str
    label: int      # 0 = real, 1 = fake
    group: str      # generator name (fake) or source corpus (real)


def scan_dir(root: str | Path) -> list[Sample]:
    """
    Expects:
        root/real/<source>/*.jpg      (the <source> level is optional)
        root/fake/<generator>/*.jpg

    <generator> is what leave-one-generator-out evaluation splits on, so put
    each generator in its own folder even if you only have two.
    """
    root = Path(root)
    out: list[Sample] = []
    for label_name, label in (("real", 0), ("fake", 1)):
        base = root / label_name
        if not base.is_dir():
            raise FileNotFoundError(f"missing {base}")
        for p in sorted(base.rglob("*")):
            if p.suffix.lower() not in IMG_EXT or not p.is_file():
                continue
            rel = p.relative_to(base).parts
            group = rel[0] if len(rel) > 1 else f"{label_name}_root"
            out.append(Sample(str(p), label, group))
    if not out:
        raise RuntimeError(f"no images found under {root}")
    return out


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------

_MEAN_T = torch.tensor(CLIP_MEAN).view(3, 1, 1)
_STD_T = torch.tensor(CLIP_STD).view(3, 1, 1)


def to_clip_tensor(img: Image.Image, preproc: str, py_rng: random.Random) -> torch.Tensor:
    """PIL RGB -> normalised (3, 224, 224) float tensor."""
    if preproc == "resize":
        w, h = img.size
        s = RES / min(w, h)
        img = img.resize((max(RES, int(round(w * s))), max(RES, int(round(h * s)))), Image.Resampling.BICUBIC)
        w, h = img.size
        left, top = (w - RES) // 2, (h - RES) // 2
        img = img.crop((left, top, left + RES, top + RES))
    elif preproc == "nativecrop":
        w, h = img.size
        if min(w, h) < RES:                       # too small to crop natively
            s = RES / min(w, h)
            img = img.resize((max(RES, int(round(w * s))), max(RES, int(round(h * s)))), Image.Resampling.BICUBIC)
            w, h = img.size
        left = py_rng.randint(0, w - RES)
        top = py_rng.randint(0, h - RES)
        img = img.crop((left, top, left + RES, top + RES))
    else:
        raise ValueError(f"unknown preproc {preproc!r}")

    arr = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255.0)
    return (arr - _MEAN_T) / _STD_T


# --------------------------------------------------------------------------
# view generation
#
# PICKLING NOTE (see also augment.py): DataLoader workers are pickled on
# Windows (spawn) but not on Linux/Mac (fork). Lambdas and closures survive
# fork but not spawn. These were plain functions returning lambdas, which
# worked in this repo's own testing (Linux) but breaks the moment a Windows
# user runs the exact same code with num_workers > 0. Fixed by using
# picklable classes with __call__ instead of closures.
# --------------------------------------------------------------------------

ViewFn = Callable[[Image.Image, np.random.Generator, random.Random], Image.Image]


class RandomView:
    """Draws a random degradation chain each call. Picklable — plain __init__ args only."""

    def __init__(self, max_ops: int = 3, p_clean: float = 0.15) -> None:
        self.max_ops = max_ops
        self.p_clean = p_clean

    def __call__(self, img: Image.Image, rng: np.random.Generator, py_rng: random.Random) -> Image.Image:
        return A.sample_chain(img, rng, py_rng, max_ops=self.max_ops, p_clean=self.p_clean)[0]


class CleanView:
    """Identity view — the un-degraded image, used as view 0."""

    def __call__(self, img: Image.Image, rng: np.random.Generator, py_rng: random.Random) -> Image.Image:
        return img


class ConditionView:
    """Wraps one NamedCondition (itself picklable — see augment.py) as a view."""

    def __init__(self, cond: A.NamedCondition) -> None:
        self.cond = cond

    def __call__(self, img: Image.Image, rng: np.random.Generator, py_rng: random.Random) -> Image.Image:
        return self.cond.fn(img, rng)


def random_view(max_ops: int = 3, p_clean: float = 0.15) -> ViewFn:
    return RandomView(max_ops=max_ops, p_clean=p_clean)


def clean_view() -> ViewFn:
    return CleanView()


def condition_view(cond: A.NamedCondition) -> ViewFn:
    return ConditionView(cond)


class ViewDataset(Dataset):
    """Yields (V, 3, 224, 224) — one tensor per view of the same source image."""

    def __init__(
        self,
        samples: Sequence[Sample],
        views: Sequence[ViewFn],
        preproc: str = "resize",
        renormalise: bool = True,
        seed: int = 0,
    ) -> None:
        self.samples = list(samples)
        self.views = list(views)
        self.preproc = preproc
        self.renormalise = renormalise
        self.seed = seed

    # We disabled Pillow's default decompression-bomb limit (see top of file)
    # because SID_Set legitimately has images up to ~6000px on a side. That
    # protection existed for a reason though: a corrupt or malicious file can
    # report absurd dimensions and blow up on the FIRST full decode -- which
    # for PIL is .convert()/.load(), not .open() (.open only reads the
    # header). An uncapped decode of a bad file can try to allocate many GB
    # regardless of how much RAM the machine actually has, which surfaces as
    # a PyTorch/C++ allocator RuntimeError, not a catchable PIL exception --
    # so the bare try/except around .convert("RGB") below does not reliably
    # catch it. This cap re-adds a sane ceiling: generous enough for any real
    # photo (200 megapixels), tight enough to reject garbage headers before
    # a single byte of pixel data is decoded.
    MAX_PIXELS = 200_000_000

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        s = self.samples[idx]
        # deterministic per-sample seeding -> reproducible across runs and workers
        seed = (self.seed * 1_000_003 + idx) % (2**31 - 1)
        rng = np.random.default_rng(seed)
        py_rng = random.Random(seed)

        try:
            raw = Image.open(s.path)          # header only, does not decode pixels
            w, h = raw.size
            if w * h > self.MAX_PIXELS:
                print(f"[skip] {s.path}: {w}x{h} = {w*h/1e6:.0f}MP exceeds the "
                     f"{self.MAX_PIXELS/1e6:.0f}MP cap, likely a corrupt file — skipping")
                return torch.zeros(len(self.views), 3, RES, RES), -1
            img = raw.convert("RGB")          # decode happens here
        except Exception as e:
            print(f"[skip] {s.path}: failed to load ({type(e).__name__}: {e})")
            return torch.zeros(len(self.views), 3, RES, RES), -1

        if self.renormalise:
            img = A.normalise_source(img, rng, py_rng)

        out = [to_clip_tensor(v(img, rng, py_rng), self.preproc, py_rng) for v in self.views]
        return torch.stack(out, 0), s.label


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


def load_clip(model_name: str = "openai/clip-vit-large-patch14", device: str | None = None):
    from transformers import CLIPVisionModelWithProjection

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # low_cpu_mem_usage skips the default HF loading path of building a full
    # random-initialised model first and THEN overwriting it with pretrained
    # weights (briefly needing ~2x the model's memory footprint). Instead
    # weights are loaded straight onto their target tensors. For a ~303M
    # param model in fp32 this is the difference between a peak of ~2.4GB
    # and ~1.2GB during load -- exactly the class of allocation that fails
    # with "not enough memory: tried to allocate 2147450880 bytes" on a
    # machine without much free RAM.
    model = CLIPVisionModelWithProjection.from_pretrained(  # type: ignore[arg-type]
        model_name, low_cpu_mem_usage=True
    )
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    n = sum(p.numel() for p in model.parameters())
    print(f"[clip] {model_name}  params={n/1e6:.1f}M  device={device}")
    if n > 2e9:
        raise RuntimeError(f"{n/1e9:.2f}B params exceeds the 2B limit")
    return model, device


@torch.no_grad()
def embed(
    model,
    device: str,
    dataset: ViewDataset,
    feature: str = "proj",
    batch_size: int = 32,
    num_workers: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (feats, labels):
        feats  (N, V, D) float16
        labels (N,) int64,  -1 marks an image that failed to load
    """
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=num_workers > 0,
    )
    n_views = len(dataset.views)
    use_amp = device == "cuda"

    feats, labels = [], []
    for i, (x, y) in enumerate(dl):
        b = x.shape[0]
        x = x.view(b * n_views, 3, RES, RES).to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            vout = model.vision_model(pixel_values=x)
            pooled = vout.pooler_output                       # (B*V, H)
            if feature == "pooled":
                f = pooled
            elif feature == "proj":
                f = model.visual_projection(pooled)           # (B*V, P)
            elif feature == "both":
                f = torch.cat([pooled, model.visual_projection(pooled)], dim=-1)
            else:
                raise ValueError(f"unknown feature {feature!r}")
        feats.append(f.float().view(b, n_views, -1).cpu().numpy().astype(np.float16))
        labels.append(y.numpy())
        if i % 20 == 0:
            print(f"  batch {i}/{len(dl)}", flush=True)

    return np.concatenate(feats, 0), np.concatenate(labels, 0)