"""Image understanding pipeline: OCR text + CLIP visual embedding + thumbnail.

Adreno/CPU: thumbnails are parallelized in batch (ThreadPool) and cached;
CLIP preprocess is batched in embeddings/clip_image.preprocess_batch.
"""
from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from .. import config
from . import ocr


def make_thumbnail(path: Path, max_side: int = 320) -> Path | None:
    """Cache a small preview under data/thumbs. Returns thumb path or None."""
    try:
        # Stable sha1 of the absolute path: `hash()` is salted per-process in
        # Python 3, so old names could never be matched for deletion later.
        # Thumbs generated with the old salted scheme are simply ignored and
        # regenerated on the next index pass — no migration needed.
        digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()
        out = config.THUMBS_DIR / f"{digest}.jpg"
        if out.exists():
            return out
        img = Image.open(path)
        img.thumbnail((max_side, max_side))
        img.convert("RGB").save(out, "JPEG", quality=80)
        return out
    except Exception:
        return None


def describe_image(path: Path) -> tuple[str, str]:
    """Return (ocr_text, note). Visual semantics come from CLIP embeddings."""
    text = ocr.ocr_image(path)
    note = "image"
    if text:
        note += " + ocr"
    backend = ocr.active_backend()
    if backend and backend != "none":
        note += f" ({backend})"
    return text, note


def make_thumbnails(paths: list[Path], max_side: int = 320, workers: int | None = None) -> list[Path | None]:
    """Batch thumbnails on the Oryon pool (IO + PIL scale, Adreno-friendly JPEG)."""
    if not paths:
        return []
    n = workers or max(2, min(8, (os.cpu_count() or 8) - 2))
    with ThreadPoolExecutor(max_workers=n, thread_name_prefix="lm-thumb") as pool:
        return list(pool.map(lambda p: make_thumbnail(p, max_side), paths))
