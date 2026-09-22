"""Image embeddings via CLIP ViT-B/32 image encoder (ONNX, QNN/NPU).

Also exposes the text-query side so natural-language queries can be compared
directly against image vectors (shared CLIP embedding space). For search, a
text query is embedded with CLIP's text encoder to retrieve images, and with
Nomic to retrieve documents — the query engine fuses both result sets.

Dev fallback mirrors nomic_text.HashingEncoder (hash-based, no real semantics).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from .. import config
from . import base
from .nomic_text import HashingEncoder

logger = logging.getLogger(__name__)

_IMG_SIZE = 224
_CLIP_VOCAB = 49408  # CLIP BPE vocab size (hash-fallback Gather bound)
_IMAGENET_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_IMAGENET_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((_IMG_SIZE, _IMG_SIZE), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return arr.transpose(2, 0, 1)[np.newaxis, ...]  # [1,3,224,224]


def preprocess_batch(imgs: list[Image.Image]) -> np.ndarray:
    """Batch preprocess to [B,3,224,224] — one vectorized normalize, Adreno-friendly layout."""
    if not imgs:
        return np.zeros((0, 3, _IMG_SIZE, _IMG_SIZE), dtype=np.float32)
    arrs = []
    for img in imgs:
        im = img.convert("RGB").resize((_IMG_SIZE, _IMG_SIZE), Image.BICUBIC)
        arrs.append(np.asarray(im, dtype=np.float32) / 255.0)
    batch = np.stack(arrs).astype(np.float32)  # [B,H,W,3]
    batch = (batch - _IMAGENET_MEAN.reshape(1, 1, 1, 3)) / _IMAGENET_STD.reshape(1, 1, 1, 3)
    return batch.transpose(0, 3, 1, 2)  # [B,3,H,W]


class ClipImageEmbedder:
    def __init__(self) -> None:
        path = config.MODELS_DIR / "clip-vit-b32-image.onnx"
        self.session = base.create_session(path)
        self.provider = base.active_provider_of(self.session)
        self.input_name = self.session.get_inputs()[0].name

    def encode_image(self, img: Image.Image) -> np.ndarray:
        out = self.session.run(None, {self.input_name: _preprocess(img)})[0]
        return base.normalize(out.reshape(-1))

    def encode_images(self, imgs: list[Image.Image], batch_size: int | None = None) -> np.ndarray:
        """Batched CLIP image inference: one NPU run per batch."""
        from . import base as _base

        if not imgs:
            return np.zeros((0, 512), dtype=np.float32)
        bs = batch_size or _base.default_batch_size()
        outs = []
        for i in range(0, len(imgs), bs):
            batch = preprocess_batch(imgs[i : i + bs])
            try:
                out = self.session.run(None, {self.input_name: batch})[0]
            except Exception:
                # Fixed-batch export: fall back to single runs.
                for im in imgs[i : i + bs]:
                    out1 = self.session.run(None, {self.input_name: _preprocess(im)})[0]
                    outs.append(_base.normalize(np.asarray(out1, dtype=np.float32).reshape(-1)))
                continue
            arr = np.asarray(out, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            for r in range(arr.shape[0]):
                outs.append(_base.normalize(arr[r].reshape(-1)))
        return _base.l2_normalize_rows(np.stack(outs))


class ClipTextEmbedder:
    """CLIP text encoder — projects a natural-language query into image space."""

    def __init__(self) -> None:
        path = config.MODELS_DIR / "clip-vit-b32-text.onnx"
        self.session = base.create_session(path)
        self.provider = base.active_provider_of(self.session)
        self.input_name = self.session.get_inputs()[0].name

    def encode_query(self, text: str, max_tokens: int = 77) -> np.ndarray:
        from . import clip_tokenizer

        if clip_tokenizer.available():
            # Real CLIP BPE ids: [SOT 49406] + bpe + [EOT 49407], padded to 77.
            ids = clip_tokenizer.encode(text, context_length=max_tokens)
        else:
            # No merges file: fall back to word-hash ids so search still runs,
            # but semantics are degraded — run scripts/setup_models.py.
            logger.warning(
                "CLIP BPE merges missing (%s) — falling back to hash tokenization; "
                "run scripts/setup_models.py for correct text queries",
                clip_tokenizer.merges_path(),
            )
            from .nomic_text import tokenize

            ids = [t % _CLIP_VOCAB for t in tokenize(text)[: max_tokens - 2]]
            if not ids:
                ids = [0]
            ids = ids[: max_tokens - 2]
        ids_arr = np.array([ids], dtype=np.int64)
        feed = {}
        for inp in self.session.get_inputs():
            low = inp.name.lower()
            if "mask" in low:
                continue  # CLIP text export takes only input_ids (verified)
            elif "token_type" in low or "segment" in low:
                feed[inp.name] = np.zeros_like(ids_arr)
            else:
                feed[inp.name] = ids_arr
        out = self.session.run(None, feed)[0]
        arr = np.asarray(out, dtype=np.float32)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        return base.normalize(arr.reshape(-1))


class _HashImageEmbedder(HashingEncoder):
    """Fallback: embed image by perceptual hashing (edges + color histogram)."""

    def encode_image(self, img: Image.Image) -> np.ndarray:
        small = img.convert("RGB").resize((16, 16))
        arr = np.asarray(small, dtype=np.float32)
        gray = arr.mean(axis=2)
        edges = np.abs(np.diff(gray, axis=0)).ravel() * 100
        hist = np.concatenate([(arr[..., c].ravel() // 32) for c in range(3)])
        text = " ".join(f"{int(x)}" for x in np.concatenate([edges, hist]))
        return self._one(text)

    def encode_images(self, imgs: list[Image.Image], batch_size: int | None = None) -> np.ndarray:
        return base.l2_normalize_rows(np.stack([self.encode_image(i) for i in imgs])) if imgs else np.zeros((0, 768), dtype=np.float32)


def get_image_embedder():
    try:
        return ClipImageEmbedder()
    except Exception:
        return _HashImageEmbedder()


def get_clip_text_embedder():
    try:
        return ClipTextEmbedder()
    except Exception:
        return _HashImageEmbedder()
