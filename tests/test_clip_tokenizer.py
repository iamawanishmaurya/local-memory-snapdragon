"""CLIP BPE tokenizer tests: golden vectors + semantic ordering (CORR-02).

Structural/golden tests run model-free (skip-with-warning when the merges
file is absent). The integration test needs models/clip-vit-b32-text.onnx
and models/clip-vit-b32-image.onnx; it is skipped when either is missing.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_memory.embeddings import clip_tokenizer as ct  # noqa: E402

_MERGES_PRESENT = ct.available()
_SOT, _EOT = 49406, 49407

needs_merges = pytest.mark.skipif(
    not _MERGES_PRESENT, reason=f"merges file missing: {ct.merges_path()}"
)


@pytest.mark.skipif(_MERGES_PRESENT, reason="merges file present")
def test_import_does_not_raise_without_merges():
    # With no merges file, encode must still return SOT/EOT framing, not raise.
    ids = ct.encode("a photo of a cat")
    assert ids[0] == _SOT and _EOT in ids


@needs_merges
def test_golden_vector_cat():
    ids = ct.encode("a photo of a cat")
    assert ids[:7] == [_SOT, 320, 1125, 539, 320, 2368, _EOT]


@needs_merges
def test_empty_text():
    assert ct.encode("")[:2] == [_SOT, _EOT]
    assert len(ct.encode("")) == 77


@needs_merges
def test_truncation_to_77():
    text = " ".join(f"word{i}" for i in range(100))
    ids = ct.encode(text)
    assert len(ids) == 77
    assert ids[-1] == _EOT
    assert ids[0] == _SOT
    # 75 content slots + SOT + EOT: no ids beyond the context window
    assert all(i < 49408 for i in ids)


@needs_merges
def test_non_ascii_round_trip():
    for text in ("café", "naïve résumé", "dog 🐱🐶"):
        ids = ct.encode(text)  # must not raise
        assert ids[0] == _SOT
        assert _EOT in ids
        assert all(0 <= i < 49408 for i in ids)


@needs_merges
def test_framing_and_range():
    for text in ("a photo of a cat", "INVOICE #1234 — TOTAL DUE", "x", "a,b; c"):
        ids = ct.encode(text)
        assert ids[0] == _SOT, text
        assert _EOT in ids, text
        assert all(0 <= i < 49408 for i in ids)
        assert all(i in (0,) or i <= _EOT for i in ids[ids.index(_EOT) :])


# --- Model integration (requires ONNX models; skipped otherwise) ---

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_NEEDS_MODELS = not (
    (_MODELS_DIR / "clip-vit-b32-text.onnx").exists()
    and (_MODELS_DIR / "clip-vit-b32-image.onnx").exists()
)


@pytest.mark.skipif(_NEEDS_MODELS, reason="CLIP ONNX models not present")
@pytest.mark.skipif(not _MERGES_PRESENT, reason="CLIP BPE merges file not present")
def test_semantic_ordering_dog_vs_cat(tmp_path):
    """cosine(dog-query, dog-image) > cosine(cat-query, dog-image).

    Impossible to pass with the old word-hash tokenizer — proves real BPE
    ids feed the text encoder in the shared CLIP space.
    """
    from PIL import Image

    from local_memory.embeddings.clip_image import (
        ClipImageEmbedder,
        ClipTextEmbedder,
    )

    # Solid-color fixture images (distinct hues).
    (tmp_path / "dog.jpg").write_bytes(b"")
    Image.new("RGB", (256, 256), (139, 69, 19)).save(tmp_path / "dog.jpg")
    Image.new("RGB", (256, 256), (40, 90, 200)).save(tmp_path / "cat.jpg")

    text_e = ClipTextEmbedder()
    img_e = ClipImageEmbedder()

    q_dog = text_e.encode_query("a photo of a dog")
    q_cat = text_e.encode_query("a photo of a cat")
    img_dog = img_e.encode_image(Image.open(tmp_path / "dog.jpg"))

    cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos(q_dog, img_dog) > cos(q_cat, img_dog)
