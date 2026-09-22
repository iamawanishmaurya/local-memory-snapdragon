"""OCR regression tests (plan 01-03 t4).

Model-free tests (a, c, d) always run; the planted-string test (b) runs only
when an OCR backend is actually usable (trocr models present or easyocr
importable) and is skipped with a clear message otherwise.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_memory.extractors.ocr as ocr  # noqa: E402

PLANTED = "INVOICE 2026-03-14"


# ---------------------------------------------------------------- test a ----
class _FakeDecSession:
    """Decoder stub: argmax walks ids 5 -> 6 -> EOS(2), emitting real numpy
    logits arrays (which HAVE .tolist — the exact shape of the old dead-return
    bug where any numpy output made ocr_image return "")."""

    def __init__(self, vocab: int = 8):
        self.vocab = vocab
        self._step = 0

    def get_inputs(self):
        class _I:
            name = "input_ids"
            shape = [None, None]
        class _E:
            name = "encoder_hidden_states"
            shape = [None, None, None]
        class _U:
            name = "use_cache_branch"
            shape = [None]
        return [_I(), _E(), _U()]

    def run(self, _outputs, _feed):
        logits = np.full((1, 1, self.vocab), -10.0, dtype=np.float32)
        seq = [5, 6, 2]  # token ids: "Hello", "world", EOS
        idx = min(self._step, len(seq) - 1)
        logits[0, 0, seq[idx]] = 10.0
        self._step += 1
        return [logits]


class _FakeEncSession:
    def get_inputs(self):
        class _I:
            name = "pixel_values"
            shape = [1, 3, 384, 384]
        return [_I()]

    def run(self, _outputs, _feed):
        return [np.zeros((1, 4, 8), dtype=np.float32)]


def test_dead_return_regression(monkeypatch, tmp_path):
    """ocr_image must NOT silently return '' when the TrOCR session returns a
    numpy logits array (old code: `if hasattr(out, 'tolist'): return ''`)."""
    tokenizer = {2: "</s>", 5: "Hello", 6: "Ġworld"}
    monkeypatch.setattr(
        ocr, "_try_trocr",
        lambda: (_FakeEncSession(), _FakeDecSession(), tokenizer, 1),
    )
    from PIL import Image
    png = tmp_path / "tiny.png"
    Image.new("RGB", (64, 32), "white").save(png)
    text = ocr.ocr_image(png)
    assert text.strip(), "dead return regression: ocr_image returned empty "
    "string despite a decodable logits stream"
    assert "Hello" in text and "world" in text


# ---------------------------------------------------------------- test b ----
def _backend_usable() -> bool:
    return ocr.ocr_available()


@pytest.mark.skipif(not _backend_usable(), reason="no OCR backend: run "
                    "'python scripts/setup_models.py --trocr' (or pip install "
                    "easyocr) to enable the planted-string OCR test")
def test_planted_string_from_rendered_png(tmp_path):
    from PIL import Image, ImageDraw, ImageFont
    font = None
    for name in ("arial.ttf", "segoeui.ttf", "calibri.ttf"):
        try:
            font = ImageFont.truetype(name, 64)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    img = Image.new("RGB", (900, 140), "white")
    ImageDraw.Draw(img).text((30, 35), PLANTED, fill="black", font=font)
    png = tmp_path / "planted.png"
    img.save(png)
    text = ocr.ocr_image(png)
    norm = lambda s: re.sub(r"\s+", " ", s or "").strip().lower()
    assert norm(PLANTED) in norm(text), f"planted string not found in {text!r}"


# ---------------------------------------------------------------- test c ----
def test_active_backend_reported():
    backend = ocr.active_backend()
    assert isinstance(backend, str) and backend
    assert backend in {"trocr", "easyocr", "none"}


# ---------------------------------------------------------------- test d ----
def test_preprocess_shape_and_normalization():
    from PIL import Image
    gray = Image.new("RGB", (100, 50), (128, 128, 128))
    arr = ocr._preprocess(gray)
    assert arr.shape == (1, 3, 384, 384)
    assert arr.dtype == np.float32
    means = arr.reshape(3, -1).mean(axis=1)
    assert np.all(np.abs(means) < 0.05), f"per-channel means not ~0: {means}"
