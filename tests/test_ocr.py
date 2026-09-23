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


# ---------------------------------------------------------------- test e ----
def test_wide_image_uses_banded_ocr(monkeypatch):
    """A wide image must be split into text-line bands and OCR'd per band,
    not fed whole to the encoder (which squeezes it to 384x384)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1600, 720), "white")
    draw = ImageDraw.Draw(img)
    for i in range(6):
        draw.rectangle([20, 60 + i * 110, 1580, 100 + i * 110], fill="black")

    calls: list[tuple[int, int]] = []

    def fake_ocr_pil(crop):
        calls.append(crop.size)
        return "line"

    monkeypatch.setattr(ocr, "ocr_pil", fake_ocr_pil)
    text = ocr.ocr_image_banded(img)
    # 6 bands, each ink-trimmed to ~1570px wide -> 2 vertical chunks each.
    assert len(calls) == 12, calls
    assert text == "\n".join(["line"] * 12)
    # Each call is a horizontal slice of the full width, far shorter than
    # the whole 720px image, and no wider than the chunk cap (+pad slack).
    assert all(h < 720 for (_, h) in calls), calls
    assert all(w <= ocr._BAND_CHUNK_PX + 20 for (w, _) in calls), calls


def test_medium_band_not_split_or_trimmed_to_death(monkeypatch):
    """A medium-wide image whose single text run is < _BAND_CHUNK_PX must
    OCR as ONE whole band: trimming happens first, so chunk boundaries must
    never cut through a short text line."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (900, 140), "white")
    ImageDraw.Draw(img).rectangle([100, 50, 600, 90], fill="black")

    calls: list[tuple[int, int]] = []

    def fake_ocr_pil(crop):
        calls.append(crop.size)
        return "line"

    monkeypatch.setattr(ocr, "ocr_pil", fake_ocr_pil)
    ocr.ocr_image_banded(img)
    assert len(calls) == 1, calls
    # trimmed to the ink box (500px + padding), not the blank 900px canvas
    w, h = calls[0]
    assert 480 <= w <= 540, calls


def test_dark_mode_image_inverted_before_ocr(monkeypatch):
    """Dark-mode screenshots (light text on dark ground) must be inverted
    before ocr_pil: TrOCR is trained on dark text on a light background."""
    import numpy as np
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (400, 120), "black")
    # a single text-height stroke (>= 10 rows so _text_line_bands emits a
    # band), thin enough that the ink-trimmed crop stays mostly light after
    # inversion
    ImageDraw.Draw(img).rectangle([50, 40, 350, 54], fill="white")

    means: list[float] = []

    def fake_ocr_pil(crop):
        means.append(float(np.asarray(crop.convert("L")).mean()))
        return "line"

    monkeypatch.setattr(ocr, "ocr_pil", fake_ocr_pil)
    text = ocr.ocr_image_banded(img)
    assert text == "line"
    assert means and all(m > 128 for m in means), means


def test_blank_band_rows_are_skipped(monkeypatch):
    """Bands whose ink is too sparse (< 30 ink px) are dropped instead of
    burning an OCR call that would only return noise."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (600, 200), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 20, 550, 60], fill="black")   # real line
    draw.rectangle([290, 120, 294, 124], fill="black")  # 4x4 speck: rows hold
    # <=5 ink px so _text_line_bands never emits a band for it

    calls: list[tuple[int, int]] = []

    def fake_ocr_pil(crop):
        calls.append(crop.size)
        return "line"

    monkeypatch.setattr(ocr, "ocr_pil", fake_ocr_pil)
    ocr.ocr_image_banded(img)
    assert len(calls) == 1, calls  # only the real line; the speck is skipped


def test_wide_image_ocr_image_routes_banded(monkeypatch, tmp_path):
    """ocr_image must route images larger than the band threshold through
    ocr_image_banded rather than a single whole-image OCR call."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1600, 200), "white")
    ImageDraw.Draw(img).rectangle([20, 60, 1580, 120], fill="black")
    png = tmp_path / "wide.png"
    img.save(png)

    banded_calls: list = []
    whole_calls: list = []

    monkeypatch.setattr(ocr, "ocr_image_banded",
                        lambda im: (banded_calls.append(im), "banded")[1])
    monkeypatch.setattr(ocr, "_ocr_pil_trocr",
                        lambda *a, **k: whole_calls.append(a) or "whole")
    monkeypatch.setattr(ocr, "_try_trocr",
                        lambda: (object(), object(), {}, 1))
    text = ocr.ocr_image(png)
    assert text == "banded"
    assert len(banded_calls) == 1 and not whole_calls

    # A small image still goes through the whole-image path.
    small = tmp_path / "small.png"
    Image.new("RGB", (200, 100), "white").save(small)
    banded_calls.clear()
    text = ocr.ocr_image(small)
    assert text == "whole" and not banded_calls
