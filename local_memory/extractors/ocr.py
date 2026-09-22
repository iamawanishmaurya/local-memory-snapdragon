"""OCR: extract text from images.

Backend priority (all on-device):
1. TrOCR-small-printed ONNX (encoder on QNN/NPU via the shared session chain,
   decoder via DirectML/CPU fallback) — `models/trocr/`, best on Snapdragon
2. EasyOCR (CPU, works well on ARM; heavy install — optional per README)
3. None → empty string (image still gets a CLIP visual embedding)

Models are loaded lazily and cached process-wide. All inference is on-device.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_reader = None
_reader_tried = False
_trocr_tried = False
_trocr = None  # (encoder_session, decoder_session, tokenizer_json_path)
_trocr_note = ""

# CLIP/ViT ImageNet normalization (same constants as embeddings/clip_image.py —
# TrOCR uses the ViT image processor).
_IMAGENET_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_IMAGENET_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_IMG_SIZE = 384

_DECODER_START_TOKEN_ID = 0
_EOS_TOKEN_ID = 2
_MAX_DECODE_STEPS = 20


def _models_dir():
    import os as _os
    return Path(_os.environ.get("LOCAL_MEMORY_MODELS",
               Path(__file__).resolve().parent.parent.parent / "models"))


def _trocr_dir() -> Path:
    return _models_dir() / "trocr"


def ocr_hint() -> str:
    """Actionable install hint when no OCR backend is active ('' otherwise)."""
    if _trocr_dir().is_dir():
        return ""
    if _reader_present():
        return ""
    return "trocr model missing — run: python scripts/setup_models.py --trocr"


def _reader_present() -> bool:
    try:
        import easyocr  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def _load_tokenizer(path: Path) -> dict[int, str]:
    """id -> token map from a HF tokenizer.json vocab."""
    data = json.loads(path.read_text(encoding="utf-8"))
    vocab = data.get("model", {}).get("vocab", {})
    return {int(i): tok for tok, i in vocab.items()}


# Roberta-style byte-level BPE unicode table (reverse map for decoding).
def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BYTE_DECODER: dict[str, bytes] | None = None


def _byte_decoder() -> dict[str, bytes]:
    global _BYTE_DECODER
    if _BYTE_DECODER is None:
        b2u = _bytes_to_unicode()
        _BYTE_DECODER = {u: bytes([b]) for b, u in b2u.items()}
    return _BYTE_DECODER


def _decode_ids(ids: list[int], tokenizer: dict[int, str]) -> str:
    """Decode token ids to text (roberta byte-level: join, map bytes back)."""
    tokens = [tokenizer[i] for i in ids if i in tokenizer and tokenizer[i]]
    text = "".join(tokens).replace("Ġ", " ")
    try:
        bd = _byte_decoder()
        raw = b"".join(bd[ch] for ch in text if ch in bd)
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        pass
    return text.strip()


def _preprocess(img) -> np.ndarray:
    """384x384 CLIP-normalized float32 CHW tensor for the TrOCR encoder."""
    from PIL import Image
    img = img.convert("RGB").resize((_IMG_SIZE, _IMG_SIZE), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return arr.transpose(2, 0, 1)[np.newaxis, ...]  # [1,3,384,384]


def _try_trocr():
    """Load TrOCR ONNX sessions via base.create_session (QNN→DML→CPU chain)."""
    global _trocr_tried, _trocr, _trocr_note
    if _trocr_tried:
        return _trocr
    _trocr_tried = True
    try:
        from ..embeddings import base as _base
        d = _trocr_dir()
        enc_p = d / "encoder_model.onnx"
        dec_p = d / "decoder_model_merged.onnx"
        tok_p = d / "trocr-tokenizer.json"
        if enc_p.exists() and dec_p.exists() and tok_p.exists():
            enc = _base.create_session(enc_p)
            dec = _base.create_session(dec_p)
            _trocr = (enc, dec, _load_tokenizer(tok_p))
            _trocr_note = "trocr"
    except Exception:
        _trocr = None
    return _trocr


def _past_init(dec_sess) -> dict[str, np.ndarray]:
    """Zero-length past_key_values for the merged decoder's first pass."""
    past: dict[str, np.ndarray] = {}
    for inp in dec_sess.get_inputs():
        name = inp.name
        if "past" not in name.lower():
            continue
        shape = []
        for dim in inp.shape:
            shape.append(0 if isinstance(dim, str) or dim is None else int(dim))
        # The seq-len axis of a zero-length past may not be symbolic; force
        # the second axis (batch, heads, seq, head_dim) to 0 regardless.
        if len(shape) == 4:
            shape[2] = 0
        past[name] = np.zeros(shape, dtype=np.float32)
    return past


def _greedy_decode(encoder_out: np.ndarray, dec_sess, tokenizer: dict[int, str],
                   max_steps: int = _MAX_DECODE_STEPS) -> str:
    """Autoregressive argmax decode over decoder_model_merged.onnx (KV fed back)."""
    ids = [_DECODER_START_TOKEN_ID]
    past = _past_init(dec_sess)
    for step in range(max_steps):
        feed: dict[str, np.ndarray] = {}
        use_cache = step > 0
        for inp in dec_sess.get_inputs():
            low = inp.name.lower()
            if low == "input_ids":
                feed[inp.name] = np.array([ids], dtype=np.int64)
            elif "encoder_hidden_states" in low:
                feed[inp.name] = encoder_out
            elif "use_cache_branch" in low:
                feed[inp.name] = np.array([use_cache], dtype=bool)
            elif "past" in low and inp.name in past:
                feed[inp.name] = past[inp.name]
        out = dec_sess.run(None, feed)
        logits = np.asarray(out[0], dtype=np.float32)
        next_id = int(np.argmax(logits[0, -1]))
        # Collect present.* outputs as the next step's past (name-matched when
        # possible, otherwise positionally in output order).
        present = [o for o, meta in zip(out, dec_sess.get_outputs()) if "present" in meta.name.lower()]
        if present:
            past = {}
            past_inputs = [i.name for i in dec_sess.get_inputs() if "past" in i.name.lower()]
            if len(present) == len(past_inputs):
                past = dict(zip(past_inputs, [np.asarray(p, dtype=np.float32) for p in present]))
        if next_id == _EOS_TOKEN_ID:
            break
        ids.append(next_id)
    return _decode_ids(ids[1:], tokenizer)


def active_backend() -> str:
    if _try_trocr() is not None:
        return str(_trocr_note or "trocr")
    if _get_reader() is not None:
        return "easyocr"
    return "none"


def _get_reader():
    global _reader, _reader_tried
    if not _reader_tried:
        _reader_tried = True
        try:
            import easyocr  # type: ignore
            _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        except Exception:
            _reader = None
    return _reader


def ocr_image(path: Path) -> str:
    # 1. TrOCR ONNX path (models downloaded via setup_models --trocr).
    trocr = _try_trocr()
    if trocr is not None:
        enc_sess, dec_sess, tokenizer = trocr
        try:
            from PIL import Image
            arr = _preprocess(Image.open(path))
            enc_input = enc_sess.get_inputs()[0].name
            encoder_out = np.asarray(
                enc_sess.run(None, {enc_input: arr})[0], dtype=np.float32
            )
            text = _greedy_decode(encoder_out, dec_sess, tokenizer)
            if text:
                return text[:2000]
        except Exception:
            pass
    reader = _get_reader()
    if reader is None:
        return ""
    try:
        results = reader.readtext(str(path), detail=0, paragraph=True)
        return "\n".join(results)
    except Exception:
        return ""


def ocr_available() -> bool:
    return _try_trocr() is not None or _get_reader() is not None
