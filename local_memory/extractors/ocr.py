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

# Normalization from trocr's DeiTFeatureExtractor preprocessor_config.json:
# mean = std = 0.5 (do_rescale 1/255 then normalize).
_IMAGENET_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_IMAGENET_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_IMG_SIZE = 384

_EOS_TOKEN_ID = 2
_PAD_TOKEN_ID = 1
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
    """id -> token map from a HF tokenizer.json vocab (dict or list form)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    vocab = data.get("model", {}).get("vocab", {})
    if isinstance(vocab, dict):
        return {int(i): tok for tok, i in vocab.items()}
    # SentencePiece-style list (index == id); entries may be [token] pairs.
    out: dict[int, str] = {}
    for i, entry in enumerate(vocab):
        tok = entry[0] if isinstance(entry, (list, tuple)) else entry
        out[i] = tok
    return out


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
    """Decode token ids to text (sentencepiece/roberta byte-level)."""
    pieces: list[bytes] = []
    bd = _byte_decoder()
    for i in ids:
        tok = tokenizer.get(i)
        if not tok:
            continue
        if tok.startswith("<0x") and tok.endswith(">"):  # byte-fallback token
            try:
                pieces.append(bytes([int(tok[3:-1], 16)]))
                continue
            except ValueError:
                pass
        pieces.append(tok.encode("utf-8"))
    text = b"".join(pieces).decode("utf-8", errors="ignore")
    text = text.replace("Ġ", " ").replace("▁", " ")
    # GPT-2 byte-level un-mangling; keep any char the table doesn't cover
    # (notably the spaces introduced above, which the table omits).
    text = "".join(bd[ch].decode("latin-1") if ch in bd else ch for ch in text)
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
            start_id = _PAD_TOKEN_ID
            gen_cfg = d / "generation_config.json"
            if gen_cfg.exists():
                try:
                    start_id = int(json.loads(gen_cfg.read_text(encoding="utf-8"))
                                   .get("decoder_start_token_id", _PAD_TOKEN_ID))
                except Exception:
                    pass
            _trocr = (enc, dec, _load_tokenizer(tok_p), start_id)
            _trocr_note = "trocr"
    except Exception:
        _trocr = None
    return _trocr


def _greedy_decode(encoder_out: np.ndarray, dec_sess, tokenizer: dict[int, str],
                   start_token_id: int = _PAD_TOKEN_ID,
                   max_steps: int = _MAX_DECODE_STEPS) -> str:
    """Autoregressive argmax decode over decoder_model_merged.onnx.

    Runs with use_cache_branch=False (full recompute per step): the merged
    KV-cache branch of this export produces corrupt continuations, while the
    recompute path decodes cleanly and the decoder is small enough that the
    quadratic cost is irrelevant at <=20 steps.
    """
    ids = [start_token_id]
    # past_key_values inputs are required by the graph even on the recompute
    # branch (use_cache_branch=False) — zero-length tensors satisfy them.
    zero_past: dict[str, np.ndarray] = {}
    for inp in dec_sess.get_inputs():
        if "past" not in inp.name.lower():
            continue
        dims = [1 if (d is None or isinstance(d, str)) else int(d) for d in inp.shape]
        if len(dims) == 4:
            dims[0], dims[2] = 1, 0  # batch=1, zero-length sequence
        zero_past[inp.name] = np.zeros(dims, dtype=np.float32)
    for _step in range(max_steps):
        feed: dict[str, np.ndarray] = {}
        for inp in dec_sess.get_inputs():
            low = inp.name.lower()
            if low == "input_ids":
                feed[inp.name] = np.array([ids], dtype=np.int64)
            elif "encoder_hidden_states" in low:
                feed[inp.name] = encoder_out
            elif "use_cache_branch" in low:
                feed[inp.name] = np.array([False], dtype=bool)
            elif "past" in low:
                feed[inp.name] = zero_past[inp.name]
        out = dec_sess.run(None, feed)
        logits = np.asarray(out[0], dtype=np.float32)
        next_id = int(np.argmax(logits[0, -1]))
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
        enc_sess, dec_sess, tokenizer, start_id = trocr
        try:
            from PIL import Image
            text = _ocr_pil_trocr(Image.open(path), enc_sess, dec_sess,
                                  tokenizer, start_id)
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


def ocr_pil(img) -> str:
    """OCR an in-memory PIL image (e.g. a PDF-embedded page image).

    Same backend chain as ocr_image: TrOCR ONNX first, then EasyOCR on the
    RGB numpy view. Returns '' when no backend is available or OCR fails —
    callers must treat '' as "no text", never as an error.
    """
    trocr = _try_trocr()
    if trocr is not None:
        enc_sess, dec_sess, tokenizer, start_id = trocr
        try:
            text = _ocr_pil_trocr(img, enc_sess, dec_sess, tokenizer, start_id)
            if text:
                return text[:2000]
        except Exception:
            pass
    reader = _get_reader()
    if reader is None:
        return ""
    try:
        import numpy as _np
        results = reader.readtext(_np.asarray(img.convert("RGB")), detail=0,
                                  paragraph=True)
        return "\n".join(results)
    except Exception:
        return ""


def _ocr_pil_trocr(img, enc_sess, dec_sess, tokenizer, start_id) -> str:
    arr = _preprocess(img)
    enc_input = enc_sess.get_inputs()[0].name
    encoder_out = np.asarray(
        enc_sess.run(None, {enc_input: arr})[0], dtype=np.float32
    )
    return _greedy_decode(encoder_out, dec_sess, tokenizer, start_id)


def ocr_available() -> bool:
    return _try_trocr() is not None or _get_reader() is not None
