"""Text embeddings via Nomic-Embed-Text (ONNX, QNN/NPU when available).

Model: nomic-embed-text-v1.5 context-768, exported to ONNX by
`scripts/setup_models.py` (Qualcomm AI Hub → ONNX → QNN context binaries).

Dev fallback: if the ONNX model is absent, `HashingEncoder` provides a cheap
deterministic bag-of-features encoder so the whole pipeline stays testable on
machines without models. Real retrieval quality requires the Nomic model.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from .. import config
from . import base

_DIM = 768
_MAX_TOKENS = 256  # per-chunk token budget for the exported context
_BERT_VOCAB = 30522  # nomic-embed-text-v1.5 = BERT uncased vocab (Gather bound)

_WORD_RE = re.compile(r"[\w']+", re.UNICODE)

_tokenizer = None
_wordpiece = None


class _WordPiece:
    """Minimal BERT-uncased WordPiece (offline, no `tokenizers` dependency).

    Vocab is read from models/nomic-tokenizer.json. Greedy longest-match,
    [CLS]/[SEP] wrapped like the HF export expects.
    """

    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab = vocab
        self.unk = vocab.get("[UNK]", 100)

    def encode(self, text: str) -> list[int]:
        ids = [101]  # [CLS]
        for word in re.findall(r"[a-z0-9']+|[^a-z0-9'\s]", text.lower()):
            if word in self.vocab:
                ids.append(self.vocab[word])
                continue
            # Greedy longest-match-first with ## continuation.
            i, n, cur = 0, len(word), []
            while i < n:
                j = n
                while j > i:
                    piece = word[i:j] if i == 0 else "##" + word[i:j]
                    if piece in self.vocab:
                        cur.append(self.vocab[piece])
                        i = j
                        break
                    j -= 1
                else:
                    cur = [self.unk]
                    break
            ids.extend(cur)
        ids.append(102)  # [SEP]
        return ids


def _lazy_wordpiece():
    global _wordpiece
    if _wordpiece is None:
        try:
            import json as _json
            for cand in ("nomic-tokenizer.json", "tokenizer.json"):
                path = config.MODELS_DIR / cand
                if path.exists():
                    data = _json.loads(path.read_text(encoding="utf-8"))
                    vocab = data.get("model", {}).get("vocab") or data.get("vocab")
                    if vocab:
                        _wordpiece = _WordPiece(vocab)
                        break
        except Exception:
            _wordpiece = None
    return _wordpiece


def _lazy_tokenizer():
    """Use HF tokenizers lib if installed (bundled with model), else simple words."""
    global _tokenizer
    if _tokenizer is None:
        try:
            from tokenizers import Tokenizer  # type: ignore
            for cand in ("nomic-tokenizer.json", "tokenizer.json"):
                path = config.MODELS_DIR / cand
                if path.exists():
                    _tokenizer = ("hf", Tokenizer.from_file(str(path)))
                    break
            else:
                _tokenizer = ("hf", None)
        except Exception:
            _tokenizer = ("hf", None)
    return _tokenizer


def tokenize(text: str) -> list[int]:
    kind, tok = _lazy_tokenizer()
    if kind == "hf" and tok is not None:
        return [min(t, _BERT_VOCAB - 1) for t in tok.encode(text).ids]
    wp = _lazy_wordpiece()
    if wp is not None:
        return [t for t in wp.encode(text) if 0 <= t < _BERT_VOCAB]
    # Fallback: hashed wordpieces clamped to BERT vocab (avoids Gather OOB).
    return [abs(hash(w)) % _BERT_VOCAB for w in _WORD_RE.findall(text.lower())]


class NomicTextEmbedder:
    """ONNX Nomic-Embed-Text encoder. Output: masked-mean-pooled, normalized [768].

    The HF ONNX export takes (input_ids, token_type_ids, attention_mask) and
    returns last_hidden_state [B, L, 768]; we mean-pool over non-pad tokens.
    Extra inputs are auto-detected so QNN-context exports (single input) work.
    """

    def __init__(self) -> None:
        path = config.MODELS_DIR / "nomic-embed-text.onnx"
        self.session = base.create_session(path)
        self.provider = base.active_provider_of(self.session)
        self.input_names = [i.name for i in self.session.get_inputs()]

    def _feed(self, ids_arr: np.ndarray) -> dict:
        mask = (ids_arr != 0).astype(np.int64)
        feed = {}
        for name in self.input_names:
            low = name.lower()
            if "mask" in low:
                feed[name] = mask
            elif "token_type" in low or "segment" in low:
                feed[name] = np.zeros_like(ids_arr)
            else:  # input_ids (or single-input QNN export)
                feed[name] = ids_arr
        return feed

    def _pool(self, out: np.ndarray, mask: np.ndarray) -> np.ndarray:
        arr = np.asarray(out, dtype=np.float32)
        if arr.ndim == 3:  # [B, L, D] -> masked mean over tokens
            w = mask.reshape(arr.shape[0], arr.shape[1], 1).astype(np.float32)
            denom = w.sum(axis=1).clip(min=1e-6)
            arr = (arr * w).sum(axis=1) / denom
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        rows = []
        for r in range(arr.shape[0]):
            row = arr[r].reshape(-1)[:_DIM] if arr[r].size > _DIM else arr[r].reshape(-1)
            if row.size < _DIM:
                row = np.pad(row, (0, _DIM - row.size))
            rows.append(base.normalize(row))
        return base.l2_normalize_rows(np.stack(rows))

    def encode_batch(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """Batched NPU inference: one session.run per batch (padded), not per chunk.

        Falls back to per-row runs only if the model rejects batched inputs
        (some early exports are fixed batch=1).
        """
        import numpy as np_mod  # local alias to avoid shadowing

        if not texts:
            return np.zeros((0, _DIM), dtype=np.float32)
        bs = batch_size or base.default_batch_size()
        tokenized = []
        for t in texts:
            ids = tokenize(t)[:_MAX_TOKENS]
            tokenized.append(ids if ids else [0])
        vecs: list[np.ndarray] = []
        for i in range(0, len(tokenized), bs):
            chunk = tokenized[i : i + bs]
            ids_arr = base.pad_batch(chunk)
            feed = self._feed(ids_arr)
            try:
                out = self.session.run(None, feed)[0]
                vecs.extend(self._pool(out, (ids_arr != 0).astype(np.int64)))
            except Exception:
                # Fixed-batch export: retry rows one at a time.
                for j, ids in enumerate(chunk):
                    ids1 = np.array([ids], dtype=np.int64)
                    out = self.session.run(None, self._feed(ids1))[0]
                    vecs.extend(self._pool(out, (ids1 != 0).astype(np.int64)))
        return base.l2_normalize_rows(np.stack(vecs))

    def encode(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]


class HashingEncoder:
    """Deterministic feature-hashing fallback for dev machines without models."""

    provider = "hashing-fallback (no model loaded)"

    def encode_batch(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        return base.l2_normalize_rows(np.stack([self._one(t) for t in texts]))

    def encode(self, text: str) -> np.ndarray:
        return self._one(text)

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(_DIM, dtype=np.float32)
        words = _WORD_RE.findall(text.lower())
        grams = words + [f"{a}_{b}" for a, b in zip(words, words[1:])] + [f"{a}_{b}_{c}" for a, b, c in zip(words, words[1:], words[2:])]
        for g in grams:
            h = int.from_bytes(hashlib.md5(g.encode("utf-8")).digest()[:8], "little")
            v[h % _DIM] += 1.0 if h & 0x100 else -1.0
        return base.normalize(v)


def get_text_embedder():
    """Prefer the NPU Nomic model; fall back so the app still works."""
    try:
        return NomicTextEmbedder()
    except Exception:
        return HashingEncoder()
