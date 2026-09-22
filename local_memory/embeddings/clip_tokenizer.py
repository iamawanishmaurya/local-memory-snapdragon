"""Vendored CLIP BPE text tokenizer (pure Python, zero third-party deps).

Ported from OpenAI CLIP's `simple_tokenizer.py` (MIT license) so text queries
embed with the exact token ids the `clip-vit-b32-text.onnx` export was trained
on: SOT 49406 ... EOT 49407, whitespace/contraction regex, byte-level BPE over
the 50k merges in `models/bpe_simple_vocab_16e6.txt.gz`, 77-token context.

The merges file is provisioned by `scripts/setup_models.py`. If it is absent,
`available()` returns False and callers fall back (nothing raises at import).
"""
from __future__ import annotations

import functools
import gzip
import re
from pathlib import Path

from .. import config

_SOT = 49406
_EOT = 49407
_VOCAB_SIZE = 49408
_CONTEXT_LENGTH = 77

# Canonical CLIP whitespace/contraction split pattern (openai/CLIP repo).
# The original uses the third-party `regex` module with \p{L}/\p{N}; stdlib
# `re` equivalents: letters = [^\W\d_], one digit = \d, punctuation runs =
# (?:[^\s\w]|_)+ (underscore is \w but not a letter).
_PAT = re.compile(
    r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|(?:[^\s\w]|_)+""",
    re.IGNORECASE,
)

_MERGES_NAME = "bpe_simple_vocab_16e6.txt.gz"
# 49408 vocab = 256 byte ids + 256 </w>-marked ids + 48894 merge ids + SOT/EOT.
_NUM_MERGES = 49408 - 512 - 2

# byte-level encoder table (bytes_to_unicode from GPT-2/CLIP).
# Values are ordered exactly as CLIP's bytes_to_unicode() emits them — that
# order IS the id vocab order for the first 256 ids.
_byte_encoder: dict[int, str] = {}
_bs = (
    list(range(ord("!"), ord("~") + 1))
    + list(range(ord("¡"), ord("¬") + 1))
    + list(range(ord("®"), ord("ÿ") + 1))
)
_cs = _bs[:]
_n = 0
for _b in range(2**8):
    if _b not in _bs:
        _bs.append(_b)
        _cs.append(2**8 + _n)
        _n += 1
for _b, _c in zip(_bs, _cs):
    _byte_encoder[_b] = chr(_c)

_byte_decoder = {v: k for k, v in _byte_encoder.items()}
_BYTE_CHARS = list(_byte_encoder.values())  # id 0..255 in canonical order

_state: dict | None = None


def merges_path() -> Path:
    return config.MODELS_DIR / _MERGES_NAME


def available() -> bool:
    """True when the merges file is present and parseable."""
    try:
        return _load()["ok"]
    except Exception:
        return False


def _load() -> dict:
    """Memoized vocab/merges load. `ok` is False when the file is missing."""
    global _state
    if _state is not None:
        return _state
    state = {"ok": False, "bpe_ranks": {}, "vocab": {}}
    path = merges_path()
    try:
        merges: list[tuple[str, str]] = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            lines = f.read().split("\n")
        # Line 0 is the "#version" header (prefixed by the filename in this
        # file); merges run from line 1. The canonical CLIP vocab is exactly
        # 49408 = 256 bytes + 256 </w>-marked bytes + 48894 merges + SOT/EOT.
        for line in lines[1:]:
            parts = line.split()
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))
            if len(merges) >= _NUM_MERGES:
                break
        bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        # Canonical CLIP id vocab (simple_tokenizer.py): 256 byte chars in
        # bytes_to_unicode order, the same chars with '</w>' appended, the
        # merged pieces from the merges file, then SOT/EOT.
        vocab_list = _BYTE_CHARS + [c + "</w>" for c in _BYTE_CHARS]
        for pair in merges:
            vocab_list.append(pair[0] + pair[1])
        vocab_list += ["<|startoftext|>", "<|endoftext|>"]
        vocab = {tok: i for i, tok in enumerate(vocab_list)}
        state = {"ok": True, "bpe_ranks": bpe_ranks, "vocab": vocab}
    except Exception:
        pass
    _state = state
    return state


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


def _bpe(token: str, ranks: dict[tuple[str, str], int], cache: dict[str, str]) -> str:
    if token in cache:
        return cache[token]
    word = tuple(token[:-1]) + (token[-1] + "</w>",)  # CLIP end-of-word marker
    pairs = _get_pairs(word)
    if not pairs:
        return " ".join(word)
    while True:
        bigram = min(pairs, key=lambda p: ranks.get(p, float("inf")))
        if bigram not in ranks:
            break
        first, second = bigram
        new_word: list[str] = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
            except ValueError:
                new_word.extend(word[i:])
                break
            new_word.extend(word[i:j])
            i = j
            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        word = tuple(new_word)
        if len(word) == 1:
            break
        pairs = _get_pairs(word)
    out = " ".join(word)
    cache[token] = out
    return out


@functools.lru_cache(maxsize=4096)
def _encode_word(word: str) -> tuple[int, ...]:
    """Byte-encode + BPE one regex token into CLIP ids."""
    state = _load()
    if not state["ok"]:
        return ()
    ranks: dict = state["bpe_ranks"]
    vocab: dict = state["vocab"]
    cache = _word_cache
    text = "".join(_byte_encoder[b] for b in word.encode("utf-8"))
    ids = []
    for piece in _bpe(text, ranks, cache).split(" "):
        tid = vocab.get(piece)
        if tid is not None and tid < _VOCAB_SIZE:
            ids.append(tid)
    return tuple(ids)


_word_cache: dict[str, str] = {}


def encode(text: str, context_length: int = _CONTEXT_LENGTH) -> list[int]:
    """Encode text to CLIP ids: [SOT] + bpe[:75] + [EOT], padded with 0.

    Always context_length long (default 77); ids are always < 49408. Empty
    text yields [SOT, EOT, 0, ...]. Callers must check `available()` first —
    with no merges file this returns just the framing tokens.
    """
    state = _load()
    ids: list[int] = [_SOT]
    if state["ok"]:
        for tok in _PAT.findall(text.lower()):
            ids.extend(_encode_word(tok))
    return ids[: context_length - 1] + [_EOT] + [0] * (context_length - len(ids) - 1)
