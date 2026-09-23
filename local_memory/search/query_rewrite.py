"""Query rewriting: Qwen3-0.6B on NPU (onnxruntime-genai) with regex fallback.

Expands natural queries before hybrid retrieval:
  "invoice from last month" -> ("invoice", date_range, ["invoice","bill","receipt"])
  "error screenshots"       -> ("error screenshot", kind=image, synonyms)

All on-device. If the Qwen GenAI bundle (models/qwen3-0.6b/ with
genai_config.json) or onnxruntime-genai is absent, a deterministic regex
rewriter provides date/kind/synonym expansion so search still improves.

Soft-fail contract (D-03): rewrite() NEVER raises. Any load/generation
failure degrades to the regex result with a logged warning, and the regex
output is byte-identical to the pre-LLM implementation when the model is
absent (merge-not-replace: the LLM only contributes synonym terms).
"""
from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(os.environ.get("LOCAL_MEMORY_MODELS",
                   Path(__file__).resolve().parent.parent.parent / "models"))

# Directory candidates probed for a GenAI bundle (must contain
# genai_config.json — a bare .onnx export is NOT loadable via og.Model).
_BUNDLE_DIRS = ("qwen3-0.6b", "qwen3-0.6b-int8", "qwen3-0.6b-qnn")

# Hard generation cap: at most 32 new tokens of synonyms per rewrite.
_MAX_NEW_TOKENS = 32

_SYNONYMS = {
    "invoice": ["invoice", "bill", "billing", "receipt", "payment"],
    "error": ["error", "exception", "traceback", "failed", "bug", "screenshot"],
    "screenshot": ["screenshot", "capture", "screen", "error"],
    "resume": ["resume", "cv", "curriculum"],
    "recipe": ["recipe", "cook", "ingredients", "bake"],
    "pen": ["pen", "pencil", "stationery"],
    "book": ["book", "notebook", "diary"],
}

_MONTHS = {m.lower(): i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june",
     "july", "august", "september", "october", "november", "december"])}

_gen_model = None
_gen_tok = None
_gen_tried = False
_gen_load_reason: str = "not tried"


def _register_qnn_ep_once() -> None:
    """Best-effort reuse of embeddings.base's cached QNN EP registration so
    OGA and embedding sessions don't race on the QNN provider DLL. If the
    helper is not importable, OGA simply runs on CPU EP and still works."""
    try:
        from ..embeddings import base as emb_base
        emb_base._ensure_qnn_registered()
    except Exception:
        pass


def _try_genai():
    """Load the Qwen3 GenAI bundle, trying ALL candidate dirs.

    Returns the loaded (model, tokenizer) tuple or None. Negative results
    are cached in _gen_tried. Every candidate failure is logged and the
    loop continues — a broken first candidate must not skip the rest
    (regression: the old early-return-inside-loop bug).
    """
    global _gen_model, _gen_tok, _gen_tried, _gen_load_reason
    if _gen_tried:
        return _gen_model if _gen_model is not None else None
    _gen_tried = True

    # Env override for tests / alternate installs.
    override = os.environ.get("LOCAL_MEMORY_QWEN_DIR")
    candidates = [Path(override)] if override else [_MODELS_DIR / d for d in _BUNDLE_DIRS]

    last_reason = "no bundle dir found"
    for cand in candidates:
        # A true GenAI bundle requires genai_config.json inside the dir.
        if not (cand / "genai_config.json").is_file():
            continue
        try:
            import onnxruntime_genai as og  # type: ignore
            _register_qnn_ep_once()
            model = og.Model(str(cand))
            tok = og.Tokenizer(model)
            _gen_model, _gen_tok = model, tok
            ep = "qnn" if _qnn_active() else "cpu"
            logger.info("rewriter: qwen3-genai loaded from %s (ep=%s)", cand, ep)
            _gen_load_reason = f"loaded from {cand.name} (ep={ep})"
            return _gen_model
        except ImportError as e:
            last_reason = f"onnxruntime_genai not importable ({e})"
            logger.warning("rewriter: bundle %s present but OGA import failed: %s", cand, e)
            continue
        except Exception as e:
            last_reason = f"{cand.name}: {type(e).__name__}: {e}"
            logger.warning("rewriter: failed to load bundle %s: %s", cand, e)
            continue
    _gen_load_reason = last_reason
    logger.info("rewriter: regex fallback (reason=%s)", _gen_load_reason)
    return None


def _qnn_active() -> bool:
    try:
        from ..embeddings import base as emb_base
        return emb_base.npu_active()
    except Exception:
        return False


def _prompt(query: str) -> str:
    """/no_think chat prompt (Qwen3 thinking-off) producing one line of
    comma-separated synonyms, optionally ending with '| image' or '| doc'."""
    return (
        "<|im_start|>system\n"
        "You expand search queries. /no_think\n"
        "Always answer with exactly ONE line: comma-separated synonym "
        "keywords for the query. If the query is clearly about images end "
        "the line with ' | image'; if clearly about documents, end with "
        "' | doc'. No explanations.\n"
        "Example: user asks 'broken screen photo' -> you answer: "
        "'cracked, display, picture, capture | image'\n"
        "Example: user asks 'contract paperwork' -> you answer: "
        "'agreement, document, signed | doc'<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _generate_terms(model, query: str, regex_terms: list[str]) -> dict | None:
    """Run one greedy Qwen3 generation and parse the synonym line.

    Returns {"terms": [...], "kind": "image"|"doc"|None} or None on any
    unparseable/empty output or exception (caller keeps the regex dict).
    """
    try:
        tok = _gen_tok if _gen_tok is not None else _try_genai_tok()
        if tok is None:
            return None
        import onnxruntime_genai as og  # type: ignore

        prompt = _prompt(query)
        ids = tok.encode(prompt)
        params = og.GeneratorParams(model)
        # Greedy decoding (deterministic rewrites) + hard token cap. OGA's
        # params style differs across versions: set_search_options() on 0.14+,
        # direct attrs / params.search.* on older builds.
        setopts = getattr(params, "set_search_options", None)
        cap = len(ids) + _MAX_NEW_TOKENS
        if setopts is not None:
            setopts(do_sample=False, max_length=cap)
        else:
            for ps in (params, getattr(params, "search", None)):
                if ps is None:
                    continue
                try:
                    ps.do_sample = False
                    ps.max_length = cap
                    break
                except AttributeError:
                    continue
        gen = og.Generator(model, params)
        gen.append_tokens(ids)
        text = ""
        for _ in range(_MAX_NEW_TOKENS):
            gen.generate_next_token()
            chunk = tok.decode([gen.get_sequence(0)[-1]])
            text += chunk
            # Stop on eos or at the end of a thinking block that Qwen3 emits
            # even under /no_think; a stray newline inside <think> must not
            # cut generation short.
            if "<|im_end|>" in text:
                break
            if "</think>" in text:
                tail = text.split("</think>", 1)[1]
                # Stop once an answer line after the thinking block is
                # complete (content followed by a newline).
                if "\n" in tail.lstrip():
                    break
        return _parse_line(text, regex_terms)
    except Exception as e:
        logger.warning("rewriter: generation failed (%s: %s) — regex output kept",
                       type(e).__name__, e)
        return None


def _try_genai_tok():
    if _try_genai() is not None:
        return _gen_tok
    return None


_KIND_VOCAB = ("image", "doc")


def _parse_line(text: str, regex_terms: list[str]) -> dict | None:
    """Parse 'bill, receipt, payment | doc' style output. Strict: anything
    malformed yields None so the regex result is used unchanged."""
    try:
        if not text:
            return None
        # Strip a Qwen3 thinking block (emitted even under /no_think).
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        elif "<think>" in text:
            return None  # still inside <think> at the token cap — unusable
        # Take the first non-empty line only; multi-line rambling is ignored.
        line = ""
        for raw in text.splitlines():
            if raw.strip():
                line = raw.strip()
                break
        if not line:
            return None
        kind = None
        m = re.search(r"\|\s*(image|doc)\s*$", line)
        if m:
            kind = m.group(1)
            line = line[: m.start()]
        # Small DQ models sometimes answer pipe-separated ("| doc | invoice")
        # — treat '|' as another separator. Kind-vocab words map to the kind
        # hint and never become search terms; any other kind word is ignored
        # (the LLM can only ADD within the image/doc vocabulary).
        line = line.replace("|", ",")
        # Weak DQ quantizations echo the query back ("invoice | last month |
        # image") — drop tokens whose every word is already covered by the
        # regex terms so the LLM only ever contributes NEW synonyms.
        query_words: set[str] = set()
        for rt in regex_terms:
            query_words.update(rt.split())
        terms: list[str] = []
        for raw_tok in line.split(","):
            t = raw_tok.strip().lower()
            t = re.sub(r"[^a-z0-9' ]+", "", t).strip()
            if t in _KIND_VOCAB:
                kind = kind or t
                continue
            if len(t) < 2 or len(t) > 24:
                continue
            if not re.match(r"^[a-z]", t):  # alphabetic-ish
                continue
            if t in regex_terms:  # drop regex-echoed originals
                continue
            if set(t.split()) <= query_words:  # echo of query words
                continue
            if t not in terms:
                terms.append(t)
        if not terms and not kind:
            return None
        return {"terms": terms, "kind": kind}
    except Exception:
        return None


def rewrite(query: str) -> dict:
    """Return {expanded, terms, kinds, date_from, date_to, backend}."""
    q = query.strip()
    low = q.lower()
    out = {"expanded": q, "terms": [q], "kinds": [], "date_from": None,
           "date_to": None, "backend": "regex"}
    # Date: "last month" / month names / "june".
    now = time.time()
    if "last month" in low:
        dt = datetime.now()
        first_this = datetime(dt.year, dt.month, 1).timestamp()
        prev_month = first_this - 86400 * 5
        d2 = datetime.fromtimestamp(prev_month)
        start = datetime(d2.year, d2.month, 1).timestamp()
        out["date_from"], out["date_to"] = start, first_this
    else:
        for name, num in _MONTHS.items():
            if name in low:
                y = datetime.now().year
                start = datetime(y, num, 1).timestamp()
                end = datetime(y + (num == 12), (num % 12) + 1, 1).timestamp()
                out["date_from"], out["date_to"] = start, end
                break
    # Kind hints.
    if any(w in low for w in ("screenshot", "screen", "capture", "photo", "image", "picture", "pen")):
        out["kinds"].append("image")
    if any(w in low for w in ("invoice", "pdf", "resume", "doc", "recipe")):
        out["kinds"].append("doc")
    # Synonym expansion.
    terms = set(re.findall(r"[\w']+", low))
    for key, syns in _SYNONYMS.items():
        if key in low:
            terms.update(syns)
    out["terms"] = sorted(terms) or [q]
    out["expanded"] = " ".join(out["terms"][:24])
    # NPU LLM synonym merge when a GenAI bundle is available (merge-not-
    # replace: regex dates/kinds are authoritative; the LLM only ADDS terms).
    # Soft-fail: _try_genai/_generate_terms are wrapped — rewrite() NEVER raises.
    model = None
    try:
        model = _try_genai()
    except Exception as e:
        logger.warning("rewriter: loader crashed (%s: %s) — regex result kept",
                       type(e).__name__, e)
    if model is not None:
        t0 = time.perf_counter()
        gen = _generate_terms(model, q, out["terms"])
        _record_rewrite_latency("qwen3-npu", (time.perf_counter() - t0) * 1000)
        if gen is not None:
            merged = sorted(set(out["terms"]) | set(gen["terms"]))
            out["terms"] = merged
            out["expanded"] = " ".join(merged[:24])
            # LLM kind hints stay within the search vocabulary (image/doc)
            # and only ever ADD — never invent or remove.
            if gen.get("kind") and gen["kind"] not in out["kinds"]:
                out["kinds"] = out["kinds"] + [gen["kind"]]
            out["backend"] = "qwen3-npu"
        else:
            logger.info("rewriter: LLM output unparseable — regex result kept")
    else:
        _record_rewrite_latency("regex", 0.0)
    return out


def rewriter_status() -> dict:
    """Doctor line source: backend in use, bundle presence, OGA import."""
    bundle_dir = None
    override = os.environ.get("LOCAL_MEMORY_QWEN_DIR")
    for cand in ([Path(override)] if override else [_MODELS_DIR / d for d in _BUNDLE_DIRS]):
        if (cand / "genai_config.json").is_file():
            bundle_dir = str(cand)
            break
    try:
        import onnxruntime_genai  # type: ignore  # noqa: F401
        genai_importable = True
    except Exception:
        genai_importable = False
    backend = "qwen3-npu" if _gen_model is not None else "regex"
    return {
        "backend": backend,
        "model": bundle_dir,
        "genai_importable": genai_importable,
        "reason": _gen_load_reason,
    }


def _record_rewrite_latency(backend: str, ms: float) -> None:
    """Per-backend rewrite latency counter (perf.py); never raises."""
    try:
        from .. import perf
        perf.append_rewrite_latency(backend, ms)
    except Exception:
        pass
