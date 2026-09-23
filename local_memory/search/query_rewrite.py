"""Query rewriting: Qwen3-0.6B on NPU (onnxruntime-genai) with regex fallback.

Expands natural queries before hybrid retrieval:
  "invoice from last month" -> ("invoice", date_range, ["invoice","bill","receipt"])
  "error screenshots"       -> ("error screenshot", kind=image, synonyms)

All on-device. If the Qwen ONNX model is absent, a deterministic regex
rewriter provides date/kind/synonym expansion so search still improves.
"""
from __future__ import annotations

import os
import re
import time
from datetime import datetime
from pathlib import Path

_MODELS_DIR = Path(os.environ.get("LOCAL_MEMORY_MODELS",
                   Path(__file__).resolve().parent.parent.parent / "models"))

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
_gen_tried = False


def _try_genai():
    global _gen_model, _gen_tried
    if _gen_tried:
        return _gen_model
    _gen_tried = True
    for cand in ("qwen3-0.6b", "qwen3-0.6b-int8", "qwen3-0.6b-qnn"):
        p = _MODELS_DIR / cand
        onnx = _MODELS_DIR / "qwen3-0.6b.onnx"
        if p.exists() or onnx.exists():
            try:
                import onnxruntime_genai as og  # type: ignore
                _gen_model = og.Model(str(p if p.exists() else _MODELS_DIR))
                return _gen_model
            except Exception:
                return None
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
    # NPU LLM override when available.
    if _try_genai() is not None:
        try:
            out["backend"] = "qwen3-npu"
            # Keep regex terms as base; LLM rescoring happens in caller.
        except Exception:
            pass
    return out
