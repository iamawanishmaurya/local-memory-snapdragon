"""Model-free contract tests for the query rewriter (06-02 / D-03).

No network, no model downloads. onnxruntime_genai is stubbed via sys.modules
where an LLM path is exercised; when it is genuinely absent the regex path
must be byte-identical to the pre-06-02 implementation.
"""
from __future__ import annotations

import logging
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import pytest

from local_memory.search import query_rewrite as qr

pytestmark = pytest.mark.usefixtures("_rw_reset")


@pytest.fixture(autouse=True)
def _rw_reset(monkeypatch):
    """Reset the module-level genai cache and env override per test."""
    monkeypatch.delenv("LOCAL_MEMORY_QWEN_DIR", raising=False)
    qr._gen_model = None
    qr._gen_tok = None
    qr._gen_tried = False
    qr._gen_load_reason = "not tried"
    yield
    qr._gen_model = None
    qr._gen_tok = None
    qr._gen_tried = False
    qr._gen_load_reason = "not tried"


def _expected_regex(query: str) -> dict:
    """The pre-06-02 regex implementation, inlined verbatim: the fallback
    contract is byte-identical output to this reference."""
    q = query.strip()
    low = q.lower()
    out = {"expanded": q, "terms": [q], "kinds": [], "date_from": None,
           "date_to": None, "backend": "regex"}
    now = time.time()
    if "last month" in low:
        dt = datetime.now()
        first_this = datetime(dt.year, dt.month, 1).timestamp()
        prev_month = first_this - 86400 * 5
        d2 = datetime.fromtimestamp(prev_month)
        start = datetime(d2.year, d2.month, 1).timestamp()
        out["date_from"], out["date_to"] = start, first_this
    else:
        for name, num in qr._MONTHS.items():
            if name in low:
                y = datetime.now().year
                start = datetime(y, num, 1).timestamp()
                end = datetime(y + (num == 12), (num % 12) + 1, 1).timestamp()
                out["date_from"], out["date_to"] = start, end
                break
    if any(w in low for w in ("screenshot", "screen", "capture", "photo", "image", "picture", "pen")):
        out["kinds"].append("image")
    if any(w in low for w in ("invoice", "pdf", "resume", "doc", "recipe")):
        out["kinds"].append("doc")
    terms = set(__import__("re").findall(r"[\w']+", low))
    for key, syns in qr._SYNONYMS.items():
        if key in low:
            terms.update(syns)
    out["terms"] = sorted(terms) or [q]
    out["expanded"] = " ".join(out["terms"][:24])
    return out


class _StubGenerator:
    """Fake og.Generator emitting a scripted token stream."""

    def __init__(self, text: str, tokenizer):
        self._text = text
        self._tok = tokenizer
        self._pos = 0

    def append_tokens(self, ids):
        pass

    def generate_next_token(self):
        self._pos += 1

    def get_sequence(self, i):
        # Each call returns a "sequence" whose decode yields the next chunk;
        # length drives the loop cap.
        seq = types.SimpleNamespace()
        seq.__len__ = lambda: min(self._pos, 40)
        self._last = self._tok.decode([self._pos])
        return list(range(min(self._pos, 40)))

    def decode_last(self):
        return ""


class _StubTokenizer:
    def __init__(self, script: str):
        self.script = script

    def encode(self, prompt):
        return list(range(10))

    def decode(self, ids):
        # Emit the script char-by-char over the first N calls, then newline
        # so the stop-on-newline triggers.
        i = ids[0] - 1 if ids else 0
        if i < len(self.script):
            return self.script[i]
        return "\n"


def _install_stub_og(monkeypatch, script: str, fail_first_dir: str | None = None,
                     raise_in_generation: Exception | None = None):
    """Inject a fake onnxruntime_genai module. If fail_first_dir is set,
    og.Model raises for that directory name only (regression: the early-return
    bug skipped remaining candidates)."""
    calls: list[str] = []

    class StubParams:
        def __init__(self, model):
            self.do_sample = True
            self.max_length = 0

    class StubOG:
        exceptions = __import__("builtins")

        def __init__(self):
            self.model_calls = calls

        @staticmethod
        def Model(path):
            p = Path(path)
            calls.append(p.name)
            if fail_first_dir and p.name == fail_first_dir:
                raise RuntimeError("boom: first candidate broken")
            assert (p / "genai_config.json").is_file(), "bundle must contain genai_config.json"
            return object()

        @staticmethod
        def Tokenizer(model):
            return _StubTokenizer(script)

        @staticmethod
        def GeneratorParams(model):
            return StubParams(model)

        @staticmethod
        def Generator(model, params):
            if raise_in_generation:
                raise raise_in_generation
            tok = _StubTokenizer(script)
            return _StubGenerator(script, tok)

    monkeypatch.setitem(sys.modules, "onnxruntime_genai", StubOG)
    return calls


def _make_bundle(tmp_path: Path, name: str) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "genai_config.json").write_text("{}", encoding="utf-8")
    return d


def test_regex_byte_identical_without_model(tmp_path, monkeypatch):
    """No OGA, no models dir: rewrite() equals the reference regex output."""
    monkeypatch.setattr(qr, "_MODELS_DIR", tmp_path)  # no bundle inside
    for q in ("invoice from last month", "error screenshots march",
              "pen with blue book", "bills from March"):
        got = qr.rewrite(q)
        exp = _expected_regex(q)
        assert got == exp
        assert got["backend"] == "regex"
        if "pen" not in q:
            assert got["date_from"] is not None
            assert got["date_to"] < got["date_from"] + 86400 * 32
        else:
            assert got["date_from"] is None


def test_regex_backend_when_oga_missing_but_bundle_present(tmp_path, monkeypatch):
    """Bundle dir present but onnxruntime_genai unimportable -> clean skip."""
    _make_bundle(tmp_path, "qwen3-0.6b")
    monkeypatch.setattr(qr, "_MODELS_DIR", tmp_path)
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", None)  # import raises
    r = qr.rewrite("invoice from last month")
    assert r["backend"] == "regex"
    assert r == _expected_regex("invoice from last month")


def test_genai_tried_all_candidates(tmp_path, monkeypatch, caplog):
    """REGRESSION (early-return bug): a broken first candidate must not
    prevent the second from being attempted."""
    _make_bundle(tmp_path, "qwen3-0.6b")
    _make_bundle(tmp_path, "qwen3-0.6b-int8")
    monkeypatch.setattr(qr, "_MODELS_DIR", tmp_path)
    calls = _install_stub_og(monkeypatch, script="ok", fail_first_dir="qwen3-0.6b")
    with caplog.at_level(logging.WARNING):
        r = qr.rewrite("hello book")
    assert "qwen3-0.6b" in calls and "qwen3-0.6b-int8" in calls
    assert r["backend"] == "qwen3-npu"
    assert any("failed to load bundle" in rec.message for rec in caplog.records)


def test_llm_merge(tmp_path, monkeypatch):
    """Stub generator emits 'bill, receipt, payment | doc' -> merge semantics."""
    _make_bundle(tmp_path, "qwen3-0.6b")
    monkeypatch.setattr(qr, "_MODELS_DIR", tmp_path)
    _install_stub_og(monkeypatch, script="bill, receipt, payment | doc")
    r = qr.rewrite("invoice from last month")
    assert r["backend"] == "qwen3-npu"
    exp = _expected_regex("invoice from last month")
    assert r["date_from"] == exp["date_from"] and r["date_to"] == exp["date_to"]
    # regex terms preserved as a subset, stub tokens added
    assert set(exp["terms"]) <= set(r["terms"])
    assert {"bill", "receipt", "payment"} <= set(r["terms"])
    assert "doc" in r["kinds"]
    # regex kind hints still present
    assert set(exp["kinds"]) <= set(r["kinds"])


def test_llm_cannot_invent_kinds(tmp_path, monkeypatch):
    """Kind suffix outside {image, doc} is dropped by the parser."""
    _make_bundle(tmp_path, "qwen3-0.6b")
    monkeypatch.setattr(qr, "_MODELS_DIR", tmp_path)
    _install_stub_og(monkeypatch, script="thing, gadget | video")
    r = qr.rewrite("book")
    assert r["backend"] == "qwen3-npu"
    assert "video" not in r["kinds"]


def test_unparseable_llm_output_falls_back(tmp_path, monkeypatch):
    """Garbage / empty output -> regex dict unchanged, backend regex."""
    _make_bundle(tmp_path, "qwen3-0.6b")
    monkeypatch.setattr(qr, "_MODELS_DIR", tmp_path)
    for bad in ("<think>let me reason about", "", "   \n  ", "!!! , ,,"):
        qr._gen_model = None
        qr._gen_tok = None
        qr._gen_tried = False
        _install_stub_og(monkeypatch, script=bad)
        r = qr.rewrite("invoice from last month")
        assert r == _expected_regex("invoice from last month")
        assert r["backend"] == "regex"
        qr._gen_tried = False  # re-arm cache for next iteration


def test_generation_exception_soft_fails(tmp_path, monkeypatch):
    """Any injected exception in the generation path -> unchanged regex dict."""
    _make_bundle(tmp_path, "qwen3-0.6b")
    monkeypatch.setattr(qr, "_MODELS_DIR", tmp_path)
    _install_stub_og(monkeypatch, script="fine", raise_in_generation=RuntimeError("NPU OOM"))
    r = qr.rewrite("error screenshots")
    assert r == _expected_regex("error screenshots")
    assert r["backend"] == "regex"


def test_parse_line_rules():
    p = qr._parse_line
    assert p("bill, Receipt!, payment | doc", ["invoice"]) == {
        "terms": ["bill", "receipt", "payment"], "kind": "doc"}
    # regex-echoed originals dropped, >24 chars dropped, non-alphabetic dropped
    assert p("invoice, a" * 1 + ", x" * 0, ["invoice"]) is None
    assert p("bill, " + "t" * 30, []) == {"terms": ["bill"], "kind": None}
    assert p(None, []) is None
    assert p("multi\nsecond line | image", []) == {"terms": ["multi"], "kind": None}


def test_rewrite_never_raises(tmp_path, monkeypatch):
    """Soft-fail contract: rewrite() returns a well-formed dict under any
    hostile stub (Model raising, decode raising, etc.)."""
    monkeypatch.setattr(qr, "_MODELS_DIR", tmp_path)
    monkeypatch.setattr(qr, "_try_genai", lambda: (_ for _ in ()).throw(RuntimeError("kaboom")))
    r = qr.rewrite("invoice from last month")
    assert set(r) == {"expanded", "terms", "kinds", "date_from", "date_to", "backend"}
    assert r["backend"] == "regex"


def test_perf_rewrite_latency_recorded(monkeypatch):
    """Per-backend rewrite latency buckets land in perf.py."""
    from local_memory import perf
    perf.reset_rewrite_latency()
    qr._record_rewrite_latency("regex", 1.5)
    qr._record_rewrite_latency("qwen3-npu", 42.0)
    stats = perf.rewrite_latency_stats()
    assert stats["regex"]["count"] == 1
    assert stats["qwen3-npu"]["count"] == 1
    assert stats["qwen3-npu"]["p50_ms"] == 42.0
    perf.reset_rewrite_latency()


def test_rewriter_status_model_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(qr, "_MODELS_DIR", tmp_path)
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", None)
    st = qr.rewriter_status()
    assert st["backend"] == "regex"
    assert st["model"] is None
    assert st["genai_importable"] is False
