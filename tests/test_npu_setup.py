"""NPU export flow mechanics (05-03): setup_models.py --npu, model-free & network-free.

Covers: token gate (fail-fast, no silent skip), target list / compile options,
idempotence (existing .serialized -> no job submitted), and the sibling
<stem>.serialized selection contract the runtime relies on
(local_memory/embeddings/base.py::context_binary_for — read-only dependency).

qai_hub is always mocked — the real API is never touched in tests/CI.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def _load_setup_models():
    spec = importlib.util.spec_from_file_location("setup_models", ROOT / "scripts" / "setup_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sm():
    return _load_setup_models()


def test_npu_token_gate():
    """--npu with QAI_HUB_API_TOKEN scrubbed must exit 1 with the actionable
    message — never silently skip (and never hit the network)."""
    env = {k: v for k, v in os.environ.items() if k != "QAI_HUB_API_TOKEN"}
    env["QAI_HUB_API_TOKEN"] = ""  # explicit empty must also fail fast
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "setup_models.py"), "--npu"],
        capture_output=True, text=True, env=env, timeout=60,
        cwd=str(ROOT),
    )
    assert r.returncode == 1, f"expected exit 1, got {r.returncode}: {r.stdout!r}"
    out = r.stdout + r.stderr
    assert "aihub.qualcomm.com" in out
    assert "QAI_HUB_API_TOKEN" in out


def test_npu_targets_and_options(sm):
    """Exactly the three encoder ONNX stems; TrOCR excluded (D-01); INT8
    context-binary options string present."""
    assert sm.NPU_TARGETS == [
        "nomic-embed-text.onnx",
        "clip-vit-b32-image.onnx",
        "clip-vit-b32-text.onnx",
    ]
    assert all(not t.startswith("trocr") for t in sm.NPU_TARGETS)
    for t in sm.NPU_TARGETS:
        assert t.endswith(".onnx")
    opts = sm.NPU_COMPILE_OPTIONS
    assert "qnn_context_binary" in opts
    assert "quantize_io" in opts
    assert "quantize_fulltype int8" in opts


def test_npu_idempotent(sm, monkeypatch, tmp_path):
    """Existing .serialized sibling -> skip everything; submit_compile_job and
    upload_model are never called."""
    stem = "_tmp_test"
    onnx = tmp_path / f"{stem}.onnx"
    ser = tmp_path / f"{stem}.serialized"
    onnx.write_bytes(b"x" * 200_000)
    ser.write_bytes(b"y" * 200_000)

    submitted = []
    monkeypatch.setattr(sm, "NPU_TARGETS", [f"{stem}.onnx"])
    monkeypatch.setattr(sm, "MODELS_DIR", tmp_path)

    import types
    fake_hub = types.SimpleNamespace(
        upload_model=lambda *a, **kw: submitted.append("upload") or (_ for _ in ()).throw(AssertionError("upload called")),
        submit_compile_job=lambda *a, **kw: submitted.append("submit") or (_ for _ in ()).throw(AssertionError("submit called")),
        get_devices=lambda: [],
        configure=lambda **kw: None,
    )
    # run_npu imports qai_hub lazily; patch sys.modules so the import resolves.
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "qai_hub", fake_hub)
    monkeypatch.setenv("QAI_HUB_API_TOKEN", "fake-token-for-test")
    monkeypatch.setattr(sm, "_npu_device", lambda hub, name: object())
    rc = sm.run_npu()
    assert rc == 0
    assert submitted == []


def test_npu_all_present_returns_zero_without_jobs(sm, monkeypatch, tmp_path):
    """Full idempotence path: every target already has a .serialized -> rc 0,
    no jobs, no network."""
    import types
    for stem in sm.NPU_TARGETS:
        (tmp_path / stem).write_bytes(b"x" * 200_000)
        (tmp_path / stem).with_suffix(".serialized").write_bytes(b"y" * 200_000)
    monkeypatch.setattr(sm, "MODELS_DIR", tmp_path)
    submitted = []
    fake_hub = types.SimpleNamespace(
        upload_model=lambda *a, **kw: submitted.append("upload"),
        submit_compile_job=lambda *a, **kw: submitted.append("submit"),
        get_devices=lambda: [],
        configure=lambda **kw: None,
    )
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "qai_hub", fake_hub)
    monkeypatch.setenv("QAI_HUB_API_TOKEN", "fake-token-for-test")
    monkeypatch.setattr(sm, "_npu_device", lambda hub, name: object())
    rc = sm.run_npu()
    assert rc == 0
    assert submitted == []


def test_context_binary_selection():
    """context_binary_for picks the sibling .serialized; npu_active reflects
    provider state (environment-dependent — informational on hosts without QNN).

    Skipped without any .serialized in models/ so CI (model-free) skips, not fails.
    """
    from local_memory.embeddings import base
    serialized = sorted(MODELS_DIR.glob("*.serialized")) if MODELS_DIR.exists() else []
    if not serialized:
        pytest.skip("no .serialized context binaries in models/ (run setup_models.py --npu with a real token)")
    onnx = MODELS_DIR / (serialized[0].stem + ".onnx")
    picked = base.context_binary_for(onnx)
    assert picked is not None
    assert picked.exists()
    assert picked.suffix == ".serialized"
    # Informational: npu_active() is True only on hosts with the QNN EP registered.
    active = base.npu_active()
    assert isinstance(active, bool)
