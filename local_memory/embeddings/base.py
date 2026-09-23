"""ONNX Runtime session helper with Snapdragon NPU (QNN) preference.

Provider priority: QNNExecutionProvider (NPU) -> CPU. GPU/DirectML is skipped
because on Windows-on-ARM the QNN EP is the supported accelerator path.

Every embedding model in this package loads through `create_session`, so the
whole embedding layer automatically runs on the NPU when the machine supports
it — including on x86 dev machines via CPU fallback so development works
anywhere.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_MODELS_DIR = Path(os.environ.get("LOCAL_MEMORY_MODELS", Path(__file__).resolve().parent.parent.parent / "models"))

# QNN / HTP tuning via env (set on Snapdragon target, harmless elsewhere).
#   LOCAL_MEMORY_QNN_CONTEXT=1  -> prefer *.serialized QNN context binaries
#   LOCAL_MEMORY_HTP_MODE=burst|sustained|balanced (default burst for bulk index)
#   LOCAL_MEMORY_BATCH=32       -> default NPU batch size for encode_batch
_HTP_MODE = os.environ.get("LOCAL_MEMORY_HTP_MODE", "burst")
_DEFAULT_BATCH = int(os.environ.get("LOCAL_MEMORY_BATCH", "32"))
_USE_CONTEXT = os.environ.get("LOCAL_MEMORY_QNN_CONTEXT", "0") == "1"


def default_batch_size() -> int:
    return max(1, _DEFAULT_BATCH)


def htp_mode() -> str:
    return _HTP_MODE


def context_binary_for(model_path: Path) -> Path | None:
    """Return sibling *.serialized QNN context binary if present and enabled."""
    if not _USE_CONTEXT:
        # Still auto-detect: if a .serialized sibling exists, prefer it.
        cand = model_path.with_suffix(".serialized")
        alt = model_path.parent / (model_path.stem + ".serialized")
        for p in (cand, alt):
            if p.exists():
                return p
        return None
    cand = model_path.with_suffix(".serialized")
    if cand.exists():
        return cand
    alt = model_path.parent / (model_path.stem + ".serialized")
    return alt if alt.exists() else None


class EmbeddingError(RuntimeError):
    pass


def available_providers() -> list[str]:
    try:
        import onnxruntime as ort
        _ensure_qnn_registered(ort)
        return list(ort.get_available_providers())
    except ImportError:
        return []


_qnn_registered = False


def _ensure_qnn_registered(ort=None) -> bool:
    """Register the QNN EP plug-in (onnxruntime-qnn 2.x ships it as a
    separate DLL: onnxruntime_providers_qnn.dll + QnnHtp.dll). Older 1.x
    builds had QNN built in — then this is a no-op. Cached after first try."""
    global _qnn_registered
    if _qnn_registered:
        return True
    try:
        import onnxruntime as _ort
        ort = ort or _ort
        if "QNNExecutionProvider" in ort.get_available_providers():
            _qnn_registered = True
            return True
        try:
            import onnxruntime_qnn as qnn_pkg  # type: ignore
        except ImportError:
            return False
        ort.register_execution_provider_library(
            qnn_pkg.get_ep_name() if hasattr(qnn_pkg, "get_ep_name") else "QNNExecutionProvider",
            qnn_pkg.get_library_path(),
        )
        _qnn_registered = "QNNExecutionProvider" in ort.get_available_providers()
        return _qnn_registered
    except Exception:
        return False


def npu_active() -> bool:
    return "QNNExecutionProvider" in available_providers()


def _qnn_backend_path() -> str:
    """Absolute path to QnnHtp.dll (ORT 2.x plug-in needs it; relative name
    fails DLL search and silently drops the QNN EP)."""
    try:
        import onnxruntime_qnn as qnn_pkg  # type: ignore
        p = qnn_pkg.get_qnn_htp_path()
        if p and Path(p).exists():
            return str(p)
    except Exception:
        pass
    return "QnnHtp.dll"


def create_session(model_path: Path):
    import onnxruntime as ort

    ctx = context_binary_for(model_path)
    # If only a context binary exists (no ONNX), report clearly.
    target = ctx if (ctx and not model_path.exists()) else model_path
    if not target.exists():
        raise EmbeddingError(
            f"Model not found: {model_path}. Run `python scripts/setup_models.py` once "
            f"(requires network) to export NPU models via Qualcomm AI Hub."
        )
    avail = available_providers()
    providers = []
    if "QNNExecutionProvider" in avail:
        # HTP perf tuning: burst = max throughput for bulk indexing,
        # sustained = thermal-safe for watcher/idle, balanced = default.
        qnn_opts = {"backend_path": _qnn_backend_path()}
        if htp_mode() in ("burst", "sustained", "balanced"):
            qnn_opts["htp_performance_mode"] = htp_mode()
        # Point at context binary when available (SoC-locked, fastest path).
        if ctx:
            qnn_opts["qnn_context_binary_path"] = str(ctx)
        providers.append(("QNNExecutionProvider", qnn_opts))
    # Adreno GPU via DirectML as middle tier when available (Windows).
    if "DmlExecutionProvider" in avail:
        providers.append("DmlExecutionProvider")
    providers.append("CPUExecutionProvider")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Oryon has many cores: let ORT use them for pre/post + CPU fallback.
    try:
        import os as _os
        opts.intra_op_num_threads = max(1, min(8, (_os.cpu_count() or 4)))
        opts.inter_op_num_threads = 1
    except Exception:
        pass
    return ort.InferenceSession(str(target), sess_options=opts, providers=providers)


def active_provider_of(session) -> str:
    try:
        return session.get_providers()[0]
    except Exception:
        return "unknown"


def normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


def pad_batch(ids_list: list[list[int]], pad_id: int = 0) -> np.ndarray:
    """Pad variable-length token id lists to a single [B, L] int64 batch."""
    if not ids_list:
        return np.zeros((0, 1), dtype=np.int64)
    max_len = max(len(x) for x in ids_list)
    max_len = max(1, max_len)
    out = np.full((len(ids_list), max_len), pad_id, dtype=np.int64)
    for i, ids in enumerate(ids_list):
        if ids:
            out[i, : len(ids)] = np.asarray(ids, dtype=np.int64)
    return out


def provider_report() -> dict:
    """Extended provider report for --doctor / /api/status (NPU+GPU+CPU)."""
    avail = available_providers()
    return {
        "available": avail,
        "qnn_available": "QNNExecutionProvider" in avail,
        "dml_available": "DmlExecutionProvider" in avail,
        "htp_mode": htp_mode(),
        "context_binaries_enabled": True,
        "default_batch": default_batch_size(),
    }
