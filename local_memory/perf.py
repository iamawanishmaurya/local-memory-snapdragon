"""Performance measurement for the pitch (DEMO-05 / D-05).

All functions are never-raise: a missing model or a failed benchmark degrades
to 0 / "unknown" so /api/perf never 500s — measurement must never break the
demo. The honest-NPU rule lives here too: the badge derives from
``active_provider_of`` (actual dispatch), never from ``available_providers``
(capability listing), so a machine that silently falls back to CPU says so.
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path

PROCESS_START = time.perf_counter()

# Where "Record these numbers" appends (DEMO-05). Module-level so tests can
# redirect it; the endpoint resolves relative to the repo root by default.
METRICS_PATH = Path(__file__).resolve().parent.parent / "docs" / "pitch-metrics.md"

_cold_start: dict | None = None
_npu_check: dict | None = None
_npu_checked_at: float = 0.0
_latencies: list[float] = []
LATENCY_WINDOW = 100


def check_npu_live() -> dict:
    """Negotiated provider for the nomic model via a real session (cached 60s)."""
    global _npu_check, _npu_checked_at
    if _npu_check is not None and time.monotonic() - _npu_checked_at < 60:
        return _npu_check
    try:
        from . import config
        from .embeddings import base as emb_base

        session = emb_base.create_session(Path(config.MODELS_DIR) / "nomic-embed-text.onnx")
        active = emb_base.active_provider_of(session)
        _npu_check = {
            "requested": "QNN",
            "active": active,
            "npu_live": active == "QNNExecutionProvider",
        }
    except Exception as exc:
        _npu_check = {"requested": "QNN", "active": "unknown", "npu_live": False,
                      "note": type(exc).__name__}
    _npu_checked_at = time.monotonic()
    return _npu_check


def record_cold_start() -> dict | None:
    """Wall seconds from process start to the first successful warmup embed,
    per embedder (text / CLIP image / CLIP text). None on failure."""
    global _cold_start
    if _cold_start is not None:
        return _cold_start
    out: dict[str, float] = {}
    try:
        from .embeddings import get_text_embedder

        t0 = time.perf_counter()
        get_text_embedder().encode_batch(["warmup"])
        out["text_embed_s"] = round(time.perf_counter() - t0, 3)
    except Exception:
        pass
    try:
        from PIL import Image

        from .embeddings.clip_image import get_image_embedder

        probe = Image.new("RGB", (64, 64))
        t0 = time.perf_counter()
        get_image_embedder().encode_image(probe)
        out["clip_image_s"] = round(time.perf_counter() - t0, 3)
    except Exception:
        pass
    try:
        from .embeddings.clip_image import get_clip_text_embedder

        t0 = time.perf_counter()
        get_clip_text_embedder().encode_query("warmup")
        out["clip_text_s"] = round(time.perf_counter() - t0, 3)
    except Exception:
        pass
    if out:
        _cold_start = out
    return _cold_start


def append_latency(ms: float) -> None:
    _latencies.append(ms)
    if len(_latencies) > LATENCY_WINDOW:
        del _latencies[: len(_latencies) - LATENCY_WINDOW]


def latency_stats() -> dict:
    if not _latencies:
        return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0}
    try:
        if len(_latencies) == 1:
            v = round(_latencies[0], 1)
            return {"count": 1, "p50_ms": v, "p95_ms": v}
        qs = statistics.quantiles(_latencies, n=20)  # nearest-rank-ish p50/p95
        return {
            "count": len(_latencies),
            "p50_ms": round(qs[9], 1),
            "p95_ms": round(qs[18], 1),
        }
    except Exception:
        return {"count": len(_latencies), "p50_ms": 0.0, "p95_ms": 0.0}


def reset_latency() -> None:
    _latencies.clear()


def throughput() -> dict:
    """Reuse bench stages for the indexing story; 0 on failure."""
    out = {"embed_chunks_per_s": 0.0, "index_files_per_s": 0.0}
    try:
        from . import bench

        r = bench.bench_text_embed(n_chunks=16)
        out["embed_chunks_per_s"] = round(float(r.get("chunks_per_s", 0.0)), 1)
    except Exception:
        pass
    try:
        from . import bench

        r = bench.bench_index_scan()
        out["index_files_per_s"] = round(float(r.get("files_per_s", 0.0)), 1)
    except Exception:
        pass
    return out
