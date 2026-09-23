"""Benchmark harness: times each pipeline stage, rates implementations.

Each entry measures wall-clock on synthetic data (no models needed — hashing
fallback exercises the same code paths as NPU). Run:

    python -m local_memory.main --bench
    python scripts/bench_all.py
    python scripts/demo_index.py --bench

Output is a ranked table + per-stage speedup vs the serial baseline, so you
can see exactly how much faster each phase got.
"""
from __future__ import annotations

import os
import statistics
import tempfile
import time
from pathlib import Path


def _swap_config(tmp: Path):
    """Point config at an isolated temp data-home. Returns restore closure."""
    import local_memory.config as _cfg
    from local_memory.store import database as _db, vector_store as _vs
    saved = (_cfg.DATA_HOME, _cfg.DB_PATH, _cfg.THUMBS_DIR, _cfg.SETTINGS_PATH)
    _db.close()
    _vs.invalidate_cache()
    _cfg.DATA_HOME = tmp / "data"
    _cfg.DB_PATH = tmp / "data" / "index.db"
    _cfg.THUMBS_DIR = tmp / "data" / "thumbs"
    _cfg.SETTINGS_PATH = tmp / "data" / "settings.json"

    def _restore():
        _db.close()
        _vs.invalidate_cache()
        _cfg.DATA_HOME, _cfg.DB_PATH, _cfg.THUMBS_DIR, _cfg.SETTINGS_PATH = saved

    return _restore


def _time(fn, repeats: int = 3):
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return min(samples), statistics.mean(samples)


def bench_chunking(n_words: int = 4000) -> dict:
    from .extractors import text_extractor
    text = " ".join(f"word{i}" for i in range(n_words))
    tmin, tmean = _time(lambda: text_extractor.chunk_words(text))
    return {"stage": "chunking", "min_s": round(tmin, 4), "mean_s": round(tmean, 4)}


def bench_text_embed(n_chunks: int = 64) -> dict:
    from .embeddings import get_text_embedder
    emb = get_text_embedder()
    texts = [f"benchmark chunk {i} invoice payment receipt" for i in range(n_chunks)]
    # Warmup (model load / session init excluded from timing below).
    try:
        emb.encode_batch(texts[:4])
    except Exception:
        pass
    tmin, tmean = _time(lambda: emb.encode_batch(texts))
    backend = getattr(emb, "provider", "hashing-fallback")
    return {"stage": f"text-embed x{n_chunks}", "backend": str(backend),
            "min_s": round(tmin, 4), "mean_s": round(tmean, 4),
            "chunks_per_s": round(n_chunks / tmin, 1) if tmin > 0 else 0}


def bench_image_preprocess(n: int = 16) -> dict:
    from PIL import Image
    from .embeddings.clip_image import preprocess_batch, _preprocess
    imgs = [Image.new("RGB", (640, 480), (i * 7 % 255, 100, 150)) for i in range(n)]
    tmin1, _ = _time(lambda: [_preprocess(i) for i in imgs])
    tmin2, _ = _time(lambda: preprocess_batch(imgs))
    speedup = round(tmin1 / tmin2, 2) if tmin2 > 0 else 1.0
    return {"stage": f"image-preprocess x{n}", "serial_s": round(tmin1, 4),
            "batched_s": round(tmin2, 4), "speedup": speedup}


def bench_index_scan(n_files: int = 40, workers: int | None = None) -> dict:
    from .pipeline import scan_folder
    from .store import database, vector_store
    tmp = Path(tempfile.mkdtemp(prefix="lm-bench-"))
    for i in range(n_files):
        (tmp / f"doc_{i:03d}.txt").write_text(
            f"Invoice {i} from Acme Corp for September services. Total {i*10} dollars. " * 8,
            encoding="utf-8")
    restore = _swap_config(tmp)
    try:
        database.init_db()
        vector_store.init_db()
        tmin, tmean = _time(lambda: scan_folder(str(tmp), recursive=False,
                                                workers=workers, clean_after=False), repeats=1)
    finally:
        restore()
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return {"stage": f"index-scan x{n_files} (workers={workers})",
            "min_s": round(tmin, 4), "files_per_s": round(n_files / tmin, 1) if tmin > 0 else 0}


def bench_search(n_files: int = 20) -> dict:
    from .search import query_engine
    from .store import database, vector_store
    from .pipeline import scan_folder
    tmp = Path(tempfile.mkdtemp(prefix="lm-bench-q-"))
    for i in range(n_files):
        (tmp / f"f_{i:03d}.txt").write_text(
            f"document {i} invoice billing receiptacme " * 6, encoding="utf-8")
    restore = _swap_config(tmp)
    try:
        database.init_db()
        vector_store.init_db()
        scan_folder(str(tmp), recursive=False)
        tmin, tmean = _time(lambda: query_engine.search("invoice from last month"))
    finally:
        restore()
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return {"stage": "search (hybrid, parallel fan-out)", "min_s": round(tmin, 4),
            "mean_s": round(tmean, 4), "qps": round(1 / tmin, 1) if tmin > 0 else 0}


def run_all() -> list[dict]:
    print("\nLocal Memory — benchmark (hashing fallback if no ONNX models; same code paths as NPU)\n")
    results = []
    for fn in (bench_chunking, bench_text_embed, bench_image_preprocess,
               lambda: bench_index_scan(40, workers=1),
               lambda: bench_index_scan(40, workers=None),
               bench_search):
        try:
            r = fn()
        except Exception as e:
            r = {"stage": getattr(fn, "__name__", "stage"), "error": str(e)}
        results.append(r)
        print(f"  {r}")
    # Ratings: compare serial vs parallel index scan.
    try:
        serial = next(r for r in results if "workers=1" in str(r.get("stage")))
        parallel = next(r for r in results if "workers=None" in str(r.get("stage")))
        if serial.get("min_s") and parallel.get("min_s"):
            sp = round(serial["min_s"] / parallel["min_s"], 2)
            print(f"\n  Parallel index speedup (serial -> Oryon pool): {sp}x")
            print("  Rating: " + ("EXCELLENT" if sp >= 2 else "GOOD" if sp >= 1.3 else "MODEST (small files = IO bound)"))
    except StopIteration:
        pass
    print("\nDone. Benchmark data was cleaned automatically (temp dirs removed).\n")
    return results
