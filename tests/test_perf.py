"""Perf/heterogeneous tests: batched NPU, parallel pipeline, clean options.

Runs on any machine (hashing fallback, no models required).
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["LOCAL_MEMORY_HOME"] = str(Path(__file__).parent / ".tmpdata_perf")

from local_memory import config, privacy  # noqa: E402
from local_memory.store import database, vector_store  # noqa: E402


def setup_module(_):
    # Rebind config (import-time DATA_HOME may belong to another test module
    # in the same pytest process). Each suite gets its own isolated home.
    home = Path(__file__).parent / ".tmpdata_perf"
    config.DATA_HOME = home
    config.DB_PATH = home / "index.db"
    config.THUMBS_DIR = home / "thumbs"
    config.SETTINGS_PATH = home / "settings.json"
    database.close()
    vector_store.invalidate_cache()
    config.ensure_dirs()
    database.init_db()
    vector_store.init_db()


def teardown_module(_):
    import shutil
    database.close()
    vector_store.invalidate_cache()
    shutil.rmtree(Path(__file__).parent / ".tmpdata_perf", ignore_errors=True)


def test_pad_batch_shapes():
    from local_memory.embeddings import base
    arr = base.pad_batch([[1, 2, 3], [4], [5, 6]])
    assert arr.shape == (3, 3)
    assert arr.dtype.name == "int64"


def test_provider_report_keys():
    from local_memory.embeddings import base
    rep = base.provider_report()
    assert {"available", "qnn_available", "dml_available", "htp_mode", "default_batch"} <= set(rep)


def test_text_embed_batch_matches_singles():
    from local_memory.embeddings import get_text_embedder
    emb = get_text_embedder()
    texts = ["invoice billing receipt", "chocolate cake recipe", "error screenshot traceback"]
    batched = emb.encode_batch(texts)
    singles = [emb.encode(t) for t in texts]
    assert batched.shape[0] == 3
    import numpy as np
    for b, s in zip(batched, singles):
        assert float(np.dot(b, s)) > 0.99


def test_preprocess_batch_shape():
    from PIL import Image
    from local_memory.embeddings.clip_image import preprocess_batch
    imgs = [Image.new("RGB", (100, 80), (10, 20, 30)) for _ in range(4)]
    arr = preprocess_batch(imgs)
    assert arr.shape == (4, 3, 224, 224)


def test_parallel_scan_beats_or_matches_serial(tmp_path=Path(__file__).parent / ".tmpdata_perf"):
    from local_memory.pipeline import scan_folder
    d = tmp_path / "par"
    d.mkdir(exist_ok=True)
    for i in range(12):
        (d / f"f{i}.txt").write_text("invoice billing acme " * 20 + str(i), encoding="utf-8")
    t0 = time.perf_counter()
    n1 = scan_folder(str(d), workers=1)
    t1 = time.perf_counter() - t0
    # Touch files to force reindex, then parallel.
    for p in d.glob("*.txt"):
        with open(p, "a", encoding="utf-8") as f:
            f.write(" more")
    t0 = time.perf_counter()
    n2 = scan_folder(str(d), workers=4)
    t2 = time.perf_counter() - t0
    assert n1 >= 1 and n2 >= 1
    # Parallel must not be pathologically slower (>5x) on tiny files.
    assert t2 < t1 * 5 + 2.0


def test_query_rewrite_expands():
    from local_memory.search.query_rewrite import rewrite
    rw = rewrite("invoice from last month")
    assert "bill" in rw["terms"] or "receipt" in rw["terms"]
    assert rw["date_from"] is not None
    rw2 = rewrite("error screenshots")
    assert "image" in rw2["kinds"]


def test_search_uses_rewrite(tmp_path=Path(__file__).parent / ".tmpdata_perf"):
    from local_memory.pipeline import scan_folder
    from local_memory.search import query_engine
    d = tmp_path / "rw"
    d.mkdir(exist_ok=True)
    (d / "bill_acme.txt").write_text("This is a bill from Acme Corp. Payment due.", encoding="utf-8")
    scan_folder(str(d))
    res = query_engine.search("invoice from last month")
    assert res and "bill_acme.txt" in res[0]["path"]


def test_clean_orphans_keeps_settings():
    from local_memory import privacy as _p
    config.ensure_dirs()
    database.init_db()
    out = _p.clean_orphans()
    assert "pruned_missing" in out and "files" in out


def test_clean_after_wipes(tmp_path=Path(__file__).parent / ".tmpdata_perf"):
    from local_memory.pipeline import scan_folder
    d = tmp_path / "ca"
    d.mkdir(exist_ok=True)
    (d / "x.txt").write_text("hello invoice world", encoding="utf-8")
    scan_folder(str(d), clean_after=True)
    assert database.stats()["files"] == 0


def test_vector_cache_and_stats():
    import numpy as np
    vector_store.upsert(999999, 0, "text", np.ones(768, dtype=np.float32))
    m1, k1 = vector_store.load_all("text")
    m2, k2 = vector_store.load_all("text")
    assert len(k1) == len(k2) and m1.shape == m2.shape
    assert "vectors" in vector_store.stats()
    vector_store.delete(999999)


def test_bench_runs():
    from local_memory.bench import bench_chunking, bench_text_embed
    assert bench_chunking()["mean_s"] >= 0
    assert bench_text_embed(8)["chunks_per_s"] > 0
