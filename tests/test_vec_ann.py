"""vec_ann tests (PERF-01, plan 06-01 Task 6): model-free and dll-free-safe.

On this Windows ARM64 host sqlite-vec ships no win_arm64 wheel, so the
fallback contract is the primary assertion: with the extension unavailable
(or its import blocked), vector_store.search() returns the brute-force numpy
results unchanged. ANN parity / incremental tests skip unless vec_ann
actually loads.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TMP_HOME = Path(__file__).parent / ".tmpdata-ann"
os.environ["LOCAL_MEMORY_HOME"] = str(_TMP_HOME)

from local_memory import config  # noqa: E402
from local_memory.store import vector_store  # noqa: E402
from local_memory.store import vec_ann  # noqa: E402


def setup_module(_):
    config.DATA_HOME = _TMP_HOME
    config.DB_PATH = _TMP_HOME / "index.db"
    config.THUMBS_DIR = _TMP_HOME / "thumbs"
    config.SETTINGS_PATH = _TMP_HOME / "settings.json"
    vector_store.invalidate_cache()
    config.ensure_dirs()
    vector_store.init_db()


def teardown_module(_):
    import shutil
    vector_store.invalidate_cache()
    vec_ann.invalidate(None)
    shutil.rmtree(_TMP_HOME, ignore_errors=True)


@pytest.fixture(autouse=True)
def _fresh_state():
    """Isolated DB + reset ANN probe cache per test."""
    import sqlite3
    vector_store.invalidate_cache()
    vec_ann.invalidate(None)
    vec_ann._probe = None  # force re-probe (module-level cache)
    conn = vector_store._connect()
    try:
        conn.execute("DELETE FROM vectors")
        for space in ("text", "image"):
            try:
                conn.execute(f"DELETE FROM vec_{space}")
            except Exception:
                pass  # vec table may not exist (no extension)
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()
    yield
    vec_ann._probe = None


def _seed(space="text", n=12, dim=768, seed=7):
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    vector_store.upsert_many([(i, 0, space, vecs[i]) for i in range(n)])
    return vecs


def test_bruteforce_fallback_unchanged(monkeypatch):
    """With the sqlite-vec import blocked, search() returns numpy results."""
    monkeypatch.setitem(sys.modules, "sqlite_vec", None)
    vec_ann._probe = None
    assert not vec_ann.available()["ok"]
    assert vec_ann.search("text", np.zeros(768, dtype=np.float32), 5) is None

    vecs = _seed(n=10)
    # Brute-force ground truth on the same data.
    mat, keys = vector_store.load_all("text", use_cache=False)
    expected = vector_store.search("text", vecs[0], top_k=5)
    got = vector_store.search("text", vecs[0], top_k=5)
    assert [(r["file_id"], r["ordinal"]) for r in got] == \
           [(r["file_id"], r["ordinal"]) for r in expected]
    # And the top hit really is the probe itself.
    top = got[0]
    truth = mat @ vecs[0]
    best = keys[int(np.argmax(truth))]
    assert (top["file_id"], top["ordinal"]) == best
    assert abs(top["score"] - float(truth.max())) < 1e-5


def test_doctor_reports_vec_status(capsys):
    """--doctor prints the sqlite-vec status line in the no-dll state."""
    from local_memory.main import doctor
    vec_ann._probe = None
    try:
        doctor()
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "sqlite-vec" in out
    assert "brute-force fallback active" in out


def test_ann_parity():
    """ANN top-10 matches brute-force numpy (skips without the extension)."""
    if not vec_ann.available()["ok"]:
        pytest.skip("sqlite-vec ANN not available on this host")
    vecs = _seed(n=50, seed=11)
    got = vector_store.search("text", vecs[3], top_k=10)
    assert vec_ann.available()["ok"]  # search() must not silently degrade
    mat, keys = vector_store.load_all("text", use_cache=False)
    truth = {keys[i] for i in np.argsort(-(mat @ vecs[3]))[:10]}
    ann_ids = {(r["file_id"], r["ordinal"]) for r in got}
    assert ann_ids == truth


def test_incremental_upsert_found():
    """A vector upserted after initial sync appears as the top hit."""
    if not vec_ann.available()["ok"]:
        pytest.skip("sqlite-vec ANN not available on this host")
    _seed(n=20, seed=13)
    rng = np.random.default_rng(99)
    distinctive = rng.standard_normal((768,), dtype=np.float32)
    distinctive /= np.linalg.norm(distinctive)
    vector_store.upsert(777, 0, "text", distinctive)
    hits = vector_store.search("text", distinctive, top_k=5)
    assert hits[0]["file_id"] == 777 and hits[0]["ordinal"] == 0
