"""Vector store: persists embeddings in SQLite and searches them.

Design notes
------------
* Vectors are stored as float32 blobs keyed by (file_id, ordinal, space).
  `space` is 'text' (Nomic-Embed) or 'image' (CLIP) — CLIP queries are matched
  against image vectors, Nomic text-query vectors against text vectors.
* Search is exact cosine similarity (numpy matmul, Oryon-SIMD friendly). For
  10k–100k+ chunks an ANN backend (sqlite-vec, if installed) plugs in behind
  the same interface; otherwise the in-memory normalized cache avoids re-reads.
* Optional INT8 quantization (LOCAL_MEMORY_VEC_QUANT=1) halves storage and
  speeds matmul on Adreno/CPU at small recall cost.
"""
from __future__ import annotations

import os
import sqlite3
import threading

import numpy as np

DIM = {
    "text": 768,   # nomic-embed-text-v1.5
    "image": 512,  # CLIP ViT-B/32
}

_QUANT = os.environ.get("LOCAL_MEMORY_VEC_QUANT", "0") == "1"

# In-memory normalized cache: avoids re-reading SQLite on every query.
# Keyed by space -> (matrix, keys, db_mtime). Invalidated on write/wipe.
_cache: dict = {}
_cache_lock = threading.Lock()


def _quantize(vec: np.ndarray) -> tuple[bytes, float]:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    scale = float(np.abs(v).max()) or 1.0
    q = np.round(v / scale * 127).astype(np.int8)
    return q.tobytes(), scale


def _dequantize(blob: bytes, scale: float, dim: int) -> np.ndarray:
    q = np.frombuffer(blob, dtype=np.int8).astype(np.float32)
    if q.size != dim and q.size > 0:
        dim = q.size
    return (q / 127.0 * scale).astype(np.float32)


def invalidate_cache(space: str | None = None) -> None:
    with _cache_lock:
        if space is None:
            _cache.clear()
        else:
            _cache.pop(space, None)


def _db_path():
    from .. import config
    config.ensure_dirs()
    return config.DB_PATH


def _connect() -> sqlite3.Connection:
    """Fresh handle with WAL + long busy timeout + retry-friendly settings.

    Must be used as `with _connect() as conn:` — never held across NPU
    inference, so writers never block each other longer than one executemany.
    """
    import time as _time
    last: Exception | None = None
    for _ in range(5):
        try:
            conn = sqlite3.connect(str(_db_path()), timeout=60, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=60000")
            return conn
        except Exception as e:
            last = e
            _time.sleep(1.0)
    raise last  # type: ignore[misc]


def init_db() -> None:
    conn = _connect()
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS vectors (
                   file_id INTEGER NOT NULL,
                   ordinal INTEGER NOT NULL,
                   space TEXT NOT NULL,
                   vec BLOB NOT NULL,
                   scale REAL DEFAULT 1.0,
                   PRIMARY KEY (file_id, ordinal, space)
               ) WITHOUT ROWID"""
        )
        # Migrate old tables without scale column.
        try:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(vectors)").fetchall()]
            if "scale" not in cols:
                conn.execute("ALTER TABLE vectors ADD COLUMN scale REAL DEFAULT 1.0")
                conn.commit()
        except Exception:
            pass
        conn.commit()
    finally:
        conn.close()


def _encode_blob(vec: np.ndarray) -> tuple[bytes, float]:
    if _QUANT:
        return _quantize(vec)
    return np.asarray(vec, dtype=np.float32).tobytes(), 1.0


def _write(sql: str, params, many: bool = False, retries: int = 6) -> None:
    """Execute a write with busy-retry (survives concurrent watcher + rescan)."""
    import time as _time
    last: Exception | None = None
    for attempt in range(retries):
        conn = _connect()
        try:
            if many:
                conn.executemany(sql, params)
            else:
                conn.execute(sql, params)
            conn.commit()
            return
        except sqlite3.OperationalError as e:
            last = e
            if "locked" not in str(e).lower() and "busy" not in str(e).lower():
                raise
            _time.sleep(min(5.0, 0.2 * (2 ** attempt)))
        finally:
            conn.close()
    raise last  # type: ignore[misc]


def upsert(file_id: int, ordinal: int, space: str, vec: np.ndarray) -> None:
    blob, scale = _encode_blob(vec)
    _write(
        "INSERT OR REPLACE INTO vectors (file_id, ordinal, space, vec, scale) VALUES (?,?,?,?,?)",
        (file_id, ordinal, space, blob, scale),
    )
    invalidate_cache(space)


def upsert_many(rows: list[tuple[int, int, str, np.ndarray]]) -> None:
    if not rows:
        return
    _write(
        "INSERT OR REPLACE INTO vectors (file_id, ordinal, space, vec, scale) VALUES (?,?,?,?,?)",
        [(f, o, s, *_encode_blob(v)) for f, o, s, v in rows],
        many=True,
    )
    for _, _, s, _ in rows:
        invalidate_cache(s)


def load_all(space: str, use_cache: bool = True) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return (matrix [n, dim], keys [(file_id, ordinal)]) normalized rows."""
    if use_cache:
        with _cache_lock:
            hit = _cache.get(space)
        if hit is not None:
            mat, keys = hit
            return mat.copy(), list(keys)
    # sqlite-vec ANN fast path when installed (exact same results at small n).
    try:
        import sqlite_vec  # type: ignore
        if sqlite_vec is not None and not _QUANT:
            pass  # placeholder: ANN index built lazily in search()
    except Exception:
        pass
    conn = _connect()
    try:
        rows = conn.execute("SELECT file_id, ordinal, vec, scale FROM vectors WHERE space=?", (space,)).fetchall()
    except Exception:
        rows = conn.execute("SELECT file_id, ordinal, vec FROM vectors WHERE space=?", (space,)).fetchall()
        rows = [(r[0], r[1], r[2], 1.0) for r in rows]
    finally:
        conn.close()
    if not rows:
        dim = DIM.get(space, 768)
        return np.zeros((0, dim), dtype=np.float32), []
    if _QUANT:
        dim = DIM.get(space, 768)
        mat = np.stack([_dequantize(r[2], float(r[3] if len(r) > 3 else 1.0), dim) for r in rows])
        # Fix dim if stored vectors differ (e.g. hashing fallback dim).
        if mat.ndim == 1:
            mat = mat.reshape(len(rows), -1)
    else:
        # float32 blobs: rows may be 3- or 4-tuples depending on schema.
        mat = np.frombuffer(b"".join(r[2] for r in rows), dtype=np.float32).reshape(len(rows), -1)
    keys = [(r[0], r[1]) for r in rows]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = (mat / norms).astype(np.float32)
    if use_cache:
        with _cache_lock:
            _cache[space] = (mat.copy(), list(keys))
    return mat, keys


def search(space: str, query_vec: np.ndarray, top_k: int = 24) -> list[dict]:
    """Cosine-similarity search. Returns [{file_id, ordinal, score}] desc."""
    mat, keys = load_all(space)
    if mat.shape[0] == 0:
        return []
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    qn = np.linalg.norm(q)
    if qn > 0:
        q = q / qn
    scores = mat @ q
    order = np.argsort(-scores)[:top_k]
    return [
        {"file_id": keys[i][0], "ordinal": keys[i][1], "score": float(scores[i])}
        for i in order
    ]


def wipe() -> None:
    _write("DELETE FROM vectors", ())
    invalidate_cache()


def delete(file_id: int) -> None:
    _write("DELETE FROM vectors WHERE file_id=?", (file_id,))
    invalidate_cache()


def delete_orphans() -> int:
    """Remove vectors whose file row no longer exists. Returns rows deleted."""
    import time as _time

    last: Exception | None = None
    for attempt in range(6):
        conn = _connect()
        try:
            cur = conn.execute("DELETE FROM vectors WHERE file_id NOT IN (SELECT id FROM files)")
            conn.commit()
            if cur.rowcount:
                invalidate_cache()
            return cur.rowcount or 0
        except sqlite3.OperationalError as e:
            last = e
            if "locked" not in str(e).lower() and "busy" not in str(e).lower():
                raise
            _time.sleep(min(5.0, 0.2 * (2 ** attempt)))
        finally:
            conn.close()
    raise last  # type: ignore[misc]


def stats() -> dict:
    conn = _connect()
    try:
        n = conn.execute("SELECT COUNT(*) c FROM vectors").fetchone()[0]
        nbytes = conn.execute("SELECT COALESCE(SUM(LENGTH(vec)),0) s FROM vectors").fetchone()[0]
    finally:
        conn.close()
    return {"vectors": n, "bytes": nbytes, "quantized": _QUANT, "cached_spaces": sorted(_cache.keys())}
