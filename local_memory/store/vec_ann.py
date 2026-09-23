"""sqlite-vec ANN backend behind the vector_store interface (PERF-01).

Self-contained, guarded module: every entry point degrades to a graceful
sentinel (None / ok: False) when sqlite-vec is unusable on this host —
most notably Windows ARM64, where sqlite-vec 0.1.9 ships no win_arm64
wheel. The brute-force numpy path in vector_store.search() stays the
default until a vendored ``vec0-arm64.dll`` is built via
``scripts/build_vec0.ps1``.

Row identity: vec0 tables use a derived rowid ``(file_id << 20) | ordinal``
(see sync()). file_id/ordinal are ALSO declared as METADATA columns so KNN
queries can select them directly. The ``vectors`` table remains the source
of truth; vec tables are additive shadow structures (D-01).
"""
from __future__ import annotations

import logging
import sqlite3
import threading

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_probe: dict | None = None          # cached available() result
_sync_ok: set[str] = set()          # spaces whose vec table matched row counts
_query_warned = False               # warn once on query failure

_VENDORED_NAMES = ("vec0-arm64.dll", "vec0.dll")


def _vendored_paths():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    return [os.path.join(here, "bin", name) for name in _VENDORED_NAMES]


def available() -> dict:
    """Probe once (cached, thread-safe) whether the vec0 extension is loadable."""
    global _probe
    if _probe is not None:
        return dict(_probe)
    with _LOCK:
        if _probe is not None:
            return dict(_probe)
        result = {"ok": False, "source": "none", "error": None}
        try:
            import sqlite_vec  # type: ignore
        except Exception as e:
            result["error"] = f"import sqlite_vec failed: {e}"
            _probe = result
            return dict(result)
        conn = None
        try:
            conn = sqlite3.connect(":memory:")
            if not hasattr(conn, "enable_load_extension"):
                raise AttributeError("sqlite3.Connection has no enable_load_extension")
            try:
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                result.update(ok=True, source="wheel")
            except Exception:
                # Wheel shim has no usable DLL for this platform — try vendored.
                conn.enable_load_extension(False)
                import os
                loaded = False
                for path in _vendored_paths():
                    if not os.path.isfile(path):
                        continue
                    try:
                        conn.enable_load_extension(True)
                        conn.load_extension(path)
                        conn.enable_load_extension(False)
                        loaded = True
                        result.update(ok=True, source="vendored")
                        break
                    except Exception:
                        conn.enable_load_extension(False)
                if not loaded:
                    result["error"] = "no loadable vec0 DLL (no win_arm64 wheel / vendored dll absent)"
        except Exception as e:
            result["error"] = str(e)
        finally:
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass
        _probe = result
        return dict(result)


def _mark_unavailable(reason: str) -> None:
    global _probe
    with _LOCK:
        _probe = {"ok": False, "source": "none", "error": reason}
        _sync_ok.clear()


def _dims() -> dict:
    from . import vector_store
    return dict(vector_store.DIM)


def ensure_tables(conn: sqlite3.Connection) -> bool:
    """Create the vec_text / vec_image shadow tables. Guarded: never raises."""
    if not available()["ok"]:
        return False
    dims = _dims()
    try:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_text USING vec0("
            f"embedding float[{dims.get('text', 768)}] distance_metric=cosine, "
            f"file_id int METADATA, ordinal int METADATA)"
        )
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_image USING vec0("
            f"embedding float[{dims.get('image', 512)}] distance_metric=cosine, "
            f"file_id int METADATA, ordinal int METADATA)"
        )
        conn.commit()
        return True
    except sqlite3.OperationalError as e:
        _mark_unavailable(f"ensure_tables failed: {e}")
        return False
    except Exception as e:
        _mark_unavailable(f"ensure_tables failed: {e}")
        return False


def _rowid(file_id: int, ordinal: int) -> int:
    # Deterministic rowid: (file_id << 20) | ordinal. The 2^20 cap on ordinal
    # is safe: ordinals are per-file chunk indices (always small). file_id must
    # fit in ~43 bits to stay within SQLite's 64-bit signed rowid — guard below.
    rowid = (file_id << 20) | ordinal
    assert 0 <= ordinal < (1 << 20) and rowid < (1 << 62), "vec_ann rowid overflow: file_id/ordinal out of range"
    return rowid


def _decode_vec(blob: bytes, scale: float, dim: int):
    import numpy as np
    from . import vector_store
    if vector_store._QUANT:
        return vector_store._dequantize(blob, scale, dim)
    return np.frombuffer(blob, dtype=np.float32)


def sync(conn: sqlite3.Connection, space: str) -> bool:
    """Rebuild vec_<space> from the vectors table if row counts differ.

    Returns True if a rebuild happened. Never raises; on failure the ANN
    path is marked unavailable so the caller falls back to numpy.
    """
    if not available()["ok"]:
        return False
    if space in _sync_ok:
        return False
    try:
        n_vec = conn.execute("SELECT COUNT(*) FROM vectors WHERE space=?", (space,)).fetchone()[0]
        n_ann = conn.execute(f"SELECT COUNT(*) FROM vec_{space}").fetchone()[0]
        if n_vec == n_ann:
            _sync_ok.add(space)
            return False
        conn.execute(f"DELETE FROM vec_{space}")
        dim = _dims().get(space, 768)
        rows = conn.execute(
            "SELECT file_id, ordinal, vec, scale FROM vectors WHERE space=?", (space,)
        ).fetchall()
        batch = []
        for file_id, ordinal, blob, scale in rows:
            vec = _decode_vec(blob, float(scale if scale is not None else 1.0), dim)
            batch.append((_rowid(int(file_id), int(ordinal)), vec, int(file_id), int(ordinal)))
            if len(batch) >= 5000:
                conn.executemany(
                    f"INSERT INTO vec_{space}(rowid, embedding, file_id, ordinal) VALUES (?,?,?,?)",
                    batch,
                )
                batch = []
        if batch:
            conn.executemany(
                f"INSERT INTO vec_{space}(rowid, embedding, file_id, ordinal) VALUES (?,?,?,?)",
                batch,
            )
        conn.commit()
        _sync_ok.add(space)
        return True
    except Exception as e:
        _mark_unavailable(f"sync({space}) failed: {e}")
        return False


def search(space: str, query_vec, top_k: int = 24) -> list[dict] | None:
    """KNN via vec0. Returns [{file_id, ordinal, score}] desc, or None to
    signal the caller to use the brute-force numpy fallback."""
    global _query_warned
    if not available()["ok"]:
        return None
    conn = None
    try:
        from . import vector_store
        conn = vector_store._connect()
        if not ensure_tables(conn):
            return None
        sync(conn, space)
        import numpy as np
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        rows = conn.execute(
            f"SELECT file_id, ordinal, distance FROM vec_{space} "
            f"WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (q, top_k),
        ).fetchall()
        return [
            {"file_id": int(r[0]), "ordinal": int(r[1]), "score": 1.0 - float(r[2])}
            for r in rows
        ]
    except Exception as e:
        _mark_unavailable(f"search failed: {e}")
        if not _query_warned:
            logger.warning("sqlite-vec ANN query failed — falling back to numpy: %s", e)
            _query_warned = True
        return None
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass


def invalidate(space: str | None = None) -> None:
    """Drop the sync-state memo so the next search re-checks row counts."""
    with _LOCK:
        if space is None:
            _sync_ok.clear()
        else:
            _sync_ok.discard(space)


def status() -> dict:
    """Doctor-facing summary."""
    avail = available()
    out = {
        "available": avail["ok"],
        "source": avail["source"],
        "vec_text_rows": 0,
        "vec_image_rows": 0,
        "error": avail["error"],
    }
    if not avail["ok"]:
        return out
    conn = None
    try:
        from . import vector_store
        conn = vector_store._connect()
        if ensure_tables(conn):
            for space in ("text", "image"):
                out[f"vec_{space}_rows"] = conn.execute(
                    f"SELECT COUNT(*) FROM vec_{space}"
                ).fetchone()[0]
    except Exception as e:
        out["error"] = str(e)
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass
    return out
