"""Statistics: disk usage, index composition, and file breakdowns.

Read-only aggregate report for GET /api/stats. Every section is guarded —
any failure degrades to zeros/empty lists rather than failing the endpoint.
All SQL uses constant/parameterized queries only (no user input reaches this
module).
"""
from __future__ import annotations

import sqlite3

from . import config


def _file_size(p) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def dir_bytes(d) -> int:
    """Total bytes of all files under a directory (thumbnails)."""
    import os as _os
    from pathlib import Path as _P

    total = 0
    try:
        for root, _, files in _os.walk(d):
            for fn in files:
                total += _file_size(_P(root) / fn)
    except OSError:
        pass
    return total


def database_usage() -> dict:
    """Sizes of the live SQLite files plus the thumbnails cache."""
    db = config.DB_PATH
    out = {
        "index_db_bytes": _file_size(db) if db.exists() else 0,
        "wal_bytes": 0,
        "shm_bytes": 0,
        "thumbnails_bytes": dir_bytes(config.THUMBS_DIR) if config.THUMBS_DIR.exists() else 0,
    }
    for suffix, key in (("-wal", "wal_bytes"), ("-shm", "shm_bytes")):
        p = config.DATA_HOME / (db.name + suffix)
        if p.exists():
            out[key] = _file_size(p)
    out["total_bytes"] = out["index_db_bytes"] + out["wal_bytes"] + out["shm_bytes"] + out["thumbnails_bytes"]
    return out


def file_breakdowns() -> dict:
    """Files count plus by-kind / by-extension / by-folder aggregates."""
    from .store import database

    out = {"total": 0, "by_kind": [], "by_extension": [], "by_folder": []}
    try:
        conn: sqlite3.Connection = database._conn()
        out["total"] = conn.execute("SELECT COUNT(*) c FROM files").fetchone()["c"]
        out["by_kind"] = [
            {"kind": r["kind"], "count": r["c"], "bytes": r["s"]}
            for r in conn.execute(
                "SELECT kind, COUNT(*) c, COALESCE(SUM(size_bytes),0) s "
                "FROM files GROUP BY kind ORDER BY s DESC"
            ).fetchall()
        ]
        out["by_extension"] = [
            {"ext": r["ext"], "count": r["c"], "bytes": r["s"]}
            for r in conn.execute(
                "SELECT ext, COUNT(*) c, COALESCE(SUM(size_bytes),0) s "
                "FROM files GROUP BY ext ORDER BY s DESC LIMIT 10"
            ).fetchall()
        ]
        # The files.folder column already holds the watched root path
        # (pipeline assigns it per scanned folder).
        out["by_folder"] = [
            {"folder": r["folder"], "count": r["c"], "bytes": r["s"]}
            for r in conn.execute(
                "SELECT folder, COUNT(*) c, COALESCE(SUM(size_bytes),0) s "
                "FROM files GROUP BY folder ORDER BY s DESC LIMIT 10"
            ).fetchall()
        ]
    except Exception:
        pass
    return out


def index_stats() -> dict:
    """Chunk/vectors/OCR composition of the index."""
    from .store import database, vector_store

    out = {
        "total_chunks": 0,
        "embedded_chunks": 0,
        "images_understood": 0,
        "ocr_files": 0,
        "db_page_count": 0,
        "db_page_size": 0,
    }
    try:
        conn = database._conn()
        out["total_chunks"] = conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
        out["images_understood"] = conn.execute(
            "SELECT COUNT(*) c FROM files WHERE kind='image'"
        ).fetchone()["c"]
        try:
            out["ocr_files"] = conn.execute(
                """SELECT COUNT(*) c FROM files
                   WHERE (ocr_text IS NOT NULL AND ocr_text != '')
                      OR (ocr_used IS NOT NULL AND ocr_used = 1)"""
            ).fetchone()["c"]
        except Exception:
            # Pre-migration DB (no ocr_used column) — count images only.
            out["ocr_files"] = conn.execute(
                "SELECT COUNT(*) c FROM files WHERE ocr_text IS NOT NULL AND ocr_text != ''"
            ).fetchone()["c"]
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        page_count = conn.execute("PRAGMA page_count").fetchone()[0]
        out["db_page_size"] = int(page_size)
        out["db_page_count"] = int(page_count)
    except Exception:
        pass
    try:
        out["embedded_chunks"] = int(vector_store.stats().get("vectors", 0))
    except Exception:
        pass
    return out


def full_report(watcher_running: bool = False, pipeline: dict | None = None) -> dict:
    """Assemble the whole /api/stats payload. Never raises."""
    try:
        db_usage = database_usage()
    except Exception:
        db_usage = {"index_db_bytes": 0, "wal_bytes": 0, "shm_bytes": 0,
                    "thumbnails_bytes": 0, "total_bytes": 0}
    try:
        files = file_breakdowns()
    except Exception:
        files = {"total": 0, "by_kind": [], "by_extension": [], "by_folder": []}
    try:
        index = index_stats()
    except Exception:
        index = {"total_chunks": 0, "embedded_chunks": 0, "images_understood": 0,
                 "ocr_files": 0, "db_page_count": 0, "db_page_size": 0}
    return {
        "database": db_usage,
        "files": files,
        "index": index,
        "activity": {
            "last_scan": pipeline or {"status": "idle", "queued": 0, "done": 0, "last_file": ""},
            "watcher_running": bool(watcher_running),
        },
    }
