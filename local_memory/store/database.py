"""SQLite metadata store for indexed files and chunks."""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from .. import config

_lock = threading.RLock()
_shared_conn: sqlite3.Connection | None = None
_shared_path: str = ""


def _conn() -> sqlite3.Connection:
    """Single shared connection (check_same_thread=False + RLock).

    Replaces the old thread-local pool: thread-local handles kept the DB
    file locked on Windows after ThreadPool workers exited, breaking wipe.
    One shared handle + lock is safe for this workload and releases cleanly.
    """
    global _shared_conn, _shared_path
    with _lock:
        target = str(config.DB_PATH)
        if _shared_conn is None or _shared_path != target:
            if _shared_conn is not None:
                try:
                    _shared_conn.close()
                except Exception:
                    pass
            config.ensure_dirs()
            _shared_conn = sqlite3.connect(target, timeout=30, check_same_thread=False)
            _shared_conn.row_factory = sqlite3.Row
            _shared_conn.execute("PRAGMA journal_mode=WAL")
            _shared_conn.execute("PRAGMA synchronous=NORMAL")
            # Make the chunks ON DELETE CASCADE schema actually enforce.
            _shared_conn.execute("PRAGMA foreign_keys=ON")
            _shared_path = target
        return _shared_conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    folder TEXT NOT NULL,
    ext TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    mtime REAL NOT NULL,
    indexed_at REAL NOT NULL,
    kind TEXT NOT NULL,              -- 'text' | 'image' | 'pdf' | 'docx' | 'other'
    ocr_text TEXT DEFAULT ''         -- extracted/OCR text for images
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    start_offset INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_files_folder ON files(folder);
"""


def init_db() -> None:
    with _lock:
        _conn().executescript(SCHEMA)


def close() -> None:
    """Close the shared connection (needed before deleting the DB on Windows)."""
    global _shared_conn, _shared_path
    with _lock:
        if _shared_conn is not None:
            try:
                _shared_conn.close()
            except Exception:
                pass
            _shared_conn = None
            _shared_path = ""


def upsert_file(path: str, folder: str, ext: str, size: int, mtime: float, kind: str, ocr_text: str = "") -> int:
    now = time.time()
    with _lock:
        conn = _conn()
        conn.execute(
            """INSERT INTO files (path, folder, ext, size_bytes, mtime, indexed_at, kind, ocr_text)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(path) DO UPDATE SET
                 size_bytes=excluded.size_bytes, mtime=excluded.mtime,
                 indexed_at=excluded.indexed_at, kind=excluded.kind, ocr_text=excluded.ocr_text""",
            (path, folder, ext, size, mtime, now, kind, ocr_text),
        )
        conn.commit()
        row = conn.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()
        return int(row["id"])


def replace_chunks(file_id: int, chunks: list[tuple[int, str]]) -> None:
    """chunks: list of (ordinal, text). Replaces any previous chunks."""
    with _lock:
        conn = _conn()
        conn.execute("DELETE FROM chunks WHERE file_id=?", (file_id,))
        conn.executemany(
            "INSERT INTO chunks (file_id, ordinal, text) VALUES (?,?,?)",
            [(file_id, o, t) for o, t in chunks],
        )
        conn.commit()


def delete_file(path: str) -> int | None:
    """Delete a file row; returns the deleted file's id (None if absent)."""
    with _lock:
        conn = _conn()
        row = conn.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()
        if row is None:
            return None
        conn.execute("DELETE FROM files WHERE path=?", (path,))
        conn.commit()
        return int(row["id"])


def get_file(path: str) -> sqlite3.Row | None:
    with _lock:
        return _conn().execute("SELECT * FROM files WHERE path=?", (path,)).fetchone()


def file_by_id(file_id: int) -> sqlite3.Row | None:
    """Row for an indexed file id (used by /api/thumbnail — D-04 index-lookup-only)."""
    with _lock:
        return _conn().execute("SELECT * FROM files WHERE id=?", (file_id,)).fetchone()


def needs_index(path: str, mtime: float, size: int) -> bool:
    row = get_file(path)
    return row is None or row["mtime"] != mtime or row["size_bytes"] != size


def get_chunk_text(file_id: int, ordinal: int) -> str:
    with _lock:
        row = _conn().execute(
            "SELECT text FROM chunks WHERE file_id=? AND ordinal=?", (file_id, ordinal)
        ).fetchone()
        return row["text"] if row else ""


def list_files(folder: str | None = None) -> list[sqlite3.Row]:
    with _lock:
        if folder:
            return _conn().execute("SELECT * FROM files WHERE folder=? ORDER BY size_bytes DESC", (folder,)).fetchall()
        return _conn().execute("SELECT * FROM files ORDER BY size_bytes DESC").fetchall()


def stats() -> dict:
    with _lock:
        conn = _conn()
        n = conn.execute("SELECT COUNT(*) c FROM files").fetchone()["c"]
        n_chunks = conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
        n_images = conn.execute("SELECT COUNT(*) c FROM files WHERE kind='image'").fetchone()["c"]
        total = conn.execute("SELECT COALESCE(SUM(size_bytes),0) s FROM files").fetchone()["s"]
        return {"files": n, "chunks": n_chunks, "images": n_images, "total_bytes": total}


def remove_missing(folder: str) -> int:
    """Delete index rows whose files no longer exist on disk."""
    gone = 0
    for row in list_files(folder):
        if not Path(row["path"]).exists():
            delete_file(row["path"])
            gone += 1
    return gone


def vacuum() -> None:
    with _lock:
        conn = _conn()
        try:
            conn.execute("VACUUM")
            conn.commit()
        except Exception:
            pass


def delete_orphan_chunks() -> int:
    with _lock:
        conn = _conn()
        cur = conn.execute("DELETE FROM chunks WHERE file_id NOT IN (SELECT id FROM files)")
        conn.commit()
        return cur.rowcount or 0
