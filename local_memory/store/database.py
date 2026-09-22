"""SQLite metadata store for indexed files and chunks."""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path

from .. import config

log = logging.getLogger(__name__)

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

-- FTS5 substrate (D-01): external-content virtual tables — no text is
-- duplicated, the index is derived data and can be dropped/rebuilt freely.
-- chunks.id IS the rowid; replace_chunks is DELETE+INSERT only, so chunks
-- need no UPDATE trigger. files is updated in place by upsert_file, so it
-- needs an AFTER UPDATE trigger (delete old + insert new).
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text)
    VALUES ('delete', old.id, old.text);
END;

-- NOTE: external-content FTS5 fetches column values from the content table
-- BY NAME, so the FTS column must be `path` (files has no `name` column).
CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
    path, ocr_text, content='files', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS files_ai AFTER INSERT ON files BEGIN
    INSERT INTO files_fts(rowid, path, ocr_text)
    VALUES (new.id, new.path, new.ocr_text);
END;
CREATE TRIGGER IF NOT EXISTS files_au AFTER UPDATE ON files BEGIN
    INSERT INTO files_fts(files_fts, rowid, path, ocr_text)
    VALUES ('delete', old.id, old.path, old.ocr_text);
    INSERT INTO files_fts(rowid, path, ocr_text)
    VALUES (new.id, new.path, new.ocr_text);
END;
CREATE TRIGGER IF NOT EXISTS files_ad AFTER DELETE ON files BEGIN
    INSERT INTO files_fts(files_fts, rowid, path, ocr_text)
    VALUES ('delete', old.id, old.path, old.ocr_text);
END;
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


# --- FTS5 substrate (Phase 3, D-01/D-02) ------------------------------------


def fts_rows(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    """Read-only helper for FTS MATCH queries from the search engine."""
    with _lock:
        return _conn().execute(sql, params).fetchall()


def chunk_snippet(chunk_id: int, match: str) -> str:
    """FTS5 snippet with <mark> highlighting for a chunk row (BM25 hits)."""
    with _lock:
        row = _conn().execute(
            "SELECT snippet(chunks_fts, 0, '<mark>', '</mark>', '…', 16) s "
            "FROM chunks_fts WHERE chunks_fts MATCH ? AND rowid=?",
            (match, chunk_id),
        ).fetchone()
        return row["s"] if row else ""


def file_snippet(file_id: int, match: str) -> str:
    """FTS5 snippet with <mark> highlighting over files_fts OCR text (col 1)."""
    with _lock:
        row = _conn().execute(
            "SELECT snippet(files_fts, 1, '<mark>', '</mark>', '…', 16) s "
            "FROM files_fts WHERE files_fts MATCH ? AND rowid=?",
            (match, file_id),
        ).fetchone()
        return row["s"] if row else ""


def backfill_fts() -> dict:
    """Rebuild the FTS indexes when they drift from the content tables (D-02).

    Drift detection: COUNT(*) on an external-content FTS vtab just counts the
    content table, so the FTS5 full integrity check —
    ``INSERT INTO t(t, rank) VALUES('integrity-check', 1)`` — is used instead;
    it raises when the index does not match the content rows. A drift triggers
    an FTS5 'rebuild', which repopulates from current content state and is
    inherently idempotent. Never raises — mirrors config.migrate_home().
    """
    out = {"rebuilt": False, "chunks": 0, "files": 0}
    try:
        with _lock:
            init_db()  # create virtual tables + triggers if missing
            conn = _conn()

            def _desync(table: str) -> bool:
                try:
                    conn.execute(f"INSERT INTO {table}({table}, rank) VALUES('integrity-check', 1)")
                    return False
                except sqlite3.DatabaseError:
                    return True

            c_desync = _desync("chunks_fts")
            f_desync = _desync("files_fts")
            if c_desync or f_desync:
                if c_desync:
                    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
                if f_desync:
                    conn.execute("INSERT INTO files_fts(files_fts) VALUES('rebuild')")
                conn.commit()
                out["rebuilt"] = True
            out["chunks"] = conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
            out["files"] = conn.execute("SELECT COUNT(*) c FROM files").fetchone()["c"]
    except Exception:
        log.warning("FTS backfill failed", exc_info=True)
    return out


def fts_status() -> dict:
    """Doctor report: FTS availability and index sync state.

    chunks/files/chunks_total/files_total are content-table counts (COUNT(*)
    on an external-content vtab mirrors the content table); index freshness
    comes from the FTS5 full integrity check (in_sync).
    """
    out = {"available": False, "in_sync": False,
           "chunks": 0, "files": 0, "chunks_total": 0, "files_total": 0}
    try:
        with _lock:
            conn = _conn()
            have = conn.execute(
                "SELECT COUNT(*) c FROM sqlite_master "
                "WHERE type='table' AND name IN ('chunks_fts','files_fts')"
            ).fetchone()["c"]
            out["chunks_total"] = conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
            out["files_total"] = conn.execute("SELECT COUNT(*) c FROM files").fetchone()["c"]
            if have == 2:
                out["available"] = True
                out["chunks"] = out["chunks_total"]
                out["files"] = out["files_total"]
                in_sync = True
                for t in ("chunks_fts", "files_fts"):
                    try:
                        conn.execute(f"INSERT INTO {t}({t}, rank) VALUES('integrity-check', 1)")
                    except sqlite3.DatabaseError:
                        in_sync = False
                out["in_sync"] = in_sync
    except Exception:
        pass
    return out
