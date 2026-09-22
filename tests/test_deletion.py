"""Zero-orphan deletion tests (CORR-03 / D-06).

Runs model-free: MODELS_DIR is pointed at an empty dir so every embedder
falls back to the deterministic hashing encoder.
"""
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TMP_HOME = Path(__file__).parent / ".tmpdata-deletion"
os.environ["LOCAL_MEMORY_HOME"] = str(TMP_HOME)

from local_memory import config  # noqa: E402

_ORIG_MODELS_DIR = config.MODELS_DIR

from local_memory.extractors import image_understanding  # noqa: E402
from local_memory.pipeline import handle_change, reconcile_deletions  # noqa: E402
from local_memory.store import database, vector_store  # noqa: E402


def setup_module(_):
    import shutil
    shutil.rmtree(TMP_HOME, ignore_errors=True)
    config.DATA_HOME = TMP_HOME
    config.DB_PATH = TMP_HOME / "index.db"
    config.THUMBS_DIR = TMP_HOME / "thumbs"
    config.SETTINGS_PATH = TMP_HOME / "settings.json"
    # Deterministic/no-model mode: hashing-encoder fallbacks, no ONNX load.
    config.MODELS_DIR = TMP_HOME / "no-models"
    database.close()
    vector_store.invalidate_cache()
    config.ensure_dirs()
    database.init_db()
    vector_store.init_db()


def teardown_module(_):
    import shutil
    database.close()
    vector_store.invalidate_cache()
    shutil.rmtree(TMP_HOME, ignore_errors=True)
    # Restore global state so sibling test modules see the real models dir.
    config.MODELS_DIR = _ORIG_MODELS_DIR


def _counts():
    conn = sqlite3.connect(str(config.DB_PATH))
    try:
        files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        vectors = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
    finally:
        conn.close()
    return files, chunks, vectors


def _make_image(tmp: Path, name: str) -> Path:
    from PIL import Image
    p = tmp / name
    img = Image.new("RGB", (64, 64), color=(120, 40, 200))
    img.save(p, "PNG")
    return p


def handle_change_index(path: Path) -> bool:
    from local_memory.pipeline import index_file
    return index_file(str(path))


def test_delete_removes_all_rows_and_thumbnail():
    tmp = TMP_HOME / "docs-a"
    tmp.mkdir(parents=True, exist_ok=True)
    p = _make_image(tmp, "shot.png")
    assert handle_change_index(p) is True

    row = database.get_file(str(p))
    assert row is not None
    file_id = int(row["id"])
    thumb = image_understanding.make_thumbnail(p)
    assert thumb and thumb.exists()

    assert _counts()[0] >= 1 and _counts()[2] >= 1

    p.unlink()
    handle_change("deleted", str(p))

    conn = sqlite3.connect(str(config.DB_PATH))
    try:
        assert conn.execute("SELECT COUNT(*) FROM files WHERE id=?", (file_id,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM chunks WHERE file_id=?", (file_id,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM vectors WHERE file_id=?", (file_id,)).fetchone()[0] == 0
    finally:
        conn.close()
    assert not thumb.exists(), "thumbnail must be deleted with the file"


def test_move_purges_old_path_and_indexes_new():
    tmp = TMP_HOME / "docs-b"
    tmp.mkdir(parents=True, exist_ok=True)
    p = _make_image(tmp, "before.png")
    assert handle_change_index(p) is True

    dest = tmp / "after.png"
    p.rename(dest)

    # Mimic watcher.on_moved: purge source, index destination.
    handle_change("deleted", str(p))
    handle_change("changed", str(dest))

    conn = sqlite3.connect(str(config.DB_PATH))
    try:
        # Key by path: SQLite may reuse the freed rowid for the new file, so
        # the old record set is identifiable by path, not by id.
        assert conn.execute("SELECT COUNT(*) FROM files WHERE path=?", (str(p),)).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM chunks c JOIN files f ON c.file_id=f.id WHERE f.path=?",
            (str(p),)).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM vectors v JOIN files f ON v.file_id=f.id WHERE f.path=?",
            (str(p),)).fetchone()[0] == 0
    finally:
        conn.close()
    new_row = database.get_file(str(dest))
    assert new_row is not None
    assert database.get_file(str(p)) is None


def test_reconcile_sweep_idempotent():
    tmp = TMP_HOME / "docs-c"
    tmp.mkdir(parents=True, exist_ok=True)
    keep = tmp / "keep.txt"
    keep.write_text("survivor content " * 20, encoding="utf-8")
    gone = tmp / "gone.txt"
    gone.write_text("doomed content " * 20, encoding="utf-8")
    assert handle_change_index(keep) and handle_change_index(gone)

    gone.unlink()  # deleted behind the app's back

    stale, orphan_vecs = reconcile_deletions([str(tmp)])
    assert stale >= 1
    assert database.get_file(str(gone)) is None
    assert database.get_file(str(keep)) is not None

    # Idempotent: second sweep is a no-op.
    assert reconcile_deletions([str(tmp)]) == (0, 0)


def test_foreign_keys_pragma_on():
    conn = database._conn()
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_vector_orphan_purge():
    import numpy as np
    # Bogus vector for a nonexistent file_id.
    vector_store.upsert(999999, 0, "text", np.ones(8, dtype=np.float32))
    conn = sqlite3.connect(str(config.DB_PATH))
    try:
        assert conn.execute("SELECT COUNT(*) FROM vectors WHERE file_id=999999").fetchone()[0] == 1
    finally:
        conn.close()

    removed = vector_store.delete_orphans()
    assert removed >= 1
    conn = sqlite3.connect(str(config.DB_PATH))
    try:
        assert conn.execute("SELECT COUNT(*) FROM vectors WHERE file_id=999999").fetchone()[0] == 0
    finally:
        conn.close()
