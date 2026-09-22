"""Search quality tests (Phase 3): FTS5 sync, BM25 fusion, snippets.

All model-free — the HashingEncoder fallback stands in for the Nomic/CLIP
ONNX models (MODELS_DIR pointed at an empty dir).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Isolated data home, chosen BEFORE local_memory/config import time paths are
# read (Phase 1 pitfall) — mirrors test_smoke.py.
os.environ["LOCAL_MEMORY_HOME"] = str(Path(__file__).parent / ".tmpdata-quality")

from local_memory import config  # noqa: E402
from local_memory.store import database, vector_store  # noqa: E402
from local_memory.pipeline import scan_folder  # noqa: E402
from local_memory.search import query_engine  # noqa: E402

_HOME = Path(__file__).parent / ".tmpdata-quality"


def setup_module(_):
    import shutil

    shutil.rmtree(_HOME, ignore_errors=True)
    config.DATA_HOME = _HOME
    config.DB_PATH = _HOME / "index.db"
    config.THUMBS_DIR = _HOME / "thumbs"
    config.SETTINGS_PATH = _HOME / "settings.json"
    config.MODELS_DIR = _HOME / "no-models"  # deterministic/no-model mode
    database.close()
    vector_store.invalidate_cache()
    config.ensure_dirs()
    database.init_db()
    vector_store.init_db()


def teardown_module(_):
    import shutil

    database.close()
    vector_store.invalidate_cache()
    shutil.rmtree(_HOME, ignore_errors=True)


def _match(table: str, term: str) -> list[int]:
    return [r["rowid"] for r in database.fts_rows(
        f"SELECT rowid FROM {table} WHERE {table} MATCH ?", (fts_term(term),))]


def fts_term(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def _make_file(folder: Path, name: str, text: str) -> Path:
    p = folder / name
    p.write_text(text, encoding="utf-8")
    return p


# --- Task 1: FTS tables, triggers, backfill ---------------------------------


def test_fts_tables_and_triggers_exist():
    names = {
        r["name"]
        for r in database.fts_rows(
            "SELECT name FROM sqlite_master WHERE name IN "
            "('chunks_fts','files_fts','chunks_ai','chunks_ad',"
            "'files_ai','files_au','files_ad')"
        )
    }
    assert names == {
        "chunks_fts", "files_fts",
        "chunks_ai", "chunks_ad",
        "files_ai", "files_au", "files_ad",
    }


def test_fts_sync_on_all_write_paths(tmp_path=None):
    docs = _HOME / "sync_docs"
    docs.mkdir(exist_ok=True)
    p = _make_file(docs, "sync_doc.txt", "alpha beta gamma delta content")

    fid = database.upsert_file(str(p), str(docs), ".txt", 100, 1.0, "text", ocr_text="")
    database.replace_chunks(fid, [(0, "alpha beta gamma"), (1, "delta content tail")])

    # INSERT path mirrored into both FTS tables (external content: no dup).
    assert len(_match("chunks_fts", "gamma")) == 1  # chunk text indexed
    assert _match("files_fts", "sync") == [fid]

    # Re-index (replace_chunks = DELETE+INSERT): old FTS rows must go.
    database.replace_chunks(fid, [(0, "alpha only window")])
    assert _match("chunks_fts", "gamma") == [], "stale chunk text still in FTS after re-index"
    assert len(_match("chunks_fts", "alpha")) == 1

    # upsert_file OCR edit (in-place UPDATE) mirrored via files_au trigger.
    database.upsert_file(str(p), str(docs), ".txt", 100, 2.0, "text", ocr_text="new ocr words")
    assert _match("files_fts", "new") == [fid]
    assert _match("files_fts", "words") == [fid]

    # delete_file (cascade) clears both tables for the file.
    assert database.delete_file(str(p)) == fid
    assert _match("chunks_fts", "alpha") == []
    assert _match("files_fts", "sync") == []


def test_fts_rebuild_never_touches_content_tables():
    """D-01 reversibility: FTS tables are derived data only."""
    docs = _HOME / "rebuild_docs"
    docs.mkdir(exist_ok=True)
    p = _make_file(docs, "keep.txt", "survivor text here")
    fid = database.upsert_file(str(p), str(docs), ".txt", 10, 1.0, "text")
    database.replace_chunks(fid, [(0, "survivor chunk")])

    before_files = database.stats()["files"]
    before_chunks = database.stats()["chunks"]
    # Drop the derived layer entirely (tables + triggers), content untouched.
    for stmt in (
        "DROP TRIGGER IF EXISTS chunks_ai", "DROP TRIGGER IF EXISTS chunks_ad",
        "DROP TRIGGER IF EXISTS files_ai", "DROP TRIGGER IF EXISTS files_au",
        "DROP TRIGGER IF EXISTS files_ad",
        "DROP TABLE IF EXISTS chunks_fts", "DROP TABLE IF EXISTS files_fts",
    ):
        database.fts_rows(stmt)
    assert database.stats()["files"] == before_files
    assert database.stats()["chunks"] == before_chunks

    # init_db restores the schema. Desync the index directly (bypass the
    # trigger) to force a real rebuild path.
    database.init_db()
    database.fts_rows("DROP TRIGGER chunks_ad")
    with database._lock:
        database._conn().execute("UPDATE chunks SET text='desynced zzz' WHERE rowid=("
                                 "SELECT id FROM chunks LIMIT 1)")
        database._conn().commit()
    out = database.backfill_fts()
    assert out["rebuilt"] is True
    assert out["chunks"] == before_chunks
    assert out["files"] == before_files
    # After the rebuild the index reflects current content again.
    assert _match("chunks_fts", "desynced") != []
    assert database.fts_status()["in_sync"] is True


def test_backfill_fts_idempotent():
    out1 = database.backfill_fts()
    assert out1["rebuilt"] is False  # healthy index from previous test
    assert out1["chunks"] == database.stats()["chunks"]
    out2 = database.backfill_fts()
    assert out2["rebuilt"] is False


def test_fts_status_shape():
    st = database.fts_status()
    assert st["available"] is True
    assert st["in_sync"] is True
    assert st["chunks"] == st["chunks_total"]
    assert st["files"] == st["files_total"]


def test_scan_folder_keeps_fts_in_sync():
    docs = _HOME / "scan_docs"
    docs.mkdir(exist_ok=True)
    _make_file(docs, "scan_me.txt", "scanner probe zebra text")
    n = scan_folder(str(docs))
    assert n >= 1
    st = database.fts_status()
    assert st["in_sync"] is True
    assert _match("chunks_fts", "zebra") != []


# --- Task 2: BM25 ranked lists in query_engine -------------------------------


def test_bm25_replaces_like_scan():
    import inspect

    src = inspect.getsource(query_engine._keyword_hits)
    assert "list_files" not in src, "_keyword_hits must not do a Python LIKE scan"
    assert "MATCH" in inspect.getsource(query_engine._bm25_hits)


def test_bm25_search_returns_keyword_scores():
    docs = _HOME / "bm25_docs"
    docs.mkdir(exist_ok=True)
    _make_file(docs, "invoice_bstar.txt", "invoice for services rendered promptly")
    _make_file(docs, "unrelated.txt", "gardening tips for spring tomatoes")
    assert scan_folder(str(docs)) >= 2

    results = query_engine.search("invoice")
    assert results, "expected results"
    hit = next((r for r in results if "invoice" in r["name"]), None)
    assert hit is not None, "BM25 list missed the planted invoice file"
    assert hit["match_keyword"] > 0

    # A lexically-empty query still returns dense results without error.
    res2 = query_engine.search("xyzzyqqqqzNothingMatchesThisLexically")
    assert isinstance(res2, list)


def test_bm25_query_plan_uses_fts_index():
    plan = database.fts_rows(
        "EXPLAIN QUERY PLAN SELECT c.file_id, c.id, rank FROM chunks_fts "
        "JOIN chunks c ON c.id = chunks_fts.rowid WHERE chunks_fts MATCH ? "
        "ORDER BY rank LIMIT 36",
        ('"invoice"',),
    )
    detail = " ".join(r["detail"] for r in plan)
    assert "VIRTUAL TABLE" in detail, detail
    assert "SCAN chunks" not in detail.replace("chunks_fts", ""), detail
    plan2 = database.fts_rows(
        "EXPLAIN QUERY PLAN SELECT rowid FROM files_fts WHERE files_fts MATCH ?",
        ('"invoice"',),
    )
    detail2 = " ".join(r["detail"] for r in plan2)
    assert "SCAN files" not in detail2.replace("files_fts", ""), detail2


# --- Task 3: fts_quote edge cases + TRACER slice -----------------------------


def test_fts_metacharacters_safely_quoted():
    """FTS5 query syntax in user input must never raise (research §6)."""
    for q in ('pen" OR (', 'blue" NOT book NEAR/10(', '"', "()*:", "a AND b OR c"):
        res = query_engine.search(q)
        assert isinstance(res, list)


def test_tracer_blue_book_end_to_end():
    """TRACER SLICE: text file → chunks → trigger → FTS index → BM25 →
    fused result with an FTS5 <mark> snippet — proven model-free."""
    docs = _HOME / "tracer_docs"
    docs.mkdir(exist_ok=True)
    _make_file(docs, "notes.txt", "blue book on the desk with a pen and a lamp nearby")
    assert scan_folder(str(docs)) == 1

    results = query_engine.search("blue book")
    assert results, "tracer query returned nothing"
    hit = next((r for r in results if r["name"] == "notes.txt"), None)
    assert hit is not None, f"notes.txt not in results: {[r['name'] for r in results]}"
    assert hit["match_keyword"] > 0
    assert "<mark>" in hit["snippet"] or "blue book" in hit["snippet"].lower(), hit["snippet"]
    assert hit["snippet_source"] in ("chunk", "ocr", "name")
