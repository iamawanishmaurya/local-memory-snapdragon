"""Smoke tests: run on any machine (CPU fallback, no models required).

    python -m pytest tests/ -q
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Isolated data home for tests
os.environ["LOCAL_MEMORY_HOME"] = str(Path(__file__).parent / ".tmpdata")

from local_memory import config, privacy  # noqa: E402
from local_memory.store import database, vector_store  # noqa: E402
from local_memory.pipeline import index_file, scan_folder  # noqa: E402
from local_memory.search import query_engine  # noqa: E402
from local_memory.extractors import text_extractor  # noqa: E402


def setup_module(_):
    home = Path(__file__).parent / ".tmpdata"
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
    shutil.rmtree(Path(__file__).parent / ".tmpdata", ignore_errors=True)


def make_file(tmp: Path, name: str, text: str) -> Path:
    p = tmp / name
    p.write_text(text, encoding="utf-8")
    return p


def test_chunking_overlaps():
    words = " ".join(f"word{i}" for i in range(500))
    chunks = text_extractor.chunk_words(words)
    assert len(chunks) >= 2
    assert all(len(c.split()) <= config.CHUNK_WORDS for c in chunks)


def test_index_and_search(tmp_path=Path(__file__).parent / ".tmpdata"):
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    make_file(docs, "invoice_acme.txt", "This is an invoice from Acme Corp for September services. Total due: 450 dollars.")
    make_file(docs, "recipe.txt", "Chocolate cake recipe: flour, sugar, cocoa, eggs. Bake for 35 minutes.")
    n = scan_folder(str(docs))
    assert n == 2

    results = query_engine.search("invoice from last month")
    assert results, "expected at least one hit"
    assert "invoice_acme.txt" in results[0]["path"]

    results2 = query_engine.search("cake recipe cocoa")
    assert results2
    assert "recipe.txt" in results2[0]["path"]


def test_reindex_skips_unchanged(tmp_path=Path(__file__).parent / ".tmpdata"):
    docs = tmp_path / "docs2"
    docs.mkdir(exist_ok=True)
    p = make_file(docs, "a.txt", "hello world content")
    assert index_file(str(p)) is True
    assert index_file(str(p)) is False  # unchanged → skipped


def test_wipe():
    from local_memory.store import database
    assert database.stats()["files"] >= 0
    removed = privacy.wipe_all()
    assert any(removed.values())
    database.init_db()
    vector_store.init_db()
    assert database.stats()["files"] == 0


def test_health_report(tmp_path=Path(__file__).parent / ".tmpdata"):
    docs = tmp_path / "docs3"
    docs.mkdir(exist_ok=True)
    make_file(docs, "big.txt", "x" * 1000)
    scan_folder(str(docs))
    report = config and __import__("local_memory.health.storage_health", fromlist=["health_report"]).health_report()
    assert "summary" in report and "suggestions" in report
