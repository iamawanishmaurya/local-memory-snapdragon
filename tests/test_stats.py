"""Statistics endpoint tests (Phase 04-4): GET /api/stats.

Uses the shared isolated-home client (tests/conftest.py) — the same
TestClient + Bearer token pattern as tests/test_security.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import make_security_client  # noqa: E402

import pytest  # noqa: E402

from local_memory.store import database  # noqa: E402


@pytest.fixture(scope="module")
def client():
    client, headers = make_security_client()
    yield client, headers


def _seed(client, headers):
    """Insert a couple of files with chunks; returns (client, headers)."""
    database.upsert_file(str(Path("C:/x") / "a.md"), str(Path("C:/x")), ".md",
                         100, 1.0, "text")
    database.upsert_file(str(Path("C:/x") / "b.jpg"), str(Path("C:/x")), ".jpg",
                         2000, 2.0, "image", ocr_text="hello sign")
    database.replace_chunks(1, [(0, "markdown body"), (1, "second chunk")])
    return client, headers


def test_stats_requires_token(client):
    c, _ = client
    r = c.get("/api/stats")
    assert r.status_code == 401


def test_stats_shape_and_seed(client):
    c, headers = _seed(*client)
    r = c.get("/api/stats", headers=headers)
    assert r.status_code == 200
    body = r.json()
    for section in ("database", "files", "index", "activity"):
        assert section in body

    db = body["database"]
    for key in ("index_db_bytes", "wal_bytes", "shm_bytes", "thumbnails_bytes", "total_bytes"):
        assert isinstance(db[key], int) and db[key] >= 0
    assert db["index_db_bytes"] > 0  # live DB was created by the fixture
    assert db["total_bytes"] >= db["index_db_bytes"]

    files = body["files"]
    assert files["total"] == 2
    kinds = {k["kind"]: k for k in files["by_kind"]}
    assert set(kinds) == {"text", "image"}
    assert kinds["image"]["bytes"] == 2000 and kinds["image"]["count"] == 1
    exts = [(e["ext"], e["count"], e["bytes"]) for e in files["by_extension"]]
    assert (".md", 1, 100) in exts and (".jpg", 1, 2000) in exts
    assert all({"ext", "count", "bytes"} == set(e) for e in files["by_extension"])
    assert len(files["by_folder"]) <= 10

    index = body["index"]
    assert index["total_chunks"] == 2
    assert index["images_understood"] == 1
    assert index["ocr_files"] == 1
    assert index["embedded_chunks"] >= 0
    assert index["db_page_size"] > 0 and index["db_page_count"] > 0

    activity = body["activity"]
    assert isinstance(activity["watcher_running"], bool)
    assert "status" in activity["last_scan"]


def test_by_extension_top10_limit(client):
    c, headers = client
    before = len(c.get("/api/stats", headers=headers).json()["files"]["by_extension"])
    for i in range(13):
        p = str(Path("C:/y") / f"f{i}.txt")
        database.upsert_file(p, str(Path("C:/y")), ".txt", i * 10 + 1, 1.0, "text")
    r = c.get("/api/stats", headers=headers)
    assert r.status_code == 200
    exts = r.json()["files"]["by_extension"]
    assert any(e["ext"] == ".txt" and e["count"] == 13 for e in exts)
    assert len(exts) <= 10  # never more than the top-10 cap
    assert len(exts) == min(10, before + 1)  # .txt groups the 13 new files into one row
