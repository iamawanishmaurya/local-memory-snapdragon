"""Tests for POST /api/open (04-3): open an indexed file with the OS default app.

Security posture: the endpoint is covered by the same middleware as every
/api/* route (loopback Host + Bearer token). Tests NEVER launch a real
application — os.startfile is monkeypatched to record the call.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import make_security_client  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(scope="module")
def client():
    client, headers = make_security_client()
    yield client, headers
    from local_memory.store import database, vector_store
    import shutil
    database.close()
    vector_store.invalidate_cache()
    shutil.rmtree(Path(__file__).parent / ".tmpdata-security", ignore_errors=True)


def _index_file(client, name="note.txt", content="hello open"):
    """Insert a real file into the isolated index; returns (file_id, path).

    ``client`` is the TestClient (not the fixture tuple).
    """
    from local_memory import config
    from local_memory.store import database

    src = config.DATA_HOME / name
    src.write_text(content, encoding="utf-8")
    file_id = database.upsert_file(
        str(src), str(config.DATA_HOME), ".txt", src.stat().st_size, src.stat().st_mtime, "text"
    )
    return file_id, src


def test_open_by_id_calls_startfile(client, monkeypatch):
    from local_memory.server import app as server_app

    client_app, headers = client
    opened: list[str] = []
    # Never launch a real app in tests: record the path instead.
    monkeypatch.setattr(server_app.os, "startfile", lambda p: opened.append(p), raising=False)

    file_id, src = _index_file(client_app)
    r = client_app.post("/api/open", json={"file_id": file_id}, headers=headers)
    assert r.status_code == 200, r.text
    assert r.json() == {"opened": True}
    assert opened == [str(src)]


def test_open_bogus_id_404(client):
    client_app, headers = client
    r = client_app.post("/api/open", json={"file_id": 999999}, headers=headers)
    assert r.status_code == 404


def test_open_missing_file_410(client):
    client_app, headers = client
    file_id, src = _index_file(client_app, name="vanishing.txt")
    src.unlink()
    r = client_app.post("/api/open", json={"file_id": file_id}, headers=headers)
    assert r.status_code == 410


def test_open_requires_token(client):
    """Unauthenticated request must 401 — the /api/* middleware covers /api/open."""
    client_app, _ = client
    r = client_app.post("/api/open", json={"file_id": 1})
    assert r.status_code == 401


def test_open_startfile_failure_500(client, monkeypatch):
    from local_memory.server import app as server_app

    client_app, headers = client

    def _boom(_p):
        raise OSError("no association")

    monkeypatch.setattr(server_app.os, "startfile", _boom, raising=False)
    file_id, _ = _index_file(client_app, name="unopenable.txt")
    r = client_app.post("/api/open", json={"file_id": file_id}, headers=headers)
    assert r.status_code == 500
    assert "error" in r.json()
