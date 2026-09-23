"""Phase 4 cleanup tests (plan 04-02): content-hash duplicates, actionable
suggestions, move endpoint safety, cleanup-folder persistence.

Model-free; uses the shared TestClient + token fixture from conftest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_memory import config  # noqa: E402
from local_memory.health import storage_health  # noqa: E402


# ------------------------------------------------- content-hash dupes ----

def _seed_db(tmp_path, monkeypatch):
    """Isolated index with a controlled file table. Returns (database,
    storage_health) — both reloaded so their module-level bindings point at
    the fresh home (storage_health binds `database` at import time)."""
    home = tmp_path / "home"
    monkeypatch.setenv("LOCAL_MEMORY_HOME", str(home))
    import importlib

    import local_memory.health.storage_health as storage_health
    import local_memory.store.database as database
    importlib.reload(config)
    importlib.reload(database)
    importlib.reload(storage_health)
    database.init_db()
    return database, storage_health


def test_content_hash_dupes_group_and_non_dupes_not(tmp_path, monkeypatch):
    """Byte-identical files with different names group; same-name files with
    different content do NOT."""
    db, sh = _seed_db(tmp_path, monkeypatch)
    watched = tmp_path / "watched"
    watched.mkdir()
    same = b"identical payload for the dupe test\n"
    for name in ("alpha_copy.txt", "beta_copy.txt", "gamma.txt"):
        p = watched / name
        p.write_bytes(same)
        db.upsert_file(str(p), str(watched), ".txt", len(same), 0.0, "text")
    diff1, diff2 = watched / "notes one.txt", watched / "notes two.txt"
    diff1.write_bytes(b"completely different content one\n")
    diff2.write_bytes(b"totally unrelated content two\n")
    for p in (diff1, diff2):
        db.upsert_file(str(p), str(watched), ".txt", p.stat().st_size, 0.0, "text")

    report = sh.health_report()
    dupe_cards = [s for s in report["suggestions"] if "duplicate content" in s["title"]]
    assert dupe_cards, "byte-identical pair must surface a duplicate card"
    dupe_paths = [f["path"] for f in dupe_cards[0]["files"]]
    # The group's first member is "kept"; the rest are actionable copies.
    assert len(dupe_paths) == 2, dupe_paths
    assert any(p.endswith("beta_copy.txt") for p in dupe_paths)
    assert any(p.endswith("gamma.txt") for p in dupe_paths)
    assert not any(p.endswith("alpha_copy.txt") for p in dupe_paths), \
        "the kept original must not be offered for moving"
    assert not any("notes" in p for p in dupe_paths), \
        "same-name different-content files must NOT be grouped"


def test_suggestion_files_capped_and_shaped(tmp_path, monkeypatch):
    db, sh = _seed_db(tmp_path, monkeypatch)
    watched = tmp_path / "watched"
    watched.mkdir()
    big = watched / "big.bin"
    big.write_bytes(b"\0" * (storage_health.LARGE_FILE_BYTES + 1))
    db.upsert_file(str(big), str(watched), ".bin", big.stat().st_size, 0.0, "binary")

    report = sh.health_report()
    large = [s for s in report["suggestions"] if "large" in s["title"]]
    assert large and large[0]["files"], "large-file card must carry actionable files"
    entry = large[0]["files"][0]
    assert {"path", "size_bytes", "size_human", "reason"} <= set(entry)
    assert len(large[0]["files"]) <= storage_health.MAX_SUGGESTION_FILES


# ------------------------------------------------------- move endpoint ----

@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Isolated server (own home + watched folder + cleanup folder)."""
    monkeypatch.setenv("LOCAL_MEMORY_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_MEMORY_TOKEN", "testtoken")
    monkeypatch.setattr(config, "DATA_HOME", tmp_path / "home", raising=False)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "home" / "index.db", raising=False)
    monkeypatch.setattr(config, "THUMBS_DIR", tmp_path / "home" / "thumbs", raising=False)
    monkeypatch.setattr(config, "TOKEN_PATH", tmp_path / "home" / "token", raising=False)
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "home" / "settings.json", raising=False)

    import importlib

    from local_memory.server import app as app_module
    import local_memory.store.database as database
    importlib.reload(database)
    importlib.reload(app_module)
    database.init_db()
    watched = tmp_path / "watched"
    watched.mkdir()
    monkeypatch.setattr(config, "watched_folders", lambda: [str(watched)], raising=False)
    monkeypatch.setattr(
        config, "cleanup_folder",
        lambda: str(tmp_path / "cleanup"), raising=False,
    )
    from fastapi.testclient import TestClient
    with TestClient(app_module.app, base_url="http://127.0.0.1:8787") as c:
        yield c, database, watched, tmp_path / "cleanup", tmp_path


AUTH = {"Authorization": "Bearer testtoken"}


def test_move_moves_indexed_file_and_rejects_others(client):
    c, database, watched, cleanup_dir, tmp = client
    src = watched / "move_me.txt"
    src.write_bytes(b"move me\n")
    database.upsert_file(str(src), str(watched), ".txt", 8, 0.0, "text")

    # Unindexed path -> 404 (no existence oracle).
    stranger = tmp / "stranger.txt"
    stranger.write_bytes(b"x")
    r = c.post("/api/cleanup/move", json={"path": str(stranger)}, headers=AUTH)
    assert r.status_code == 404

    # Indexed but outside watched roots -> 400.
    outside = tmp / "outside.txt"
    outside.write_bytes(b"x")
    database.upsert_file(str(outside), str(tmp), ".txt", 1, 0.0, "text")
    r = c.post("/api/cleanup/move", json={"path": str(outside)}, headers=AUTH)
    assert r.status_code == 400

    # Happy path: file lands in the cleanup folder, source gone.
    r = c.post("/api/cleanup/move", json={"path": str(src)}, headers=AUTH)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["moved"] is True
    assert str(cleanup_dir) in body["dest"]
    assert not src.exists() and Path(body["dest"]).is_file()

    # Moving a now-missing indexed file -> 410.
    r = c.post("/api/cleanup/move", json={"path": str(src)}, headers=AUTH)
    assert r.status_code == 410


def test_move_requires_auth(client):
    c, _database, watched, _cleanup, _tmp = client
    src = watched / "auth_probe.txt"
    src.write_bytes(b"x")
    r = c.post("/api/cleanup/move", json={"path": str(src)})
    assert r.status_code == 401


def test_cleanup_config_roundtrip(client, monkeypatch):
    c, _database, _watched, _cleanup, tmp = client
    r = c.get("/api/cleanup/config", headers=AUTH)
    assert r.status_code == 200 and "cleanup_folder" in r.json()

    new_dir = tmp / "my_cleanup"
    r = c.put("/api/cleanup/config", json={"cleanup_folder": str(new_dir)}, headers=AUTH)
    assert r.status_code == 200
    assert new_dir.is_dir(), "set must create the directory"
    assert config.load_settings()["cleanup_folder"] == str(new_dir), \
        "choice must persist to settings.json (survives restart)"
