"""Migration tests: copy-not-move relocation to %LOCALAPPDATA%\\LocalMemory.

Runs model-free. Follows the env-var-before-import convention: LOCAL_MEMORY_HOME
is unset before importing config so the LOCALAPPDATA resolution chain is used.
"""
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure the resolution chain starts clean before config is imported.
os.environ.pop("LOCAL_MEMORY_HOME", None)

from local_memory import config  # noqa: E402


@pytest.fixture(autouse=True)
def _no_explicit_home(monkeypatch):
    """test_smoke sets LOCAL_MEMORY_HOME at import; migrate_home must see it unset."""
    monkeypatch.delenv("LOCAL_MEMORY_HOME", raising=False)


def _seed_legacy(legacy: Path) -> bytes:
    """Seed a fake legacy data/ dir; return the raw index.db bytes for comparison."""
    thumbs = legacy / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    db_bytes = b"fake sqlite database bytes"
    (legacy / "index.db").write_bytes(db_bytes)
    (legacy / "index.db-wal").write_bytes(b"wal bytes")
    (legacy / "settings.json").write_text('{"watched_folders": []}', encoding="utf-8")
    (thumbs / "img.png").write_bytes(b"thumb bytes")
    return db_bytes


def test_migrate_copies_and_keeps_legacy(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy_data"
    new_home = tmp_path / "appdata" / "LocalMemory"
    db_bytes = _seed_legacy(legacy)
    monkeypatch.setattr(config, "LEGACY_DATA_HOME", legacy)
    monkeypatch.setattr(config, "DATA_HOME", new_home)
    monkeypatch.setattr(config, "DB_PATH", new_home / "index.db")
    monkeypatch.setattr(config, "THUMBS_DIR", new_home / "thumbs")
    monkeypatch.setattr(config, "SETTINGS_PATH", new_home / "settings.json")

    result = config.migrate_home()
    assert result is None

    # New home contains all copied files.
    assert (new_home / "index.db").read_bytes() == db_bytes
    assert (new_home / "index.db-wal").exists()
    assert (new_home / "settings.json").exists()
    assert (new_home / "thumbs" / "img.png").read_bytes() == b"thumb bytes"

    # Legacy dir is byte-identical — copy, never move.
    assert (legacy / "index.db").read_bytes() == db_bytes
    assert (legacy / "index.db-wal").exists()
    assert (legacy / "settings.json").exists()
    assert (legacy / "thumbs" / "img.png").read_bytes() == b"thumb bytes"


def test_migrate_idempotent(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy_data"
    new_home = tmp_path / "appdata" / "LocalMemory"
    _seed_legacy(legacy)
    monkeypatch.setattr(config, "LEGACY_DATA_HOME", legacy)
    monkeypatch.setattr(config, "DATA_HOME", new_home)
    monkeypatch.setattr(config, "DB_PATH", new_home / "index.db")
    monkeypatch.setattr(config, "THUMBS_DIR", new_home / "thumbs")
    monkeypatch.setattr(config, "SETTINGS_PATH", new_home / "settings.json")

    config.migrate_home()
    mtime_first = (new_home / "index.db").stat().st_mtime_ns

    # Second call must be a no-op: index.db already exists in the new home.
    config.migrate_home()
    assert (new_home / "index.db").stat().st_mtime_ns == mtime_first


def test_migrate_failure_falls_back_to_legacy(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy_data"
    new_home = tmp_path / "appdata" / "LocalMemory"
    _seed_legacy(legacy)
    monkeypatch.setattr(config, "LEGACY_DATA_HOME", legacy)
    monkeypatch.setattr(config, "DATA_HOME", new_home)
    monkeypatch.setattr(config, "DB_PATH", new_home / "index.db")
    monkeypatch.setattr(config, "THUMBS_DIR", new_home / "thumbs")
    monkeypatch.setattr(config, "SETTINGS_PATH", new_home / "settings.json")

    def boom(src, dst):
        raise OSError("disk on fire")

    monkeypatch.setattr(config.shutil, "copy2", boom)
    result = config.migrate_home()  # must not raise
    assert result == str(legacy)
    assert config.DATA_HOME == legacy
    assert config.DB_PATH == legacy / "index.db"
    assert config.THUMBS_DIR == legacy / "thumbs"
    assert config.SETTINGS_PATH == legacy / "settings.json"


def test_is_cloud_synced():
    assert config.is_cloud_synced(r"C:\Users\me\OneDrive\Documents\data") is True
    assert config.is_cloud_synced(r"C:\Users\me\Dropbox\stuff") is True
    assert config.is_cloud_synced(r"C:\Users\me\iCloud\Drive") is True
    assert config.is_cloud_synced(Path(r"C:\Users\me\AppData\Local\LocalMemory")) is False


def test_env_home_takes_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_MEMORY_HOME", str(tmp_path / "explicit"))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    # Re-resolve through the documented chain without reloading the module.
    assert config._default_data_home() == Path(tmp_path / "explicit")
    monkeypatch.delenv("LOCAL_MEMORY_HOME")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    assert config._default_data_home() == Path(tmp_path / "appdata") / "LocalMemory"
    monkeypatch.delenv("LOCALAPPDATA")
    assert config._default_data_home() == config.LEGACY_DATA_HOME
