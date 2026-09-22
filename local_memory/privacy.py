"""Privacy controls: pause/resume indexing and full local wipe.

The wipe is intentionally blunt: it removes the entire data home (database,
vectors, thumbnails, settings). Nothing is recoverable afterwards, and nothing
was ever copied off the device in the first place.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from . import config


def set_paused(paused: bool) -> None:
    settings = config.load_settings()
    settings["paused"] = paused
    config.save_settings(settings)


def is_paused() -> bool:
    return bool(config.load_settings().get("paused", False))


def wipe_all() -> dict:
    """Destroy the entire local index. Returns what was removed for the UI."""
    removed = {"database": False, "thumbnails": False, "settings": False}
    from .store import database
    from .store import vector_store as _vs
    database.close()  # release SQLite handles so the file can be deleted on Windows
    try:
        _vs.invalidate_cache()
    except Exception:
        pass
    if config.DB_PATH.exists():
        config.DB_PATH.unlink()
        removed["database"] = True
    for suffix in ("-wal", "-shm", "-journal"):
        p = Path(str(config.DB_PATH) + suffix)
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass
    if config.THUMBS_DIR.exists():
        shutil.rmtree(config.THUMBS_DIR, ignore_errors=True)
        removed["thumbnails"] = True
    if config.SETTINGS_PATH.exists():
        config.SETTINGS_PATH.unlink()
        removed["settings"] = True
    return removed


def clean_orphans() -> dict:
    """Clean WITHOUT full wipe: prune missing files, orphan thumbs/chunks, vacuum.

    Safe to run anytime; keeps watched-folder list + settings intact.
    """
    from pathlib import Path as _P
    from .store import database as _db
    from .store import vector_store as _vs
    _db.init_db()
    _vs.init_db()
    removed_missing = 0
    for f in config.watched_folders():
        try:
            removed_missing += _db.remove_missing(f)
        except Exception:
            pass
    # Orphan thumbnails (hash-named, can't reverse-map): drop all if DB empty,
    # else keep (thumbs are tiny). Report counts.
    n_thumbs = 0
    if config.THUMBS_DIR.exists():
        n_thumbs = len(list(config.THUMBS_DIR.glob("*.jpg")))
        if _db.stats().get("files", 1) == 0 and n_thumbs:
            shutil.rmtree(config.THUMBS_DIR, ignore_errors=True)
            config.THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    # Vacuum to reclaim space.
    try:
        _db.vacuum()
    except Exception:
        pass
    try:
        _vs.invalidate_cache()
    except Exception:
        pass
    return {"pruned_missing": removed_missing, "thumbs_present": n_thumbs,
            "files": _db.stats().get("files", 0)}


def storage_usage() -> dict:
    import os as _os
    from pathlib import Path as _P  # was using clean_orphans' function-local alias — NameError
    total = 0
    nfiles = 0
    for root, _, files in _os.walk(config.DATA_HOME):
        for fn in files:
            try:
                fp = _P(root) / fn
                total += fp.stat().st_size
                nfiles += 1
            except OSError:
                pass
    return {"data_home": str(config.DATA_HOME), "bytes": total, "files": nfiles}
