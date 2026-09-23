"""Application configuration: paths, defaults, and the watched-folder registry.

All user data lives under a single directory so that "full wipe" is a single
folder delete. The default home is `%LOCALAPPDATA%\\LocalMemory` (outside any
cloud-synced folder — SQLite WAL on OneDrive corrupts databases); the legacy
repo `data/` directory is supported via automatic copy-based migration.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

APP_NAME = "Local Memory"
APP_HOST = "127.0.0.1"  # never bind to a public interface
APP_PORT = 8787

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_DATA_HOME = REPO_ROOT / "data"


def _default_data_home() -> Path:
    """Resolution chain: LOCAL_MEMORY_HOME env > %LOCALAPPDATA%\\LocalMemory > legacy data/."""
    env = os.environ.get("LOCAL_MEMORY_HOME")
    if env:
        return Path(env)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "LocalMemory"
    return LEGACY_DATA_HOME


DATA_HOME = _default_data_home()
MODELS_DIR = Path(os.environ.get("LOCAL_MEMORY_MODELS", REPO_ROOT / "models"))

DB_PATH = DATA_HOME / "index.db"
THUMBS_DIR = DATA_HOME / "thumbs"
SETTINGS_PATH = DATA_HOME / "settings.json"
# Auth token (D-01): lives in the data home, never in the repo.
TOKEN_PATH = DATA_HOME / "token"


def is_cloud_synced(path: Path | str) -> bool:
    """True if any part of the path sits under OneDrive, Dropbox, or iCloud."""
    p = str(path)
    return any(marker in p.lower() for marker in ("onedrive", "dropbox", "icloud"))


def _repoint_data_home(home: Path) -> None:
    """Re-point the module-level path attributes at `home`."""
    global DATA_HOME, DB_PATH, THUMBS_DIR, SETTINGS_PATH, TOKEN_PATH
    DATA_HOME = home
    DB_PATH = home / "index.db"
    THUMBS_DIR = home / "thumbs"
    SETTINGS_PATH = home / "settings.json"
    TOKEN_PATH = home / "token"


def migrate_home() -> str | None:
    """Copy (never move) the legacy repo data/ index into the new home.

    Idempotent: runs once — after the new home has an index.db, this is a no-op.
    Returns the path actually used when a fallback happened, else None.
    """
    if os.environ.get("LOCAL_MEMORY_HOME"):
        return None  # explicit home chosen by the user — nothing to migrate
    if (DATA_HOME / "index.db").exists():
        return None  # already migrated / fresh install
    legacy_db = LEGACY_DATA_HOME / "index.db"
    if not legacy_db.exists():
        return None  # nothing legacy to migrate
    try:
        DATA_HOME.mkdir(parents=True, exist_ok=True)
        THUMBS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy_db, DB_PATH)
        for name in ("index.db-wal", "index.db-shm", "settings.json"):
            legacy_file = LEGACY_DATA_HOME / name
            if legacy_file.exists():
                shutil.copy2(legacy_file, DATA_HOME / name)
        legacy_thumbs = LEGACY_DATA_HOME / "thumbs"
        if legacy_thumbs.is_dir():
            for f in legacy_thumbs.iterdir():
                if f.is_file():
                    shutil.copy2(f, THUMBS_DIR / f.name)
        logger.info("migrated index from %s to %s", LEGACY_DATA_HOME, DATA_HOME)
        return None
    except Exception as e:  # noqa: BLE001 — migration must never crash startup
        logger.warning("migration failed, continuing with legacy path: %s", e)
        _repoint_data_home(LEGACY_DATA_HOME)
        return str(LEGACY_DATA_HOME)

# Chunking for long documents: overlapping windows of ~200 words.
CHUNK_WORDS = 200
CHUNK_OVERLAP_WORDS = 40

# Files we attempt text extraction from, by extension.
TEXT_EXTS = {".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".json", ".xml", ".yaml", ".yml", ".html", ".htm",
             ".ps1", ".sh", ".bat", ".cmd"}
PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
# Jupyter notebooks: JSON, ingested via markdown headers / first cells.
IPYNB_EXTS = {".ipynb"}

# Binary types we never open for content but still make findable by
# filename/metadata (accuracy eval gap #1). ext/compound-suffix -> kind.
BINARY_KINDS = {
    ".zip": "archive", ".7z": "archive", ".rar": "archive", ".tar": "archive",
    ".gz": "archive", ".tgz": "archive", ".bz2": "archive", ".xz": "archive",
    ".jar": "archive", ".whl": "archive",
    ".exe": "executable", ".dll": "executable", ".msi": "executable",
    ".msix": "executable", ".appx": "executable", ".apk": "executable",
    ".iso": "disk-image", ".img": "disk-image", ".dmg": "disk-image",
    ".vhd": "disk-image", ".vhdx": "disk-image",
    ".parquet": "data", ".xlsx": "data", ".pptx": "data",
}
# Compound suffixes checked against the full filename before the bare ext
# (Path("x.tar.gz").suffix is ".gz").
BINARY_COMPOUND = {".tar.gz": "archive", ".tar.bz2": "archive", ".tar.xz": "archive"}

# Deep inspection (archive entry listing) caps — never open huge files.
ARCHIVE_INSPECT_MAX_BYTES = 100 * 1024 * 1024  # 100 MB
ARCHIVE_ENTRY_LIMIT = 20

# Notebook ingestion caps.
IPYNB_MAX_CHARS = 8000

# OCR of PDF-embedded page images (accuracy eval gap #2).
PDF_OCR_MAX_PAGES = 5          # OCR at most the first N text-free pages
PDF_OCR_MAX_BYTES = 60 * 1024 * 1024  # skip OCR for PDFs over 60 MB


def binary_kind(path: str | Path) -> str | None:
    """Kind for a binary file ('archive'/'executable'/...), or None.

    Compound suffixes (.tar.gz) win over the bare extension.
    """
    name = str(path).lower()
    for suffix, kind in BINARY_COMPOUND.items():
        if name.endswith(suffix):
            return kind
    return BINARY_KINDS.get(Path(name).suffix)


def indexable_exts() -> set[str]:
    """Every extension the scanner should pick up."""
    return TEXT_EXTS | PDF_EXTS | DOCX_EXTS | IMAGE_EXTS | IPYNB_EXTS | set(BINARY_KINDS) | set(BINARY_COMPOUND)

DEFAULT_WATCHED = ["Downloads", "Documents", "Desktop", "Pictures"]


def user_profile_dir(name: str) -> Path:
    """Resolve a well-known folder under the user profile (works on Windows)."""
    return Path.home() / name


def ensure_dirs() -> None:
    DATA_HOME.mkdir(parents=True, exist_ok=True)
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)


def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    return {"watched_folders": [], "paused": False}


def save_settings(settings: dict) -> None:
    ensure_dirs()
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def auth_token() -> str:
    """Single token factory (D-01). Create-on-read, idempotent.

    DEV-OVERRIDE RULE (pinned for the whole project): the LOCAL_MEMORY_TOKEN
    environment variable overrides the persisted token file. This is the only
    sanctioned override path — it exists for the Vite dev proxy (ui/vite.config.ts)
    and tests; never log or embed the token value anywhere else.
    """
    override = os.environ.get("LOCAL_MEMORY_TOKEN")
    if override:
        return override
    if TOKEN_PATH.exists():
        token = TOKEN_PATH.read_text(encoding="utf-8").strip()
        if token:
            return token
    token = secrets.token_urlsafe(32)
    ensure_dirs()
    # Best-effort restrictive ACL (0600): the user-profile data home is already
    # per-user on Windows; os.open keeps the file from being world-readable.
    fd = os.open(TOKEN_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, token.encode("utf-8"))
    finally:
        os.close(fd)
    return token


def watched_folders() -> list[str]:
    return load_settings().get("watched_folders", [])


def add_watched_folder(path: str) -> list[str]:
    settings = load_settings()
    folders = settings.setdefault("watched_folders", [])
    p = str(Path(path).resolve())
    if p not in folders:
        folders.append(p)
    save_settings(settings)
    return folders


def remove_watched_folder(path: str) -> list[str]:
    settings = load_settings()
    p = str(Path(path).resolve())
    settings["watched_folders"] = [f for f in settings.get("watched_folders", []) if f != p]
    save_settings(settings)
    return settings["watched_folders"]


def cleanup_folder() -> str:
    """Reversible-move target for Storage Health cleanup (D-03)."""
    settings = load_settings()
    p = settings.get("cleanup_folder")
    if p:
        return str(Path(p).expanduser())
    return str(user_profile_dir("Documents") / "LocalMemoryCleanup")


def set_cleanup_folder(path: str) -> str:
    """Persist the cleanup folder choice; create the directory on set."""
    resolved = str(Path(path).expanduser().resolve())
    Path(resolved).mkdir(parents=True, exist_ok=True)
    settings = load_settings()
    settings["cleanup_folder"] = resolved
    save_settings(settings)
    return resolved
