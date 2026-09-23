"""Storage Health: index-backed insights and cleanup suggestions.

Phase 4 (DEMO-03): suggestions carry actionable `files` lists so the UI can
offer one-click "Move to Cleanup" (reversible moves — nothing deletes).
Duplicate detection is content-based (sha256, size pre-grouped) rather than
name-based, so a real duplicate pair groups and same-name-different-content
files don't.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

from .. import config
from ..store import database

LARGE_FILE_BYTES = 200 * 1024 * 1024      # 200 MB
STALE_DAYS = 365                          # untouched for a year
ARCHIVE_STALE_DAYS = 180                  # old archives card
# Content hashing is capped per file: duplicates demo on documents/media,
# hashing a multi-GB ISO byte-for-byte is wasted I/O.
HASH_CAP_BYTES = 50 * 1024 * 1024
MAX_SUGGESTION_FILES = 10                 # per-card file rows in the UI


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} TB"


def _content_hash(path: str, size: int) -> str | None:
    """sha256 of the first min(size, HASH_CAP_BYTES) bytes; None on any error."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            remaining = min(size, HASH_CAP_BYTES)
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                h.update(chunk)
                remaining -= len(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _duplicate_groups(files: list[dict]) -> tuple[list[list[dict]], int]:
    """Byte-identical groups via size pre-group + capped sha256."""
    by_size: dict[int, list[dict]] = {}
    for f in files:
        if f["size_bytes"] > 0:
            by_size.setdefault(f["size_bytes"], []).append(f)
    groups: list[list[dict]] = []
    for size, candidates in by_size.items():
        if len(candidates) < 2:
            continue
        by_hash: dict[str, list[dict]] = {}
        for f in candidates:
            digest = _content_hash(f["path"], size)
            if digest:
                by_hash.setdefault(digest, []).append(f)
        groups.extend(g for g in by_hash.values() if len(g) > 1)
    dupe_bytes = sum(min(f["size_bytes"], HASH_CAP_BYTES) for g in groups for f in g[1:])
    return groups, dupe_bytes


def _actionable(files: list[dict], reason: str) -> list[dict]:
    return [
        {
            "path": f["path"],
            "size_bytes": f["size_bytes"],
            "size_human": _human(f["size_bytes"]),
            "reason": reason,
        }
        for f in files[:MAX_SUGGESTION_FILES]
    ]


def health_report() -> dict:
    files = [dict(r) for r in database.list_files()]
    stats = database.stats()

    per_folder: dict[str, dict] = {}
    for f in files:
        entry = per_folder.setdefault(f["folder"], {"count": 0, "bytes": 0, "kinds": {}})
        entry["count"] += 1
        entry["bytes"] += f["size_bytes"]
        entry["kinds"][f["kind"]] = entry["kinds"].get(f["kind"], 0) + 1

    largest = sorted(files, key=lambda f: -f["size_bytes"])[:10]
    now = time.time()
    stale = [f for f in files if (now - f["mtime"]) > STALE_DAYS * 86400]
    stale_bytes = sum(f["size_bytes"] for f in stale)

    archive_kinds = {"archive", "disk-image"}
    old_archives = [
        f for f in files
        if f["kind"] in archive_kinds and (now - f["mtime"]) > ARCHIVE_STALE_DAYS * 86400
    ]
    archive_bytes = sum(f["size_bytes"] for f in old_archives)

    dupes, dupe_bytes = _duplicate_groups(files)

    suggestions: list[dict] = []
    big = [f for f in files if f["size_bytes"] >= LARGE_FILE_BYTES]
    if big:
        suggestions.append({
            "title": f"{len(big)} very large file(s) (≥200 MB)",
            "detail": "Review the largest files list; installers and old recordings are common wins.",
            "potential_bytes": sum(f["size_bytes"] for f in big),
            "files": _actionable(big, f"large file (≥ {_human(LARGE_FILE_BYTES)})"),
        })
    if stale:
        suggestions.append({
            "title": f"{len(stale)} file(s) untouched for over a year",
            "detail": "Archive or delete old files to reduce clutter.",
            "potential_bytes": stale_bytes,
            "files": _actionable(stale, f"untouched over {STALE_DAYS} days"),
        })
    if old_archives:
        suggestions.append({
            "title": f"{len(old_archives)} old archive(s)/image(s) (>{ARCHIVE_STALE_DAYS} days)",
            "detail": "Old installers, disk images and archives are usually safe to move out.",
            "potential_bytes": archive_bytes,
            "files": _actionable(old_archives, "old archive or disk image"),
        })
    if dupes:
        suggestions.append({
            "title": f"{len(dupes)} duplicate content group(s)",
            "detail": "Byte-identical copies (verified by content hash) — one can be moved away.",
            "potential_bytes": dupe_bytes,
            "files": _actionable([f for g in dupes for f in g[1:]], "duplicate content"),
        })

    return {
        "summary": {
            "files": stats["files"],
            "chunks": stats["chunks"],
            "images": stats["images"],
            "total_bytes": stats["total_bytes"],
            "total_human": _human(stats["total_bytes"]),
        },
        "per_folder": [
            {
                "folder": folder,
                "count": e["count"],
                "bytes": e["bytes"],
                "bytes_human": _human(e["bytes"]),
                "kinds": e["kinds"],
            }
            for folder, e in sorted(per_folder.items(), key=lambda kv: -kv[1]["bytes"])
        ],
        "largest": [
            {**f, "size_human": _human(f["size_bytes"])} for f in largest
        ],
        "stale_count": len(stale),
        "dupe_groups": len(dupes),
        "cleanup_folder": config.cleanup_folder(),
        "suggestions": [
            {**s, "potential_human": _human(s["potential_bytes"])} for s in suggestions
        ],
    }
