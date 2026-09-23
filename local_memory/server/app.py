"""Layer 6 — Local web UI server.

Binds to 127.0.0.1 ONLY: the interface is never reachable from the network.
All state lives in the local SQLite database; there is no external service.
"""
from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .. import config, privacy
from ..embeddings import base as emb_base
from ..extractors import image_understanding
from ..health import storage_health
from ..pipeline import handle_change, reconcile_deletions, rescan_all_async, scan_folder, status as pipeline_status
from ..search import query_engine
from ..stats import full_report as stats_report
from ..watcher import FolderWatcher

app = FastAPI(title=config.APP_NAME, docs_url=None, redoc_url=None)
watcher = FolderWatcher(handle_change)

# Built shadcn-admin UI (ui/dist). Falls back to the bundled single-file UI
# when the frontend hasn't been built yet.
UI_DIST = Path(__file__).resolve().parent.parent.parent / "ui" / "dist"


class FolderIn(BaseModel):
    path: str


class SearchIn(BaseModel):
    """POST /api/search body (D-04/D-07 contract).

    Empty/whitespace query -> 400 ``{"error": "empty query"}``. Queries are
    truncated to 200 chars server-side before reaching the engine (response
    echoes the effective query). FTS5 metacharacters are neutralized by
    ``query_engine.fts_quote`` — they can never 500. No ``?mode=`` parameter:
    the fused ranking is the single API surface.
    """

    query: str
    top_k: int = 12


@app.get("/")
def index():
    return _serve_index()


def _serve_index():
    """Serve the SPA shell with the auth token injected invisibly (D-01).

    Token injection is per-response: the token is NEVER written into ui/dist
    on disk. Both the index route and the SPA catch-all fallback use this so
    client-route reloads also receive the token.
    """
    candidates = [UI_DIST / "index.html", Path(__file__).parent / "static" / "index.html"]
    for candidate in candidates:
        if candidate.exists():
            html = candidate.read_text(encoding="utf-8")
            script = f'<script>window.__LM_TOKEN__="{config.auth_token()}"</script>'
            html = html.replace("</head>", f"{script}</head>", 1)
            from fastapi.responses import HTMLResponse
            return HTMLResponse(html)
    return JSONResponse({"error": "UI not built"}, status_code=404)


_LOOPBACK_HOSTNAMES = {"localhost", "127.0.0.1", "::1"}
_security_log = logging.getLogger("local_memory.server.security")


def _host_hostname(host_header: str) -> str | None:
    """Hostname (no port, no brackets) from a Host header, or None if unparseable."""
    try:
        return urlparse(f"//{host_header}").hostname
    except ValueError:
        return None


def _origin_port(origin: str) -> int | None:
    try:
        return urlparse(origin).port
    except ValueError:
        return None


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Host/Origin defense (D-03) + Bearer token on /api/* (D-02) in ONE handler.

    Ordering is explicit and load-bearing: Host check -> Origin check -> token
    check, so 403 always wins over 401 and a spoofed-Host request can never
    reach an endpoint (nor the SPA shell, whose HTML embeds the token).

    Fail closed: missing Host, non-loopback Host, or a foreign Origin is 403
    and LOGGED (ROADMAP: rejected rebinding/CSRF attempts are a demo talking
    point). Missing Origin passes (curl, same-origin GET navigations).
    """
    host_header = request.headers.get("Host", "")
    hostname = _host_hostname(host_header) if host_header else None
    if hostname is None or hostname.lower() not in _LOOPBACK_HOSTNAMES:
        _security_log.warning(
            "rejected: forbidden host method=%s path=%s host=%s", request.method, request.url.path, host_header or "<missing>"
        )
        return JSONResponse({"error": "forbidden host"}, status_code=403)

    origin = request.headers.get("Origin")
    if origin:
        o = urlparse(origin)
        # Same server port as the Host header the client used (default: APP_PORT).
        try:
            host_port = urlparse(f"//{host_header}").port
        except ValueError:
            host_port = None
        expected_port = host_port or config.APP_PORT
        ok = (
            o.scheme == "http"
            and (o.hostname or "").lower() in _LOOPBACK_HOSTNAMES
            and _origin_port(origin) == expected_port
        )
        # DEV FLAG (pinned rule): LOCAL_MEMORY_DEV=1 additionally allows the
        # Vite dev server origin (http://localhost:5173 / 127.0.0.1:5173). This
        # is the only sanctioned dev-origin override; production rejects it.
        if not ok and os.environ.get("LOCAL_MEMORY_DEV") == "1" and o.scheme == "http":
            dev_ok = (o.hostname or "").lower() in _LOOPBACK_HOSTNAMES and _origin_port(origin) in (5173, None)
            ok = dev_ok
        if not ok:
            _security_log.warning(
                "rejected: forbidden origin method=%s path=%s host=%s origin=%s",
                request.method, request.url.path, host_header, origin,
            )
            return JSONResponse({"error": "forbidden origin"}, status_code=403)

    if request.url.path.startswith("/api/"):
        expected = f"Bearer {config.auth_token()}"
        if request.headers.get("Authorization", "") != expected:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
    return await call_next(request)


# Serve the built SPA assets; the client-side-route catch-all is registered
# at the bottom of this module, after all /api routes, so it never shadows them.
if UI_DIST.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/assets", StaticFiles(directory=UI_DIST / "assets"), name="assets")
    app.mount("/images", StaticFiles(directory=UI_DIST / "images"), name="images")


@app.get("/api/status")
def api_status():
    import os as _os
    from ..store import database
    from ..extractors import ocr as _ocr
    prov = emb_base.provider_report()
    return {
        "app": config.APP_NAME,
        "version": "0.1.0",
        "paused": privacy.is_paused(),
        "pipeline": pipeline_status(),
        "watched_folders": config.watched_folders(),
        "npu": {
            "providers_available": prov["available"],
            "qnn_available": prov["qnn_available"],
            "htp_mode": prov["htp_mode"],
            "batch": prov["default_batch"],
        },
        "cpu": {
            "cores": _os.cpu_count(),
            "workers": max(2, min(10, (_os.cpu_count() or 8) - 2)),
        },
        "gpu": {"dml_available": prov["dml_available"]},
        "ocr_backend": _ocr.active_backend(),
        "stats": database.stats() if config.DB_PATH.exists() else {"files": 0, "chunks": 0, "images": 0, "total_bytes": 0},
    }


@app.post("/api/folders")
def api_add_folder(body: FolderIn):
    p = Path(body.path)
    if not p.is_dir():
        return JSONResponse({"error": f"Not a folder: {body.path}"}, status_code=400)
    folders = config.add_watched_folder(str(p))
    _restart_watcher(folders)
    rescan_all_async(folders)
    return {"watched_folders": folders}


@app.delete("/api/folders")
def api_remove_folder(body: FolderIn):
    folders = config.remove_watched_folder(body.path)
    _restart_watcher(folders)
    return {"watched_folders": folders}


@app.post("/api/index")
def api_index_now():
    folders = config.watched_folders()
    rescan_all_async(folders)
    return {"started": True, "folders": folders}


@app.post("/api/pause")
def api_pause():
    privacy.set_paused(True)
    return {"paused": True}


@app.post("/api/resume")
def api_resume():
    privacy.set_paused(False)
    return {"paused": False}


MAX_QUERY_CHARS = 200  # D-07: longer queries are truncated, not rejected


@app.post("/api/search")
def api_search(body: SearchIn):
    if not body.query.strip():
        return JSONResponse({"error": "empty query"}, status_code=400)
    q = body.query[:MAX_QUERY_CHARS]
    # Zero results stays a clean 200 {"results": []} — the UI adds the hint.
    return {"query": q, "results": query_engine.search(q, body.top_k)}


class OpenIn(BaseModel):
    file_id: int


@app.post("/api/open")
def api_open(body: OpenIn):
    """Open an indexed file with the OS default application.

    localhost-only by design: the middleware (D-02/D-03) already enforces a
    loopback Host and a Bearer token on every /api/* route. The caller supplies
    a numeric file id — never a path — so only files that are actually in the
    index can be opened, and only if they still exist on disk.
    """
    from ..store import database

    row = database.file_by_id(body.file_id)
    if row is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    path = Path(row["path"])
    if not path.is_file():
        return JSONResponse({"error": "file no longer on disk"}, status_code=410)
    try:
        os.startfile(str(path))  # Windows: opens with the default associated app
    except Exception as exc:  # short error string only — no path leakage
        return JSONResponse({"error": f"could not open file ({type(exc).__name__})"}, status_code=500)
    return {"opened": True}


@app.get("/api/stats")
def api_stats():
    """Aggregate statistics for the Statistics page. All sections are guarded
    in local_memory/stats.py — this endpoint never 500s on partial failure."""
    watcher_running = False
    try:
        watcher_running = bool(watcher.observer and watcher.observer.is_alive())
    except Exception:
        pass
    try:
        pipeline = dict(pipeline_status())
    except Exception:
        pipeline = None
    return stats_report(watcher_running=watcher_running, pipeline=pipeline)


@app.get("/api/health")
def api_health():
    return storage_health.health_report()


class MoveIn(BaseModel):
    path: str


class CleanupFolderIn(BaseModel):
    cleanup_folder: str


def _watched_root(path: Path) -> str | None:
    """The watched folder containing `path`, or None (roots are exact prefixes)."""
    resolved = str(path.resolve())
    for root in config.watched_folders():
        root_resolved = str(Path(root).resolve())
        if resolved == root_resolved or resolved.startswith(root_resolved.rstrip("\\/") + "\\"):
            return root
    return None


@app.post("/api/cleanup/move")
def api_cleanup_move(body: MoveIn):
    """Move ONE indexed file into the cleanup folder (D-03: reversible, never
    deletes). Mirrors the /api/thumbnail index-only rule: the path must be
    inside a watched folder AND present in the files table, so arbitrary local
    paths can never be moved. The index row disappears via the existing
    deletion propagation (watcher moved-event purge + startup reconcile).
    """
    import shutil

    from ..store import database
    src = Path(body.path)
    row = database.get_file(str(src))
    if row is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    if _watched_root(src) is None:
        return JSONResponse({"error": "outside watched folders"}, status_code=400)
    if not src.is_file():
        return JSONResponse({"error": "file no longer on disk"}, status_code=410)
    dest_dir = Path(config.cleanup_folder())
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    stem, suffix = src.stem, src.suffix
    n = 1
    while dest.exists():
        dest = dest_dir / f"{stem} ({n}){suffix}"
        n += 1
    try:
        shutil.move(str(src), str(dest))
    except OSError as exc:
        return JSONResponse({"error": f"move failed ({type(exc).__name__})"}, status_code=500)
    return {"moved": True, "dest": str(dest)}


@app.get("/api/cleanup/config")
def api_cleanup_config():
    return {"cleanup_folder": config.cleanup_folder()}


@app.put("/api/cleanup/config")
def api_cleanup_set_config(body: CleanupFolderIn):
    try:
        saved = config.set_cleanup_folder(body.cleanup_folder)
    except OSError as exc:
        return JSONResponse({"error": f"could not create folder ({type(exc).__name__})"}, status_code=400)
    return {"cleanup_folder": saved}


@app.get("/api/thumbnail")
def api_thumbnail(file_id: int):
    """Serve the cached thumbnail for an INDEXED image (D-04: index-lookup only).

    The caller supplies a file id — never a path. The served file is resolved
    from the files table and lives under config.THUMBS_DIR, so arbitrary local
    paths can never reach a filesystem call. Non-indexed / non-image -> 404
    (no existence oracle).

    NOTE (Phase 4): if the UI ever previews images it must use fetch ->
    URL.createObjectURL — a Bearer header cannot ride <img src>.
    """
    import hashlib

    from ..store import database
    row = database.file_by_id(file_id)
    if row is None or row["kind"] != "image":
        return JSONResponse({"error": "not found"}, status_code=404)
    digest = hashlib.sha1(row["path"].encode("utf-8")).hexdigest()
    thumb = config.THUMBS_DIR / f"{digest}.jpg"
    # Belt-and-braces: the served file must always be inside THUMBS_DIR —
    # the base is a constant and the name is a hex digest, so this can never
    # fail, but fail loudly (500) rather than serve outside the sandbox.
    assert thumb.resolve().parent == config.THUMBS_DIR.resolve(), "thumbnail escaped THUMBS_DIR"
    if not thumb.is_file():
        # Regenerate ONLY from the indexed path; a deleted source 404s —
        # no regeneration from (and no read of) a caller-named path.
        source = Path(row["path"])
        if not source.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        thumb = image_understanding.make_thumbnail(source) or thumb
    if thumb.is_file():
        return FileResponse(thumb, media_type="image/jpeg")
    return JSONResponse({"error": "not found"}, status_code=404)


@app.post("/api/wipe")
def api_wipe():
    from ..store import database, vector_store
    watcher.stop()
    removed = privacy.wipe_all()
    try:
        vector_store.wipe()
    except Exception:
        pass
    database.init_db()
    return {"wiped": removed}


@app.post("/api/clean")
def api_clean():
    """Safe cleanup: prune missing files, orphan chunks, vacuum. Keeps settings."""
    return {"cleaned": privacy.clean_orphans(), "usage": privacy.storage_usage()}


@app.get("/api/bench")
def api_bench():
    from ..bench import run_all
    return {"results": run_all()}


@app.get("/api/suggest-folders")
def api_suggest_folders():
    suggestions = []
    for name in config.DEFAULT_WATCHED:
        p = config.user_profile_dir(name)
        if p.is_dir():
            suggestions.append(str(p))
    return {"suggestions": suggestions}


def _restart_watcher(folders: list[str]) -> None:
    watcher.start(folders)


@app.on_event("startup")
def startup() -> None:
    from ..store import database, vector_store
    # Storage relocation (D-04/D-05): copy legacy repo data/ into the new home
    # BEFORE any SQLite connection opens, then create fresh dirs.
    fallback = config.migrate_home()
    if fallback:
        print(f"[config] migration failed — operating on legacy path: {fallback}")
    config.ensure_dirs()
    # Prime the auth token (D-01) so the token file exists before the UI is
    # ever served and the middleware always has a value to compare against.
    config.auth_token()
    database.init_db()
    vector_store.init_db()
    # FTS backfill (D-02): create/repair the FTS5 indexes BEFORE the reconcile
    # sweep, so deletions fire into a populated index. Idempotent — a healthy
    # index skips the rebuild.
    try:
        import logging

        fts = database.backfill_fts()
        if fts.get("rebuilt"):
            logging.getLogger(__name__).info("FTS backfill rebuilt: %s", fts)
    except Exception:
        import logging

        logging.getLogger(__name__).exception("FTS backfill failed")
    folders = config.watched_folders()
    # Reconcile sweep (D-06): self-heal deletions that happened while the app
    # was off — before the watcher starts, so events never race the sweep.
    try:
        reconcile_deletions(folders)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("startup reconcile sweep failed")
    if folders:
        watcher.start(folders)
        rescan_all_async(folders)


# Client-side route fallback for the built SPA — registered LAST so every
# /api route defined above wins over it.
if (UI_DIST / "index.html").exists():

    @app.get("/{spa_path:path}")
    def spa(spa_path: str):
        candidate = UI_DIST / spa_path
        if candidate.is_file():
            return FileResponse(candidate)
        return _serve_index()
