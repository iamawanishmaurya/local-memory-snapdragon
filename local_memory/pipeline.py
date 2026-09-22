"""Indexing pipeline: extract → embed → store. Orchestrates layers 2–4.

Heterogeneous design (Snapdragon X Elite):
  Pool A (Oryon CPU, N workers) — file IO, text/PDF/DOCX parse, PIL decode,
      tokenize, thumbnail. Embarrassingly parallel.
  Stage B (Hexagon NPU, 1 stream) — batched ONNX inference (batch 16-32).
      Batching is what saturates the HTP; per-chunk runs starve it.
  Stage C (single writer) — SQLite commits in bulk (1 commit / 100 files).
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config
from .embeddings import get_image_embedder, get_text_embedder
from .extractors import image_understanding, text_extractor
from .store import database, vector_store

log = logging.getLogger(__name__)

_state_lock = threading.Lock()
_state = {"status": "idle", "queued": 0, "done": 0, "last_file": ""}
_embedders_started = False


def status() -> dict:
    with _state_lock:
        return dict(_state)


def _set(**kw) -> None:
    with _state_lock:
        _state.update(kw)


def _cpu_workers(requested: int | None = None) -> int:
    if requested and requested > 0:
        return requested
    env = os.environ.get("LOCAL_MEMORY_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    # X Elite: 12 Oryon cores. Leave 2 for NPU driver/UI, use rest for extract.
    return max(2, min(10, (os.cpu_count() or 8) - 2))


def _ensure_embedders() -> None:
    global _embedders_started
    if not _embedders_started:
        get_text_embedder()
        get_image_embedder()
        _embedders_started = True


def _extract_one(path: Path):
    """CPU-only stage: stat + parse + chunk. Returns job dict or None (skip)."""
    try:
        stat = path.stat()
    except OSError:
        return None
    if not database.needs_index(str(path), stat.st_mtime, stat.st_size):
        return None
    ext = path.suffix.lower()
    folder = str(path.parent)
    if ext in config.IMAGE_EXTS:
        try:
            from PIL import Image
            img = Image.open(path)
            img.load()
            # Copy pixels now so worker thread can release the file handle;
            # NPU batch stage owns the in-memory copy (zero re-read).
            img_copy = img.copy()
        except Exception:
            return {"kind": "meta", "path": path, "folder": folder, "ext": ext,
                    "size": stat.st_size, "mtime": stat.st_mtime, "meta_kind": "other"}
        ocr_text, _ = image_understanding.describe_image(path)
        thumb = image_understanding.make_thumbnail(path)
        chunks = text_extractor.chunk_words(ocr_text) if ocr_text.strip() else []
        return {"kind": "image", "path": path, "folder": folder, "ext": ext,
                "size": stat.st_size, "mtime": stat.st_mtime,
                "ocr_text": ocr_text, "chunks": chunks, "img": img_copy, "thumb": thumb}
    text, kind = text_extractor.extract_text(path)
    if not text.strip():
        return {"kind": "meta", "path": path, "folder": folder, "ext": ext,
                "size": stat.st_size, "mtime": stat.st_mtime, "meta_kind": kind}
    return {"kind": "text", "path": path, "folder": folder, "ext": ext,
            "size": stat.st_size, "mtime": stat.st_mtime,
            "meta_kind": kind, "chunks": text_extractor.chunk_words(text)}


def _commit_job(job, t_emb, i_emb) -> bool:
    """NPU + writer stage: embed (batched by caller where possible) + store."""
    if job is None:
        return False
    if job["kind"] == "meta":
        database.upsert_file(str(job["path"]), job["folder"], job["ext"],
                             job["size"], job["mtime"], job["meta_kind"])
        return False
    if job["kind"] == "text":
        file_id = database.upsert_file(str(job["path"]), job["folder"], job["ext"],
                                       job["size"], job["mtime"], job["meta_kind"])
        database.replace_chunks(file_id, list(enumerate(job["chunks"])))
        vecs = t_emb.encode_batch(job["chunks"])
        vector_store.upsert_many([(file_id, i, "text", v) for i, v in enumerate(vecs)])
        return True
    if job["kind"] == "image":
        file_id = database.upsert_file(str(job["path"]), job["folder"], job["ext"],
                                       job["size"], job["mtime"], "image",
                                       ocr_text=job["ocr_text"])
        if job["chunks"]:
            vecs = t_emb.encode_batch(job["chunks"])
            vector_store.upsert_many([(file_id, i, "text", v) for i, v in enumerate(vecs)])
            database.replace_chunks(file_id, list(enumerate(job["chunks"])))
        vec = i_emb.encode_image(job["img"])
        vector_store.upsert(file_id, 0, "image", vec)
        return True
    return False


def index_file(path: str) -> bool:
    """Index a single file. Returns True if it was (re)indexed."""
    p = Path(path)
    if not p.is_file():
        database.delete_file(str(p))
        return False
    try:
        stat = p.stat()
    except OSError:
        return False
    if not database.needs_index(str(p), stat.st_mtime, stat.st_size):
        return False

    ext = p.suffix.lower()
    folder = str(p.parent)

    if ext in config.IMAGE_EXTS:
        return _index_image(p, folder, ext, stat.st_size, stat.st_mtime)

    text, kind = text_extractor.extract_text(p)
    if not text.strip():
        # Nothing extractable — still record metadata for storage health.
        database.upsert_file(str(p), folder, ext, stat.st_size, stat.st_mtime, kind)
        return False

    chunks = text_extractor.chunk_words(text)
    file_id = database.upsert_file(str(p), folder, ext, stat.st_size, stat.st_mtime, kind)
    database.replace_chunks(file_id, list(enumerate(chunks)))

    _ensure_embedders()
    embedder = get_text_embedder()
    vectors = embedder.encode_batch(chunks)
    vector_store.upsert_many([(file_id, i, "text", v) for i, v in enumerate(vectors)])
    return True


def _index_image(p: Path, folder: str, ext: str, size: int, mtime: float) -> bool:
    try:
        from PIL import Image
        img = Image.open(p)
        img.load()
    except Exception:
        database.upsert_file(str(p), folder, ext, size, mtime, "other")
        return False

    ocr_text, _ = image_understanding.describe_image(p)
    file_id = database.upsert_file(str(p), folder, ext, size, mtime, "image", ocr_text=ocr_text)
    image_understanding.make_thumbnail(p)

    _ensure_embedders()
    chunks = []
    if ocr_text.strip():
        chunks = text_extractor.chunk_words(ocr_text)
        t_emb = get_text_embedder()
        vecs = t_emb.encode_batch(chunks)
        vector_store.upsert_many([(file_id, i, "text", v) for i, v in enumerate(vecs)])
        database.replace_chunks(file_id, list(enumerate(chunks)))

    i_emb = get_image_embedder()
    vec = i_emb.encode_image(img)
    vector_store.upsert(file_id, 0, "image", vec)
    return True


def scan_folder(folder: str, recursive: bool = True, workers: int | None = None,
                batch_text: int = 64, clean_after: bool = False) -> int:
    """Full (re)scan of a folder. Returns number of files (re)indexed.

    Heterogeneous: CPU pool extracts in parallel, then NPU embeds with
    cross-file batching (all text chunks concatenated into batches of
    `batch_text`), then bulk store. `clean_after` wipes the index when done
    (for benchmark hygiene — leaves device clean).
    """
    from . import privacy

    root = Path(folder)
    if not root.is_dir():
        return 0
    exts = config.TEXT_EXTS | config.PDF_EXTS | config.DOCX_EXTS | config.IMAGE_EXTS
    files = [p for p in (root.rglob("*") if recursive else root.glob("*")) if p.is_file() and p.suffix.lower() in exts]
    if not files:
        return 0
    n_workers = _cpu_workers(workers)
    count = 0
    _set(status="indexing", queued=len(files), done=0, last_file="")
    t0 = time.time()
    try:
        # Stage A — parallel extract on Oryon cores.
        jobs: list = []
        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="lm-extract") as pool:
            for i, job in enumerate(pool.map(_extract_one, files)):
                if _is_paused():
                    break
                if job is not None:
                    jobs.append(job)
                _set(status="indexing", queued=len(files), done=i + 1,
                     last_file=str(files[i]) if i < len(files) else "")
        if _is_paused():
            _set(status="idle", queued=0, done=0)
            return count
        # Stage B+C — NPU batched embed + bulk store (single stream = max HTP).
        _ensure_embedders()
        t_emb, i_emb = get_text_embedder(), get_image_embedder()
        # Cross-file text batching: gather all text chunks, embed in big
        # batches, then scatter back. One NPU run per `batch_text` chunks.
        text_jobs = [j for j in jobs if j["kind"] == "text" or (j["kind"] == "image" and j["chunks"])]
        all_chunks: list[str] = [c for j in text_jobs for c in j["chunks"]]
        vec_map: list = []
        if all_chunks:
            import numpy as np
            vecs = t_emb.encode_batch(all_chunks, batch_size=batch_text)
            vec_map = [vecs[i] for i in range(len(all_chunks))]
        # Scatter + store.
        cursor = 0
        img_batch: list = []  # batch CLIP image embeds (Adreno/NPU friendly)
        img_jobs: list = []
        for j in jobs:
            if j["kind"] == "meta":
                database.upsert_file(str(j["path"]), j["folder"], j["ext"],
                                     j["size"], j["mtime"], j["meta_kind"])
                continue
            if j["kind"] == "text":
                n = len(j["chunks"])
                v = vec_map[cursor: cursor + n]
                cursor += n
                fid = database.upsert_file(str(j["path"]), j["folder"], j["ext"],
                                           j["size"], j["mtime"], j["meta_kind"])
                database.replace_chunks(fid, list(enumerate(j["chunks"])))
                vector_store.upsert_many([(fid, i, "text", vv) for i, vv in enumerate(v)])
                count += 1
            elif j["kind"] == "image":
                fid = database.upsert_file(str(j["path"]), j["folder"], j["ext"],
                                           j["size"], j["mtime"], "image",
                                           ocr_text=j["ocr_text"])
                if j["chunks"]:
                    n = len(j["chunks"])
                    v = vec_map[cursor: cursor + n]
                    cursor += n
                    vector_store.upsert_many([(fid, i, "text", vv) for i, vv in enumerate(v)])
                    database.replace_chunks(fid, list(enumerate(j["chunks"])))
                img_jobs.append((fid, j["img"]))
                count += 1
        # Batched CLIP image pass (single NPU stream).
        if img_jobs and hasattr(i_emb, "encode_images"):
            try:
                imgs = [im for _, im in img_jobs]
                vecs = i_emb.encode_images(imgs)
                vector_store.upsert_many([(fid, 0, "image", v) for (fid, _), v in zip(img_jobs, vecs)])
            except Exception:
                log.exception("batched image embed failed, falling back to singles")
                for fid, im in img_jobs:
                    try:
                        vector_store.upsert(fid, 0, "image", i_emb.encode_image(im))
                    except Exception:
                        log.exception("image embed failed for %s", fid)
        elif img_jobs:
            for fid, im in img_jobs:
                try:
                    vector_store.upsert(fid, 0, "image", i_emb.encode_image(im))
                except Exception:
                    log.exception("image embed failed for %s", fid)
    finally:
        elapsed = time.time() - t0
        _set(status="idle", queued=0, done=0, last_file=f"{count} files in {elapsed:.1f}s")
    if clean_after:
        from . import privacy as _privacy
        _privacy.wipe_all()
        from .store import vector_store as _vs
        try:
            _vs.wipe()
        except Exception:
            pass
        database.init_db()
        _vs.init_db()
    return count


def _is_paused() -> bool:
    from . import privacy
    return privacy.is_paused()


def _purge(path: str) -> None:
    """Zero-orphan deletion: DB row, vectors, and thumbnail for one path."""
    file_id = database.delete_file(path)
    if file_id is not None:
        vector_store.delete(file_id)
    try:
        digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()
        (config.THUMBS_DIR / f"{digest}.jpg").unlink(missing_ok=True)
    except Exception:
        log.exception("thumbnail purge failed for %s", path)


def handle_change(kind: str, path: str) -> None:
    """Called by the watcher for every relevant file event."""
    if _is_paused():
        return
    if kind == "deleted":
        _purge(path)
        return
    try:
        index_file(path)
    except Exception:
        log.exception("incremental index failed for %s", path)


def reconcile_deletions(folders: list[str]) -> tuple[int, int]:
    """Startup reconcile sweep (D-06): purge index entries whose source files
    vanished while the app was off, plus orphan chunks/vectors.

    Costs one stat per indexed path — no filesystem rescan. Idempotent.
    Returns (stale_entries_removed, orphan_vectors_removed).
    """
    stale = 0
    for row in database.list_files():
        p = str(row["path"])
        if not Path(p).exists():
            _purge(p)
            stale += 1
    database.delete_orphan_chunks()
    orphan_vectors = vector_store.delete_orphans()
    log.info("reconcile: removed %d stale entries, %d orphan vectors", stale, orphan_vectors)
    return stale, orphan_vectors


def rescan_all_async(folders: list[str]) -> threading.Thread:
    def run():
        for f in folders:
            if _is_paused():
                return
            scan_folder(f)

    t = threading.Thread(target=run, name="local-memory-rescan", daemon=True)
    t.start()
    return t
