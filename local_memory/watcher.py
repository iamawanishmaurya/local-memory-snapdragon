"""Layer 1 — File Watcher: monitors chosen folders and feeds the pipeline."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from . import config

log = logging.getLogger(__name__)

IGNORE_NAMES = ("~$", "~", ".tmp", ".crdownload", ".part")  # office locks + partial downloads
DEBOUNCE_SECONDS = 2.0


def _should_process(path: str) -> bool:
    p = Path(path)
    if p.name.startswith(IGNORE_NAMES):
        return False
    ext = p.suffix.lower()
    if (
        ext in config.TEXT_EXTS
        or ext in config.PDF_EXTS
        or ext in config.DOCX_EXTS
        or ext in config.IMAGE_EXTS
        or ext in config.IPYNB_EXTS
    ):
        return True
    # Binary types (incl. compound suffixes like .tar.gz) — metadata ingestion.
    return config.binary_kind(path) is not None


class _Handler(FileSystemEventHandler):
    def __init__(self, on_change) -> None:
        self.on_change = on_change

    def on_created(self, event):
        if not event.is_directory:
            self._maybe(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._maybe(event.src_path)

    def on_moved(self, event):
        dest = getattr(event, "dest_path", "")
        if dest and not event.is_directory:
            # A rename orphans the old record set — purge the source first.
            self.on_change("deleted", event.src_path)
            self._maybe(dest)

    def on_deleted(self, event):
        if not event.is_directory:
            self.on_change("deleted", event.src_path)

    def _maybe(self, path: str):
        if _should_process(path):
            self.on_change("changed", path)


class FolderWatcher:
    """Watches a set of folders; callbacks are debounced per-file."""

    def __init__(self, on_change) -> None:
        self.on_change = on_change
        self.observer: Observer | None = None
        self._lock = threading.Lock()
        self._pending: dict[str, float] = {}
        self._timer: threading.Timer | None = None
        self.watching: set[str] = set()
        # Oryon pool for incremental indexing so burst saves don't serialize.
        from concurrent.futures import ThreadPoolExecutor
        import os as _os
        self._pool = ThreadPoolExecutor(max_workers=max(2, min(8, (_os.cpu_count() or 8) - 2)),
                                        thread_name_prefix="lm-watch")

    def start(self, folders: list[str]) -> None:
        self.stop()
        self.observer = Observer(timeout=1.0)
        self.watching = set()
        for f in folders:
            p = Path(f)
            if p.is_dir():
                self.observer.schedule(_Handler(self._debounced), str(p), recursive=True)
                self.watching.add(str(p))
        self.observer.start()
        log.info("watching %d folder(s): %s", len(self.watching), self.watching)

    def _debounced(self, kind: str, path: str) -> None:
        with self._lock:
            self._pending[path] = kind
            if self._timer is None:
                self._timer = threading.Timer(DEBOUNCE_SECONDS, self._flush)
                self._timer.daemon = True
                self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            pending, self._pending = self._pending, {}
            self._timer = None
        for path, kind in pending.items():
            try:
                # Offload to pool: NPU single-stream is preserved inside
                # pipeline.index_file via embedder locks; CPU extracts parallel.
                self._pool.submit(self._safe_change, kind, path)
            except Exception:
                log.exception("handler failed for %s", path)

    def _safe_change(self, kind: str, path: str) -> None:
        try:
            self.on_change(kind, path)
        except Exception:
            log.exception("handler failed for %s", path)

    def stop(self) -> None:
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5)
            self.observer = None
        self.watching = set()
