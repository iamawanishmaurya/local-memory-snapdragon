"""Local Memory entry point.

    python -m local_memory.main            # start server + watchers
    python -m local_memory.main --doctor   # show NPU/provider diagnostics
"""
from __future__ import annotations

import argparse
import logging
import os
import sys


def doctor() -> None:
    import os
    import sys as _sys
    for _s in (_sys.stdout, _sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    from .embeddings import base
    from .extractors import ocr
    from . import config

    print("Local Memory — diagnostics")
    print(f"  data home      : {config.DATA_HOME}")
    print(f"  legacy data    : {config.LEGACY_DATA_HOME}")
    if config.is_cloud_synced(config.DATA_HOME):
        print(f"  WARNING: data path is under a cloud-synced folder: {config.DATA_HOME}")
    migrated = config.migrate_home()
    if migrated:
        print(f"  migration      : FAILED — operating on legacy path: {migrated}")
    elif (config.DATA_HOME / "index.db").exists():
        print(f"  migration      : complete (index present at {config.DATA_HOME})")
    else:
        print("  migration      : not needed (no legacy index found)")
    print(f"  models dir     : {config.MODELS_DIR}")
    print(f"  onnxruntime    : available" if base.available_providers() else "  onnxruntime    : NOT INSTALLED")
    rep = base.provider_report()
    print(f"  providers      : {', '.join(rep['available']) or '-'}")
    qnn = rep["qnn_available"]
    print(f"  QNN (NPU)      : {'AVAILABLE — models will run on the Snapdragon NPU' if qnn else 'not available (CPU fallback)'}")
    print(f"  HTP mode       : {rep['htp_mode']}  (LOCAL_MEMORY_HTP_MODE=burst|sustained|balanced)")
    print(f"  NPU batch      : {rep['default_batch']}  (LOCAL_MEMORY_BATCH)")
    print(f"  DML (GPU)      : {'AVAILABLE — Adreno via DirectML' if rep['dml_available'] else 'not available'}")
    import os as _os
    workers = max(2, min(10, (_os.cpu_count() or 8) - 2))
    print(f"  CPU workers    : {workers}  (Oryon pool, LOCAL_MEMORY_WORKERS overrides)")
    for name in ("nomic-embed-text.onnx", "clip-vit-b32-image.onnx", "clip-vit-b32-text.onnx",
                 "nomic-embed-text.serialized", "clip-vit-b32-image.serialized",
                 "trocr/encoder_model.onnx", "trocr/decoder_model_merged.onnx", "qwen3-0.6b.onnx"):
        path = config.MODELS_DIR / name
        tag = "OK" if path.exists() else ("missing — run scripts/setup_models.py" if name.endswith(".onnx") else "optional")
        print(f"  model {name:28s}: {tag}")
    print(f"  OCR backend    : {ocr.active_backend()}")
    hint = ocr.ocr_hint()
    if hint:
        print(f"  OCR setup      : {hint}")
    try:
        from .search.query_rewrite import rewrite as _rw
        print(f"  query rewrite  : {_rw('invoice from last month')['backend']}")
    except Exception:
        pass
    try:
        from .store import vector_store as _vs
        print(f"  vector store   : { _vs.stats()}")
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(prog="local-memory")
    parser.add_argument("--doctor", action="store_true", help="print NPU/provider diagnostics and exit")
    parser.add_argument("--bench", action="store_true", help="run throughput benchmark and exit")
    parser.add_argument("--clean", action="store_true", help="prune orphans + vacuum (keeps settings) and exit")
    parser.add_argument("--wipe", action="store_true", help="destroy the entire local index and exit")
    parser.add_argument("--no-deps", action="store_true", help="skip dependency validation (or set LOCAL_MEMORY_NO_DEPS=1)")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    # Console encoding: Windows cp1252 crashes on unicode status glyphs.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # Fail-fast dependency validation (D-07): no pip, no network — just imports.
    # Skipped with --no-deps or LOCAL_MEMORY_NO_DEPS=1 (explicit opt-out).
    if not args.no_deps and os.environ.get("LOCAL_MEMORY_NO_DEPS") != "1":
        from .deps import fail_fast
        fail_fast()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.doctor:
        doctor()
        return
    if args.clean:
        from . import privacy
        print(privacy.clean_orphans())
        return
    if args.wipe:
        from . import privacy
        from .store import database, vector_store
        print("Wiped:", privacy.wipe_all())
        try:
            vector_store.wipe()
        except Exception:
            pass
        database.init_db()
        vector_store.init_db()
        return
    if args.bench:
        from .bench import run_all
        run_all()
        return

    import uvicorn

    from . import config
    # Storage relocation (D-04/D-05): copy legacy repo data/ into the new home
    # BEFORE any SQLite connection opens.
    fallback = config.migrate_home()
    if fallback:
        print(f"[config] migration failed — operating on legacy path: {fallback}")
    config.ensure_dirs()
    host, port = config.APP_HOST, args.port or config.APP_PORT
    # Single-instance guard: a second server would double-index and lock the DB.
    import socket as _sock
    _probe = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
    _probe.settimeout(2.0)
    try:
        if _probe.connect_ex((host, port)) == 0:
            print(f"  Local Memory is already running at http://{host}:{port} — not starting a duplicate.")
            print(f"  Stop the other instance first (or use --port <other>).")
            return
    finally:
        _probe.close()
    print(f"\n  Local Memory running at http://{host}:{port}  (localhost only — nothing leaves this PC)\n")
    uvicorn.run("local_memory.server.app:app", host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
