"""CLI demo: index a folder, search it, inspect health, or wipe — no UI needed.

    python scripts/demo_index.py --index "C:/Users/you/Documents"
    python scripts/demo_index.py --search "invoice from last month"
    python scripts/demo_index.py --health
    python scripts/demo_index.py --doctor
    python scripts/demo_index.py --bench
    python scripts/demo_index.py --clean
    python scripts/demo_index.py --wipe
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", metavar="FOLDER", help="index a folder once")
    ap.add_argument("--workers", type=int, default=None, help="CPU extract workers (default: auto Oryon pool)")
    ap.add_argument("--batch", type=int, default=64, help="NPU text batch size (default 64)")
    ap.add_argument("--clean-after", action="store_true", help="wipe index after --index/--search (benchmark hygiene)")
    ap.add_argument("--search", metavar="QUERY", help="natural-language search")
    ap.add_argument("--health", action="store_true", help="storage health report")
    ap.add_argument("--doctor", action="store_true", help="NPU diagnostics")
    ap.add_argument("--bench", action="store_true", help="run benchmark table + ratings")
    ap.add_argument("--clean", action="store_true", help="prune missing/orphans + vacuum (keeps settings)")
    ap.add_argument("--wipe", action="store_true", help="destroy the local index")
    ap.add_argument("--no-deps", action="store_true", help="skip dependency validation (or set LOCAL_MEMORY_NO_DEPS=1)")
    args = ap.parse_args()

    # Fail-fast dependency validation (no pip, no network). Skipped with
    # --no-deps or LOCAL_MEMORY_NO_DEPS=1.
    if not args.no_deps and os.environ.get("LOCAL_MEMORY_NO_DEPS") != "1":
        from local_memory.deps import fail_fast
        fail_fast()

    if args.doctor:
        from local_memory.main import doctor
        doctor()
    elif args.bench:
        from local_memory.bench import run_all
        run_all()
    elif args.clean:
        from local_memory import privacy
        import json
        print(json.dumps(privacy.clean_orphans(), indent=2))
    elif args.wipe:
        from local_memory import privacy
        from local_memory.store import database, vector_store
        print("Wiped:", privacy.wipe_all())
        vector_store.wipe()
        database.init_db()
        print("Local index destroyed. Nothing remains on disk.")
    elif args.index:
        from local_memory.pipeline import scan_folder
        from local_memory.store import database, vector_store
        import time
        database.init_db()
        vector_store.init_db()
        t0 = time.perf_counter()
        n = scan_folder(args.index, workers=args.workers, batch_text=args.batch,
                        clean_after=args.clean_after)
        dt = time.perf_counter() - t0
        print(f"Indexed {n} new/updated file(s) from {args.index} in {dt:.2f}s "
              f"({n/dt:.1f} files/s, workers={args.workers or 'auto'}, batch={args.batch})")
        if args.clean_after:
            print("Index cleaned after run (--clean-after). Device left clean.")
    elif args.search:
        from local_memory.store import database, vector_store
        database.init_db()
        vector_store.init_db()
        from local_memory.search import query_engine
        results = query_engine.search(args.search)
        if not results:
            print("No results.")
        for r in results:
            print(f"\n[{r['score']*100:5.1f}%] {r['name']}  ({r['kind']}, {r['size_bytes']} bytes)")
            print(f"        {r['path']}")
            if r["snippet"]:
                print(f"        {r['snippet'][:140]}")
        if args.clean_after:
            from local_memory import privacy
            from local_memory.store import database as _db, vector_store as _vs
            privacy.wipe_all()
            _vs.wipe()
            _db.init_db()
            _vs.init_db()
            print("\nIndex cleaned after search (--clean-after).")
    elif args.health:
        from local_memory.health import storage_health
        import json
        print(json.dumps(storage_health.health_report(), indent=2))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
