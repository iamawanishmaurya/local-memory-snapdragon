"""PERF-01 benchmark: 100k synthetic vectors, search latency + recall gate.

Builds a TEMP-database index of 100,000 x 768-dim L2-normalized float32
vectors (seeded RNG, never the real DB), then measures vector_store.search()
latency over 200 probes and checks recall@10 against numpy brute force.

- If the sqlite-vec ANN path is active: asserts p50 < 100 ms and
  recall@10 == 1.0 (vec0 v0.1.9 KNN is exact), exits non-zero on failure.
- If ANN is inactive (expected default on win-arm64 without a vendored
  vec0.dll): records brute-force timings, prints SKIPPED, exits 0.

Appends one results line to .planning/phases/06-*/06-BENCHMARK.md.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_VECTORS = 100_000
DIM = 768
BATCH = 5_000
N_PROBES = 200
N_RECALL = 100
TOP_K = 10
GATE_MS = 100.0


def _percentile(sorted_ms: list[float], q: float) -> float:
    idx = min(len(sorted_ms) - 1, int(round(q * (len(sorted_ms) - 1))))
    return sorted_ms[idx]


def main() -> int:
    # Isolate the benchmark from the real data home BEFORE importing
    # local_memory.config (module-level DB_PATH binds at import).
    tmpdir = tempfile.mkdtemp(prefix="lm-bench-100k-")
    # config resolves DATA_HOME/DB_PATH from LOCAL_MEMORY_HOME at import.
    os.environ["LOCAL_MEMORY_HOME"] = tmpdir
    os.environ.setdefault("LOCAL_MEMORY_NO_DEPS", "1")
    for mod in [m for m in list(sys.modules) if m.startswith("local_memory")]:
        del sys.modules[mod]

    from local_memory import config
    config.ensure_dirs()
    import local_memory.store.vector_store as vs

    from local_memory import perf as perf_mod
    perf_mod.reset_latency()

    vs.init_db()
    rng = np.random.default_rng(42)
    mat = rng.standard_normal((N_VECTORS, DIM), dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat /= norms

    t0 = time.perf_counter()
    for start in range(0, N_VECTORS, BATCH):
        rows = []
        for i in range(start, min(start + BATCH, N_VECTORS)):
            # Spread over 20 synthetic "files" x 5000 chunks.
            file_id = i // (N_VECTORS // 20)
            rows.append((file_id, i % (N_VECTORS // 20), "text", mat[i]))
        vs.upsert_many(rows)
    insert_s = time.perf_counter() - t0
    print(f"inserted {N_VECTORS} vectors in {insert_s:.1f}s")

    probes = mat[:N_PROBES]  # first 200 rows of the same RNG stream

    timings = []
    results = []
    for i, q in enumerate(probes):
        t = time.perf_counter()
        res = vs.search("text", q, top_k=TOP_K)
        timings.append((time.perf_counter() - t) * 1000.0)
        if i < N_RECALL:
            results.append(res)
    timings.sort()
    p50, p95 = _percentile(timings, 0.50), _percentile(timings, 0.95)

    # Brute-force ground truth for the recall sample.
    ann = vs._ann_status()
    ann_active = bool(ann.get("available"))
    recall = None
    if ann_active:
        full_mat, keys = vs.load_all("text", use_cache=False)
        hits = 0
        for i in range(N_RECALL):
            truth = full_mat @ probes[i]
            top = {keys[j] for j in np.argsort(-truth)[:TOP_K]}
            got = {(r["file_id"], r["ordinal"]) for r in results[i]}
            hits += len(top & got)
        recall = hits / (N_RECALL * TOP_K)

    backend = "vec0-ann" if ann_active else "brute-force-numpy"
    print(f"backend={backend} n_vectors={N_VECTORS} probes={N_PROBES}")
    print(f"p50={p50:.1f}ms p95={p95:.1f}ms" + ("" if recall is None else f" recall@10={recall:.4f}"))

    # Record into the perf latency window too.
    for ms in timings:
        perf_mod.append_latency(ms)

    # Report line (append) to the phase benchmark report.
    report = None
    plan_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".planning", "phases",
        "06-search-performance-npu-query-rewriter-and-ann-vector-index",
    )
    if os.path.isdir(plan_dir):
        report = os.path.join(plan_dir, "06-BENCHMARK.md")
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        line = (f"- {stamp} | backend={backend} | n={N_VECTORS} | probes={N_PROBES} | "
                f"p50={p50:.1f}ms | p95={p95:.1f}ms | "
                f"recall@10={'n/a (brute-force is ground truth)' if recall is None else format(recall, '.4f')}\n")
        with open(report, "a", encoding="utf-8") as f:
            f.write(line)

    if not ann_active:
        print("SKIPPED: ANN inactive, brute-force timings recorded")
        print("(deferred: ANN inactive on this device - <100ms + recall gates evaluate once vec0.dll is built)")
        return 0

    ok = True
    if p50 >= GATE_MS:
        print(f"FAIL: p50 {p50:.1f}ms >= {GATE_MS}ms gate")
        ok = False
    if recall != 1.0:
        print(f"FAIL: recall@10 {recall} != 1.0")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
