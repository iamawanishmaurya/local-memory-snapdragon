# Pitch Metrics

Append-only run log recorded at demo time.


## Run — 2026-09-23T16:40:05.505037+00:00

| Metric | Value |
|--------|-------|
| NPU active | CPUExecutionProvider (live: False) |
| Cold start — text_embed_s | 1.748 s |
| Cold start — clip_image_s | 1.29 s |
| Cold start — clip_text_s | 1.107 s |
| Query latency p50 / p95 | 3010.2 / 3968.3 ms (n=3) |
| Throughput | 25.5 chunks/s embed · 3.4 files/s index |
| Corpus | 241 files · 10074 chunks |
| CPU cores / onnxruntime | 8 / 1.30.0 |

```json
{
  "recorded_at": "2026-09-23T16:40:05.505037+00:00",
  "device": {
    "cpu_cores": 8,
    "onnxruntime": "1.30.0"
  },
  "npu": {
    "requested": "QNN",
    "active": "CPUExecutionProvider",
    "npu_live": false
  },
  "cold_start_s": {
    "text_embed_s": 1.748,
    "clip_image_s": 1.29,
    "clip_text_s": 1.107
  },
  "latency": {
    "count": 3,
    "p50_ms": 3010.2,
    "p95_ms": 3968.3
  },
  "throughput": {
    "embed_chunks_per_s": 25.5,
    "index_files_per_s": 3.4
  },
  "corpus": {
    "files": 241,
    "chunks": 10074
  }
}
```
