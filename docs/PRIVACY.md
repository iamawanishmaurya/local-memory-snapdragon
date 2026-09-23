# Privacy Model

Local Memory's entire value proposition is that your files never leave your
PC. This document states exactly what the software does, so it can be audited.

## Data flow

```
your files ──read──▶ local pipeline ──write──▶ data/ (SQLite + vectors + thumbs)
                        │
                        └── never: network, telemetry, cloud, accounts
```

- **Reads:** only files inside folders you explicitly added. The watcher
  never traverses outside them; deletions remove DB rows immediately.
- **Writes:** one directory tree, `data/` (configurable via
  `LOCAL_MEMORY_HOME`): `index.db` (metadata + text chunks + vectors) and
  `thumbs/` (image previews). Nothing else is created anywhere on disk.
- **Network:** the server binds to `127.0.0.1` only and makes **zero outbound
  connections**. The single exception in the whole product is the one-time
  model export in `scripts/setup_models.py`, which you run deliberately
  during setup. After that the machine can be offline forever.

## User controls

| Control | Effect |
|---|---|
| Add/remove folders | Scoping is per-folder, opt-in, reversible |
| Pause / Resume | Halts watcher callbacks and all indexing |
| **Wipe all data** | Deletes the entire data home: database, vectors, thumbnails, settings. No confirmation prompt can be bypassed; nothing is recoverable afterwards — and nothing was ever copied elsewhere |

## Guarantees we can defend to judges

1. `grep` the codebase for `requests`/`httpx`/`urllib` in `local_memory/`:
   none (urllib appears only in the setup script).
2. The API server binds `127.0.0.1` — not reachable from the network.
3. Uninstalling = delete the folder. No registry entries, no services,
   no scheduled tasks, no files outside `data/` and `models/`.
