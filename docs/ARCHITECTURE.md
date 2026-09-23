# Architecture

Local Memory is a six-layer on-device pipeline. Every layer runs locally; the
only network use in the product's lifetime is the one-time model export during
setup (`scripts/setup_models.py`).

```
watch → extract → embed → store → rank → UI
```

## 1. File Watcher Layer (`watcher.py`)

`watchdog` observers, one per user-chosen folder, recursive. Events are
debounced (2s) per file so saving a document repeatedly triggers one index
operation. Only files with extensions in `config.py`'s allow-list (text, PDF,
DOCX, images) are processed; Office lock files (`~$…`) are ignored. Deletions
remove the file's rows immediately.

## 2. Content Extraction Layer (`extractors/`)

| Input | Extractor | Output |
|---|---|---|
| `.txt .md .csv .log …` | direct read | plain text |
| `.pdf` | `pypdf` | page text |
| `.docx` | `python-docx` | paragraphs + tables |
| images | OCR (EasyOCR backend) + PIL | OCR text + thumbnail cache |

Long documents are split into overlapping 200-word windows
(`CHUNK_WORDS`/`CHUNK_OVERLAP_WORDS`) so retrieval points at the *relevant
passage*, and the UI can show a snippet around the match.

## 3. Embedding Layer (`embeddings/`)

All models load through `embeddings/base.create_session`, which requests
providers in priority order:

1. **`QNNExecutionProvider`** → Snapdragon NPU (Hexagon HTA/HTP via QNN)
2. `CPUExecutionProvider` → universal fallback (dev machines, x86)

This means the *same code* demonstrably runs on the NPU when present — check
with `python -m local_memory.main --doctor` — while remaining runnable anywhere.

- **Text:** Nomic-Embed-Text v1.5 (768-dim). One vector per chunk.
- **Images:** CLIP ViT-B/32 image encoder (512-dim), one vector per image.
- **Queries:** Nomic for document retrieval; CLIP text encoder so a text query
  can land in the image-embedding space.

When the ONNX files are absent (fresh clone, non-target machine), the package
falls back to a deterministic feature-hashing encoder: the full pipeline still
runs and the tests pass, but with toy retrieval quality. Real quality requires
the one-time model setup.

## 4. Local Vector Store + Metadata (`store/`)

- **Metadata** (`files`, `chunks` tables): SQLite with WAL mode. Path, size,
  mtime, kind, OCR text, chunk texts.
- **Vectors** (`vectors` table): float32 blobs keyed by `(file_id, ordinal,
  space)` where `space ∈ {text, image}`.

Search is exact cosine similarity via one numpy matmul over all vectors —
for the prototype's scale (tens of thousands of chunks) this is sub-10ms on
X Elite. The interface (`upsert`, `search`, `wipe`) is intentionally tiny so a
`sqlite-vec`/FAISS ANN backend can replace the matmul without touching callers.

Re-indexing is content-aware: `needs_index(path, mtime, size)` skips files
whose stat signature is unchanged, so a rescan of 10k files touches only what
changed.

## 5. Query & Ranking Layer (`search/query_engine.py`)

One query fans out to three retrievers, fused with fixed weights:

```
score = 0.60·semantic(Nomic cosine) + 0.25·visual(CLIP cosine) + 0.15·keyword(LIKE overlap)
      + up to 0.10 recency boost (files < 30 days old)
```

The keyword and recency components are what make queries like "invoice from
last month" work even when the embedding model hasn't loaded, and they boost
precision on exact terms (invoice numbers, project names). The result payload
carries each component score separately so the UI can *show* why a result
ranked where it did.

## 6. UI (`server/` + `ui/`)

FastAPI bound to **127.0.0.1 only**. The dashboard is built from the
**shadcn-admin** template (Vite + React + shadcn/ui + TanStack Router), adapted
in `ui/` with four screens:

- **Search** (`/`) — status badges (NPU active, indexing progress, paused),
  stat cards, hybrid results with thumbnails, snippets and per-component scores.
- **Folders** (`/folders`) — add/remove watched folders, quick-add for
  Downloads/Documents/Desktop/Pictures, re-index now.
- **Storage Health** (`/storage`) — totals, per-folder breakdown, largest
  files, stale and duplicate-prone groups, cleanup suggestions.
- **Privacy & Data** (`/settings`) — privacy status badges, pause/resume
  indexing, type-WIPE-to-confirm full local wipe. Appearance is inherited from
  the template (dark/light, fonts).

`vite build` emits static files to `ui/dist`; FastAPI serves them plus an SPA
fallback registered *after* every `/api` route. A dependency-free single-file
UI (`server/static/index.html`) remains as fallback when the bundle is absent.
No CDN, no Google Fonts, no analytics — the UI works with Wi-Fi off.

## Privacy posture (recap)

- Data home is a single folder (`data/`): DB + vectors + thumbnails + settings.
  **Wipe = delete the folder.** No hidden state elsewhere.
- Watchers only see folders the user explicitly added.
- No telemetry, no accounts, no analytics, no CDN. The UI works with Wi-Fi off.
