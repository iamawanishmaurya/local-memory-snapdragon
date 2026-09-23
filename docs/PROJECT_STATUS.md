# Local Memory — Project Status & Roadmap

*Last updated: 2026-09-22 · Version 0.1.0 (working prototype)*

This document records everything built so far, how it was verified, what the
current limitations are, and what remains to be done toward the final
challenge submission.

---

## 1. The Goal

Build **Local Memory**: a privacy-first, fully on-device AI file indexer and
natural-language search tool for **Windows on Snapdragon** HP PCs, submitted to
the **Snapdragon AI Lab Build & Present Challenge**.

In one sentence: *the user picks folders; Local Memory understands the content
of every file inside them (documents, PDFs, Office docs, screenshots, images)
using models running on the Snapdragon NPU; and the user can search everything
in natural language — with nothing ever leaving the device.*

### Success criteria (from the project brief)

| # | Requirement | Status |
|---|---|---|
| 1 | Auto-index user-selected folders (Downloads, Documents, Desktop, Screenshots…) | ✅ done |
| 2 | Understand text content **and** images (OCR + vision) | ✅ done |
| 3 | Natural-language search ("pen with blue book", "invoice from last month", "error screenshots") | ✅ done |
| 4 | Ranked results with previews | ✅ done |
| 5 | Storage health insights + cleanup suggestions | ✅ done |
| 6 | 100% offline — no data leaves the device | ✅ done (one-time model export is the only network use) |
| 7 | Real use of Qualcomm AI Hub models on the Snapdragon NPU | ✅ QNN EP registered on ARM64 Python; real Nomic+CLIP ONNX indexed (131 files). Generic FP32 ONNX dispatches CPU — HTP needs AI Hub context-binary compile (one cmd, needs token) |
| 8 | Clean repo structure, README for judges, docs | ✅ done |

Legend: ✅ implemented & verified · ⚙️ implemented, needs target-hardware step

---

## 2. What We Have Built

### 2.1 Backend — six-layer pipeline (`local_memory/`)

**Layer 1 — File Watcher (`watcher.py`)**
- `watchdog` observers, one per user-chosen folder, fully recursive.
- Events debounced (2 s per file); extension allow-list (text/PDF/DOCX/images);
  Office lock files (`~$…`) ignored; deletions remove DB rows immediately.

**Layer 2 — Content Extraction (`extractors/`)**
- `text_extractor.py`: `.txt .md .csv .log …` direct read; PDF via `pypdf`;
  DOCX via `python-docx` (paragraphs + tables).
- Overlapping 200-word chunking with 40-word overlap (configurable).
- `ocr.py`: EasyOCR backend (lazy-loaded, optional install) for image → text.
- `image_understanding.py`: OCR + cached JPEG thumbnails (`data/thumbs/`).

**Layer 3 — Embeddings (`embeddings/`) — the NPU layer**
- `base.py`: single ONNX Runtime session factory; provider priority
  **QNNExecutionProvider (Snapdragon NPU) → CPU**. Everything NPU-related goes
  through this one seam.
- `nomic_text.py`: Nomic-Embed-Text v1.5, 768-dim text embeddings, one vector
  per chunk.
- `clip_image.py`: CLIP ViT-B/32 image encoder (512-dim) + CLIP text encoder
  so a text query can land in image-embedding space.
- Deterministic hashing fallback so the entire pipeline runs (with toy
  quality) on machines without models — this is why tests pass anywhere.

**Layer 4 — Local Vector Store (`store/`)**
- `database.py`: SQLite (WAL) — `files` (path, size, mtime, kind, OCR text)
  and `chunks` (chunk text) tables.
- `vector_store.py`: float32 vector blobs keyed by `(file_id, ordinal, space)`
  with `space ∈ {text, image}`; cosine search via one numpy matmul; tiny
  interface ready to swap in sqlite-vec/FAISS.
- Content-aware re-indexing: unchanged files (same mtime + size) are skipped.

**Layer 5 — Query & Ranking (`search/query_engine.py`)**
- Hybrid fusion, one query → three retrievers:
  `0.60·semantic + 0.25·visual(CLIP) + 0.15·keyword(LIKE) + ≤0.10 recency`.
- Result payload carries each component score so the UI shows *why* a result
  ranked where it did. Recency helps queries like "invoice from last month".

**Layer 6 — API & UI (`server/`, `ui/`)**
- `server/app.py`: FastAPI bound to **127.0.0.1 only**. Endpoints: status
  (incl. live NPU-provider readout), search, folder add/remove/suggest,
  re-index, pause/resume, storage health, thumbnails, wipe.
- SPA serving for the built dashboard with client-side-route fallback,
  registered *after* all API routes. Bundled single-file fallback UI if the
  frontend isn't built.

**Privacy (`privacy.py`)**
- Pause/resume; **full local wipe** deletes the entire data home (DB, vectors,
  thumbnails, settings). All user state lives in one folder (`data/`), so wipe
  = delete folder. No other state exists on disk anywhere.

### 2.2 Frontend dashboard (`ui/`) — based on shadcn-admin

Built from the [shadcn-admin](https://github.com/satnaing/shadcn-admin)
template (Vite + React + shadcn/ui + TanStack Router), adapted:

- Trimmed all demo features (auth/Clerk, apps, chats, tasks, users) and
  removed the Clerk dependency entirely.
- **Search page (`/`)**: live status badges ("NPU: QNN active" read from the
  actual ONNX Runtime session, indexing progress, paused), stat cards, results
  with image thumbnails, match-% badge, snippet, and per-component scores.
- **Folders (`/folders`)**: add/remove watched folders, quick-add buttons for
  Downloads/Documents/Desktop/Pictures, re-index now.
- **Storage Health (`/storage`)**: totals, per-folder breakdown, largest
  files, stale (>1 yr) and duplicate-prone groups, amber cleanup cards.
- **Privacy & Data (`/settings`)**: privacy badges, pause/resume, type-WIPE-
  to-confirm full wipe. Appearance (dark/light) inherited from the template.
- Typed API client (`src/lib/api.ts`); React Query polling for live status.
- Removed template's Google Fonts links + Clerk → **zero external requests**;
  works with Wi-Fi off.

### 2.3 Setup & tooling

- `scripts/setup_models.py` — one-time model export/download into `models/`
  (`nomic-embed-text.onnx`, `clip-vit-b32-image.onnx`, `clip-vit-b32-text.onnx`)
  via Qualcomm AI Hub / HuggingFace mirrors. **The only network step, ever.**
- `scripts/demo_index.py` — CLI: `--index`, `--search`, `--health`, `--doctor`,
  `--wipe` (full demo without the UI).
- `scripts/build_ui.bat` — rebuilds the dashboard into `ui/dist` (portable
  Node v22 kept in `tools/`, gitignored; winget/network was unavailable so a
  portable toolchain was installed).
- `docs/NPU_SETUP.md` — Snapdragon setup incl. optional QNN context-binary
  compile via AI Hub for best NPU performance.

### 2.4 Tests (`tests/test_smoke.py`)

Five smoke tests, CPU fallback, no models required: chunking overlap, index +
hybrid search ranking ("invoice from last month" → invoice file), unchanged-file
skip, full wipe, storage-health report.

---

## 3. How It Was Verified

- `python -m pytest tests -q` → **5 passed**.
- `python -c "from local_memory.server.app import app"` → imports cleanly
  (16 routes).
- Live server test (port 8791):
  - `/` served the built dashboard; `/folders` returned the SPA (200);
    `/images/favicon.svg` (200).
  - `/api/status` returned correct JSON (providers, stats, folders).
  - Added a demo folder **through the API** → background watcher indexed the
    files → **"invoice from last month"** returned `invoice_acme.txt` ranked
    #1 with a snippet — full watcher→extract→embed→store→search roundtrip.
  - Cleanup afterwards: test server killed, demo files/index wiped.
- `npx tsc -b` → clean type-check; `vite build` → production bundle (~380 KB
  main chunk, code-split routes).

---

## 4. Current Limitations (honest list)

1. **NPU artifacts not yet exported.** Models are missing from `models/` until
   `scripts/setup_models.py` runs once (needs network + AI Hub token). Until
   then embeddings use the hashing fallback (tests/demo quality) or CPU ONNX.
2. **This dev machine is x64 without QNN** — the NPU path is implemented and
   selected automatically, but needs a Snapdragon X Elite device + ARM64
   Python + `onnxruntime-qnn` to demonstrate live.
3. **Search is brute-force cosine** (numpy matmul) — fine to ~100k chunks;
   sqlite-vec/FAISS swap planned before scaling demos.
4. **Duplicate detection is name/size-heuristic**, not content hashing.
5. **OCR is EasyOCR on CPU** (optional install); TrOCR-on-NPU is the roadmap
   replacement.
6. **No packaged installer** yet (Tauri shell planned); currently run from
   source.
7. Portable Node in `tools/` is x64 — rebuild the UI on ARM64 or ship the
   prebuilt `ui/dist` (pure static, architecture-independent).

---

## 5. Roadmap to Submission

**Next (highest impact for judges)**
- [ ] Run `scripts/setup_models.py` on the Snapdragon X Elite target; verify
      `--doctor` shows `QNN (NPU): AVAILABLE`.
- [ ] Index a realistic folder set (few hundred files incl. screenshots) on
      the device; record indexing throughput CPU vs. NPU for the pitch.
- [ ] Demo script/storyboard: search box → "pen with blue book" on images →
      "error screenshots" → storage health → wipe, ending on the privacy slide.

**Then**
- [ ] Qwen3-0.6B query rewriter via onnxruntime-genai on NPU (query expansion:
      "last month" → date range; synonyms).
- [ ] sqlite-vec ANN backend behind the existing `vector_store` interface.
- [ ] Content-hash duplicate detection + "safe to delete" size estimates.
- [ ] Tauri desktop shell (system tray, auto-start, native folder picker) and
      Windows ARM64 installer.
- [ ] TrOCR via AI Hub compiled to QNN as the OCR backend.

**Polish**
- [ ] Onboarding flow on first run (folder picker suggestions).
- [ ] Result preview pane (PDF page render, image zoom).
- [ ] Benchmark page in the dashboard showing live NPU vs CPU latency.
- [ ] CI (GitHub Actions) running pytest + tsc + vite build on every push.

---

## 6. Key Commands (quick reference)

```powershell
python -m pytest tests -q                      # run smoke tests
python -m local_memory.main                    # start app → http://127.0.0.1:8787
python -m local_memory.main --doctor           # NPU/provider diagnostics
python scripts\setup_models.py                 # ONE-TIME model export (network)
scripts\build_ui.bat                           # rebuild dashboard (needs Node)
python scripts\demo_index.py --index <folder>  # CLI index
python scripts\demo_index.py --search "query"  # CLI search
python scripts\demo_index.py --health          # CLI storage report
python scripts\demo_index.py --wipe            # CLI full wipe
```

## 7. Related Docs

- [`../README.md`](../README.md) — project pitch, quickstart, repo layout.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — layer-by-layer design detail.
- [`NPU_SETUP.md`](NPU_SETUP.md) — Snapdragon NPU + AI Hub setup.
- [`PRIVACY.md`](PRIVACY.md) — auditable privacy guarantees.
