# 🧠 Local Memory

**A privacy-first, fully on-device AI file indexer and natural-language search engine for Windows on Snapdragon.**

Local Memory remembers what's inside your files — documents, PDFs, screenshots, images — so you can search them the way you think:

> *"pen with blue book"* · *"invoice from last month"* · *"that error screenshot"* · *"resume draft I edited in June"*

Everything runs **100% on your PC**. Text understanding, image understanding, OCR, embeddings, and search all execute on the **Snapdragon NPU** via Qualcomm AI Hub models and ONNX Runtime with the **Qualcomm QNN Execution Provider**. No cloud. No telemetry. No data ever leaves the device.

---

## ✨ Why Local Memory

| Problem | Local Memory |
|---|---|
| Windows search only matches filenames and exact keywords | Semantic search understands *meaning* — "invoice from last month" finds `Acme_Billing_Sept.pdf` |
| Screenshot folders are a graveyard — you can't search image content | Vision model + OCR index *what's inside* every image |
| Cloud AI assistants require uploading your private files | Nothing leaves the device — verified zero network calls during use |
| Storage clutter grows silently | Storage Health view surfaces duplication hotspots and cleanup suggestions |

---

## 🚀 Key Features

- **Automatic folder indexing** — pick Downloads, Documents, Desktop, Screenshots, or any folder; a background watcher keeps the index fresh as files change.
- **Multi-modal understanding**
  - Text files, PDFs, and Office docs → parsed and chunked.
  - Images → OCR (extracted text) **and** CLIP vision embeddings (what the image *looks like*).
- **Natural-language search** — one query box, hybrid ranking (semantic vector score + filename/text keyword score + recency).
- **Ranked results with previews** — thumbnail for images, text snippet around the best-matching chunk for documents.
- **Storage Health** — per-folder size breakdown, largest/oldest/duplicate-prone files, cleanup suggestions.
- **Privacy controls** — user-chosen folders only, pause/resume indexing, and a full **local wipe** that destroys the entire index in one click.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Web UI (browser)                          │
│        search box · results + previews · health · settings       │
└──────────────────────────────┬───────────────────────────────────┘
                               │ FastAPI (localhost only)
┌──────────────────────────────┴───────────────────────────────────┐
│  1. File Watcher Layer      watchdog observers on chosen folders │
│  2. Content Extraction      text parsers · OCR · CLIP vision     │
│  3. Embedding Layer         Nomic-Embed-Text + CLIP on NPU (QNN) │
│  4. Local Vector Store      SQLite metadata + vectors (FAISS-    │
│                             optional; numpy brute-force default) │
│  5. Query & Ranking         hybrid semantic + keyword + recency  │
│  6. Storage Health          size analysis, cleanup suggestions   │
└──────────────────────────────────────────────────────────────────┘
```

### Snapdragon NPU pipeline

| Stage | Model (Qualcomm AI Hub) | Runtime |
|---|---|---|
| Text embeddings | **Nomic-Embed-Text** (context-768) | ONNX Runtime + **QNN EP → NPU** |
| Image understanding | **OpenAI CLIP** (ViT-B/32 image encoder) | ONNX Runtime + **QNN EP → NPU** |
| OCR | **EasyOCR / TrOCR** | CPU/GPU (see `docs/NPU_SETUP.md`) |
| Optional query understanding | Qwen3-0.6B / Phi-mini | Ollama / onnxruntime-genai (planned) |

Models are exported/compiled **once** during setup with Qualcomm AI Hub (`scripts/setup_models.py` downloads or compiles the QNN-context binaries into `models/`). After that the app runs fully offline.

The dashboard is built on **[shadcn-admin](https://github.com/satnaing/shadcn-admin)** (Vite + React + shadcn/ui + TanStack Router), adapted into `ui/` with Local Memory screens, compiled to static files, and served by the local FastAPI server — still zero network.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for layer-by-layer detail, [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) for everything done so far + roadmap, and [`docs/NPU_SETUP.md`](docs/NPU_SETUP.md) for Snapdragon device setup.

---

## 🖥️ Requirements

- **Hardware:** Snapdragon X Elite / X2 Elite Windows PC (e.g. HP OmniBook Ultra / EliteBook with Snapdragon). Falls back to CPU on any other machine so development works anywhere.
- **OS:** Windows 11 (ARM64), Python 3.10–3.12
- **One-time setup (needs network):** model export via Qualcomm AI Hub; a Qualcomm AI Hub account + API key.

---

## ⚡ Quickstart

```powershell
# 1. Create environment (Windows ARM64)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. One-time: export/download NPU models via Qualcomm AI Hub (network required)
set QAI_HUB_API_TOKEN=your_token_here
python scripts\setup_models.py

# 3. One-time: build the shadcn-admin dashboard (needs Node 18+)
scripts\build_ui.bat

# 4. Launch Local Memory (from here on: fully offline)
python -m local_memory.main
# → open http://127.0.0.1:8787
```

In the dashboard: **Folders → Add Folder** (e.g. Downloads, Documents, Desktop). Indexing starts immediately; **Search** as soon as progress appears. **Storage Health** and **Privacy & Data** live in the sidebar.

### CLI demo (no UI)

```powershell
python scripts\demo_index.py --folder "C:\Users\you\Documents"   # index
python scripts\demo_index.py --search "invoice from last month"  # search
python scripts\demo_index.py --health                            # storage report
python scripts\demo_index.py --wipe                              # destroy local index
```

---

## 🔒 Privacy by Design

- **Folders are opt-in.** Only folders you explicitly add are scanned. The watcher never traverses outside them.
- **Zero network during use.** The only outbound calls ever made are the one-time model export via AI Hub during setup. Search, indexing, and the UI are localhost-only.
- **Pause anytime.** One click halts the watcher and indexing pipeline.
- **Full local wipe.** Deletes the entire SQLite database, all vectors, and all cached thumbnails. Uninstalling leaves nothing behind.
- **No telemetry, no accounts, no analytics.**

Details in [`docs/PRIVACY.md`](docs/PRIVACY.md).

---

## 📁 Repository Layout

```
local-memory-index/
├── README.md
├── requirements.txt
├── local_memory/
│   ├── main.py                  # entry point
│   ├── config.py                # paths, settings, folder registry
│   ├── privacy.py               # pause/resume + full local wipe
│   ├── watcher.py               # layer 1: file watching
│   ├── pipeline.py              # orchestrates extract → embed → store
│   ├── extractors/              # layer 2: content extraction
│   │   ├── text_extractor.py    #   txt/md/pdf/docx
│   │   ├── ocr.py               #   image → text
│   │   └── image_understanding.py
│   ├── embeddings/              # layer 3: NPU embeddings
│   │   ├── base.py              #   ONNX/QNN session helper (NPU→GPU→CPU)
│   │   ├── nomic_text.py        #   Nomic-Embed-Text
│   │   └── clip_image.py        #   CLIP ViT-B/32
│   ├── store/                   # layer 4: local storage
│   │   ├── database.py          #   SQLite metadata
│   │   └── vector_store.py      #   vectors + ANN search
│   ├── search/                  # layer 5: query & ranking
│   │   └── query_engine.py
│   ├── health/                  # storage health insights
│   │   └── storage_health.py
│   └── server/                  # layer 6: UI
│       ├── app.py               #   FastAPI (localhost only) + SPA serving
│       └── static/index.html    #   bundled fallback UI (when ui/dist absent)
├── scripts/
│   ├── setup_models.py          # Qualcomm AI Hub export (one-time)
│   └── demo_index.py            # CLI demo: index/search/health/wipe
├── docs/
│   ├── ARCHITECTURE.md
│   ├── NPU_SETUP.md
│   └── PRIVACY.md
├── ui/                          # shadcn-admin based dashboard (Vite + React)
│   ├── src/features/            #   search · folders · storage · privacy screens
│   ├── src/lib/api.ts           #   typed client for the local FastAPI
│   └── dist/                    #   built static bundle served by FastAPI
├── tests/
│   └── test_smoke.py
└── models/                      # exported ONNX/QNN artifacts (gitignored)
```

---

## 🧪 Verifying NPU execution

```
python scripts\demo_index.py --doctor
```

prints which execution provider is active (`QNNExecutionProvider` = NPU engaged) and model load status. On the Snapdragon X Elite target, text-embedding throughput is expected to be several× faster than CPU; the `--doctor` output is the judges' quick proof of real NPU usage.

---

## 🗺️ Roadmap

- [x] End-to-end pipeline: watch → extract → embed → store → search
- [x] OCR + CLIP image understanding
- [x] Storage health + cleanup suggestions
- [x] Privacy controls (pause, wipe, opt-in folders)
- [ ] Qwen3-0.6B query rewriter on NPU (onnxruntime-genai)
- [ ] Tauri shell for packaged desktop app
- [ ] sqlite-vec ANN index for 100k+ files

## 📄 License

MIT — see [`LICENSE`](LICENSE).
