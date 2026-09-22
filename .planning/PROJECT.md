# Local Memory

## What This Is

A fully private, on-device AI file indexer and natural-language search application for Snapdragon-powered HP (Windows on ARM) PCs, built for the **Snapdragon® AI Lab Build & Present Challenge** (Unstop competition).

Local Memory automatically indexes user-selected folders, understands both text and image content using Qualcomm AI Hub models running on the Hexagon NPU, and lets users search and organise files through natural language — like a personal memory feed. Zero data leaves the PC.

**Core Value:** A user can type what they remember ("pen with blue book", "invoice from last month", "screenshot of that error") and get ranked, previewed results — even when filenames are useless — with proof that everything stayed on-device.

## Context

- **Competition:** Snapdragon AI Lab Build & Present Challenge (Qualcomm / Unstop) — judged on Technical Implementation, Application Use Case & Innovation, Deployment & Accessibility, Presentation & Documentation.
- **Audience:** Students and young professionals in India who cannot/will not upload personal documents to cloud AI. Judges value NPU utilisation, offline privacy, and a clean demo.
- **Deadline-driven:** This is a competition submission — demo-readiness and polish beat exhaustive features.
- **Current state:** The codebase is substantially built and working (see `.planning/codebase/`). A 12-slide pitch deck and detailed project brief already exist (the brief describes exactly this codebase).
- **Known debt:** `.planning/codebase/CONCERNS.md` documents 20 concerns — most importantly an unauthenticated destructive localhost API (arbitrary-file `/api/thumbnail`, CSRF/DNS-rebinding exposure), a dead TrOCR branch that returns empty OCR, semantically broken CLIP text tokenization, orphaned vectors/thumbnails on file deletion, and startup pip-install duplication.

## Existing Capabilities (validated by codebase map)

- ✓ Background indexing pipeline: file watcher → extraction (PDF/DOCX/text) → embedding (`local_memory/pipeline.py`)
- ✓ Nomic-Embed-Text text embeddings on Snapdragon NPU via ONNX Runtime + QNN EP (QNN → DirectML → CPU fallback) (`local_memory/embeddings/`)
- ✓ CLIP image embeddings for visual search (`local_memory/embeddings/clip_image.py`)
- ✓ SQLite + WAL local vector store with thumbnails (`local_memory/store/database.py`)
- ✓ FastAPI server on localhost:8787 with search/rank/thumbnail APIs (`local_memory/server/app.py`)
- ✓ React 19 + Vite + shadcn-admin UI with search, results, storage views (`ui/`)
- ✓ Image understanding (caption/OCR extraction path) (`local_memory/extractors/`)
- ✓ Smoke + performance test suite runnable model-free via deterministic fallbacks (`tests/`)
- ✓ Smart ranking by relevance + recency (`local_memory/` query layer)

## Requirements

### Validated

- ✓ Automatic background indexing of selected folders — existing
- ✓ NPU-accelerated text embeddings (Nomic-Embed-Text, QNN EP) — existing
- ✓ Local vector store, fully on-device — existing
- ✓ Natural-language semantic search with ranked results — existing
- ✓ React UI with search, previews, storage views — existing

### Active

- [ ] Fix the dead TrOCR/OCR branch so OCR genuinely contributes indexed text (or remove it cleanly and rely on CLIP + captions)
- [ ] Fix CLIP text tokenization so text-side queries work correctly for image search
- [ ] Secure the local API: require a local auth token, restrict `/api/thumbnail` to indexed files, add CSRF/DNS-rebinding protections
- [ ] Clean up index/thumbnails when files are deleted (no orphaned rows)
- [ ] Remove auto pip-install on startup; validate dependencies deterministically instead
- [ ] One-command setup/demo path: fresh machine → models present → indexing → search working (installer/launcher)
- [ ] Demo kit: curated cluttered test folder, scripted demo queries matching the brief ("pen with blue book", "invoices from March", "screenshots of that error")
- [ ] Storage health + one-click cleanup suggestions surfaced in UI (per brief §4.1)
- [ ] Zero-network privacy proof for the demo (verifiable offline operation)

### Out of Scope

- Cloud sync or any network AI calls — the entire value proposition is on-device privacy
- Hindi/multilingual query support — brief lists it as enhanced/future; defer past demo
- Small LLM query rewriting (Qwen3/Phi via GenieX) — optional in brief; only if time allows
- Packaging as official Qualcomm AI Hub sample app — post-challenge future scope
- Memory Timeline (Instagram-style feed grouping) — enhanced feature; defer past demo unless trivial

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use existing codebase as-is; harden and polish rather than rewrite | Working end-to-end pipeline already matches the brief; competition rewards a finished demo | — Pending |
| QNN → DirectML → CPU fallback chain stays | Demo must run even if NPU driver hiccups | — Pending |
| Security fixes target localhost-only threat model | Single-user demo device; no need for full multi-user auth | — Pending |
| Deterministic model fallbacks kept for tests/CI | Test suite must run model-free on any machine | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-09-22 after initialization*
