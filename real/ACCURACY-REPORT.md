# Offline vs Online Accuracy Evaluation — Local Memory (Snapdragon NPU)

**Date:** 2026-09-23 · **Server:** localhost:8792 · **Index:** 141 files, 9,828 chunks (Downloads folder)
**Method:** 69 representative Downloads files analyzed independently ("online-AI view", the `real/` answer sheet, 3 content-based queries each = 207 queries), fired at the fully offline `/api/search` (RRF hybrid, NPU QNN embeddings, TrOCR), rank of the correct file recorded.

## Results

| Metric | All 69 files | Indexable files only (39) |
|--------|-------------|---------------------------|
| Top-1 accuracy | 27.5% | **48.7%** |
| Top-3 accuracy | 31.9% | **56.4%** |
| Top-5 accuracy | 34.8% | **61.5%** |
| MRR | 0.320 | **0.567** |

The gap is a **coverage** finding, not a search-quality finding: 30 of 69 files are file types the pipeline never ingests (`.exe`, `.iso`, `.zip`, `.tar.gz`, `.msix`, `.parquet`, `.ps1`, `.ipynb`, `.sh`, `.xlsx`) — they are unfindable by any engine.

## What works well offline (beats filename search)

- CSV datasets: 7/8 found top-3 (content/column-based queries)
- Docs with distinctive content: assignments, rubrics, calendars, logs, receipts
- OCR'd images: WhatsApp photos found from visible content
- Explains itself: every result carries "via keyword/semantic/both" + highlighted snippet

## Where offline falls short of an online AI

1. **Binary file types not indexed at all** (30 files) — biggest gap; the brief promises "everything you create or download"
2. **Image-only PDFs**: scanned notes (2401331720013, NDA FORM, dhruv_mishra ID, Quiz) — embedded page images never reach TrOCR
3. **Small-image OCR vs long docs**: marksheet.jpeg ranked 12 — content queries lose to a competing Marksheet PDF
4. **Score formatting bug**: negative RRF+tiebreaker scores render as "−2% match" in UI
5. **Generic-template content**: conference-template-a4 unfindable (no distinctive content — correct behavior but expected by eval)

## E2E (browser) findings

- UI loads authenticated (invisible token), badge "Local · Private" present, "NPU: QNN active"
- Search renders ranked cards with match badges, `<mark>` highlights, thumbnails ✓
- Bug: Enter key does not submit the search (button-only)
- Bug: negative "−0% match … −2% match" percentages shown

## Verdict

For content-bearing files, offline NPU search ≈ 0.57 MRR — solid for a demo when queries are content-phrased, with clear before/after wins vs filename search. The two highest-leverage fixes before the competition demo: (1) ingest binaries at least by name/metadata (turns 30 unfindable files into findable), (2) route PDF-embedded page images through TrOCR.

## Post-Fix Re-Evaluation (same 69 files, 207 queries, fully offline)

Fixes applied: binary metadata ingestion (zip/tar entry listing, ipynb text), PDF page-render OCR fallback (PyMuPDF 150 DPI -> TrOCR), UI Enter-submit + match-% clamp.

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Files findable at all | 39/69 | **69/69** | +30 |
| Top-1 accuracy | 27.5% | **40.6%** | +13.1pp |
| Top-3 accuracy | 31.9% | **60.9%** | +29.0pp |
| Top-5 accuracy | 34.8% | **69.6%** | +34.8pp |
| MRR | 0.320 | **0.541** | +0.221 |

Now-found examples: Kali ISO, Burp Suite, VirtualBox, Windows.iso, ZCode installer (name-based); dhruv_mishra ID card, NDA FORM, Quiz (render-OCR rank 1); archive/pcc zips via entry listings. Remaining misses: content-based queries on pptx slides (metadata-only path) and 2-3 generic-content files (template, huge book ranked 10). TrOCR-small accuracy on dense rendered pages is a model-quality ceiling (handwriting/dense text is approximate), noted for the pitch.

Commits: 927aaeb, 59e5e01, 9511bde, b66107a, 876efe9 (suite: 94 passed, 2 skipped).
