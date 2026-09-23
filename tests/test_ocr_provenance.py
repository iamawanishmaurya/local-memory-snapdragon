"""OCR provenance chain (Phase 4 fix): a PDF whose text comes from the
render/embedded-image OCR fallback must be counted as an OCR file in stats.

Model-free: the OCR backend is faked, and the PDF is a blank pypdf page
(no text layer), which forces the OCR path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("LOCAL_MEMORY_HOME", str(home))
    import importlib

    from local_memory.health import storage_health
    from local_memory.store import database
    importlib.reload(database)
    importlib.reload(storage_health)
    database.init_db()
    return database, storage_health


def _blank_pdf(path: Path) -> None:
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(width=612, height=792)
    with open(path, "wb") as f:
        w.write(f)


def test_pdf_ocr_provenance_counts_in_stats(tmp_path, monkeypatch, isolated):
    database, storage_health = isolated
    from local_memory.extractors import text_extractor
    from local_memory.extractors import ocr as ocr_module

    monkeypatch.setattr(ocr_module, "ocr_available", lambda: True)
    monkeypatch.setattr(ocr_module, "ocr_pil", lambda img: "INVOICE 2026-03-14 planted text for stats")
    monkeypatch.setattr(text_extractor, "_ocr_pdf_page_render", lambda p, i: "RENDER OCR text for stats test")

    pdf = tmp_path / "scanned.pdf"
    _blank_pdf(pdf)
    text, kind = text_extractor.extract_text(pdf)
    # A blank page has no embedded image, so the RENDER fallback fires.
    assert "RENDER OCR text" in text, "faked OCR text must flow into extracted text"
    assert text_extractor.ocr_was_used(pdf), "OCR provenance must be recorded"

    watched = tmp_path / "watched"
    watched.mkdir()
    real = watched / "scanned.pdf"
    real.write_bytes(pdf.read_bytes())
    file_id = database.upsert_file(str(real), str(watched), ".pdf",
                                   real.stat().st_size, 0.0, "pdf", ocr_used=True)
    database.replace_chunks(file_id, list(enumerate(text_extractor.chunk_words(text))))

    from local_memory import stats as stats_module
    idx = stats_module.index_stats()
    assert idx["ocr_files"] == 1, f"expected 1 OCR file, got {idx['ocr_files']}"
