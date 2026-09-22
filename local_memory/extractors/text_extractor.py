"""Content extraction for text-like documents: txt/md/csv/log, PDF, DOCX,
.ipynb notebooks — plus metadata-only chunks for binary files.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from .. import config

log = logging.getLogger(__name__)

_KIND_DESCRIPTION = {
    "archive": "archive",
    "executable": "executable installer or application",
    "disk-image": "disk image",
    "data": "data file",
}


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _has_real_text(text: str) -> bool:
    """True when extracted text looks meaningful.

    Some image-only PDFs (print-to-PDF output) still yield a few junk glyphs
    per page from pypdf (':' , digits, boxes) — enough to defeat a plain
    `text.strip()` check, but far below real prose. Require a minimum count
    of alphanumeric characters before trusting the text layer.
    """
    return sum(ch.isalnum() for ch in text) >= 20


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        parts: list[str] = []
        size_ok = path.stat().st_size <= config.PDF_OCR_MAX_BYTES
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if not size_ok or _has_real_text(text):
                parts.append(text)
                continue
            # Image-only page (scanned notes): route the embedded page image
            # through the OCR backend. Capped at the first PDF_OCR_MAX_PAGES
            # text-free pages so a 300-page scan doesn't stall indexing.
            if i < config.PDF_OCR_MAX_PAGES:
                ocr_text = _ocr_pdf_page(page)
                if not _has_real_text(ocr_text):
                    # Embedded-image OCR gave nothing usable — fall back to
                    # rendering the page: print-to-PDF files embed no
                    # decodable page images (or a tiny junk one).
                    ocr_text = _ocr_pdf_page_render(path, i)
                if _has_real_text(ocr_text):
                    text = ocr_text
            parts.append(text)
        return "\n".join(parts)
    except Exception:
        return ""


def _ocr_pdf_page(page) -> str:
    """OCR the largest image embedded in a pypdf page. '' on any failure."""
    try:
        from . import ocr
        if not ocr.ocr_available():
            return ""
        images = list(page.images)  # guarded: pypdf raises on corrupt XObjects
        if not images:
            return ""
        best = max(images, key=lambda im: len(im.data))
        pil = best.image  # PIL image decoded by pypdf
        return ocr.ocr_pil(pil)
    except Exception:
        log.debug("pdf page image OCR failed", exc_info=True)
        return ""


def _ocr_pdf_page_render(path: Path, page_index: int) -> str:
    """OCR a rasterized rendering of a PDF page (PyMuPDF, modest DPI).

    Fallback for image-only PDFs whose embedded images pypdf cannot decode
    (e.g. Windows print-to-PDF output). The OCR backends are line-level
    models — a whole page squeezed to their input size yields garbage — so
    the render is split into horizontal text bands (dark-pixel row runs)
    and each band is OCR'd separately. '' on any failure; pymupdf itself is
    an optional dependency so a missing wheel degrades silently.
    """
    try:
        import io

        try:
            import pymupdf as fitz  # PyMuPDF — lazy/optional import
        except ImportError:  # older wheels only expose the legacy name
            import fitz
        import numpy as np
        from PIL import Image

        from . import ocr
        if not ocr.ocr_available():
            return ""
        with fitz.open(str(path)) as doc:
            page = doc.load_page(page_index)
            png = page.get_pixmap(dpi=150).tobytes("png")
        pil = Image.open(io.BytesIO(png))
        gray = np.asarray(pil.convert("L"))
        dark_rows = (gray < 128).sum(axis=1)
        # Text-line bands = runs of rows containing dark pixels, each padded,
        # so the OCR model sees lines at near-native resolution.
        bands: list[tuple[int, int]] = []
        start = None
        for y, count in enumerate(dark_rows):
            if count > 5 and start is None:
                start = y
            elif count <= 5 and start is not None:
                if y - start >= 10:
                    bands.append((start, y))
                start = None
        if start is not None and len(dark_rows) - start >= 10:
            bands.append((start, len(dark_rows)))
        if not bands:  # blank page or odd render — try the full page once
            return ocr.ocr_pil(pil)
        w, h = pil.size
        texts: list[str] = []
        for y0, y1 in bands:
            crop = pil.crop((0, max(0, y0 - 5), w, min(h, y1 + 5)))
            text = ocr.ocr_pil(crop)
            if text:
                texts.append(text)
        return "\n".join(texts)
    except ImportError:
        log.debug("pymupdf not installed — render-based PDF OCR skipped")
        return ""
    except Exception:
        log.debug("pdf page render OCR failed for %s page %d",
                  path, page_index, exc_info=True)
        return ""


def _read_ipynb(path: Path) -> str:
    """Markdown headers + first lines of each cell — enough to find a notebook
    by topic without embedding cell outputs (which can be huge/binary)."""
    try:
        import json
        nb = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return ""
    lines: list[str] = []
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        if isinstance(src, str):
            src = src.splitlines()
        text = "\n".join(src).strip()
        if not text:
            continue
        if cell.get("cell_type") == "markdown":
            lines.append(text)
        else:
            lines.extend(text.splitlines()[:3])  # first lines of code cells
        if sum(len(x) for x in lines) > config.IPYNB_MAX_CHARS:
            break
    return "\n".join(lines)[: config.IPYNB_MAX_CHARS]


def _read_docx(path: Path) -> str:
    try:
        import docx  # python-docx
        d = docx.Document(str(path))
        parts = [p.text for p in d.paragraphs]
        for table in d.tables:
            for row in table.rows:
                parts.append(" | ".join(c.text for c in row.cells))
        return "\n".join(parts)
    except Exception:
        return ""


def extract_text(path: Path) -> tuple[str, str]:
    """Return (text, kind). Empty text means nothing usable extracted."""
    ext = path.suffix.lower()
    if ext in config.TEXT_EXTS:
        return _read_text_file(path), "text"
    if ext in config.PDF_EXTS:
        return _read_pdf(path), "pdf"
    if ext in config.DOCX_EXTS:
        return _read_docx(path), "docx"
    if ext in config.IPYNB_EXTS:
        return _read_ipynb(path), "notebook"
    return "", "other"


# ----------------------------------------------------------- binary metadata --

def _name_tokens(path: Path) -> list[str]:
    """Split a filename stem (and compound ext) into search-friendly tokens.

    'kali-linux-2026.2-installer-amd64.iso' -> ['kali','linux','2026.2',
    'installer','amd64','iso']
    """
    raw = path.name.lower()
    exts = ""
    for suffix in sorted(config.BINARY_COMPOUND, key=len, reverse=True):
        if raw.endswith(suffix):
            exts = suffix
            raw = raw[: -len(suffix)]
            break
    else:
        exts = path.suffix.lower()
        raw = path.stem.lower()
    tokens = [t for t in re.split(r"[-_\s]+|(?<=\D)\.|\.(?=\D)", raw) if t]
    if exts.startswith("."):
        exts = exts[1:]
    if exts:
        for part in exts.split("."):
            if part:
                tokens.append(part)
    return tokens


def _archive_entries(path: Path) -> list[str]:
    """Top-level entries of a zip/tar archive (capped). Never raises."""
    import zipfile
    import tarfile
    try:
        if path.stat().st_size > config.ARCHIVE_INSPECT_MAX_BYTES:
            return []
        name = path.name.lower()
        if zipfile.is_zipfile(str(path)):
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
        elif name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
            with tarfile.open(path, "r:*") as tf:
                names = tf.getnames()
        else:
            return []
        return [n for n in names[: config.ARCHIVE_ENTRY_LIMIT]]
    except Exception:
        log.debug("archive entry listing failed for %s", path, exc_info=True)
        return []


def metadata_chunks(path: Path) -> tuple[str, list[str]]:
    """Synthesize text chunks for a binary file from name/metadata.

    Returns (kind, chunks). Chunk 1: raw tokens (exact FTS matches).
    Chunk 2: a natural-language phrase ("Kali Linux 2026.2 installer amd64 ISO
    disk image") so Nomic embeddings match natural queries, plus top-level
    archive entries when the file is a small zip/tar.
    """
    kind = config.binary_kind(str(path)) or "binary"
    tokens = _name_tokens(path)
    chunks: list[str] = []
    if tokens:
        chunks.append(f"{path.name} " + " ".join(tokens))
        phrase = " ".join(t if any(c.isdigit() for c in t) else t.capitalize()
                          for t in tokens[:-1])
        desc = _KIND_DESCRIPTION.get(kind, kind)
        ext_word = tokens[-1].upper()
        natural = f"{phrase} {ext_word} {desc}".strip()
        entries = _archive_entries(path) if kind == "archive" else []
        if entries:
            natural += ". Contents: " + " ".join(
                e.rsplit("/", 1)[-1].replace("-", " ").replace("_", " ")
                for e in entries)
        chunks.append(natural)
    return kind, chunks


def chunk_words(text: str) -> list[str]:
    """Split into overlapping word-window chunks (see config.CHUNK_WORDS)."""
    words = text.split()
    if not words:
        return []
    step = config.CHUNK_WORDS - config.CHUNK_OVERLAP_WORDS
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + config.CHUNK_WORDS])
        if len(chunk.split()) >= config.CHUNK_OVERLAP_WORDS or i == 0:
            chunks.append(chunk)
        if i + config.CHUNK_WORDS >= len(words):
            break
    return chunks or [" ".join(words)]
