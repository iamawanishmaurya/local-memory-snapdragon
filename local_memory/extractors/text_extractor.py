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
    "media": "audio or video recording",
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
                    _pdf_ocr_used[str(path)] = True
            parts.append(text)
        return "\n".join(parts)
    except Exception:
        return ""


# OCR provenance (DEMO-05 stats): per-path record of PDFs whose text came
# from the render/embedded-image OCR fallback. Keyed by path so parallel
# extraction workers never cross-contaminate.
_pdf_ocr_used: dict[str, bool] = {}


def ocr_was_used(path: str | Path) -> bool:
    return bool(_pdf_ocr_used.get(str(path)))


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
    the render is split into horizontal text bands via the shared
    `ocr.ocr_image_banded` helper. '' on any failure; pymupdf itself is an
    optional dependency so a missing wheel degrades silently.
    """
    try:
        import io

        try:
            import pymupdf as fitz  # PyMuPDF — lazy/optional import
        except ImportError:  # older wheels only expose the legacy name
            import fitz
        from PIL import Image

        from . import ocr
        if not ocr.ocr_available():
            return ""
        with fitz.open(str(path)) as doc:
            page = doc.load_page(page_index)
            png = page.get_pixmap(dpi=150).tobytes("png")
        pil = Image.open(io.BytesIO(png))
        return ocr.ocr_image_banded(pil)
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


def _read_pptx(path: Path) -> str:
    """Slide text in order (D-02): text frames + table cells + notes,
    each slide prefixed 'Slide N:'. Capped; '' on any failure."""
    try:
        from pptx import Presentation
        prs = Presentation(str(path))
        parts: list[str] = []
        for i, slide in enumerate(prs.slides, start=1):
            if i > config.PPTX_MAX_SLIDES:
                break
            body: list[str] = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    text = shape.text.strip()
                    if text:
                        body.append(text)
                if shape.has_table:
                    for row in shape.table.rows:
                        cells = [c.text.strip() for c in row.cells]
                        row_text = " | ".join(c for c in cells if c)
                        if row_text:
                            body.append(row_text)
            if getattr(slide, "has_notes_slide", False):
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    body.append(notes)
            if body:
                parts.append(f"Slide {i}: " + "\n".join(body))
            if sum(len(x) for x in parts) > config.OFFICE_MAX_CHARS:
                break
        return "\n".join(parts)[: config.OFFICE_MAX_CHARS]
    except Exception:
        return ""


def _read_xlsx(path: Path) -> str:
    """Cell text per sheet (D-02): read_only streaming, capped by sheet/row/col.
    '' on any failure."""
    try:
        from openpyxl import load_workbook
        wb = load_workbook(str(path), read_only=True, data_only=True)
        parts: list[str] = []
        try:
            for name in wb.sheetnames[: config.XLSX_MAX_SHEETS]:
                ws = wb[name]
                rows: list[str] = []
                for i, row in enumerate(ws.iter_rows(values_only=True)):
                    if i >= config.XLSX_MAX_ROWS:
                        break
                    cells = [
                        str(v) for v in row[:50]
                        if v is not None and str(v).strip()
                    ]
                    if cells:
                        rows.append(" | ".join(cells))
                if rows:
                    parts.append(f"Sheet {name}: " + "\n".join(rows))
                if sum(len(x) for x in parts) > config.OFFICE_MAX_CHARS:
                    break
        finally:
            wb.close()
        return "\n".join(parts)[: config.OFFICE_MAX_CHARS]
    except Exception:
        return ""


def _read_media(path: Path) -> str | None:
    """Tag text for a media file (D-03): 'Title by Artist — Album (date)
    audio recording, M:SS'. None when mutagen is missing or no tags —
    the caller falls back to plain name tokens. Never raises."""
    try:
        import mutagen
        f = mutagen.File(str(path), easy=True)
        if f is None or not f.tags:
            return None

        def tag(name: str) -> str:
            vals = f.tags.get(name) or []
            return str(vals[0]).strip() if vals else ""

        title, artist = tag("title"), tag("artist")
        album, date = tag("album"), tag("date")
        if not (title or artist or album):
            return None
        ext = path.suffix.lower()
        if ext == ".mp4":
            kind_word = "MP4 video"
        elif ext in (".mkv", ".mov", ".avi", ".webm"):
            kind_word = "video recording"
        else:
            kind_word = "audio recording"
        length = getattr(f.info, "length", None)
        dur = ""
        if length and length > 0:
            secs = int(round(length))
            h, rem = divmod(secs, 3600)
            m, s = divmod(rem, 60)
            dur = f", {h}:{m:02d}:{s:02d}" if h else f", {m}:{s:02d}"
        pieces = []
        head = " by ".join(x for x in (title, artist) if x)
        if head:
            pieces.append(head)
        if album:
            tail = f"Album {album}" + (f" ({date})" if date else "")
            pieces.append("— " + tail)
        pieces.append(kind_word)
        text = " ".join(pieces) + dur
        return text.strip()
    except Exception:
        return None


def extract_text(path: Path) -> tuple[str, str]:
    """Return (text, kind). Empty text means nothing usable extracted."""
    ext = path.suffix.lower()
    if ext in config.TEXT_EXTS:
        return _read_text_file(path), "text"
    if ext in config.PDF_EXTS:
        return _read_pdf(path), "pdf"
    if ext in config.DOCX_EXTS:
        return _read_docx(path), "docx"
    if ext in config.PPTX_EXTS:
        return _read_pptx(path), "slides"
    if ext in config.XLSX_EXTS:
        return _read_xlsx(path), "spreadsheet"
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
        if kind == "media":
            media_text = _read_media(path)
            if media_text:
                natural += ". " + media_text
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
