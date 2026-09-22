"""Tests for binary metadata ingestion (accuracy eval gap #1) and
OCR of PDF-embedded page images (accuracy eval gap #2).

Model-free: the HashingEncoder text fallback + fake TrOCR sessions keep
every assertion deterministic on a machine with no models installed.
"""
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["LOCAL_MEMORY_HOME"] = str(Path(__file__).parent / ".tmpdata-binary")

from local_memory import config  # noqa: E402
from local_memory.extractors import ocr, text_extractor  # noqa: E402
from local_memory.pipeline import index_file, scan_folder  # noqa: E402
from local_memory.search import query_engine  # noqa: E402
from local_memory.store import database, vector_store  # noqa: E402


def setup_module(_):
    home = Path(__file__).parent / ".tmpdata-binary"
    config.DATA_HOME = home
    config.DB_PATH = home / "index.db"
    config.THUMBS_DIR = home / "thumbs"
    config.SETTINGS_PATH = home / "settings.json"
    database.close()
    vector_store.invalidate_cache()
    config.ensure_dirs()
    database.init_db()
    vector_store.init_db()


def teardown_module(_):
    import shutil
    database.close()
    vector_store.invalidate_cache()
    shutil.rmtree(Path(__file__).parent / ".tmpdata-binary", ignore_errors=True)


# ------------------------------------------------------------ classification --

def test_binary_kind_classification():
    assert config.binary_kind("kali-linux-2026.2-installer-amd64.iso") == "disk-image"
    assert config.binary_kind("setup.exe") == "executable"
    assert config.binary_kind("app.msix") == "executable"
    assert config.binary_kind("data.parquet") == "data"
    assert config.binary_kind("backup.tar.gz") == "archive"  # compound suffix
    assert config.binary_kind("notes.txt") is None
    assert config.binary_kind("photo.jpg") is None


def test_name_tokens_split_separators():
    toks = text_extractor._name_tokens(Path("kali-linux-2026.2-installer-amd64.iso"))
    assert toks == ["kali", "linux", "2026.2", "installer", "amd64", "iso"]


# --------------------------------------------------------------- metadata -----

def test_metadata_chunks_iso(tmp_path):
    p = tmp_path / "kali-linux-2026.2-installer-amd64.iso"
    p.write_bytes(b"\x00" * 16)
    kind, chunks = text_extractor.metadata_chunks(p)
    assert kind == "disk-image"
    assert len(chunks) == 2
    assert "kali linux 2026.2 installer amd64 iso" in chunks[0]
    assert "Kali" in chunks[1] and "ISO" in chunks[1] and "disk image" in chunks[1]


def test_metadata_chunks_zip_lists_entries(tmp_path):
    p = tmp_path / "project-sources.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("src/main.py", "print('hi')")
        zf.writestr("README.md", "hi")
    kind, chunks = text_extractor.metadata_chunks(p)
    assert kind == "archive"
    assert "main.py" in chunks[-1]


def test_archive_inspect_size_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARCHIVE_INSPECT_MAX_BYTES", 4)
    p = tmp_path / "tiny.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("a.txt", "x")
    _, chunks = text_extractor.metadata_chunks(p)
    assert len(chunks) == 2 and "a.txt" not in chunks[1]


# ------------------------------------------------------------- ipynb ----------

def test_ipynb_extracts_markdown_and_first_code_lines(tmp_path):
    nb = {"cells": [
        {"cell_type": "markdown", "source": "# Gradient Descent Notes\n\nconverges fast"},
        {"cell_type": "code", "source": "import numpy as np\nlr = 0.01\nwhile True:\n    pass"},
    ]}
    p = tmp_path / "lecture.ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")
    text, kind = text_extractor.extract_text(p)
    assert kind == "notebook"
    assert "Gradient Descent" in text and "lr = 0.01" in text
    assert "while True" in text and "pass" not in text  # only first lines kept


# --------------------------------------------------------- pipeline/search ----

def test_scan_folder_indexes_binaries_and_finds_by_name():
    docs = Path(__file__).parent / ".tmpdata-binary" / "bin"
    docs.mkdir(exist_ok=True)
    (docs / "kali-linux-2026.2-installer-amd64.iso").write_bytes(b"\x00" * 32)
    (docs / "nvidia-driver-570.exe").write_bytes(b"MZ\x00\x00")
    (docs / "sales.parquet").write_bytes(b"PAR1")
    n = scan_folder(str(docs))
    assert n == 3
    row = database.get_file(str(docs / "kali-linux-2026.2-installer-amd64.iso"))
    assert row is not None and row["kind"] == "disk-image"

    hits = query_engine.search("kali linux installer amd64 iso")
    assert hits and "kali-linux" in hits[0]["path"], hits[:1]
    hits2 = query_engine.search("nvidia driver exe")
    assert hits2 and "nvidia-driver" in hits2[0]["path"]


def test_index_file_binary_returns_true(tmp_path):
    p = tmp_path / "tool.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("x.txt", "x")
    assert index_file(str(p)) is True
    assert index_file(str(p)) is False  # unchanged -> skipped


# --------------------------------------------------- PDF page-image OCR -------

class _FakeDecSession:
    def get_inputs(self):
        class _I:
            name = "input_ids"
            shape = [None, None]
        class _E:
            name = "encoder_hidden_states"
            shape = [None, None, None]
        class _U:
            name = "use_cache_branch"
            shape = [None]
        return [_I(), _E(), _U()]

    def run(self, _outputs, _feed):
        logits = np.full((1, 1, 8), -10.0, dtype=np.float32)
        seq = [5, 6, 5, 6, 5, 6, 2]  # "Hello world" x3 (must look like real text)
        idx = min(self._step, len(seq) - 1)
        logits[0, 0, seq[idx]] = 10.0
        self._step += 1
        return [logits]

    _step = 0


class _FakeEncSession:
    def get_inputs(self):
        class _I:
            name = "pixel_values"
            shape = [1, 3, 384, 384]
        return [_I()]

    def run(self, _outputs, _feed):
        return [np.zeros((1, 4, 8), dtype=np.float32)]


def _image_only_pdf(tmp_path: Path) -> Path:
    """A PDF whose only content is a raster image (what a scanner produces)."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (400, 200), "white")
    ImageDraw.Draw(img).rectangle([10, 10, 390, 190], outline="black")
    p = tmp_path / "scan.pdf"
    img.save(p, "PDF")
    return p


def test_image_only_pdf_ocrs_embedded_page_image(monkeypatch, tmp_path):
    monkeypatch.setattr(ocr, "_try_trocr",
                        lambda: (_FakeEncSession(), _FakeDecSession(), {2: "</s>", 5: "Hello", 6: "Ġworld"}, 1))
    p = _image_only_pdf(tmp_path)
    text, kind = text_extractor.extract_text(p)
    assert kind == "pdf"
    assert "Hello" in text and "world" in text


def test_pdf_ocr_page_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(ocr, "_try_trocr",
                        lambda: (_FakeEncSession(), _FakeDecSession(), {2: "</s>", 5: "Hello", 6: "Ġworld"}, 1))
    monkeypatch.setattr(config, "PDF_OCR_MAX_PAGES", 0)
    p = _image_only_pdf(tmp_path)
    text, _ = text_extractor.extract_text(p)
    assert "Hello" not in text


def test_pdf_ocr_size_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(ocr, "_try_trocr",
                        lambda: (_FakeEncSession(), _FakeDecSession(), {2: "</s>", 5: "Hello", 6: "Ġworld"}, 1))
    monkeypatch.setattr(config, "PDF_OCR_MAX_BYTES", 10)
    p = _image_only_pdf(tmp_path)
    text, _ = text_extractor.extract_text(p)
    assert "Hello" not in text


def test_pdf_ocr_no_backend_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(ocr, "_try_trocr", lambda: None)
    monkeypatch.setattr(ocr, "_get_reader", lambda: None)
    p = _image_only_pdf(tmp_path)
    text, _ = text_extractor.extract_text(p)
    assert text.strip() == ""  # deterministic fallback, no crash
