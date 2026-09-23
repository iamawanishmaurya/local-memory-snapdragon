"""Real PPTX/XLSX extraction tests (Phase 5, EXTRACT-01, plan 05-01).

Fixtures are generated in-test with python-pptx/openpyxl themselves —
model-free and dependency-optional: every test skips when the library is
missing so the model-free CI suite stays green even if requirements drift.
"""
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["LOCAL_MEMORY_HOME"] = str(Path(__file__).parent / ".tmpdata-office")

from local_memory import config  # noqa: E402
from local_memory.extractors import text_extractor  # noqa: E402
from local_memory.store import database, vector_store  # noqa: E402
from local_memory.pipeline import scan_folder  # noqa: E402
from local_memory.search import query_engine  # noqa: E402

try:
    import pptx  # noqa: F401
    HAVE_PPTX = True
except ImportError:
    HAVE_PPTX = False

try:
    import openpyxl  # noqa: F401
    HAVE_XLSX = True
except ImportError:
    HAVE_XLSX = False

_HOME = Path(__file__).parent / ".tmpdata-office"


def setup_module(_):
    import shutil

    shutil.rmtree(_HOME, ignore_errors=True)
    config.DATA_HOME = _HOME
    config.DB_PATH = _HOME / "index.db"
    config.THUMBS_DIR = _HOME / "thumbs"
    config.SETTINGS_PATH = _HOME / "settings.json"
    config.MODELS_DIR = _HOME / "no-models"  # model-free: hashing fallback
    database.close()
    vector_store.invalidate_cache()
    config.ensure_dirs()
    database.init_db()
    vector_store.init_db()


def teardown_module(_):
    import shutil

    database.close()
    vector_store.invalidate_cache()
    shutil.rmtree(_HOME, ignore_errors=True)


def _make_pptx(path: Path) -> None:
    from pptx import Presentation

    prs = Presentation()
    layout = prs.slide_layouts[6]  # blank
    for n in (1, 2):
        slide = prs.slides.add_slide(layout)
        box = slide.shapes.add_textbox(0, 0, 4_000_000, 1_000_000)
        box.text_frame.text = (
            f"quantum falcon presentation slide {n} of the quarterly review"
        )
    slide2 = prs.slides[1]
    rows, cols = 2, 2
    table = slide2.shapes.add_table(rows, cols, 0, 2_000_000, 4_000_000,
                                    1_500_000).table
    table.cell(0, 0).text = "roadmap item"
    table.cell(0, 1).text = "owner falcon"
    table.cell(1, 0).text = "status"
    table.cell(1, 1).text = "on track"
    prs.save(str(path))


def _make_xlsx(path: Path, rows: int = 3) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Budget"
    ws["A1"] = "xylophone budget 4281"
    ws["A2"] = "travel line"
    ws["A3"] = "totals below"
    for i in range(4, rows + 1):
        ws.cell(row=i, column=1, value=f"filler row {i}")
    wb.save(str(path))


# --- extraction units ---------------------------------------------------------


@pytest.mark.skipif(not HAVE_PPTX, reason="python-pptx not installed")
def test_pptx_extraction_slide_order_and_table(tmp_path):
    p = tmp_path / "deck.pptx"
    _make_pptx(p)
    text, kind = text_extractor.extract_text(p)
    assert kind == "slides"
    assert kind != "data"
    assert "quantum falcon presentation slide 1" in text
    assert "quantum falcon presentation slide 2" in text
    assert text.index("slide 1") < text.index("slide 2"), "slide order broken"
    assert "Slide 2:" in text
    assert "owner falcon" in text and "on track" in text, "table cells missing"
    assert len(text) <= config.OFFICE_MAX_CHARS


@pytest.mark.skipif(not HAVE_XLSX, reason="openpyxl not installed")
def test_xlsx_extraction_sheets_and_caps(tmp_path):
    p = tmp_path / "wb.xlsx"
    _make_xlsx(p)
    text, kind = text_extractor.extract_text(p)
    assert kind == "spreadsheet"
    assert kind != "data"
    assert "Sheet Budget:" in text
    assert "xylophone budget 4281" in text
    assert len(text) <= config.OFFICE_MAX_CHARS


@pytest.mark.skipif(not HAVE_XLSX, reason="openpyxl not installed")
def test_xlsx_row_cap_fast(tmp_path):
    """A 3000-row workbook is capped (rows + OFFICE_MAX_CHARS) and fast."""
    from openpyxl import Workbook

    p = tmp_path / "big.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Big"
    for i in range(1, 3001):
        ws.cell(row=i, column=1, value=f"rowvalue{i}")
    wb.save(str(p))
    t0 = time.perf_counter()
    text, kind = text_extractor.extract_text(p)
    elapsed = time.perf_counter() - t0
    assert kind == "spreadsheet"
    assert "rowvalue1" in text
    # Capped twice over: at XLSX_MAX_ROWS and at OFFICE_MAX_CHARS.
    assert len(text) <= config.OFFICE_MAX_CHARS
    assert elapsed < 10.0, f"row-capped extraction too slow: {elapsed:.1f}s"


@pytest.mark.skipif(not HAVE_XLSX, reason="openpyxl not installed")
def test_xlsx_sheet_cap(tmp_path):
    from openpyxl import Workbook

    p = tmp_path / "many.xlsx"
    wb = Workbook()
    wb.active.title = "One"
    wb.active["A1"] = "content one"
    for i in range(15):
        name = f"Extra{i}"
        ws = wb.create_sheet(title=name)
        ws["A1"] = f"content {name}"
    wb.save(str(p))
    text, _ = text_extractor.extract_text(p)
    assert "Sheet One:" in text
    assert "Sheet Extra8:" in text  # 10th sheet, inside the cap
    assert "Sheet Extra9:" not in text  # 11th sheet beyond XLSX_MAX_SHEETS


@pytest.mark.parametrize("ext", [".pptx", ".xlsx"])
def test_corrupt_office_file_yields_empty_not_crash(tmp_path, ext):
    p = tmp_path / ("broken" + ext)
    p.write_bytes(b"not really an office file")
    text, kind = text_extractor.extract_text(p)
    assert text == ""
    assert kind in ("slides", "spreadsheet")


# --- end-to-end: index + FTS query finds the generated files ------------------


@pytest.mark.skipif(not (HAVE_PPTX and HAVE_XLSX),
                    reason="python-pptx/openpyxl not installed")
def test_office_files_found_by_content_search():
    docs = _HOME / "office_docs"
    docs.mkdir(exist_ok=True)
    deck = docs / "quarterly_deck.pptx"
    wb = docs / "planning_workbook.xlsx"
    _make_pptx(deck)
    _make_xlsx(wb)
    assert scan_folder(str(docs)) >= 2

    row_deck = database.get_file(str(deck))
    row_wb = database.get_file(str(wb))
    assert row_deck["kind"] == "slides", row_deck["kind"]
    assert row_wb["kind"] == "spreadsheet", row_wb["kind"]

    results = query_engine.search("quantum falcon presentation")
    hit = next((r for r in results if r["name"] == "quarterly_deck.pptx"), None)
    assert hit is not None, "slide phrase query missed the deck"
    assert hit["match_keyword"] > 0

    results = query_engine.search("xylophone budget")
    hit = next((r for r in results if r["name"] == "planning_workbook.xlsx"), None)
    assert hit is not None, "cell-value query missed the workbook"
    assert hit["match_keyword"] > 0
