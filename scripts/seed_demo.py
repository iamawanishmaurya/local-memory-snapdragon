"""Curated demo seed corpus (DEMO-02 / D-02).

Generates a reproducible cluttered demo folder on any machine — nothing binary
committed to git. Mirrors the D-08 gate payloads VERBATIM so
tests/test_search_quality.py::test_brief_queries_top3_seeded_demo stays green.

Usage:
    python scripts/seed_demo.py [--target <dir>] [--clean]

Idempotent per target folder: files whose bytes already match are skipped,
others overwritten. --clean wipes the target first.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time
import zipfile
from pathlib import Path

# Gate payloads — VERBATIM from tests/test_search_quality.py (D-08). Do not edit.
PEN_TEXT = "pen with blue book on the desk, morning study notes"
INVOICE_TEXT = "invoice for web design work, dated 12 March, total 900 dollars"
ERROR_TEXT = "screenshot of the error: TypeError cannot read property id of undefined, full traceback below"

SCRIPTED_QUERIES = [
    "pen with blue book",
    "invoices from March",
    "screenshots of that error",
]

# Gate distractor texts (reused verbatim for realism).
DISTRACTOR_TEXTS = {
    "recipe.txt": "chocolate chip cookie recipe: butter, sugar, flour, oven at 180",
    "resume.txt": "curriculum vitae: software engineer, five years experience, python",
    "meeting.txt": "meeting notes: budget review on Thursday, attendees listed below",
    "travel.txt": "travel itinerary: flights to Lisbon, hotel near the coast",
    "workout.txt": "weekly workout plan: running, swimming, rest day Sunday",
    "garden.txt": "spring gardening checklist: prune roses, plant tomatoes, mulch beds",
}


def _default_target() -> Path:
    profile = os.environ.get("USERPROFILE") or str(Path.home())
    return Path(profile) / "Documents" / "LocalMemoryDemo"


def _write_text(path: Path, text: str) -> str:
    """Write text file idempotently. Returns 'wrote' or 'skipped'."""
    data = text.encode("utf-8")
    if path.is_file() and path.read_bytes() == data:
        return "skipped"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return "wrote"


def _write_bytes(path: Path, data: bytes) -> str:
    if path.is_file() and path.read_bytes() == data:
        return "skipped"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return "wrote"


def _make_image(path: Path, lines: list[str], dark: bool = False) -> str:
    """Generate a 1200x800 PIL image with text. Idempotent via marker check."""
    from PIL import Image, ImageDraw

    bg = (20, 20, 28) if dark else (255, 255, 255)
    fg = (240, 240, 240) if dark else (0, 0, 0)
    # Deterministic render: same pixels every run.
    img = Image.new("RGB", (1200, 800), color=bg)
    d = ImageDraw.Draw(img)
    y = 60
    for line in lines:
        d.text((60, y), line, fill=fg)
        y += 48
    if "OK" in lines[-1] or any("OK" in ln for ln in lines):
        # OK button rect for the error dialog.
        d.rectangle([540, 640, 660, 710], outline=fg, width=3)
        d.text((580, 660), "OK", fill=fg)
    # Compare bytes: save to memory first.
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return _write_bytes(path, buf.getvalue())


def _backdate(path: Path, days_ago: int) -> None:
    ts = time.time() - days_ago * 86400
    os.utime(path, (ts, ts))


def _march_mtime() -> float:
    import datetime

    year = datetime.datetime.now().year
    dt = datetime.datetime(year, 3, 12, 10, 0, 0)
    return dt.timestamp()


def main() -> int:
    random.seed(42)
    ap = argparse.ArgumentParser(description="Generate curated Local Memory demo folder")
    ap.add_argument("--target", default=str(_default_target()))
    ap.add_argument("--clean", action="store_true", help="delete target first")
    args = ap.parse_args()

    target = Path(args.target)
    if args.clean and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    wrote = 0
    skipped = 0

    def track(status: str) -> None:
        nonlocal wrote, skipped
        if status == "wrote":
            wrote += 1
        else:
            skipped += 1

    # --- 1. Planted winners (gate payloads VERBATIM) ---
    track(_write_text(target / "pen_note.txt", PEN_TEXT + "\n"))
    track(_write_text(target / "invoice_march.txt", INVOICE_TEXT + "\n"))
    track(_write_text(target / "error_shot.txt", ERROR_TEXT + "\n"))
    # March mtime so the query-rewriter date path ("March") also fires.
    march_ts = _march_mtime()
    os.utime(target / "invoice_march.txt", (march_ts, march_ts))

    # --- 2. PIL OCR targets (1200x800, one side >= 800px for banded OCR) ---
    track(_make_image(
        target / "error_dialog.png",
        ["TypeError: Cannot read property 'id' of undefined", "at renderList (app.js:42)", "An unexpected error occurred.", "OK"],
    ))
    track(_make_image(
        target / "invoice_scan.png",
        ["INVOICE March 2026", "Web design services", "Total $900", "Due on receipt"],
    ))
    track(_make_image(
        target / "desk_photo.png",
        ["blue book and pen on desk", "morning study setup", "photo caption strip"],
    ))
    track(_make_image(
        target / "error_dialog_dark.png",
        ["TypeError: Cannot read property 'id' of undefined", "dark mode screenshot", "OK"],
        dark=True,
    ))

    # --- 3. Distractors (12-15, gate texts reused) ---
    for name, text in DISTRACTOR_TEXTS.items():
        track(_write_text(target / name, text + "\n"))
    extra_distractors = {
        "IMG_0001.txt": "photo dump from the weekend trip, beach sunset and friends",
        "IMG_0002.txt": "scanned warranty card for the kitchen mixer, valid two years",
        "resume (1).txt": "curriculum vitae copy: software engineer, five years experience, python",
        "book_notes.txt": "notes on deep work: focus blocks, no phone mornings, evening review",
        "shopping_list.txt": "groceries: milk, eggs, bread, spinach, oats, coffee beans",
        "rent_agreement.txt": "rental agreement summary: deposit two months, notice period thirty days",
        "flight_confirm.txt": "booking reference XYZ123, departure Friday morning, seat 14A",
        "gym_log.txt": "monday chest day, tuesday rest, wednesday legs, thursday cardio",
        "plant_care.txt": "water money plant weekly, rotate pot for even sunlight",
    }
    for name, text in extra_distractors.items():
        track(_write_text(target / name, text + "\n"))

    # --- 4. Cleanup-demo plants ---
    # Byte-identical duplicate pair.
    dup_bytes = b"duplicate file content for cleanup demo: same bytes, two names\n"
    track(_write_bytes(target / "report_final.txt", dup_bytes))
    track(_write_bytes(target / "report_final_copy.txt", dup_bytes))
    # 250MB sparse large file (seek + single null write, NTFS sparse).
    big = target / "old_backup_blob.bin"
    if big.is_file() and big.stat().st_size == 250 * 1024 * 1024:
        track("skipped")
    else:
        with open(big, "wb") as f:
            f.seek(250 * 1024 * 1024 - 1)
            f.write(b"\0")
        track("wrote")
    _backdate(big, 400)
    # Old archive with 3 text entries, mtime backdated 2 years.
    arch = target / "archive_2023.zip"
    arch_bytes_buf = io_bytes = None
    import io as _io

    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("old_notes.txt", "archived notes from two years ago\n")
        z.writestr("old_receipt.txt", "archived receipt: total 150 dollars\n")
        z.writestr("old_list.txt", "archived packing list\n")
    arch_bytes_buf = buf.getvalue()
    status = _write_bytes(arch, arch_bytes_buf)
    track(status)
    _backdate(arch, 730)
    # 2-3 stale files (mtime > 365 days old).
    for name in ("stale_report_2022.txt", "stale_photo_note.txt"):
        p = target / name
        track(_write_text(p, f"stale file {name}: untouched for over a year\n"))
        _backdate(p, 400)

    total = len(list(target.iterdir()))
    print(f"[seed] target: {target} ({total} files, {wrote} wrote, {skipped} skipped)")
    print("[seed] scripted queries:")
    for q in SCRIPTED_QUERIES:
        print(f"  - {q}")
    print("[seed] next: add this folder as a watched folder via the UI (Folders page)")
    print("        or POST /api/folders {\"path\": \"<target>\"} then POST /api/index")
    return 0


if __name__ == "__main__":
    sys.exit(main())
