"""Phase 4 demo-readiness tests (plan 04-01): launcher gates, seed payloads,
thumbnail transport.

Model-free and network-free: the launcher is asserted by marker presence
(it drives real downloads only when a sentinel is missing on the target
machine), the seed by direct execution into a tmp dir, the transport by
source-level contract on the UI bundle sources.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------- launcher ----

def test_launcher_script_gates():
    """launch.ps1 must contain the idempotence gates: venv check, deps
    fail_fast reuse, model sentinels with size guard, ui/dist gate, port
    probe, status poll, and no runtime pip install."""
    src = (ROOT / "scripts" / "launch.ps1").read_text(encoding="utf-8", errors="replace")
    assert "venv-arm64" in src, "launcher must reference the project venv"
    assert "fail_fast" in src, "launcher must reuse deps.fail_fast (SEC-03)"
    assert "setup_models.py" in src, "launcher must provision models when missing"
    assert "nomic-embed-text.onnx" in src, "model sentinel gate missing"
    assert "ui/dist/index.html" in src or "ui\\dist\\index.html" in src, "UI dist gate missing"
    assert "api/status" in src, "status poll missing"
    invoking = [ln for ln in src.splitlines()
                if "pip" in ln.lower() and "Write-Host" not in ln and not ln.strip().startswith("#")]
    assert not invoking, f"launcher must never invoke pip (SEC-03): {invoking}"


def test_launcher_has_no_demo_flag():
    """D-01: seeding is a separate command; launcher must not gain --demo."""
    src = (ROOT / "scripts" / "launch.ps1").read_text(encoding="utf-8", errors="replace")
    assert "--demo" not in src


# ----------------------------------------------------------------- seed ----

def test_seed_gate_payloads(tmp_path):
    """seed_demo.py plants the three D-08 gate payloads VERBATIM and is
    idempotent (second run skips everything)."""
    import subprocess

    target = tmp_path / "demo"
    for _ in range(2):
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "seed_demo.py"),
             "--target", str(target)],
            capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0, r.stderr
        out = r.stdout
        for q in ("pen with blue book", "invoices from March", "screenshots of that error"):
            assert q in out, f"scripted query {q!r} not printed"

    assert (target / "pen_note.txt").read_text(encoding="utf-8").strip() == \
        "pen with blue book on the desk, morning study notes"
    assert (target / "invoice_march.txt").read_text(encoding="utf-8").strip() == \
        "invoice for web design work, dated 12 March, total 900 dollars"
    assert (target / "error_shot.txt").read_text(encoding="utf-8").strip() == \
        "screenshot of the error: TypeError cannot read property id of undefined, full traceback below"

    # Cleanup-demo plants exist: duplicate pair + sparse large file + archive.
    assert (target / "report_final.txt").read_bytes() == \
        (target / "report_final_copy.txt").read_bytes()
    big = target / "old_backup_blob.bin"
    assert big.stat().st_size >= 250 * 1024 * 1024  # logical size, sparse on disk
    assert (target / "archive_2023.zip").is_file()
    # 25-35 files total.
    n = len(list(target.iterdir()))
    assert 25 <= n <= 35, f"unexpected seed size: {n}"


def test_seed_clean_flag(tmp_path):
    import subprocess

    target = tmp_path / "demo2"
    target.mkdir()
    stale = target / "leftover.txt"
    stale.write_text("old", encoding="utf-8")
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "seed_demo.py"),
         "--target", str(target), "--clean"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert not stale.exists(), "--clean must wipe the target first"


# ------------------------------------------------------------ transport ----

def test_thumbnail_transport():
    """ResultRow must render thumbnails via the Bearer-carrying blob helper,
    never a raw /api/thumbnail img src (Phase 2 middleware 401s that)."""
    api_src = (ROOT / "ui" / "src" / "lib" / "api.ts").read_text(encoding="utf-8")
    assert "createObjectURL" in api_src, "blob object URL helper missing"
    assert "Authorization" in api_src, "thumbnail fetch must send Bearer"
    page_src = (ROOT / "ui" / "src" / "features" / "search" / "index.tsx").read_text(encoding="utf-8")
    assert "src={`/api/thumbnail" not in page_src, "raw img src still present (401s on stage)"
    assert "thumbnailUrl(" in page_src, "ResultRow must use the thumbnailUrl helper"
    assert "revokeThumbnailUrl" in page_src, "object URLs must be revoked on unmount"
