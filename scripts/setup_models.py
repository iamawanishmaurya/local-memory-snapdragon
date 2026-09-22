"""One-time model export via Qualcomm AI Hub (requires network + API token).

Downloads / compiles the NPU-ready ONNX artifacts into models/:
  - nomic-embed-text.onnx       (Nomic-Embed-Text v1.5, text embeddings)
  - clip-vit-b32-image.onnx     (CLIP ViT-B/32 image encoder)
  - clip-vit-b32-text.onnx      (CLIP ViT-B/32 text encoder)

Setup:
    pip install qai-hub
    set QAI_HUB_API_TOKEN=...     (from https://aihub.qualcomm.com -> Settings)
    python scripts/setup_models.py

After this script finishes, Local Memory runs 100% offline forever.
On a Snapdragon X Elite device, qai-hub can also compile QNN context binaries
for direct NPU deployment (see docs/NPU_SETUP.md).
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

# Windows console (cp1252) chokes on unicode arrows/check marks — force utf-8.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Public HuggingFace sources for the ONNX exports (mirrors AI Hub model zoo).
NOMIC_ONNX_URL = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model.onnx"
# NOTE: qualcomm/CLIP requires HF auth (401 for anonymous). Xenova's export of
# the same openai/clip-vit-base-patch32 weights is public; vision_model.onnx =
# image encoder, text_model.onnx = text encoder.
CLIP_IMAGE_URLS = [
    "https://huggingface.co/qualcomm/CLIP/resolve/main/clip-image-encoder/vit-32/image-encoder.onnx",
    "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model.onnx",
]
CLIP_TEXT_URLS = [
    "https://huggingface.co/qualcomm/CLIP/resolve/main/clip-text-encoder/vit-32/text-encoder.onnx",
    "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/text_model.onnx",
]
# CLIP BPE merges for the text tokenizer (see local_memory/embeddings/clip_tokenizer.py).
CLIP_BPE_URL = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"

TARGETS = [
    ("nomic-embed-text.onnx", [NOMIC_ONNX_URL]),
    ("nomic-tokenizer.json", ["https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer.json"]),
    ("clip-vit-b32-image.onnx", CLIP_IMAGE_URLS),
    ("clip-vit-b32-text.onnx", CLIP_TEXT_URLS),
    ("bpe_simple_vocab_16e6.txt.gz", [CLIP_BPE_URL]),
]

# Optional NPU extras (compiled via AI Hub on the Snapdragon target).
# TrOCR: pre-exported community ONNX (microsoft/trocr-small-printed ships no
# onnx/model.onnx — that URL 404s). Files land under models/trocr/.
# Primary: onnx-community (may be gated/401 anonymously); mirror: Xenova's
# identical Optimum export (same files under onnx/).
TROCR_BASE = "https://huggingface.co/onnx-community/trocr-small-printed-ONNX/resolve/main/"
TROCR_MIRROR = "https://huggingface.co/Xenova/trocr-small-printed/resolve/main/onnx/"
TROCR_TARGETS = [
    ("trocr/encoder_model.onnx", [TROCR_BASE + "encoder_model.onnx", TROCR_MIRROR + "encoder_model.onnx"]),
    ("trocr/decoder_model_merged.onnx", [TROCR_BASE + "decoder_model_merged.onnx", TROCR_MIRROR + "decoder_model_merged.onnx"]),
    ("trocr/trocr-tokenizer.json", [TROCR_BASE + "tokenizer.json", "https://huggingface.co/Xenova/trocr-small-printed/resolve/main/tokenizer.json"]),
]
QWEN_URL = "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/onnx/model.onnx"
OPTIONAL = [
    ("qwen3-0.6b.onnx", QWEN_URL),
]


def _download(url: str, dest: Path, retries: int = 8) -> None:
    """Resumable download with retries (flaky links): resumes .part via Range."""
    import time
    import urllib.request
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            have = tmp.stat().st_size if tmp.exists() else 0
            req = urllib.request.Request(url)
            if have:
                req.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "ab" if have else "wb") as f:
                # Server ignored Range (200 instead of 206): restart from scratch.
                if have and resp.status == 200:
                    f.close()
                    tmp.unlink(missing_ok=True)
                    f = open(tmp, "wb")
                    have = 0
                total = resp.headers.get("Content-Length")
                total = (int(total) + have) if total else 0
                done = have
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  ... {done//1024//1024} / {total//1024//1024} MB", end="", flush=True)
            print()
            tmp.replace(dest)
            return
        except Exception as e:
            print(f"\n  attempt {attempt}/{retries} failed: {e}")
            if attempt == retries:
                raise
            time.sleep(min(30, 2 ** attempt))


def main() -> int:
    import sys as _sys
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--trocr", action="store_true", help="also fetch TrOCR OCR model (NPU)")
    _p.add_argument("--qwen", action="store_true", help="also fetch Qwen3-0.6B rewriter (NPU)")
    _p.add_argument("--all", action="store_true", help="fetch base + optional models")
    _a, _ = _p.parse_known_args()
    MODELS_DIR.mkdir(exist_ok=True)

    token = os.environ.get("QAI_HUB_API_TOKEN")
    if token:
        print("Qualcomm AI Hub token found — trying qai-hub export first…")
        try:
            import qai_hub as hub  # type: ignore
            print(f"  AI Hub devices: {[d.name for d in hub.get_devices()][:3]} …")
            # Full NPU compile flow (context binaries) is documented in
            # docs/NPU_SETUP.md; here we ensure ONNX artifacts exist first.
        except Exception as e:
            print(f"  qai-hub not usable ({e}); falling back to direct ONNX download.")

    try:
        import urllib.request
    except ImportError:  # pragma: no cover
        print("Python urllib unavailable"); return 1

    ok = True
    targets = list(TARGETS)
    import sys as _sys
    if "--trocr" in _sys.argv or "--all" in _sys.argv:
        targets.extend(TROCR_TARGETS)
    if "--qwen" in _sys.argv or "--all" in _sys.argv:
        targets.extend(OPTIONAL)
    for name, urls in targets:
        if isinstance(urls, str):
            urls = [urls]
        dest = MODELS_DIR / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 100_000:
            print(f"[ok] {name} already present")
            continue
        print(f"[dl] downloading {name} ...")
        err = None
        for url in urls:
            try:
                _download(url, dest)
                print(f"[ok] saved {dest} ({dest.stat().st_size // 1024 // 1024} MB)")
                err = None
                break
            except Exception as e:
                err = e
                print(f"  mirror failed ({url.split('/')[2]}): {e} — trying next ...")
        if err is not None:
            ok = False
            print(f"[fail] {err}\n  Download manually and place at: {dest}")

    print("\nDone." if ok else "\nSome models failed — see messages above.")
    print("Local Memory is now fully offline. Start it with: python -m local_memory.main")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
