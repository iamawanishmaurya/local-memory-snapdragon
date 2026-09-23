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
    ("trocr/generation_config.json", [TROCR_BASE + "generation_config.json", "https://huggingface.co/Xenova/trocr-small-printed/resolve/main/generation_config.json"]),
]
# 06-02: the bare Qwen/Qwen3-0.6B ONNX export is NOT a GenAI bundle (no
# genai_config.json) and cannot load via onnxruntime_genai.Model. The --qwen
# branch below fetches the onnx-community discQuant GenAI bundle instead.
QWEN_REPO = "onnx-community/Qwen3-0.6B-DQ-ONNX"
QWEN_BUNDLE_DIR = "qwen3-0.6b"
# Tokenizer/config files flattened next to the model; prefer the int4 model
# variant (fallback q4f16) when present in the repo layout.
QWEN_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def _hf_list_repo_files(repo: str) -> list[str]:
    """List files of a public HF repo via the API (no auth needed)."""
    import json
    import urllib.request
    url = f"https://huggingface.co/api/models/{repo}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.load(resp)
    return [s["rfilename"] for s in data.get("siblings", [])]


def _pick_qwen_model_files(files: list[str]) -> list[str]:
    """Choose the int4 model files (plus genai_config.json / tokenizer) from
    the repo listing. GenAI bundles keep genai_config.json next to the model
    (usually under onnx/); we flatten everything into one bundle dir."""
    cfg = [f for f in files if f.endswith("genai_config.json")]
    tok = [f for f in files if f.rsplit("/", 1)[-1] in QWEN_TOKENIZER_FILES]
    int4 = [f for f in files if "/model_int4" in f or f.startswith("model_int4")]
    q4f16 = [f for f in files if "/model_q4f16" in f or f.startswith("model_q4f16")]
    chosen = int4 or q4f16
    # .onnx, .onnx.data and .onnx_data (HF external-data naming) all count —
    # an ONNX without its external data file fails to load.
    model_files = [f for f in chosen
                   if f.endswith(".onnx") or f.endswith(".onnx.data") or f.endswith(".onnx_data")]
    picked: list[str] = []
    seen: set[str] = set()
    for group in (cfg, tok, model_files):
        for f in group:
            base = f.rsplit("/", 1)[-1]
            if base not in seen:
                seen.add(base)
                picked.append(f)
    return picked


def _sanitize_qwen_bundle(dest_dir: Path) -> None:
    """Make the fetched bundle loadable by a CPU/QNN OGA build on this machine:
    point decoder.filename at the downloaded .onnx and strip foreign EP options
    (onnx-community configs ship webgpu provider_options that a CPU-only OGA
    build rejects with 'WebGPU execution provider is not supported')."""
    import json
    cfg_path = dest_dir / "genai_config.json"
    if not cfg_path.is_file():
        return
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        decoder = cfg["model"]["decoder"]
        onnx_name = next(
            (f.name for f in dest_dir.glob("*.onnx") if not f.name.endswith(".data")),
            None)
        if onnx_name:
            decoder["filename"] = onnx_name
        decoder.setdefault("session_options", {})["provider_options"] = [{"cpu": {}}]
        cfg_path.write_text(json.dumps(cfg, indent=4), encoding="utf-8")
        print(f"[ok] genai_config.json sanitized (filename={decoder.get('filename')}, ep=cpu)")
    except Exception as e:
        print(f"[warn] could not sanitize genai_config.json: {e}")


def run_qwen() -> int:
    """Fetch the Qwen3-0.6B GenAI bundle into models/qwen3-0.6b/ and verify
    genai_config.json landed. Idempotent: skips if the bundle already exists.
    If the HF layout changed, prints the builder fallback command."""
    import json
    import urllib.request

    dest_dir = MODELS_DIR / QWEN_BUNDLE_DIR
    try:
        files = _hf_list_repo_files(QWEN_REPO)
    except Exception as e:
        if (dest_dir / "genai_config.json").is_file():
            print(f"[ok] {QWEN_BUNDLE_DIR}/ GenAI bundle already present — skipping (HF unreachable)")
            return 0
        print(f"[fail] cannot list HF repo {QWEN_REPO}: {e}")
        print("  fallback: python -m onnxruntime_genai.models.builder -m Qwen/Qwen3-0.6B "
              f"-o {dest_dir.as_posix()} -p int4 -e cpu")
        return 1
    picked = _pick_qwen_model_files(files)
    if not any(f.endswith("genai_config.json") for f in picked) or not picked:
        print(f"[fail] unexpected repo layout for {QWEN_REPO}: {files[:20]} ...")
        print("  fallback: python -m onnxruntime_genai.models.builder -m Qwen/Qwen3-0.6B "
              f"-o {dest_dir.as_posix()} -p int4 -e cpu")
        return 1
    dest_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for rel in picked:
        base = rel.rsplit("/", 1)[-1]
        dest = dest_dir / base
        if dest.exists() and dest.stat().st_size > 100:
            print(f"[ok] {QWEN_BUNDLE_DIR}/{base} already present")
            continue
        url = f"https://huggingface.co/{QWEN_REPO}/resolve/main/{rel}"
        print(f"[dl] {QWEN_BUNDLE_DIR}/{base} ...")
        try:
            _download(url, dest)
            print(f"[ok] saved {dest} ({dest.stat().st_size // 1024 // 1024} MB)")
        except Exception as e:
            ok = False
            print(f"[fail] {base}: {e}")
    # Post-download verification: a bundle without genai_config.json is useless.
    if not (dest_dir / "genai_config.json").is_file():
        ok = False
        print(f"[fail] {dest_dir / 'genai_config.json'} missing after download")
    _sanitize_qwen_bundle(dest_dir)
    # Clean up the old stray bare export if present (it never worked with OGA).
    stray = MODELS_DIR / "qwen3-0.6b.onnx"
    if stray.exists():
        print(f"[note] removing stale bare export {stray.name} (not a GenAI bundle)")
        try:
            stray.unlink()
        except OSError:
            pass
    if ok:
        print(f"\nDone. GenAI bundle at {dest_dir} — restart Local Memory; the")
        print("rewriter should report backend=qwen3-npu (see --doctor).")
    else:
        print("\nBundle incomplete — re-run `python scripts/setup_models.py --qwen` to retry.")
    return 0 if ok else 1


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


# ---------------------------------------------------------------------------
# NPU mode (--npu): AI Hub compile of ONNX -> QNN context binaries (.serialized)
# ---------------------------------------------------------------------------
# The runtime (local_memory/embeddings/base.py::context_binary_for) auto-detects
# a sibling <stem>.serialized next to each .onnx — no runtime changes needed.
# TrOCR decoder stays on CPU in v1 (autoregressive + past-KV = hardest to
# compile; the encoder is the small share anyway) — see 05-CONTEXT.md D-01.
NPU_TARGETS = [
    "nomic-embed-text.onnx",
    "clip-vit-b32-image.onnx",
    "clip-vit-b32-text.onnx",
]
NPU_COMPILE_OPTIONS = "--target_runtime qnn_context_binary --quantize_io --quantize_fulltype int8"
NPU_DEFAULT_DEVICE = "Snapdragon X Elite CRD"
TOKEN_HELP = (
    "Qualcomm AI Hub API token not set. Create a free account at "
    "https://aihub.qualcomm.com, get a token from Settings, then:\n"
    "  set QAI_HUB_API_TOKEN=<token>"
)


def _npu_device(hub, name: str):
    """Match a device name against hub.get_devices(); None if not found."""
    for d in hub.get_devices():
        if d.name == name:
            return d
    return None


def run_npu() -> int:
    """Submit AI Hub compile jobs for the three NPU ONNX models and download
    sibling .serialized context binaries. Idempotent: skips models whose
    .serialized already exists. Returns 1 on any failure (jobs are
    resubmittable — just re-run)."""
    token = os.environ.get("QAI_HUB_API_TOKEN")
    if not token:
        print(TOKEN_HELP)
        return 1

    try:
        import qai_hub as hub  # type: ignore
    except ImportError:
        print("qai-hub is not installed. Run: pip install qai-hub")
        return 1
    try:
        hub.configure(api_token=token)
    except Exception as e:  # pragma: no cover - network/HTTP error path
        print(f"AI Hub configure failed: {e}")
        return 1

    device_name = os.environ.get("LOCAL_MEMORY_HUB_DEVICE", NPU_DEFAULT_DEVICE)
    device = _npu_device(hub, device_name)
    if device is None:
        try:
            known = [d.name for d in hub.get_devices()]
        except Exception:
            known = []
        print(f"Device {device_name!r} not found on AI Hub.")
        if known:
            print("Closest available devices:")
            for n in known[:20]:
                print(f"  - {n}")
            print("Re-run with: set LOCAL_MEMORY_HUB_DEVICE=<exact name>")
        return 1

    MODELS_DIR.mkdir(exist_ok=True)
    failures: list[str] = []
    skipped = 0
    for stem in NPU_TARGETS:
        onnx_path = MODELS_DIR / stem
        out_path = onnx_path.with_suffix(".serialized")
        if out_path.exists() and out_path.stat().st_size > 100_000:
            print(f"[ok] {out_path.name} already present — skipping")
            skipped += 1
            continue
        if not onnx_path.exists() or onnx_path.stat().st_size <= 100_000:
            failures.append(stem)
            print(f"[fail] {onnx_path.name} missing — run `python scripts/setup_models.py` first")
            continue
        print(f"[npu] compiling {stem} -> {out_path.name} on {device.name} ...")
        try:
            model = hub.upload_model(str(onnx_path))
            job = hub.submit_compile_job(
                model=model,
                device=device,
                options=NPU_COMPILE_OPTIONS,
            )
            print(f"  job: {getattr(job, 'url', job)}  (watch queue status in the browser)")
            target = job.get_target_model()  # blocks until compile finishes
            target.download(str(out_path))
            print(f"[ok] saved {out_path} ({out_path.stat().st_size // 1024 // 1024} MB)")
        except Exception as e:
            failures.append(stem)
            print(f"[fail] {stem}: {e}")
            if stem.startswith("nomic") and "shape" in str(e).lower():
                print("  hint: Nomic's HF ONNX export has dynamic seq len — fix the")
                print("  input specs to static max_len 256 and re-export, then retry.")
            print("  jobs are resubmittable: fix/re-run `python scripts/setup_models.py --npu`")

    if failures:
        print(f"\n{len(failures)} model(s) failed: {', '.join(failures)}")
        print("Re-run `python scripts/setup_models.py --npu` to retry (already-downloaded .serialized files are skipped).")
        return 1
    if skipped == len(NPU_TARGETS):
        print("\nAll NPU context binaries already present — nothing to do.")
    else:
        print("\nDone. Restart Local Memory; the Statistics badge should flip to")
        print("QNN active (check_npu_live() -> QNNExecutionProvider / npu_live: true).")
    return 0


def main() -> int:
    import sys as _sys
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--trocr", action="store_true", help="also fetch TrOCR OCR model (NPU)")
    _p.add_argument("--qwen", action="store_true", help="fetch Qwen3-0.6B GenAI bundle "
        "(onnx-community/Qwen3-0.6B-DQ-ONNX int4) into models/qwen3-0.6b/; if the HF "
        "layout differs, build locally with: python -m onnxruntime_genai.models.builder "
        "-m Qwen/Qwen3-0.6B -o models/qwen3-0.6b -p int4 -e cpu")
    _p.add_argument("--all", action="store_true", help="fetch base + optional models")
    _p.add_argument(
        "--npu",
        action="store_true",
        help="compile ONNX models to QNN context binaries via Qualcomm AI Hub "
        "(requires QAI_HUB_API_TOKEN; TrOCR decoder stays on CPU in v1)",
    )
    _a, _ = _p.parse_known_args()
    MODELS_DIR.mkdir(exist_ok=True)

    if _a.npu:
        return run_npu()
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
        # 06-02: fetch the GenAI bundle (directory) rather than a bare .onnx.
        rc = run_qwen()
        if rc != 0:
            ok = False
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
