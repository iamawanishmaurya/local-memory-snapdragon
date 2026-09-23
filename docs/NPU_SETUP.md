# Snapdragon NPU Setup Guide

Goal: all embeddings execute on the **Snapdragon NPU (Hexagon)** via ONNX
Runtime's **Qualcomm QNN Execution Provider**, with models exported through
**Qualcomm AI Hub**.

## 0. You need ARM64-native Python (important!)

The default python.org installer is **x64 (emulated)** — QNN can never load
there. On your Snapdragon PC:

```powershell
winget install Python.Python.3.12 --architecture arm64 --silent --accept-package-agreements --accept-source-agreements
# → C:\Users\<you>\AppData\Local\Programs\Python\Python312-arm64\python.exe
#    [MSC v.1943 64 bit (ARM64)]
python -m venv .venv-arm64
.venv-arm64\Scripts\activate
pip install -r requirements.txt   # pulls onnxruntime-qnn (ARM64 wheel)
```

Verify: `python -c "import onnxruntime_qnn"` imports cleanly. The app
auto-registers the QNN plug-in (`register_execution_provider_library`) —
no QNN SDK install needed; `QnnHtp.dll` ships inside the wheel.

> Status on X Plus (X1P42100), Sep 2026: QNN EP registers (`QNNExecutionProvider`
> in provider list). Generic FP32 ONNX exports (Nomic/CLIP from HuggingFace)
> still initialize on **CPU** — the HTP needs INT8-quantized, SoC-specific
> **context binaries** (step 3) for real NPU dispatch.

## 1. Verify your device

On the target Snapdragon X Elite / X2 Elite PC:

```powershell
python scripts\demo_index.py --doctor
```

If `QNNExecutionProvider` appears in the provider list, the NPU path is live.
The app automatically prefers it for every embedding call.

## 2. One-time model setup (network needed once)

```powershell
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
pip install qai-hub                     # optional, for on-device compile
set QAI_HUB_API_TOKEN=<token from aihub.qualcomm.com → Settings>
python scripts\setup_models.py
```

This populates `models/` with:

| File | Model | Role |
|---|---|---|
| `nomic-embed-text.onnx` | Nomic-Embed-Text v1.5 | text embeddings |
| `clip-vit-b32-image.onnx` | CLIP ViT-B/32 image encoder | image embeddings |
| `clip-vit-b32-text.onnx` | CLIP ViT-B/32 text encoder | text→image-space queries |

## 3. NPU compile via AI Hub (one command, real NPU dispatch)

AI Hub compiles the models into INT8 QNN context binaries targeted at the exact
SoC. The runtime auto-detects them — `local_memory/embeddings/base.py::
context_binary_for()` picks up a sibling `<stem>.serialized` next to each
`.onnx` with zero configuration, and `create_session()` passes it to the QNN EP.

One-time setup:

```powershell
pip install qai-hub
set QAI_HUB_API_TOKEN=<token from aihub.qualcomm.com → Settings>
python scripts\setup_models.py --npu
```

The script submits compile jobs for exactly these three ONNX models
(TrOCR's autoregressive decoder intentionally stays on CPU in v1):

| Model | Artifact | Expected size (INT8) |
|---|---|---|
| Nomic-Embed-Text v1.5 | `models/nomic-embed-text.serialized` | ~35–40 MB |
| CLIP ViT-B/32 image | `models/clip-vit-b32-image.serialized` | ~40 MB |
| CLIP ViT-B/32 text | `models/clip-vit-b32-text.serialized` | ~20 MB |

Compile options used: `--target_runtime qnn_context_binary --quantize_io
--quantize_fulltype int8`.

Notes:
- **Idempotent / resumable**: re-running the command skips any model whose
  `.serialized` already exists, so you can retry after AI Hub queue hiccups
  (queue times range minutes to ~1 h — the script prints job URLs to watch).
- **Device selection**: defaults to `Snapdragon X Elite CRD`. Override with
  `set LOCAL_MEMORY_HUB_DEVICE=<exact name from hub.get_devices()>` — context
  binaries are SoC-locked, so it must match your machine (X1P42100 → X Elite).
- Compilation is remote (any internet-connected host works); only *inference*
  needs the ARM64 Python + `onnxruntime-qnn` from §0.

Manual verification gate (after the download finishes):

1. Restart the app (`python -m local_memory.main`).
2. The Statistics badge must show **"NPU: QNN active ⚡"** — i.e.
   `check_npu_live()` reports `QNNExecutionProvider` / `npu_live: true`
   (`active_provider_of(create_session(...))` returns `QNNExecutionProvider`).
3. Compare `/api/perf` p50 encode latency against the ~3.0 s CPU-fallback
   baseline from Phase 4.

## 4. OCR

EasyOCR detection/recognition runs on CPU (and is fine at screenshot volume);
it is an optional install because of its PyTorch dependency:

```powershell
pip install easyocr
```

An NPU-compiled TrOCR via AI Hub is on the roadmap; `extractors/ocr.py` is the
single seam to swap backends.

## 5. Demonstrating NPU usage to judges

1. `python scripts\demo_index.py --doctor` → shows `QNN (NPU): AVAILABLE` and
   provider order.
2. Start the app; the header badge shows **"NPU: QNN active ⚡"** read from the
   live ONNX Runtime session (`/api/status`).
3. Index a folder with a few hundred files and compare wall-clock indexing
   time with `LOCAL_MEMORY_MODELS` pointing at NPU artifacts vs. CPU.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `QNNExecutionProvider` missing | Install `onnxruntime-qnn` (ARM64 wheel) and Qualcomm QNN runtime; confirm ARM64 Python |
| Model load error | Run `scripts/setup_models.py`; check `models/` paths |
| Slow indexing | Confirm `--doctor` shows QNN, not CPU; reduce OCR (EasyOCR) volume |
