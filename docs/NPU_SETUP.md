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

## 3. NPU compile (optional, best performance)

AI Hub can compile the models into QNN context binaries targeted at the exact
SoC, which load faster and run more efficiently than generic ONNX:

```python
import qai_hub as hub
model = hub.upload_model("models/nomic-embed-text.onnx")
job = hub.submit_compile_job(
    model=model,
    device=hub.Device("Snapdragon X Elite CRD"),   # or your device name
    options="--target_runtime qnn_context_binary",
)
```

Place the resulting `.serialized` binaries in `models/` and set
`LOCAL_MEMORY_QNN_CONTEXT=1`; `embeddings/base.py` will request the QNN EP
with the context-binary path.

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
