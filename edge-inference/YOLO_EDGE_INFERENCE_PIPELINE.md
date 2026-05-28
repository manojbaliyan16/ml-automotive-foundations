# YOLOv8 Edge Inference Pipeline
## From PyTorch to TensorRT — P1.4

This document covers the full edge inference pipeline for object detection on automotive-grade hardware, using YOLOv8n as the model and NVIDIA DRIVE Orin as the target deployment platform.

---

## Pipeline Overview

```
PyTorch (.pt)
      │
      │  model.export(format='onnx')        [Ultralytics exporter]
      ▼
ONNX (.onnx)                                [Portable, framework-independent]
      │
      │  TensorRT builder                   [Hardware-specific compilation]
      ▼
TensorRT Engine (.engine / .plan)           [Target-GPU binary]
      │
      │  context.execute_async_v3()
      ▼
Inference on Target Hardware
(NVIDIA T4 / DRIVE Orin DLA + GPU)
```

---

## Model: YOLOv8n

- **Variant:** YOLOv8n — nano, smallest and fastest in the YOLOv8 family
- **Use case:** Edge deployment, prototyping, latency-critical systems
- **File sizes:**
  - `.pt` — 6 MB (PyTorch compact binary, weights only)
  - `.onnx` — 12.3 MB (full computation graph + weights explicitly described)
- **Input shape:** `[1, 3, 640, H]` — batch × RGB × height × width. YOLOv8 preserves aspect ratio; a 640×480 source image stays `[1, 3, 640, 480]`, not forced to 640×640 like MobileNetV2. The TensorRT engine is built with `[1, 3, 640, 640]` as the fixed input tensor for benchmarking.
- **Output shape:** `[1, 84, 8400]` — 84 class scores × 8400 candidate boxes

**Why is .onnx 2× larger than .pt?**

The `.pt` file stores weights compactly in PyTorch's binary format. The `.onnx` file stores the complete computation graph — every Conv layer, BatchNorm, activation, detection head, and NMS operation is described explicitly as a graph of operations. The format is designed for portability across frameworks, not storage efficiency.

---

## Stage 1 — PyTorch Inference (Development Baseline)

**Hardware:** Apple M4 MacBook (MPS backend)

**Benchmark:** `17.10ms ±1.37ms`

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model('bus.jpg')
```

Ultralytics timing breakdown from Cell 1:
- Preprocess: 1.6ms
- Inference: 11ms
- Postprocess (NMS): 3.4ms

**Test result on bus.jpg:** 6 objects detected — bus (0.87 confidence), 4 persons (0.83–0.87), stop sign (0.26).

---

## Stage 2 — ONNX Export

```python
model.export(format='onnx')
```

**Why `model.export()` and not `torch.onnx.export()`?**

YOLOv8 includes custom NMS (Non-Maximum Suppression) layers in its detection head. Raw `torch.onnx.export()` cannot trace these correctly. Ultralytics' built-in exporter handles the custom layers and includes NMS inside the ONNX graph.

**NMS — Non-Maximum Suppression:**

YOLOv8 generates approximately 8,400 candidate bounding boxes per image. NMS filters these down to final detections:
1. Keep the box with highest confidence per detected object
2. Delete all overlapping boxes with >50% IoU (Intersection over Union)

NMS is included inside the ONNX graph by default in Ultralytics export. TensorRT replaces it with a hardware-optimized NMS plugin during compilation.

---

## Stage 3 — ONNX Runtime (CPU Baseline)

**Hardware:** Apple M4 MacBook (CPUExecutionProvider)

**Benchmark:** `37.4ms`

```bash
yolo predict model=yolov8n.onnx
# Uses ONNX Runtime 1.25.0 with CPUExecutionProvider
```

**Key observation:** ONNX Runtime defaults to `CPUExecutionProvider` even on hardware with a GPU. To use GPU acceleration, `CUDAExecutionProvider` must be set explicitly. This is a common reason teams see high ONNX Runtime latency on Jetson and other edge platforms — the GPU is not being used.

---

## Stage 4 — TensorRT Compilation and Inference

**Hardware:** NVIDIA T4 GPU (Google Colab)

**Benchmark:** `4.00ms ±1.07ms`  
**Speedup over PyTorch MPS:** 4.3×  
**Speedup over ONNX Runtime CPU:** 9.4×

**TensorRT does not run the model — it compiles it:**

1. Reads the ONNX graph
2. Fuses layers (Conv + BatchNorm + ReLU → single kernel)
3. Selects hardware-specific CUDA kernels for the target GPU
4. Runs INT8/FP16 calibration if configured
5. Writes a compiled binary engine optimized for one GPU architecture

The resulting `.engine` file is not portable — it runs only on the GPU it was compiled for.

**TensorRT 11 API (correct usage):**

```python
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network()          # TRT 10+: no EXPLICIT_BATCH flag needed
config = builder.create_builder_config()
parser = trt.OnnxParser(network, TRT_LOGGER)

with open('yolov8n.onnx', 'rb') as f:
    parser.parse(f.read())

serialized = builder.build_serialized_network(network, config)
runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(serialized)
```

Note: `EXPLICIT_BATCH` flag was removed in TRT 10+. Explicit batch is the default behavior now.

---

## Benchmark Comparison

| Backend | Hardware | Mean Latency | Std Dev |
|---|---|---|---|
| ONNX Runtime | CPU (Apple M4) | 37.4ms | — |
| PyTorch MPS | GPU (Apple M4) | 17.10ms | ±1.37ms |
| TensorRT | T4 GPU (Colab) | 4.00ms | ±1.07ms |

Same model. Same `.onnx` file as input. Only the execution backend changes.

---

## ADAS Context — Why These Numbers Matter

At 30fps, a vehicle's perception system has **33ms per frame**. This is the total budget for the entire stack:

```
Object Detection (YOLOv8)
+ Sensor Fusion (camera + LiDAR/radar)
+ Planning (path prediction, collision check)
+ Safety check (ASIL-D validation)
+ Actuator command (SOME/IP → CAN → brake/steer)
─────────────────────────────────────────────
Total: 33ms
```

| Backend | Detection budget used | Remaining for rest of stack |
|---|---|---|
| ONNX Runtime CPU | 37.4ms | **Over budget** |
| PyTorch MPS | 17.10ms | 15.9ms |
| TensorRT | 4.00ms | **29ms** |

With TensorRT at 4ms, detection uses ~12% of the frame budget. The remaining 29ms is available for sensor fusion, planning, and safety validation.

**Confidence threshold in ADAS:**
- Default YOLOv8 threshold: 0.25 (allows low-confidence detections)
- Production ADAS: 0.5+ required to prevent false positives from reaching the brake controller
- This is a safety decision, not a performance decision

---

## ONNX vs TensorRT — Portability vs Performance

| Property | ONNX | TensorRT Engine |
|---|---|---|
| Runs on | Any framework with ONNX Runtime | One specific GPU architecture |
| Portable | Yes | No |
| Compilation step | No | Yes (minutes) |
| Latency (YOLOv8n) | 37.4ms CPU / faster with CUDA EP | 4.00ms |
| Layer fusion | No | Yes |
| Use case | Development, cross-platform | Production deployment |

In production automotive (DRIVE Orin), TensorRT is not optional. The ONNX file is the intermediate format used to produce the engine. The `.onnx` goes anywhere; the `.engine` ships to the vehicle.

---

## Hardware Notes

**Development (this project):**
- Apple M4 MacBook — PyTorch MPS backend for development iteration
- TensorRT does NOT run on Mac (CUDA required)

**Deployment target:**
- NVIDIA DRIVE Orin — DLA (Deep Learning Accelerator) × 2 for INT8 inference, GPU for FP16
- TensorRT engine compiled specifically for Orin GPU architecture
- DRIVE Orin SDK: developer.nvidia.com/drive/downloads

**Cloud (TensorRT compilation):**
- Google Colab T4 GPU (free tier) — used for TensorRT benchmarking
- Install: `pip install tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs`
- Note: pip-installed packages are wiped on Colab runtime restart

---

## Files in This Project

| File | Description |
|---|---|
| `edge-inference/p1_4_yolo_edge_inference.ipynb` | Full pipeline notebook (5 cells) |
| `yolov8n.pt` | YOLOv8n PyTorch weights (6 MB) |
| `yolov8n.onnx` | ONNX export (12.3 MB) — portable |
| `p1_4_result.jpg` | Inference output — bounding boxes on bus.jpg |

---

## References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com)
- [NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
- [NVIDIA DRIVE Orin Product Page](https://www.nvidia.com/en-us/self-driving-cars/drive-orin/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
