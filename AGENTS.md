# ML Automotive Foundations

Machine learning foundations for automotive applications including perception, edge inference, functional safety, and cybersecurity.

## Project Structure

- `digit_recognizer/` - MNIST digit recognition experiments (PyTorch/TensorFlow)
- `image_classifier/` - Image classification models and experiments
- `edge-inference/` - Edge deployment optimization:
  - `YOLO_EDGE_INFERENCE_PIPELINE.md` - YOLO on edge devices
  - TensorRT, ONNX Runtime, TFLite conversion pipelines
- `p1_3_automotive_soc/` - Automotive SoC ML workloads:
  - `DRIVE_ORIN_ARCHITECTURE.md` - NVIDIA Drive Orin architecture
- `p2_1_cybersecurity/` - ML for automotive cybersecurity:
  - `AUTOMOTIVE_TCU_TARA.md` - TCU Threat Analysis & Risk Assessment
- `p2_2_functional_safety/` - ML functional safety (ISO 26262):
  - `AI_INFERENCE_SAFETY_ASIL.md` - ASIL analysis for AI inference
- `linkedInPost.md` - Public-facing content
- `README.md` - Project overview

## Working Style

- **local-only** delivery mode
- Python/NumPy/PyTorch/TensorFlow as appropriate
- Document model architectures, hyperparameters, metrics
- Include inference latency and memory metrics for edge targets
- Reference automotive standards (ISO 26262, ISO 21448, ISO 21434) where applicable

## Coding Conventions

- Python with type hints
- Jupyter notebooks for exploration, scripts for production
- Requirements.txt / pyproject.toml for dependencies
- Document dataset sources, preprocessing, augmentation
- Track experiments with metrics (accuracy, latency, memory, FLOPs)

## Harness Compatibility

- Works with **Claude Code** (no `.claude/` needed)
- Works with **opencode** (no `opencode.json` needed)
- Works with **VS Code** / Jupyter / any Python IDE
- No harness-specific configuration required