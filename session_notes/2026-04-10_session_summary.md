# Session Summary — April 10, 2026

## What Was Completed Today

### P1.2 — Edge Inference + Quantization — COMPLETE ✓

All 4 cells of `edge-inference/p1_2_edge_inference_quantization.ipynb` done.

**Cell 4 — Inference Speed Benchmark:**

| Model | Size | Latency | Std Dev | Speedup |
|-------|------|---------|---------|---------|
| FP32 | 13.6 MB | 16.22 ms | ±2.09 | baseline |
| Dynamic INT8 | 9.9 MB (1.4x) | 15.98 ms | ±0.33 | ~1.0x |
| Static INT8 | 3.8 MB (3.6x) | SKIPPED | — | — |

**Static inference skipped reason:** MobileNetV2 has residual `torch.add()` ops in its inverted residual blocks. Static quantization cannot quantize these without `FloatFunctional` handling. In production, TensorRT / SNPE handle this automatically at compile time.

---

## Key Concepts Learned Today

### 1. Why Dynamic Quantization Showed No Speedup on M4
- Dynamic quantization only targets **Linear layers**
- MobileNetV2 is 95% **Conv layers** — the heavy compute is untouched
- Apple M4 has very fast FP32 compute + high memory bandwidth — quantization advantage doesn't show here
- On automotive ECUs (Cortex-A55 + NPU): INT8 hardware paths + constrained memory bandwidth → 2–4x real speedup

### 2. Standard Deviation = Consistency = Safety
- FP32: ±2.09ms (noisy, variable)
- Dynamic INT8: ±0.33ms (tight, consistent)
- In real-time automotive systems, consistency matters as much as raw speed
- A model that's 16ms every time is safer than one that's sometimes 14ms, sometimes 19ms

### 3. MobileNetV2 Static Quantization Limitation
- Residual blocks use `torch.add()` for skip connections
- Static quantization can't handle this directly — throws `NotImplementedError`
- Fix requires: `torch.nn.quantized.FloatFunctional` — model-level surgery
- Production compilers (TensorRT, SNPE) abstract this away

### 4. `time.perf_counter()` vs `time.time()`
- `time.time()` — millisecond resolution, affected by OS clock adjustments
- `time.perf_counter()` — nanosecond resolution, always use this for benchmarking

### 5. Warmup Runs in Benchmarking
- First inference is always slow — CPU cache cold, memory pages not loaded, PyTorch JIT
- Always warm up with 10 runs before measuring — otherwise FP32 looks artificially slow

---

## NVIDIA Deadline
- JR1985837 + JR1983469 — deadline was April 8, already 2 days late
- Manoj committed to applying **today (April 10)**
- Do NOT let this slip further

---

## Pending Actions

- [ ] Apply to NVIDIA JR1985837 + JR1983469 — TODAY
- [ ] Push notebook + session notes to GitHub
- [ ] DSA tonight: Majority Element, Design HashSet, Design HashMap (7:45–9:00 PM IST)
- [ ] Saturday 9 AM: System Design — Edge AI inference pipeline on automotive SoC (camera → NPU → output)
- [ ] Update Excel sheet — mark P1.2 complete

---

## GitHub Push Commands

```bash
cd ~/path-to-repo/ml-automotive-foundations
git add edge-inference/p1_2_edge_inference_quantization.ipynb
git add session_notes/2026-04-10_session_summary.md
git add session_notes/2026-04-09_session_summary.md
git add session_notes/interview_prep_QA.md
git commit -m "Complete P1.2: Edge inference quantization benchmark — all 4 cells done"
git push
```

---

## P1.2 Final Status: COMPLETE ✓

All deliverables:
- Cell 1: FP32 baseline — 13.6 MB ✓
- Cell 2: Dynamic INT8 — 9.9 MB, 1.4x ✓
- Cell 3: Static INT8 — 3.8 MB, 3.6x ✓
- Cell 4: Latency benchmark — 16.22ms FP32, 15.98ms Dynamic ✓
