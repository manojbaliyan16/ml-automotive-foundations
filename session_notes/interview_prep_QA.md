# Interview Prep — Q&A Revision Notes
> Core concepts for Principal Architect interviews at NVIDIA / Qualcomm / ARM
> Revise these before every interview. Explain out loud without looking.

---

## 1. What is the difference between Dynamic and Static Quantization?

**Dynamic Quantization:**
Applies only to Linear layers. Weights are converted from FP32 to INT8 on the fly at runtime. No calibration data needed. Size reduction is modest — ~1.4x in practice.

**Static Quantization:**
Applies to all layers including Conv layers. Requires a calibration step — ~100 real images are passed through the model to measure the activation value ranges per layer. With those ranges known, all layers are converted to INT8. Size reduction is significant — ~3.6x in practice.

**Key difference in one line:**
> Dynamic = on the fly, Linear only, no data needed. Static = calibrated upfront, all layers, needs real images.

---

## 2. Why does Automotive AI use MobileNet-style architectures instead of ResNet?

Three specific reasons:

**Size:** MobileNetV2 is ~14MB. ResNet50 is ~100MB. An automotive ECU has limited flash storage — ResNet doesn't fit alongside the OS and middleware. MobileNet does.

**Speed:** MobileNet uses depthwise separable convolutions — 8–9x fewer calculations than standard convolutions. Faster inference means more camera frames processed per second. At 120 km/h, 50ms slower inference = 1.6 metres of blindness.

**Power:** Fewer calculations = less power consumption. Automotive compute budgets are measured in watts. Running ResNet continuously violates those constraints.

**One line answer:**
> MobileNet-style architectures deliver acceptable accuracy within the strict size, latency, and power constraints of embedded ECUs and NPUs — ResNet is too heavy for real-time automotive deployment.

---

## 3. What happens at 100 km/h if inference takes 100ms instead of 50ms?

**The math:**
- 100 km/h = 27.7 metres per second
- At 50ms inference: car travels 1.4 metres during inference
- At 100ms inference: car travels 2.8 metres during inference
- **Extra 50ms = 1.4 metres of blindness**

**Why it matters:**
During those extra 1.4 metres the perception system has not updated. The car is moving but blind. In pedestrian detection, that gap can be the difference between a timely emergency brake and a collision.

**Interview answer:**
> At 100 km/h the car covers 27.7 metres per second. An extra 50ms of inference latency means the vehicle travels an additional 1.4 metres without a perception update. In ADAS systems, that gap can be the difference between detecting a pedestrian in time to brake and not. This is why MobileNet's faster inference and INT8 quantization are not performance optimisations — they are safety requirements.

---

## 4. Why do we Train in FP32 but Deploy in INT8?

**Training needs FP32:**
Gradient updates during backpropagation are tiny decimal values — sometimes 0.0000001. INT8 only stores whole numbers (-128 to +127). Those gradient updates would round to zero and the model would stop learning entirely.

**Inference can use INT8:**
During inference we are not updating weights — we are just multiplying and summing fixed values. A small precision loss in the weights does not significantly change the final prediction. A cat still looks like a cat with slightly rounded weights.

**The payoff:**
4x size reduction + 2–4x speed gain on the NPU. On an ECU with limited flash storage, it is not optional — it is the only way to deploy.

**Key phrase to remember:**
> Gradient updates need decimal precision — inference doesn't.

---

## 5. What is Transfer Learning and Why Does it Work?

**What it is:**
You take a model already trained on a large dataset (ResNet/MobileNet trained on 1.2M ImageNet images) and adapt it for your specific task — instead of training from scratch.

**Why it works:**
A CNN learns in layers. Early layers learn universal features — edges, curves, textures — that are the same regardless of the task. A cat image and a pedestrian image both have edges and textures. These don't need to be relearned.

Only the final layer is task-specific. You retrain just that layer for your classes.

**The result:**
With only 450 images you get 90%+ accuracy — because you are standing on top of 1.2 million images worth of learning.

**OTA parallel:**
You don't rewrite entire ECU firmware to update one feature. You update only the component that changed. Transfer learning does the same — retrain only what needs to change.

**One line answer:**
> Transfer learning works because early CNN layers learn universal visual features — edges, textures, shapes — that transfer across any visual task, so we only retrain the final classification layer for our specific problem.

---

## 6. How do you Detect Overfitting from a Loss Curve?

**What overfitting looks like:**

| Epoch | train_loss | valid_loss |
|-------|-----------|------------|
| 1 | 1.2 | 1.1 | ← close, good |
| 2 | 0.7 | 0.8 | ← still learning |
| 3 | 0.3 | 1.4 | ← gap appearing |
| 4 | 0.1 | 2.1 | ← clear overfitting |

Train loss keeps falling. Val loss starts rising. The gap is the signal.

**What it means:**
The model is memorising training data instead of learning generalisable patterns. It performs perfectly on data it has seen but fails on new data.

**How to fix it:**
More training data, data augmentation, dropout, early stopping, reduce model complexity.

**One line answer:**
> Overfitting shows on the loss curve when training loss continues to decrease but validation loss starts increasing — the gap between them means the model is memorising, not learning.

---

## 7. What is Weight Sharing in a CNN and Why Does it Matter?

**The problem with regular NN:**
A regular NN has separate weights for every pixel position. If it learns to detect an eye at position (50, 60), that knowledge is locked to that position. A different set of weights handles position (100, 120). The model must relearn the same feature for every location. Doesn't scale.

**What CNN does:**
One filter (e.g. 3×3) slides across the entire image. The same weights scan every position. If the filter learned to detect a vertical edge, it detects vertical edges everywhere — top-left, bottom-right, centre.

**Why it matters:**

- **Fewer parameters** — one filter for the whole image instead of separate weights per position. MobileNet has 3.4M weights instead of hundreds of millions.
- **Position invariance** — a cat is a cat whether it is in the corner or the centre. The model does not need to relearn this.
- **Scales to any image size** — same filter works on 224×224 or 1080×1920.

**One line answer:**
> Weight sharing means the same filter weights are applied at every spatial position — so features learned in one location automatically generalise everywhere, reducing parameters dramatically and making CNNs practical for real images.

---

## Quantization Numbers to Know Cold

| Model | Size | Reduction |
|-------|------|-----------|
| MobileNetV2 FP32 | 13.6 MB | baseline |
| Dynamic INT8 (Linear only) | 9.9 MB | 1.4x |
| Static INT8 (Conv + Linear) | 3.8 MB | 3.6x |

---

---

## 8. Why did Dynamic Quantization not improve inference speed on Apple M4?

**The answer:**
Dynamic quantization only targets **Linear layers**. MobileNetV2's compute is dominated by **Conv layers** — depthwise separable convolutions. Quantizing only the tiny Linear classifier at the end leaves the heavy computation unchanged.

Apple M4 also has extremely fast FP32 arithmetic and high memory bandwidth, so the memory savings from INT8 Linear weights don't translate to a measurable latency difference.

**The contrast — what WOULD happen on automotive silicon:**
On a Cortex-A55 based ECU or dedicated NPU, memory bandwidth is severely constrained and INT8 hardware accelerators are present. In that environment, proper INT8 quantization of Conv layers yields 2–4x real speedup.

**One line answer:**
> Dynamic quantization sped up only the Linear layers — which are 5% of MobileNetV2's compute. The 95% that matters (Conv) was still FP32. Zero speedup on M4's fast FP32 silicon; real speedup would appear on memory-constrained automotive NPUs.

---

## 9. Why does Standard Deviation matter in automotive inference benchmarking?

**The numbers from our experiment:**
- FP32: 16.22ms ± 2.09ms (noisy)
- Dynamic INT8: 15.98ms ± 0.33ms (consistent)

**Why consistency = safety:**
In hard real-time systems (ADAS, autonomous driving), missing a deadline is not just a performance problem — it is a safety violation. At 100 km/h, variance of ±2ms means the perception system sometimes updates in 14ms, sometimes in 18ms. That unpredictability makes it harder to design safe timing margins.

A model that runs in 16ms every single time is safer to certify than one that sometimes hits 14ms but occasionally spikes to 19ms.

**One line answer:**
> In automotive real-time systems, timing consistency (low std deviation) is a safety property, not just a performance metric — unpredictable latency makes deadline guarantees impossible.

---

## 10. Why does Static Quantization of MobileNetV2 throw NotImplementedError?

**The cause:**
MobileNetV2 uses inverted residual blocks with skip connections. Inside each block there is a `torch.add()` operation — it adds the block input to the block output (residual addition). Static quantization needs to quantize every operation including this `add`. But `torch.add()` on tensors with different quantization scales cannot be directly quantized without using `FloatFunctional`.

**The fix in research settings:**
Replace `torch.add()` with `torch.nn.quantized.FloatFunctional().add()` — this allows the quantizer to handle the addition correctly. Requires modifying the model architecture.

**The fix in production:**
TensorRT (NVIDIA), SNPE/QNN (Qualcomm), and TFLite handle this automatically at compile time. You don't manually patch the model — the compiler does it.

**Interview insight:**
> Hitting this error shows you understand that model architecture and quantization are not independent — certain ops require explicit support. This is why production compilers exist; manual PyTorch quantization breaks on real architectures.

---

---

## 11. What is an Edge AI Inference Pipeline and why does automotive need it?

**Definition:**
End-to-end data flow on an automotive SoC — from raw sensor capture through pre-processing, model inference, and post-processing — producing actionable outputs within a hard real-time latency budget. Runs entirely on-device. No cloud, no network dependency.

**Why not cloud:**
- Camera runs at 30fps → 33ms per frame (1000ms ÷ 30)
- Cloud round trip = 100–500ms minimum
- At 100 km/h = 27.7 m/s → 100ms = 2.77 metres of blindness
- Safety violation, not performance issue

**One line answer:**
> Edge AI runs inference on the vehicle's SoC because cloud latency physically cannot meet the 33ms per frame budget at highway speeds — missing that deadline means the car is blind for metres, not milliseconds.

---

## 12. Walk me through the complete Edge AI inference pipeline on an automotive SoC.

**The full pipeline:**
```
Camera Sensor (raw Bayer data)
     ↓ DMA — moves to RAM without CPU
ISP — demosaic → RGB, noise reduction, white balance
     ↓ Pre-processing — resize to 224×224, normalize 0–1
NPU — model loaded at boot, frame loaded at runtime
     ↓ Inference — forward pass, bounding boxes output
NMS — removes duplicate overlapping boxes
     ↓
SOME/IP → ADAS Decision Engine
CAN/CAN FD → Actuators (brake, steer)
```

**Key points to hit:**
- DMA moves camera data — CPU can't handle 187MB/s at 30fps
- ISP is dedicated hardware — microseconds vs milliseconds on CPU
- Model loads ONCE at boot — not every frame (saves latency)
- NMS cleans up duplicate detections before output
- SOME/IP for perception data, CAN for safety-critical actuator commands

---

## 13. Why is DMA used instead of CPU to move camera data?

**The math:**
- 1080p @ 30fps = 1920 × 1080 × 3 bytes × 30 = ~187 MB/s of continuous data
- If CPU handled this: 100% of CPU cycles spent copying bytes, nothing left for inference
- DMA is dedicated hardware — moves data between camera and RAM autonomously
- CPU gets an interrupt when frame is ready, then proceeds with processing

**One line answer:**
> At 187MB/s continuous throughput, CPU-driven data transfer would consume all available cycles — DMA offloads this entirely, freeing CPU and NPU for actual compute work.

---

## 14. Why does the model load at boot time, not every frame?

**The reason:**
- A neural network like MobileNetV2 = 13.6MB of weights
- Loading 13.6MB from storage to NPU memory takes 50–200ms
- At 30fps, you have 33ms per frame
- Loading model every frame = impossible — 200ms > 33ms budget

**The solution:**
- Load model once during ECU boot (happens before car starts driving)
- Model stays resident in NPU memory
- Each frame: only the image data (1, 3, 224, 224 tensor) gets loaded — kilobytes, not megabytes

---

## Key Terms — Know These Cold

| Term | Definition |
|------|-----------|
| FP32 | 32-bit float, 4 bytes, decimal precision, used for training |
| INT8 | 8-bit integer, 1 byte, whole numbers only, used for inference |
| Calibration | Running ~100 real images to measure activation value ranges per layer |
| Depthwise separable convolutions | MobileNet's core innovation — 8–9x fewer calculations than standard Conv |
| Weight sharing | Same filter weights used at every spatial position in CNN |
| Transfer learning | Borrowing pretrained model weights and retraining only the final layer |
| Overfitting | Train loss down, val loss up — memorisation not learning |
| model.eval() | Inference mode — dropout off, all neurons active, deterministic |
| Discriminative LR | Different learning rates per layer — low for early, high for final |
| FloatFunctional | PyTorch class needed to quantize `torch.add()` in residual blocks |
| TensorRT | NVIDIA's production inference compiler — handles quantization, layer fusion, residual ops automatically |
| SNPE/QNN | Qualcomm's neural processing SDK — equivalent to TensorRT for Snapdragon NPUs |
| perf_counter | Python's nanosecond-resolution timer — always use this for latency benchmarking |
| Warmup runs | First 10 inference passes discarded — CPU cache cold, JIT not settled |
| DMA | Direct Memory Access — moves data between peripherals and RAM without CPU |
| ISP | Image Signal Processor — dedicated hardware converts raw Bayer to clean RGB |
| Demosaicing | Reconstructs full RGB from single-colour-per-pixel Bayer sensor data |
| NMS | Non-Maximum Suppression — removes duplicate bounding boxes, keeps highest confidence |
| Bayer pattern | Raw sensor format — one colour (R/G/B) per pixel, not full RGB |
| Edge AI | AI inference running on-device (SoC), not in cloud |
| SOME/IP | Automotive Ethernet protocol for service-oriented communication (e.g. perception → ADAS) |
| Boot vs Runtime | Model loads at boot (once), frame loads at runtime (every 33ms) |
| Boyer-Moore | O(n) time O(1) space majority element algorithm — candidate + count, cancel pairs |
