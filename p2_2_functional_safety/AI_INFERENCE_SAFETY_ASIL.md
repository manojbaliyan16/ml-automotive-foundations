# Designing Safe AI Inference Systems for Automotive ASIL-B
## ISO 26262 Functional Safety Applied to Edge AI Perception Pipelines

### Author: Manoj Kumar | Date: June 2026
### Standard: ISO 26262:2018 Road Vehicles — Functional Safety

---

## The Problem

A YOLOv8n model running on NVIDIA DRIVE Orin detects pedestrians at 30fps. If it fails to detect a pedestrian at highway speed, the vehicle cannot brake in time. That failure scenario is:

- **Severity:** S3 — life-threatening, fatal
- **Exposure:** E4 — highway driving is continuous
- **Controllability:** C3 — driver cannot react in time at >80 km/h

**Result: ASIL-D safety goal.**

But neural networks are statistical systems. YOLOv8n produces a confidence score — not a guarantee. A model with 99.9% mAP still misses 1 in 1000 detections. How do you build an ASIL-D certified system around a component that is inherently probabilistic?

**The answer: you don't certify the model to ASIL-D. You architect around it.**

---

## Core Principle — ASIL Decomposition for AI Systems

ISO 26262 Clause 9 allows a single ASIL-D safety goal to be decomposed into two independent requirements of lower ASIL, provided those requirements are implemented on independent channels.

```
Safety goal: "Vehicle shall not collide with detected pedestrians" = ASIL-D

Decomposed into:
  Channel A: AI Perception Pipeline      → ASIL-B
  Channel B: Safety Monitor + Fallback   → ASIL-B

Independence verified → ASIL-B × ASIL-B = 10⁻⁴ × 10⁻⁴ = 10⁻⁸/hr = ASIL-D ✓
```

The AI model never needs to be ASIL-D. The architecture ensures that even when the model fails, the safety monitor catches it before it causes harm.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DRIVE Orin SoC                               │
│                                                                 │
│  ┌──────────────────────────────┐  ┌────────────────────────┐  │
│  │  CHANNEL A — AI Pipeline     │  │  CHANNEL B — Monitor   │  │
│  │  (ASIL-B)                    │  │  (ASIL-B)              │  │
│  │                              │  │                        │  │
│  │  ISP → DLA (TensorRT INT8)   │  │  Radar cross-check     │  │
│  │  YOLOv8n → NMS → Output     │  │  Physics TTC model     │  │
│  │  Confidence threshold >0.5   │  │  Watchdog timer        │  │
│  │  Runs on: A78AE + DLA        │  │  Runs on: Cortex-R52   │  │
│  │                              │  │  (Safety Island)       │  │
│  └──────────────┬───────────────┘  └──────────┬─────────────┘  │
│                 │                             │                 │
│                 └──────────┬──────────────────┘                 │
│                            ▼                                    │
│              ┌─────────────────────────┐                        │
│              │  SAFE STATE CONTROLLER  │                        │
│              │  (ASIL-D — Safety       │                        │
│              │   Island Cortex-R52)    │                        │
│              │                         │                        │
│              │  Both channels agree?   │                        │
│              │  → Execute normally     │                        │
│              │  Monitor flags fault?   │                        │
│              │  → Trigger safe state   │                        │
│              └─────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

**Independence requirements between Channel A and B:**
- Different processor cores (A78AE vs Cortex-R52)
- Different memory regions (MPU-enforced)
- Different power domains
- No shared software components
- Freedom from Interference enforced by QNX partitioning

---

## Channel A — AI Perception Pipeline (ASIL-B)

### Hardware
- ISP → DLA ×2 (TensorRT INT8) → GPU post-processing
- Inference latency: 4ms on T4 GPU equivalent (TensorRT compiled)
- Within 33ms / 30fps hard deadline

### ASIL-B Compliance Measures

**1. Confidence Threshold Enforcement**
```
Default YOLOv8 threshold: 0.25 (development use only)
Production ADAS threshold: ≥ 0.5

Rationale: A 0.26 confidence detection reaching the brake
controller is a safety risk. Threshold is a safety decision,
not a performance optimization.
```

**2. Input Data Integrity**
- CRC check on DMA transfer from ISP to DLA memory
- Timestamp validation — reject frames older than 2× frame period
- Sensor health monitoring — ISP fault detection before inference

**3. Output Plausibility**
- Bounding box coordinates within valid image dimensions
- Object size physically plausible (pedestrian cannot be 1×1 pixel at 5m)
- Velocity vector consistent with previous frame (continuity check)

**4. Timing Watchdog**
- Inference must complete within 25ms (leaving 8ms margin in 33ms budget)
- If timeout → Channel A raises fault flag to Safety Monitor
- Safety Monitor responds within one frame period

**5. Model Monitoring**
- Periodic forward pass with known test vector — output compared to expected
- Detects DLA hardware degradation over time
- Run every 100 frames (~3.3 seconds at 30fps)

---

## Channel B — Safety Monitor (ASIL-B)

The Safety Monitor operates independently of Channel A. It does not depend on AI output for its own integrity — it provides an independent assessment of the environment.

### Radar Cross-Validation
```
Channel A says: "Pedestrian at 8m, confidence 0.87"
Radar says:     "Object at 8.2m, relative velocity -5 km/h"

Agreement within tolerance → confirmed detection
Channel A says pedestrian, radar sees nothing → flag for review
```

Radar is not affected by lighting, occlusion, or neural network failures. It is the independent sensing channel.

### Physics-Based Time-to-Collision (TTC)
```
TTC = distance / relative_velocity

If TTC < 1.5 seconds AND Channel A has no detection:
  → Safety Monitor triggers emergency brake regardless of AI output
```

This is the fallback: if the AI misses a detection entirely, physics-based TTC using radar alone can still trigger a safe response.

### Watchdog for Channel A
- Monitors Channel A heartbeat signal every frame
- If Channel A stops producing output → fault declared → safe state triggered
- Prevents silent failures where AI crashes without raising an error

### Freedom from Interference
- Runs on Cortex-R52 cores (Safety Island) — physically separate from A78AE
- QNX adaptive partition guarantees CPU time regardless of A78AE load
- Memory Protection Unit prevents any A78AE process from writing to R52 memory space
- Linux on the perception side cannot interfere with QNX on the safety side

---

## Safe State Definition

When Channel B detects a fault in Channel A, or when both channels disagree beyond tolerance, the system must transition to a safe state.

**Fail-Operational Architecture (Level 1 degradation):**
```
Channel A fault detected
→ Disable AI-based perception
→ Enable radar-only minimal risk manoeuvre
→ Alert driver with audible + visual warning
→ Limit speed to 30 km/h
→ Maintain lane
```

**Fail-Safe (Level 2 — both channels unavailable):**
```
Both Channel A and Channel B fail
→ Controlled stop (minimal risk manoeuvre)
→ Hazard lights on
→ Emergency services notification via TCU
```

**Why not hard stop immediately:**
At 100 km/h, a hard stop causes rear-end collisions. The safe state must itself be safe. Controlled deceleration and driver handover is the ASIL-D-compliant approach for highway speeds.

---

## ASIL Decomposition Validity

For decomposition to be valid under ISO 26262 Clause 9:

| Requirement | Status |
|---|---|
| Independent hardware execution contexts | ✅ A78AE vs Cortex-R52 |
| Independent memory regions | ✅ MPU-enforced |
| Independent power domains | ✅ Separate power rails |
| Freedom from Interference | ✅ QNX partitioning + MPU |
| Independent sensors | ✅ Camera (Ch A) + Radar (Ch B) |
| No shared software | ✅ Different OS partitions |
| Diagnostic coverage >90% | ✅ Watchdog + heartbeat + CRC |

**No common cause failure exists between Channel A and Channel B.**

---

## Why Linux Alone is Insufficient for Safety Channels

Linux does not provide Freedom from Interference:
- Shared kernel — kernel panic affects all processes simultaneously
- No guaranteed CPU time partitioning — safety monitor can be starved
- Not ASIL certified — no formal proof of spatial/temporal isolation
- Dynamic memory allocation — unpredictable behavior under memory pressure

**For ASIL-B safety functions, a certified RTOS (QNX) or hypervisor with ASIL-certified partitioning is required.**

Linux is acceptable for QM functions (telematics, OTA, data logging) running alongside the safety partition, provided Freedom from Interference is enforced at the hypervisor/hardware level.

---

## Benchmark Reference

| Stage | Latency | Hardware |
|---|---|---|
| ISP processing | ~1ms | Dedicated ISP block |
| DLA inference (YOLOv8n INT8) | ~4ms | DLA ×2 |
| GPU post-processing (NMS) | ~2ms | GPU |
| Safety Monitor check | ~1ms | Cortex-R52 |
| Total perception to decision | **~8ms** | — |
| Remaining budget (33ms frame) | **25ms** | For fusion + planning + actuation |

4ms on a T4 GPU (Colab benchmark). On DRIVE Orin DLA with INT8: target ≤5ms deterministic latency.

---

## Key Takeaways

1. **AI models cannot be directly certified to ASIL-D** — they are statistical, not deterministic.

2. **ASIL decomposition is the architectural solution** — ASIL-B AI channel + ASIL-B safety monitor = ASIL-D system.

3. **Independence is non-negotiable** — shared hardware, shared power, or shared software creates common cause failures that invalidate decomposition.

4. **Freedom from Interference is the implementation guarantee** — QNX + MPU + separate cores make independence provable, not assumed.

5. **The safety monitor must work without the AI** — radar + physics-based TTC provides the fallback path that makes the system fail-operational.

6. **Safe state is designed, not default** — a hard stop at 100 km/h is also dangerous. Safe state = controlled deceleration + driver alert + speed limit.

---

*Standard: ISO 26262:2018 Road Vehicles — Functional Safety*
*Reference hardware: NVIDIA DRIVE Orin (167 TOPS GPU + 87 TOPS DLA ×2 + Safety Island 4× Cortex-R52)*
*Reference model: YOLOv8n — TensorRT INT8, 4ms latency on T4 GPU*
