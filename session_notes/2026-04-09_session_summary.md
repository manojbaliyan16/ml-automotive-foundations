# Session Summary — 9 April 2026
> Use this for revision. Read before starting next session.

---

## What We Completed Today

### P1.1 — CLOSED ✓
- Image classifier (ResNet34 fine-tuned, cats/dogs/birds) — live on GitHub
- Digit recognizer (CNN from scratch, MNIST) — live on GitHub
- Repo: github.com/manojbaliyan16/ml-automotive-foundations

### P1.2 — Started (3 of 4 cells done)
- Notebook: `edge-inference/p1_2_edge_inference_quantization.ipynb`
- Cell 1: FP32 baseline measured
- Cell 2: Dynamic quantization applied
- Cell 3: Static quantization applied
- **Next session starts at Cell 4: Inference speed benchmark**

### DSA — 5 Problems Done
- All in C++, pushed to github.com/manojbaliyan16/manoj-algorithms-systematic
- Concatenation of Array, Contains Duplicate, Valid Anagram, Two Sum, Longest Common Prefix
- Following NeetCode 250 — Arrays & Hashing pattern

### Calendar
- DSA block set: Mon–Fri 7:45–9:00 PM IST (recurring)
- System Design set: Every Saturday 9:00–9:45 AM IST (recurring)

---

## Key Numbers to Remember

| Model | Size | Reduction |
|-------|------|-----------|
| MobileNetV2 FP32 | 13.6 MB | baseline |
| Dynamic INT8 (Linear only) | 9.9 MB | 1.4x |
| Static INT8 (Conv + Linear) | 3.8 MB | 3.6x |

---

## Concepts You Learned Today — Explain These Without Notes

### 1. CNN vs Regular NN
A regular NN has separate weights for every pixel position — what it learns at one location cannot be used elsewhere. A CNN uses the same filter weights everywhere (weight sharing) — so if it learns to detect an eye, it detects that eye anywhere in the image.

### 2. Transfer Learning
ResNet/MobileNet was trained on 1.2 million ImageNet images. It already knows edges, textures, shapes, object parts. You borrow this knowledge and retrain only the final layer for your classes. You don't need millions of images because you're standing on top of someone else's training.

### 3. Fine-Tuning — Two Phases
- **Phase 1 (frozen):** Only new final layer trains. High LR (1e-3). Backbone weights untouched.
- **Phase 2 (unfrozen):** All layers train with low LR (1e-4). Small nudges to pretrained weights.
- Why low LR in Phase 2? Pretrained weights are valuable — you nudge them, not overwrite them.

### 4. Discriminative Learning Rates
`slice(1e-6, 6e-5)` — early layers get tiny LR, final layers get full LR. Early layers already know universal features. Final layers need more adaptation.

### 5. Loss Function (CrossEntropyLoss)
Measures how wrong the model is. If model says cat=0.1 (but answer is cat), loss = -log(0.1) = 2.3 — high penalty. If model says cat=0.9, loss = -log(0.9) = 0.1 — small penalty. Backpropagation uses this to adjust weights.

### 6. Overfitting
Train loss going down, val loss going up = model memorising training data, not learning patterns. Fix: more data, augmentation, dropout, early stopping.

### 7. lr_find()
Shows loss vs learning rate curve. Pick the steepest downward slope — NOT the minimum. Minimum = cliff edge. Steep slope = optimal learning speed.

### 8. model.eval()
Switches to inference mode. Turns off dropout (all neurons active). BatchNorm uses fixed statistics. Result: deterministic predictions every time. Without it — same image gives different predictions each run.

### 9. FP32 vs INT8
- FP32 = 4 bytes per value, decimal precision, used for training (gradients need decimals)
- INT8 = 1 byte per value, whole numbers only, used for inference (precision loss acceptable)
- 4x fewer bytes = 4x smaller model = fits on ECU

### 10. Dynamic Quantization
Converts Linear layer weights from FP32 to INT8 at runtime. No calibration needed. Activations quantized on the fly. Result: 1.4x size reduction. Easy but limited.

### 11. Static Quantization
Runs ~100 calibration images through model first. Records actual activation value ranges per Conv layer. Then converts all weights to INT8 with informed ranges. Result: 3.6x size reduction. More effort, much better compression.

### 12. Why Calibration?
Conv layer activations vary depending on input image. You need real data to measure the typical range before mapping to INT8. Without calibration = guessing the range = clipping values = accuracy loss. Same as your OTA validation — test before deploying.

### 13. copy.deepcopy()
Preserves original FP32 model untouched. Quantize only the copy. Same principle as OTA — never modify the running firmware in place. Keep the original safe until the new version is validated.

---

## Interview Questions You Can Now Answer

1. What is the difference between dynamic and static quantization?
2. Why does automotive AI use MobileNet-style architectures instead of ResNet?
3. What happens at 120 km/h if inference takes 100ms instead of 50ms?
4. Why do we train in FP32 but deploy in INT8?
5. What is transfer learning and why does it work?
6. How do you detect overfitting from a loss curve?
7. What is weight sharing in a CNN and why does it matter?

---

## What's Pending — Do Tonight

1. **NVIDIA application — JR1985837 + JR1983469. Deadline TOMORROW April 8.**
   - URL: nvidia.wd5.myworkdayjobs.com
   - 15 minutes. Non-negotiable.

2. **DSA — NeetCode 250 (7:45 PM block)**
   - Remove Element
   - Majority Element

---

## Next Session Agenda

1. Read CLAUDE.md + Excel sheet first
2. P1.2 Cell 4 — Inference speed benchmark (FP32 vs Dynamic vs Static latency in ms)
3. Connect latency numbers to real automotive use case
4. Push completed notebook to GitHub
5. Continue NeetCode 250 — Design HashSet, Design HashMap
