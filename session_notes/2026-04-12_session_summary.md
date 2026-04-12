# Session Summary — April 12, 2026 (Sunday)

## DSA Completed Today

### 1. Majority Element — Two Approaches

**Approach 1: HashMap — O(n) time, O(n) space**
```cpp
int majorityElement(vector<int>& nums) {
    int n = nums.size();
    unordered_map<int, int> mymap;
    for(int i = 0; i < n; ++i) mymap[nums[i]]++;
    int maj_element = 0;
    for(auto it = mymap.begin(); it != mymap.end(); ++it)
        if(it->second > n/2) maj_element = it->first;
    return maj_element;
}
```

**Approach 2: Boyer-Moore Voting Algorithm — O(n) time, O(1) space**
```cpp
int majorityElement(vector<int>& nums) {
    int n = nums.size();
    int candidate = 0, count = 0;
    for(int i = 0; i < n; ++i) {
        if(count == 0) {
            candidate = nums[i];
            count = 1;
        } else if(candidate == nums[i]) {
            count++;
        } else {
            count--;
        }
    }
    return candidate;
}
```

**Key insight:** Boyer-Moore works because majority element appears > n/2 times — it can never be fully cancelled out by non-majority elements.

**Interview edge case to mention:** If no majority element is guaranteed, add a validation pass at end to confirm candidate exceeds n/2 — otherwise return -1.

---

### 2. Design HashSet — O(1) all operations

**Approach: Boolean array (works because keys bounded 0–1,000,000)**
```cpp
class MyHashSet {
public:
    bool arr[1000001] = {false};
    MyHashSet() {}
    void add(int key) { arr[key] = true; }
    void remove(int key) { arr[key] = false; }
    bool contains(int key) { return arr[key]; }
};
```

**Follow-up to know:** If keys were unbounded — use hash table with buckets + linked lists for collision handling. `key % bucket_size` as hash function.

---

## System Design — Edge AI Inference Pipeline (Taught Today)

### What is Edge AI Inference Pipeline?
End-to-end data flow on an automotive SoC — from raw sensor capture through pre-processing, model inference, and post-processing — producing actionable outputs within a hard real-time latency budget. No cloud. No network dependency.

### Why Edge, Not Cloud?
- At 100 km/h = 27.7 m/s
- Camera runs at 30fps → 33ms per frame (1000ms ÷ 30)
- Cloud round trip = 100–500ms minimum
- 100ms cloud latency = 2.77 metres of blindness
- Safety violation, not just performance issue

### The Complete Pipeline

```
Camera Sensor (raw Bayer data — RAW10/RAW12)
     ↓
DMA Controller (moves data to RAM — no CPU involvement)
     ↓
Frame Buffer in RAM
     ↓
ISP — Image Signal Processor (dedicated hardware)
   - Demosaicing: Bayer pattern → RGB
   - Noise reduction
   - White balance
   - Tone mapping
     ↓
Pre-processing (CPU/GPU)
   - Resize: 1920×1080 → 224×224
   - Normalize: pixel values 0–255 → 0.0–1.0 (÷255)
     ↓
Tensor (1, 3, 224, 224) in RAM
     ↓
NPU — Neural Processing Unit
   - Model loaded at BOOT TIME (not every frame — saves latency)
   - Frame loaded at RUNTIME (every 33ms)
   - Runs inference: forward pass only, no weight updates
     ↓
Raw detections: [x, y, width, height, confidence, class]
     ↓
NMS — Non-Maximum Suppression (post-processing)
   - Removes overlapping duplicate bounding boxes
   - Keeps highest confidence detection per object
     ↓
Clean detections
     ↓
SOME/IP (Ethernet) → ADAS Decision Engine
CAN/CAN FD       → Actuators (brake, steer) — safety critical
```

### Key Concepts Learned

| Concept | Explanation |
|---------|-------------|
| DMA | Moves camera data to RAM without CPU — at 187MB/s CPU can't keep up |
| ISP | Dedicated hardware converts raw Bayer to clean RGB — microseconds vs ms on CPU |
| Demosaicing | Reconstructs full RGB from single-colour-per-pixel sensor |
| Boot vs Runtime loading | Model loads once at boot. Frame loads every 33ms |
| Inference | Model making prediction on new data — forward pass only, no backprop |
| NMS | Removes duplicate bounding boxes — keeps highest confidence per object |
| SOME/IP | Service-oriented protocol over Ethernet — perception to ADAS |
| CAN for actuators | Deterministic, guaranteed latency — required for safety-critical brake/steer commands |

### Input tensor dimensions (from P1.2 knowledge)
```
torch.randn(1, 3, 224, 224)
            ↑  ↑   ↑   ↑
            │  │   │   └── width: 224 pixels
            │  │   └─────── height: 224 pixels
            │  └─────────── channels: 3 (R, G, B)
            └────────────── batch size: 1 image
```

---

## Pending Actions

- [ ] Apply to NVIDIA JR1985837 + JR1983469 — TOMORROW (Monday April 13) — 4 days past deadline
- [ ] Push DSA solutions to GitHub (Majority Element both approaches + Design HashSet)
- [ ] Refine resume per JD for NVIDIA + company list (booked for tomorrow)
- [ ] Next Sunday: Draw Edge AI inference pipeline from memory — no notes
- [ ] P1.3 starts next week — NPU Architecture + Automotive SoC Deep Dive

---

## NVIDIA Status
- Deadline: April 8 (PASSED — 4 days ago)
- Applications: NOT submitted yet
- Tomorrow is the absolute last chance — portal may close any day
