17ms → 4ms. Same model. Same weights. No retraining.
Three ways to run YOLOv8n this week:
ONNX Runtime CPU — 37.4ms
PyTorch MPS (Apple M4) — 17.10ms
TensorRT (NVIDIA T4) — 4.00ms
Same model. Same .onnx file as input. Only the backend changes.
Here's what these numbers mean in an actual ADAS system:
At 30fps you have 33ms per frame. That's the entire budget — object detection, sensor fusion, planning, safety check, actuator command. All of it.
37ms means you're over budget before the car has done anything else.
17ms leaves 16ms for everything downstream.
4ms leaves 29ms. Enough to run a second model. Or a third.
Most engineers I see stop at ONNX Runtime or PyTorch and then spend weeks trying to optimize the model itself — pruning layers, reducing input resolution, changing architecture. The latency problem isn't in the model. It's in the execution.
TensorRT doesn't run the model. It compiles it. Fuses layers. Picks hardware-specific kernels. Rewrites the execution graph for the target GPU. The .onnx file is 12.3MB and runs anywhere. The compiled engine runs on one GPU only.
That's the tradeoff: portability vs performance. In production automotive, you pick performance.
On DRIVE Orin, 4ms on detection means you're using 12% of your frame budget to see the world. The other 88% is left for deciding what to do about it.
Where are you in the pipeline? ONNX Runtime, PyTorch, or already on TensorRT?