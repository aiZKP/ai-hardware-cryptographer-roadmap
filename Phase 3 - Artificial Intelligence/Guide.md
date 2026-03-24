# Phase 3: Artificial Intelligence

Standalone phase after **Phase 1** (digital + HDL, architecture, OS, C++/CUDA) and **Phase 2** (PCB, MCU/RTOS, embedded Linux): learn **what** networks compute, **where** they run on-device, **how** classic vision pipelines work, and **multi-sensor perception** (calibration, filtering, fusion) before you map workloads to **Phase 4 Track A** (Xilinx FPGA) or **Track B** (Jetson).

| Topic | Guide |
|-------|--------|
| **Neural networks** | [Neural Networks/Guide.md](Neural Networks/Guide.md) — MLPs, CNNs, training, tinygrad/PyTorch, [pytorch-and-micrograd/](Neural Networks/pytorch-and-micrograd/Guide.md) |
| **Edge AI** | [Edge AI/Guide.md](Edge%20AI/Guide.md) — on-device tiers, latency/privacy, train → optimize → deploy pipeline (complements Neural Networks) |
| **Computer vision** | [Computer Vision/Guide.md](Computer Vision/Guide.md) — image processing, detection, OpenCV; pairs with Track A (FPGA) or Track B (Jetson) in Phase 4 |
| **Sensor fusion** | [Sensor Fusion/Guide.md](Sensor%20Fusion/Guide.md) — camera/LiDAR/IMU, Kalman, BEVFusion, MOT; Jetson/ROS2 labs tie to **Phase 4 Track B** |

**Suggested order:** Neural networks (theory + tinygrad) → Edge AI (deployment context) → Computer vision (or CV in parallel once tensors are familiar) → Sensor fusion (often after CV; overlaps with Phase 4 Jetson when you run stacks on hardware).

**Previous:** [Phase 1 §4 — C++ and Parallel Computing](../Phase 1 - Foundational Knowledge/4. C++ and Parallel Computing/Guide.md) · **Next:** [Phase 4 Track A — Xilinx FPGA](../Phase 4 - Track A - Xilinx FPGA/1. Xilinx FPGA Development/Guide.md) or [Phase 4 Track B — Jetson](../Phase 4 - Track B - Nvidia Jetson and Edge AI/1. Nvidia Jetson Platform/Guide.md).
