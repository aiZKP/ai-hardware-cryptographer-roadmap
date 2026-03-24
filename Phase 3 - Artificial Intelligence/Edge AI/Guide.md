# Edge AI

**Phase 3 — Artificial Intelligence** (standalone topic). Learn **where** and **why** models run on-device; learn **what** they compute in **[Neural Networks](../Neural%20Networks/Guide.md)**.

> **Goal:** Map the edge stack—latency, privacy, power tiers, and the train → optimize → deploy pipeline—so Phase 4 (Xilinx or Jetson) and specialization tracks have clear context.

**Previous:** [Phase 1 §4 — C++ and Parallel Computing](../../Phase 1 - Foundational Knowledge/4. C++ and Parallel Computing/Guide.md) · **Companion:** [Neural Networks](../Neural%20Networks/Guide.md) (MLPs, CNNs, training, tinygrad) · **Next (deployment depth):** [Phase 4 Track B — Jetson](../../Phase 4 - Track B - Nvidia Jetson and Edge AI/1. Nvidia Jetson Platform/Guide.md)

---

## Table of Contents

1. [Definition](#1-definition)
2. [Why edge AI exists](#2-why-edge-ai-exists)
3. [Where edge AI runs (tiers)](#3-where-edge-ai-runs-tiers)
4. [The edge AI pipeline](#4-the-edge-ai-pipeline)
5. [How this fits the roadmap](#5-how-this-fits-the-roadmap)

---

## 1. Definition

**Edge AI** = running AI algorithms **locally on a device** (the "edge") instead of sending data to a remote cloud server.

```
Traditional Cloud AI:
  Device → [internet] → Cloud Server (GPU farm) → [internet] → Result
  Latency: 50–300ms   Privacy risk   Needs connectivity

Edge AI:
  Device → Local Chip (CPU/GPU/NPU/FPGA) → Result
  Latency: <1ms       Data stays local   Works offline
```

---

## 2. Why edge AI exists

| Problem with Cloud AI        | Edge AI Solution                        |
|------------------------------|-----------------------------------------|
| Network latency (~100ms)     | Sub-millisecond local inference         |
| Bandwidth cost (video data)  | Only send results, not raw data         |
| Privacy (face/voice/medical) | Data never leaves the device            |
| Reliability (no internet)    | Works fully offline                     |
| Cloud cost at scale          | One-time hardware cost                  |

---

## 3. Where edge AI runs (tiers)

```
Tier 1 — Microcontrollers (MCU):
  STM32, Arduino, RP2040
  RAM: 256KB–512KB
  Power: <1W
  Use: keyword spotting, gesture detection

Tier 2 — Embedded Linux SBCs:
  Raspberry Pi, BeagleBone
  RAM: 1–8GB
  Power: 2–10W
  Use: image classification, object detection

Tier 3 — AI Accelerator SoCs:
  Nvidia Jetson, Google Coral (TPU), Apple Neural Engine
  RAM: 4–64GB
  Power: 5–30W
  Use: real-time video inference, NLP, robotics

Tier 4 — Edge Servers:
  FPGA + GPU combinations, industrial PCs
  Power: 50–300W
  Use: factory automation, autonomous vehicles
```

---

## 4. The edge AI pipeline

```
1. Train model on a powerful workstation/cloud (large data, many epochs)
2. Optimize model for edge (quantization, pruning, distillation)
3. Convert model to edge runtime format (ONNX, TensorRT, TFLite)
4. Deploy to edge device
5. Run inference locally in real-time
```

Step 1 is grounded in **[Neural Networks](../Neural%20Networks/Guide.md)**. Steps 2–5 are expanded in **Phase 4 Track B** ([Edge AI Optimization](../../Phase 4 - Track B - Nvidia Jetson and Edge AI/2. Edge AI Optimization/Guide.md)) and, for custom silicon, **Phase 4 Track A** and **Phase 5 — AI Chip Design**.

---

## 5. How this fits the roadmap

| You want… | Start here | Then |
|-----------|------------|------|
| Intuition for tensors, backprop, CNNs | [Neural Networks](../Neural%20Networks/Guide.md) | tinygrad hands-on, [pytorch-and-micrograd](../Neural%20Networks/pytorch-and-micrograd/Guide.md) |
| Product and deployment context (this guide) | Skim tiers + pipeline above | Phase 4 Jetson or FPGA track |
| Vision preprocessing and classical CV | [Computer Vision](../Computer%20Vision/Guide.md) | Phase 4 perception pipelines |
| Multi-sensor calibration, tracking, BEV fusion | [Sensor Fusion](../Sensor%20Fusion/Guide.md) | Phase 4 Jetson + [ROS2](../../Phase 4 - Track B - Nvidia Jetson and Edge AI/3. ROS2/Guide.md) for integration |

**Hub:** [Phase 3 — Artificial Intelligence](../Guide.md)
