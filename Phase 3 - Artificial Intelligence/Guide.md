# Phase 3: Artificial Intelligence — The Workloads Your Hardware Must Run

> *Before you design hardware, you must deeply understand the software it accelerates.*

**Layer mapping:** **L1** (Application & Framework) — this entire phase teaches you what AI chips compute.

**Prerequisites:** Phase 1 (Digital Foundations), Phase 2 (Embedded Systems).

**What comes after:** Phase 4 Track A (FPGA), Track B (Jetson), Track C (ML Compiler).

---

## Why This Phase Exists

Every decision in the 8-layer stack is driven by workload requirements. If you skip this phase, you'll design hardware without knowing what it needs to run. Phase 3 gives you the workload intuition that informs every hardware decision downstream.

---

## Structure: Core + Two Tracks

**Modules 1–2** are mandatory for everyone. Then you choose **Track A**, **Track B**, or both.

```
Module 1: Neural Networks          ← mandatory (what accelerators compute)
Module 2: Deep Learning Frameworks ← mandatory (micrograd, PyTorch, tinygrad)
        ↓                    ↓
   Track A                Track B
   Hardware &             Agentic AI &
   Edge AI                ML Engineering
        ↓                    ↓
   Phase 4               Phase 4 or
   (FPGA/Jetson/          Phase 5
    Compiler)             (HPC/GenAI)
```

---

## Core Modules (Mandatory)

| # | Module | What you learn | Why it matters for hardware |
|---|--------|---------------|---------------------------|
| **1** | [Neural Networks](1.%20Neural%20Networks/Guide.md) | MLPs, CNNs, training, backpropagation, loss functions | What accelerators compute — tensors, matmul, activations |
| **2** | [Deep Learning Frameworks](2.%20Deep%20Learning%20Frameworks/Guide.md) | [micrograd](2.%20Deep%20Learning%20Frameworks/micrograd/Guide.md) → [PyTorch](2.%20Deep%20Learning%20Frameworks/PyTorch/Guide.md) → [tinygrad](2.%20Deep%20Learning%20Frameworks/tinygrad/Guide.md): autograd, ops, compiler pipeline | How software generates workloads — the interface between models and hardware |

---

## Track A — Hardware & Edge AI

> *For engineers heading to Phase 4 (FPGA, Jetson, ML Compiler) and Phase 5 (Autonomous Vehicles, AI Chip Design).*

This track teaches the **perception and deployment workloads** that drive edge inference hardware.

| # | Module | What you learn | Leads to |
|---|--------|---------------|----------|
| **3** | [Computer Vision](Track%20A%20-%20Hardware%20and%20Edge%20AI/3.%20Computer%20Vision/Guide.md) | Image processing, detection, segmentation, 3D vision, OpenCV | Phase 4A (FPGA vision), Phase 5E (AV perception) |
| **4** | [Sensor Fusion](Track%20A%20-%20Hardware%20and%20Edge%20AI/4.%20Sensor%20Fusion/Guide.md) | Camera/LiDAR/IMU, Kalman filtering, BEVFusion, MOT | Phase 4B (Jetson + ROS2), Phase 5E (AV) |
| **5** | [Edge AI & Model Optimization](Track%20A%20-%20Hardware%20and%20Edge%20AI/5.%20Edge%20AI%20and%20Model%20Optimization/Guide.md) | Quantization, pruning, knowledge distillation, deployment pipeline | Phase 4 (bridge to all hardware tracks) |
| **6** | [Voice AI](Track%20A%20-%20Hardware%20and%20Edge%20AI/6.%20Voice%20AI/Guide.md) | STT (Whisper), TTS (VITS/Piper), VAD, keyword spotting, noise suppression | Phase 4A (FPGA audio DSP), Phase 4B (Jetson voice pipeline) |

**Build:** OpenCV detection, sensor calibration, INT8 quantization, tinygrad on-device inference, Whisper on Jetson, edge voice pipeline (VAD→STT→TTS).

---

## Track B — Agentic AI & ML Engineering

> *For engineers heading to Phase 5 (HPC, GPU Infrastructure) or building AI applications that generate the inference demand your hardware serves.*

This track teaches the **AI application and infrastructure workloads** — the demand side of the chip market.

| # | Module | What you learn | Leads to |
|---|--------|---------------|----------|
| **3** | [Agentic AI & GenAI](Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/3.%20Agentic%20AI%20and%20GenAI/Guide.md) | LLM agents, RAG pipelines, tool use, multi-step reasoning, GenAI products | Phase 5A/B (GPU Infrastructure, HPC) |
| **4** | [ML Engineering & MLOps](Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/4.%20ML%20Engineering%20and%20MLOps/Guide.md) | Training pipelines, experiment tracking, model serving, CI/CD for models | Phase 5A/B (HPC, distributed training) |
| **5** | [LLM Application Development](Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/5.%20LLM%20Application%20Development/Guide.md) | Prompt engineering, fine-tuning, RAG architecture, evaluation, production deployment | L1d/L1e roles (highest job volume) |

**Build:** RAG pipeline with vector search, agent with tool calling, fine-tune a small LLM, deploy model behind Triton/vLLM.

---

## Why Two Tracks?

| | Track A (Hardware & Edge AI) | Track B (Agentic AI & ML Eng) |
|---|---|---|
| **Goal** | Understand workloads that run on your chip | Understand workloads that create inference demand |
| **Focus** | Perception, sensors, optimization, deployment | LLMs, agents, training pipelines, serving |
| **Hardware connection** | Direct — you deploy on FPGA/Jetson/NPU | Indirect — you generate the traffic the chip serves |
| **Job market** | L1a, L1b, L1c roles (~4,500/month) | L1d, L1e roles (~15,000/month) |
| **Remote %** | 10–15% (hardware access needed) | 20–25% (cloud/API-based) |
| **Phase 4 path** | Track A → B → C (all hardware) | Track C (compiler) or Phase 5 directly |

**Do both?** If you have time, Track A → Track B gives you full L1 coverage. Most hardware-focused engineers do Track A first, then add Track B topics as needed.

---

## How This Phase Connects to the Stack

| What you learn | How it informs hardware design |
|---------------|-------------------------------|
| Matrix multiply in neural networks | L5: systolic array dimensions, dataflow strategy |
| Conv2D, attention, pooling ops | L2: what the compiler must fuse and tile |
| Quantization (INT8, FP8) | L6: precision support in PE design |
| LLM inference (KV-cache, batching) | L5: memory hierarchy, HBM bandwidth requirements |
| Model computational graphs | L2: graph IR representation, fusion opportunities |
| Training at scale (distributed) | L3: NCCL, multi-GPU runtime |

---

## Additional Resources

- [CMU AI Courses Reference](CMU-AI-Courses.md)

---

## Next

→ [**Phase 4 Track A — Xilinx FPGA**](../Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/1.%20Xilinx%20FPGA%20Development/Guide.md) · [**Phase 4 Track B — Jetson**](../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Guide.md) · [**Phase 4 Track C — ML Compiler**](../Phase%204%20-%20Track%20C%20-%20ML%20Compiler%20and%20Graph%20Optimization/Guide.md)
