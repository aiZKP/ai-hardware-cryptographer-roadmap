# DL Inference Optimization (inside HPC with Nvidia GPU)

**Role target:** [Step 2 — DL Inference Optimization Engineer](../../../../README.md#the-four-career-steps)

**Also aligns with:** Kernel-focused roles such as **Member of Technical Staff, Kernels** (e.g. AGI/LLM companies): designing and implementing high-performance kernels for training and inference, long-context optimization, and production deployment on NVIDIA GPUs and alternative accelerators (TPU, etc.).

**Prerequisite:** Phases 1–2, Phase 4 (Jetson, TensorRT, Edge AI Optimization).

---

## How to use this track

Study the topics **in order**. Each folder is one unit; do them one by one. The order is chosen so the most important and foundational topics come first, and each unit builds on the previous.

| Order | Folder | What you learn |
|:-----:|--------|----------------|
| **1** | [01 — Graph and Operator Optimization](01%20-%20Graph%20and%20Operator%20Optimization/Guide.md) | What we're optimizing: graph, ops, fusion, profiling, bottlenecks. **Do this first.** |
| **2** | [02 — Kernel Engineering](02%20-%20Kernel%20Engineering/Guide.md) | How to write and own kernels: Triton, CUTLASS, Flash-Attention, long-context, NCCL, production. **Core of MTS Kernels.** |
| **3** | [03 — Compiler Stack](03%20-%20Compiler%20Stack/Guide.md) | How compilers produce kernels: IR, scheduling (BEAM), codegen, TVM, MLIR. |
| **4** | [04 — Quantization](04%20-%20Quantization/Guide.md) | Low-precision inference: PTQ, QAT, INT8/INT4, kernel and runtime integration. |
| **5** | [05 — Inference Runtimes and Deployment](05%20-%20Inference%20Runtimes%20and%20Deployment/Guide.md) | Production: TensorRT, ONNX Runtime, Triton server; latency, throughput, methodology. |
| **6** | [06 — tinygrad Deep Dive](06%20-%20tinygrad%20Deep%20Dive/Guide.md) | Optional: hands-on IR, scheduler, backends; compiler–kernel interface. |

---

## Learning path (one by one)

1. **Start:** [01 — Graph and Operator Optimization](01%20-%20Graph%20and%20Operator%20Optimization/Guide.md) — Understand the graph and find bottlenecks before writing kernels.
2. **Then:** [02 — Kernel Engineering](02%20-%20Kernel%20Engineering/Guide.md) — Design and implement high-performance kernels (Triton, CUTLASS, attention, long-context).
3. **Then:** [03 — Compiler Stack](03%20-%20Compiler%20Stack/Guide.md) — See how IR, scheduling, and codegen feed kernel selection and fusion.
4. **Then:** [04 — Quantization](04%20-%20Quantization/Guide.md) — Add low-precision kernels and deployment (PTQ, QAT).
5. **Then:** [05 — Inference Runtimes and Deployment](05%20-%20Inference%20Runtimes%20and%20Deployment/Guide.md) — Deploy and measure in production-like settings.
6. **Optional:** [06 — tinygrad Deep Dive](06%20-%20tinygrad%20Deep%20Dive/Guide.md) — Hands-on compiler and backend work.

---

## Summary

| Area | Key skills |
|------|------------|
| Graph & operators | Fusion, constant folding, profiling, bottleneck analysis |
| **Kernel engineering** | Triton, CUTLASS, CuTe; Flash-Attention, long-context; NCCL/MSCCLPP; TPU/Pallas/Mojo; testing, correctness, porting |
| Compiler | IR, scheduling (e.g. BEAM), codegen, TVM/MLIR concepts |
| Quantization | PTQ, QAT, INT8/INT4, tooling (TensorRT, ONNX Runtime) |
| Runtimes | TensorRT, ONNX Runtime, Triton; latency/throughput methodology |

This track supports the **DL Inference Optimization Engineer** role and **kernel-focused roles** (e.g. Member of Technical Staff, Kernels): from graph-level optimization and custom kernel design to deployment with measurable latency and throughput, including porting to alternative hardware and production reliability.
