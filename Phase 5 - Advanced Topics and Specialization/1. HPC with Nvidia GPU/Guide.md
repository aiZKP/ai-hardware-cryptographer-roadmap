# 1. HPC with Nvidia GPU (Phase 5 Track C)

**Timeline:** 12–24 months (fundamentals and deep dives); 24–48 months for advanced phase.

This track has two main parts:

| Part | Description | Guide |
|------|--------------|-------|
| **HPC Setup** | Fundamentals, virtualization, interconnects, advanced CUDA/distributed training/performance — plus hardware-specific deep dives (8x H200, L40S, NCCL, CUDA Advanced, GPUDirect Storage) | [HPC Setup →](./HPC%20Setup/Guide.md) |
| **DL Inference Optimization** | Graph/ops, kernel engineering (Triton, CUTLASS, Flash-Attention), compiler (IR, BEAM), quantization, runtimes. *Track F; MTS Kernels–style roles.* | [DL Inference Optimization →](./DL%20Inference%20Optimization/Guide.md) |

---

## How to use this track

1. **Start with [HPC Setup](./HPC%20Setup/Guide.md)** — Covers Nvidia GPU HPC fundamentals, virtualization (vGPU, KVM), interconnects and storage (InfiniBand, GDS, Slurm, Kubernetes), and Phase 2 advanced topics (advanced CUDA, distributed training, performance modeling). Use the deep dives (8x H200, L40S, NCCL, CUDA Advanced, GDS) for your target hardware and stack.
2. **Add [DL Inference Optimization](./DL%20Inference%20Optimization/Guide.md)** — If your goal is kernel/inference optimization (e.g. MTS Kernels, DL Inference Optimization Engineer), work through the six units there in order (graph/ops → kernels → compiler → quantization → runtimes → tinygrad).

**Prerequisite:** Phase 4 (Jetson, TensorRT, CUDA) is assumed for both parts.
