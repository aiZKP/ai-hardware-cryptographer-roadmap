# 1. GPU Infrastructure (Phase 5)

**Timeline:** 12–24 months (fundamentals); 24–48 months for advanced phase.

**Prerequisites:** Phase 4 Track B (Jetson, CUDA stack), Phase 4 Track C (ML compiler + DL inference optimization).

---

## What this track covers

**HPC** = high-performance computing: solving problems that need massive compute and memory by using many machines (or many GPUs) working together. In AI, "HPC" usually means **large-scale training** and **high-throughput inference** on GPU clusters, not single workstations.

This track is organized by GPU vendor:

| Sub-track | Focus | Guide |
|-----------|-------|-------|
| **Nvidia GPU** | CUDA, NCCL, NVLink/NVSwitch, InfiniBand, GPUDirect, Slurm/K8s, multi-GPU/multi-node clusters | [Nvidia GPU →](Nvidia%20GPU/Guide.md) |
| **AMD GPU** | ROCm, HIP, RCCL, AMD Instinct (MI300X), RDNA/CDNA architecture, porting CUDA workloads | [AMD GPU →](AMD%20GPU/Guide.md) |

---

## How to use this track

1. **Start with [Nvidia GPU](Nvidia%20GPU/Guide.md)** — the dominant ecosystem for AI HPC. Covers fundamentals, virtualization, interconnects, storage, distributed training, and performance modeling.
2. **Add [AMD GPU](AMD%20GPU/Guide.md)** — for portability, alternative hardware, and understanding the growing AMD AI ecosystem (MI300X, ROCm 6+).

Both sub-tracks assume you've completed Phase 4 Track C (compiler + inference optimization). The HPC content here focuses on **infrastructure, multi-GPU scaling, and distributed systems** — the compiler and kernel optimization skills come from Track C.
