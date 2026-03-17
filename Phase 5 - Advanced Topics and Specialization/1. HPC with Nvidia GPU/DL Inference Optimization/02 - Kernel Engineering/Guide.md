# 02 — Kernel Engineering for Training & Inference

**Order:** Second. After you know the graph and bottlenecks (01), you implement and own the kernels.

**Role target:** Core of **MTS Kernels** and **DL Inference Optimization Engineer** — design, implement, deploy, and maintain high-performance kernels; production reliability; co-design with training/inference/RL.

---

## Why this is second

01 tells you *what* to optimize. This unit is *how*: writing and tuning the actual GPU/accelerator kernels that run the heavy ops. This is the heart of kernel-engineer roles (e.g. Magic, NVIDIA-style teams).

---

## 1. Custom kernel authoring frameworks

* **Triton**
    * Write GPU kernels in Python; tile-based matmul and attention; automatic tuning.
    * Integration with PyTorch (`torch.compile`, custom ops). Study official tutorials and Flash-Attention–style patterns.
* **CUTLASS / CuTeDSL**
    * NVIDIA's CUDA template libraries for GEMM, conv, and custom ops.
    * CuTe: layout and copy abstractions. Essential for pushing performance on datacenter GPUs (e.g. Blackwell).
* **Flash-Attention (v2/v3), Quack**
    * Fused attention: Q/K/V ops, online softmax, memory-efficient attention, long-context.
    * Understand tiling, memory utilization, and data movement.
* **Mojo, Pallas/Mosaic (JAX)**
    * Mojo: performance portability; Pallas/Mosaic: GPU and TPU kernels. Useful for porting to non-NVIDIA hardware.

---

## 2. Long-context and attention kernels

* **Challenges** — Memory utilization, KV-cache layout, data movement, bandwidth. Kernels must scale to 1M+ context.
* **Patterns** — Flash-Attention, FlashInfer, Magic-Attention (GTC 2026): fused attention, variable length, production-grade correctness and testing.
* **Analysis** — Roofline and occupancy for attention; avoiding memory-bound bottlenecks; measuring sustained throughput.

---

## 3. Collective communication

* **NCCL** — All-reduce, all-gather, reduce-scatter; tuning for multi-node, multi-GPU. Overlap of communication with compute.
* **MSCCLPP** — Microsoft's collective library; compare with NCCL for portability or alternative backends when relevant.

---

## 4. Production and portability

* **Robustness and testing** — Extensive tests; functional correctness; numerical stability (especially custom attention/softmax). Reproducible benchmarks and CI.
* **Porting to alternative hardware** — Evaluate/port kernels to TPU (Pallas/Mosaic), other accelerators. Abstraction layers and backend-specific trade-offs.
* **Co-design** — Clear contracts and APIs with training, inference, and RL teams; versioning and deployment for production reliability.

---

## 5. Computer architecture and code generation

* **Low-level expertise** — Memory hierarchy, warp/SM behavior, occupancy, instruction throughput. How your kernels map to hardware.
* **Code generation** — From high-level specs (e.g. MLIR, Triton IR) to GPU/TPU code; how compilers map ops to kernels.

---

## Resources

* [Triton Documentation](https://triton-lang.org/) — Language and GPU kernel patterns.
* [CUTLASS](https://github.com/NVIDIA/cutlass) — CUDA templates for GEMM and more.
* [CuTe](https://github.com/NVIDIA/cutlass/tree/main/cute) — Layout and copy DSL.
* [Flash-Attention](https://github.com/Dao-AILab/flash-attention) — Memory-efficient attention.
* [NCCL](https://developer.nvidia.com/nccl) — NVIDIA Collective Communications Library.
* [Pallas / Mosaic (JAX)](https://github.com/google/jax/tree/main/jax/experimental/pallas) — GPU/TPU kernel authoring.
* GTC 2026: Magic-Attention and long-context kernel talks.

---

## Projects

1. **Triton fused kernel** — Implement a fused operator (e.g. layer norm + residual, or a custom attention variant) in Triton. Benchmark vs PyTorch and profile with Nsight Compute.
2. **Long-context attention** — Study Flash-Attention's tiling and online softmax. Implement a simplified long-context attention kernel; measure memory vs throughput trade-offs.
3. **NCCL at scale** — Run NCCL all-reduce at scale (multi-GPU or multi-node if available). Tune and document overlap opportunities with compute.
4. **Portability report** — One-page write-up: what it would take to port one of your kernels to TPU via Pallas or to another backend (e.g. Mojo).

---

## Next

→ **[03 — Compiler Stack](../03%20-%20Compiler%20Stack/Guide.md)** — How IR, scheduling, and codegen produce and select these kernels (tinygrad, TVM, MLIR).
