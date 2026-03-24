# CUDA and SIMT

Part of [Phase 1 section 4 — C++ and Parallel Computing](../Guide.md).

**Goal:** Learn NVIDIA’s **SIMT** programming model: grids, blocks, warps, and explicit device memory—minimal kernels with **CPU reference checks**, matching how production stacks pair host C++ with GPU code.

---

## 1. Programming model and memory

* **Hierarchy:** Grids, blocks, warps, threads — occupancy vs latency hiding (qualitative).
* **Kernels:** `__global__`, launch configuration, error checking (`cudaGetLastError`, `cudaDeviceSynchronize` in learning builds).
* **Memory spaces:** Global, shared, constant, registers — tie to Phase 1 section 2 (memory hierarchy).
* **Patterns:** Element-wise ops, 1D/2D indexing, tiled matmul (conceptual), reduction vs atomics.
* **Host/device:** `cudaMalloc`, `cudaMemcpy`, pinned memory; **streams** and async copies (intro).
* **Limits:** Warp divergence, misaligned access, CPU/GPU coherence requires explicit sync.

**Official:** [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).

---

## 2. Suggested projects (in order)

| Project | Skills |
|--------|--------|
| Vector add / SAXPY | Launch config, indexing, vs CPU reference |
| 2D grayscale transform | 2D `threadIdx` / `blockIdx`, bounds |
| Naive square matmul | Nested loops on GPU; compare to cuBLAS |
| Parallel reduction (sum) | Shared memory, sync, bank conflicts (intro) |
| Small CPU+GPU pipeline | Reuse buffers; time copy vs compute |

Use one folder or repo with CMake or `nvcc` + script; **keep CPU goldens** for every kernel.

---

## 3. Connections

* **Phase 1 section 2:** GPU is another memory hierarchy; warps behave like SIMD + multithreading.
* **Phase 3:** Tensors map to these kernels and to frameworks.
* **Phase 4 Track B (Jetson):** Same model with unified memory and thermal caps.
* **Phase 5 (HPC):** Multi-GPU, NCCL, deeper optimization.

---

## Tooling

* **Hardware:** NVIDIA GPU + driver; Jetson learners may develop on x64 first.
* **Software:** CUDA Toolkit; CMake `CUDA` language or `nvcc` directly.

---

## Next in this section

**[OpenCL](../OpenCL/Guide.md)** — portable compute across vendors.

---

## Then Phase 3

**[Neural Networks](../../Phase%203%20-%20Artificial%20Intelligence/Neural%20Networks/Guide.md)** · **[Edge AI](../../Phase%203%20-%20Artificial%20Intelligence/Edge%20AI/Guide.md)**
