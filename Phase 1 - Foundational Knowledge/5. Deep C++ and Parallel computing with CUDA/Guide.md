# Deep C++ and Parallel computing with CUDA

> **Goal:** Write **correct, efficient C++** on the CPU side and **minimal CUDA kernels** on the GPU — the same pairing used in real inference stacks (host code + kernels), HPC libraries, and later phases of this roadmap (Jetson, TensorRT, custom backends).

**Placement:** Phase 1 **§5**, after **Operating Systems** (address spaces, processes/threads) and before **AI Fundamentals** (§6), so you understand *how parallel hardware executes* before you study *what neural networks compute*.

---

## 1. Modern C++ for systems and numerics

Focus on language features you will use beside CUDA host code and in performance-critical tools:

* **Core:** Value vs reference, `const`/`constexpr`, enums and strong typing, namespaces, headers vs modules (awareness).
* **Memory and ownership:** RAII, constructors/destructors, rule of five / rule of zero, smart pointers (`unique_ptr`, `shared_ptr`), when *not* to heap-allocate in hot paths.
* **Move semantics:** Rvalues, `std::move`, avoiding copies of large buffers (weight tensors, staging buffers).
* **Templates (pragmatic):** Function templates, `std::vector`, `std::array`; reading simple template errors.
* **STL algorithms:** `std::copy`, transforms, reductions — map to parallel patterns (map, reduce) you will see on GPU.
* **Error handling:** Exceptions vs error codes in HPC/embedded-adjacent code; `std::optional` / `expected` (C++23) as patterns.

**Resources:** *A Tour of C++* (Stroustrup); [cppreference.com](https://en.cppreference.com/); your compiler’s warnings (`-Wall -Wextra`) treated as mandatory.

---

## 2. CPU parallel programming (before CUDA)

CUDA is easier if CPU threading and memory models are not mysterious:

* **Threads:** `std::thread`, joining, task-based thinking.
* **Synchronization:** Mutexes, condition variables, lock granularity and contention.
* **Atomics (intro):** `std::atomic`, memory ordering at a high level — connects to GPU visibility and fence concepts later.
* **Data races and UB:** Why “it works on my machine” is not enough.
* **Profiling:** `perf` (Linux), Visual Studio / VTune (Windows), or `tracy`/similar — find real hotspots before rewriting in CUDA.

**Stretch:** OpenMP parallel for loops on CPU as a bridge to “parallel loop” mental model.

---

## 3. CUDA: programming model and memory

* **Hierarchy:** Grids, blocks, warps, threads — occupancy vs latency hiding (qualitative).
* **Kernels:** `__global__`, launch configuration, error checking (`cudaGetLastError`, `cudaDeviceSynchronize` in learning/debug builds).
* **Memory spaces:** Global, shared, constant, registers — latency/bandwidth intuition tied to Phase 1 §3 (memory hierarchy).
* **Patterns:** Element-wise ops, 1D/2D indexing, tiled matrix multiply (conceptual), reduction tree vs atomic (tradeoffs).
* **Host/device:** Allocation (`cudaMalloc`, `cudaMemcpy` / pinned memory awareness), async copies and **streams** (intro).
* **Limits:** Warp divergence, misaligned access, lack of cache coherency between CPU and GPU without explicit sync.

**Official:** [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) (NVIDIA).

---

## 4. Suggested projects (in order)

| Project | Skills |
|--------|--------|
| Vector add / SAXPY | Launch config, indexing, correctness vs CPU reference |
| 2D grayscale transform (e.g. clamp, scale) | 2D `threadIdx`/`blockIdx`, bounds checks |
| Naive square matmul | Nested loops on GPU; measure vs cuBLAS to see why libraries exist |
| Parallel reduction (sum) | Shared memory, synchronization, bank conflicts (intro) |
| One small CPU+GPU pipeline | Allocate once, reuse buffers; time copy vs compute |

Use a **single pinned repo or folder** with CMake or a simple `nvcc` + compiler script; keep CPU reference implementations for every kernel.

---

## 5. How this connects to the rest of the roadmap

* **Phase 1 §3 (Architecture & hardware):** GPU is another memory hierarchy; warps behave like SIMD + multithreading.
* **Phase 1 §6 (AI Fundamentals):** Tensors and ops are what you will eventually run through CUDA, TensorRT, or tinygrad backends.
* **Phase 4 (Jetson / TensorRT):** Same CUDA model, with unified memory and power/thermal constraints.
* **Phase 4 (HPC / GPU specialization):** Everything here scales to multi-GPU, NCCL, and kernel optimization workflows.

---

## 6. Prerequisites and tooling

* **Prerequisites:** Comfortable C syntax (from OS / embedded work); Phase 1 §4 lectures on virtual memory and threads help for CUDA host behavior.
* **Hardware:** Any recent NVIDIA GPU with a supported driver; for Jetson-only learners, develop on x64 + discrete GPU if possible, then cross-check concepts on device later.
* **Software:** CUDA Toolkit matching your driver; build with `nvcc` or CMake `CUDA` language.

---

## Next section

**[6. AI Fundamentals — Neural Networks and Edge AI](../6. AI Fundamentals - Neural Networks and Edge AI/Guide.md)** — what to compute once parallel execution is familiar.
