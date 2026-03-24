# OpenMP and OneTBB

Part of [Phase 1 section 4 — C++ and Parallel Computing](../Guide.md).

**Goal:** Shared-memory **CPU parallelism** with **OpenMP** (directive-based) and **oneTBB** (tasks and flow graphs) so structured parallel loops feel familiar before CUDA/OpenCL.

---

## 1. Baseline: `std::thread` and synchronization

* **Threads:** `std::thread`, joining, task-style thinking.
* **Sync:** Mutexes, condition variables, lock granularity.
* **Atomics:** `std::atomic`, memory ordering (high level); links to GPU visibility later.
* **Data races:** Undefined behavior when sharing is wrong.
* **Profiling:** Find CPU hotspots before rewriting as kernels.

---

## 2. OpenMP

* **Parallel for:** `#pragma omp parallel for`, schedules (`static`, `dynamic`, `guided`).
* **Reductions:** `reduction(+:sum)`.
* **Sections / single / master.**
* **SIMD:** `simd` pragma with compiler vectorization.
* **Tasks:** `task` / `taskwait` for irregular parallelism.

**Resources:** [OpenMP specifications](https://www.openmp.org/specifications/).

---

## 3. oneTBB (oneAPI Threading Building Blocks)

* **Algorithms:** `tbb::parallel_for`, `parallel_reduce`, `parallel_scan`.
* **Flow graph:** Pipeline-style dependencies (useful mental model for staged inference).
* **Containers / allocators:** For concurrent CPU services.

**Resources:** [oneTBB documentation](https://oneapi-src.github.io/oneTBB/).

---

## Next in this section

**[CUDA and SIMT](../CUDA%20and%20SIMT/Guide.md)**
