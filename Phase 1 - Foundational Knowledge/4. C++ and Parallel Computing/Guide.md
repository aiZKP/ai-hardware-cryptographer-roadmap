# C++ and Parallel Computing (Phase 1 §4)

> *From a single instruction to a thousand GPU threads — how computing became parallel, and why every AI hardware engineer must think in parallel.*

**Layer mapping:** **L1** (application — you write the code that runs on hardware), **L3** (runtime — CUDA runtime, OpenCL runtime are the bridge to the GPU driver).

**Prerequisites:** Phase 1 §2 (Computer Architecture — memory hierarchy, pipelining, caches), Phase 1 §3 (Operating Systems — threads, processes, scheduling).

**What comes after:** Phase 3 (Neural Networks — tensors map directly to these execution models), Phase 4 Track B (CUDA on Jetson with power limits), Phase 4 Track A (HLS/RTL — you design the hardware these kernels run on).

---

## Why This Module Exists

Every AI inference chip — from NVIDIA's H100 to your future custom NPU — exists because **sequential computing hit a wall**. Understanding *why* parallelism is necessary, *how* it evolved, and *how to program it* is the foundation for everything in this roadmap.

If you can't think in parallel, you can't design hardware that runs parallel workloads.

---

## Part 1 — How Computing Became Parallel (4 Steps)

Before diving into code, understand the historical pressure that created the hardware you'll design.

### Step 1: Sequential Computing (Single-Core Era)

**The simple model:** one instruction executes, then the next, then the next.

```
Instruction 1 → Instruction 2 → Instruction 3 → ...
```

For decades, performance came from **clock speed scaling** — run the same sequential code faster:
- 1980s: 4.77 MHz (Intel 8088)
- 1990s: 200 MHz (Pentium)
- 2000s: 3+ GHz (Pentium 4)

**What stopped it:** The **power wall**. Power scales roughly as `P = C * V^2 * f` (capacitance x voltage-squared x frequency). By ~2004, clock speeds plateaued at 3-4 GHz because chips couldn't dissipate the heat. This is called **Dennard scaling breakdown**.

**Connection to L5/L7/L8:** This is why AI chips exist. If clock speed still scaled freely, you'd just run everything on a faster CPU. The power wall forced the industry into parallelism and specialization — which is exactly what you're building toward in this roadmap.

---

### Step 2: Instruction-Level Parallelism (ILP / SIMD)

**Key idea:** do multiple things *inside* a single core per clock cycle.

**Pipelining** — overlap stages of different instructions:
```
Clock 1:  [Fetch A]
Clock 2:  [Decode A] [Fetch B]
Clock 3:  [Execute A] [Decode B] [Fetch C]
```
One instruction completes per cycle even though each takes multiple cycles. (You studied this in Phase 1 §2 — Computer Architecture.)

**Superscalar execution** — issue multiple independent instructions per cycle. Modern CPUs can execute 4-6 instructions per cycle if they don't depend on each other.

**SIMD (Single Instruction, Multiple Data)** — apply one operation to a vector of data:

```
Without SIMD:       With SIMD (4-wide):
a[0] = b[0] + c[0]  a[0..3] = b[0..3] + c[0..3]   ← ONE instruction
a[1] = b[1] + c[1]
a[2] = b[2] + c[2]
a[3] = b[3] + c[3]
```

This is the first taste of **data parallelism** — the same concept that GPUs take to the extreme.

**Connection to the stack:**
- **L1/L2:** AI compilers (TVM, MLIR) emit SIMD instructions when targeting CPUs.
- **L5:** When you design an accelerator's vector unit, you're designing custom SIMD hardware.

---

### Step 3: Multi-Core / Shared Memory Parallelism

**Key idea:** put multiple independent CPU cores on one chip.

When clock speed stopped scaling, chip designers added more cores:
- 2005: Intel Core 2 Duo (2 cores)
- 2010: Intel Core i7 (4 cores)
- 2024: AMD EPYC 9004 (96 cores)

But **software must explicitly use multiple cores**. A single-threaded program uses one core; the rest sit idle. This is why OpenMP, pthreads, and oneTBB exist.

```
Core 0: Process chunk [0..N/4]
Core 1: Process chunk [N/4..N/2]     ← all running simultaneously
Core 2: Process chunk [N/2..3N/4]
Core 3: Process chunk [3N/4..N]
```

**The challenge:** shared memory means **race conditions**, **deadlocks**, and **cache coherence** — all topics from Phase 1 §3 (OS).

**Connection to the stack:**
- **L4:** Firmware on your AI chip's embedded cores (ARM/RISC-V) uses multi-core scheduling.
- **L5:** NoC (Network-on-Chip) connects multiple compute tiles — same shared-memory model at hardware level.

---

### Step 4: Heterogeneous Computing (CPU + GPU)

**Key idea:** use *specialized* hardware for *specific* workload types.

CPUs are optimized for **latency** — do one thing fast (branch prediction, out-of-order execution, large caches). GPUs are optimized for **throughput** — do many simple things simultaneously (thousands of small ALUs, minimal control logic).

| | CPU | GPU |
|---|---|---|
| Cores | 4–96 (complex) | 1,000–16,000 (simple) |
| Optimized for | Latency (one task fast) | Throughput (many tasks at once) |
| Control logic | ~50% of die area | ~5% of die area |
| Cache per core | Large (MB) | Small (KB) |
| Best for | Sequential code, branching | Data-parallel: matrix math, image processing |

**Why this matters for AI:** Neural network inference is almost entirely matrix multiplication and element-wise operations — perfectly data-parallel. A GPU can process 1000x more multiply-accumulate operations per second than a CPU at the same power.

**Connection to the stack:**
- **L5:** Your custom AI chip is the next step beyond GPU — even more specialized for tensor operations.
- **L6:** The systolic array you'll design in Phase 5 (AI Chip Design) is purpose-built for the exact workloads GPUs handle with general-purpose SIMT.

---

## Part 2 — The Five Sub-Tracks

Study these **in order**. Each builds on the previous, and each maps to a layer of the chip stack.

| Order | Sub-track | What you learn | Layer | Guide |
|:-----:|-----------|---------------|:-----:|-------|
| 1 | **C++ and SIMD** | Data-level parallelism inside a single core | L1 | [Guide →](C%2B%2B%20and%20SIMD/Guide.md) |
| 2 | **OpenMP and oneTBB** | Thread-level parallelism across CPU cores | L1/L3 | [Guide →](OpenMP%20and%20OneTBB/Guide.md) |
| 3 | **CUDA and SIMT** | Massive parallelism on NVIDIA GPU (the main focus) | L1/L3 | [Guide →](CUDA%20and%20SIMT/Guide.md) |
| 4 | **ROCm and HIP** | AMD GPU programming — portable CUDA-like API | L1/L3 | [Guide →](ROCm%20and%20HIP/Guide.md) |
| 5 | **OpenCL and SYCL** | Portable compute across GPU/FPGA/CPU, modern C++ abstraction | L1/L3 | [Guide →](OpenCL%20and%20SYCL/Guide.md) |

---

### Sub-Track 1: C++ and SIMD

> *The gateway to GPU thinking — same operation, multiple data.*

**What SIMD is:** CPU vector instructions that process 4, 8, 16, or 32 data elements in one instruction.

| Instruction Set | Width | Data per instruction |
|----------------|-------|---------------------|
| SSE (1999) | 128-bit | 4x float32 |
| AVX (2011) | 256-bit | 8x float32 |
| AVX-512 (2017) | 512-bit | 16x float32 |
| ARM NEON | 128-bit | 4x float32 |

**Manual intrinsics example:**
```cpp
#include <immintrin.h>

void add_vectors(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i += 8) {
        __m256 va = _mm256_load_ps(&a[i]);   // Load 8 floats
        __m256 vb = _mm256_load_ps(&b[i]);   // Load 8 floats
        __m256 vc = _mm256_add_ps(va, vb);   // Add 8 pairs at once
        _mm256_store_ps(&c[i], vc);           // Store 8 results
    }
}
```

**Key concepts:**
- **Aligned memory:** SIMD loads require 16/32/64-byte alignment (`alignas(32)`)
- **Auto-vectorization:** Compilers can auto-vectorize simple loops (`-O2 -march=native`)
- **Manual intrinsics:** For control over exactly which instructions execute

**Why it matters for AI hardware:**
- cuDNN and CUTLASS use vectorized memory loads internally
- When you design an accelerator's vector unit (L5), you're designing custom SIMD
- MLIR's `vector` dialect (Phase 4C) targets exactly this level of abstraction

**Projects:**
- Implement vector addition with raw loops vs AVX intrinsics. Benchmark the speedup.
- Write a dot product using `_mm256_fmadd_ps` (fused multiply-add). Compare with scalar.

---

### Sub-Track 2: OpenMP and oneTBB

> *Scale from one core to many — the CPU parallelism layer.*

#### OpenMP — Pragma-Based (Easiest Start)

Add one line to a loop, and OpenMP distributes iterations across all CPU cores:

```cpp
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
}
```

The OpenMP runtime creates a **thread pool** (one thread per core), divides the loop range equally, and joins when done. You control scheduling strategy:

```cpp
// Static: divide N evenly across cores (predictable, best for uniform work)
#pragma omp parallel for schedule(static)

// Dynamic: each thread grabs a chunk when idle (better for uneven work)
#pragma omp parallel for schedule(dynamic, 64)

// Reduction: safely sum across threads
double total = 0.0;
#pragma omp parallel for reduction(+:total)
for (int i = 0; i < N; i++) {
    total += data[i];
}
```

**OpenMP strengths:** zero boilerplate, easy to add to existing code, widely supported (GCC, Clang, MSVC).

**OpenMP weakness:** limited control over task decomposition. For irregular workloads (tree traversals, graph algorithms, nested parallelism), you need something smarter.

---

#### oneTBB — Task-Based (More Control)

**oneTBB** (oneAPI Threading Building Blocks) is a C++ template library developed by Intel for parallel programming on multi-core processors. Originally known as TBB, it was rebranded as oneTBB in 2020 to integrate into the broader oneAPI ecosystem.

**Key difference from OpenMP:** Instead of parallelizing *loops*, oneTBB parallelizes *tasks*. You describe **what can run in parallel** — the runtime decides *how* to schedule it across cores.

**What oneTBB provides:**

| Component | What it does | Example |
|-----------|-------------|---------|
| **Parallel algorithms** | Ready-to-use parallel patterns | `parallel_for`, `parallel_reduce`, `parallel_sort`, `parallel_pipeline` |
| **Concurrent containers** | Thread-safe data structures | `concurrent_hash_map`, `concurrent_vector`, `concurrent_queue` |
| **Task-based runtime** | Breaks work into small tasks, schedules across cores | Work-stealing scheduler (see below) |
| **Flow graph** | Dataflow-style parallelism | Connect processing nodes into a DAG |

**Basic parallel_for:**

```cpp
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>

// Simple lambda version
tbb::parallel_for(0, N, [&](int i) {
    C[i] = A[i] + B[i];
});

// blocked_range version (more control over grain size)
tbb::parallel_for(
    tbb::blocked_range<size_t>(0, N, /*grain_size=*/1024),
    [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            C[i] = A[i] + B[i];
        }
    }
);
```

**parallel_reduce (thread-safe accumulation):**

```cpp
#include <oneapi/tbb/parallel_reduce.h>

double total = tbb::parallel_reduce(
    tbb::blocked_range<size_t>(0, N),
    0.0,  // identity value
    [&](const tbb::blocked_range<size_t>& r, double partial_sum) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            partial_sum += data[i];
        }
        return partial_sum;
    },
    std::plus<double>()  // combine partial results
);
```

**parallel_sort:**

```cpp
#include <oneapi/tbb/parallel_sort.h>

std::vector<int> data(10000000);
// ... fill data ...
tbb::parallel_sort(data.begin(), data.end());
// Automatically parallelizes across all cores
```

---

#### The Work-Stealing Scheduler — How oneTBB Actually Works

The work-stealing scheduler is the "brain" of oneTBB. It's what makes oneTBB automatically balance uneven workloads across cores — and it's a fundamental concept in parallel computing that appears in many systems (Go runtime, Tokio in Rust, Java ForkJoinPool).

**How it works — step by step:**

```
Initial state: 4 cores, work divided into tasks

Core 0 queue:  [T1] [T2] [T3] [T4] [T5]  ← lots of work
Core 1 queue:  [T6] [T7]                   ← some work
Core 2 queue:  [T8]                         ← little work
Core 3 queue:  (empty)                      ← idle!
```

**Step 1 — Private queues:** Each worker thread (one per CPU core) maintains its own private **deque** (double-ended queue) of tasks.

**Step 2 — LIFO execution (own work):** A thread pops tasks from the **bottom** of its own deque (newest first). This is like a stack — Last-In, First-Out. Why? The most recently created task likely has its data still in the CPU's L1/L2 cache. LIFO maximizes **cache locality**.

```
Core 0 works on its own tasks (bottom → top):
  Executes T5 (newest, cache-hot)
  Then T4
  Then T3...
```

**Step 3 — The "steal" (when idle):** When a thread finishes all its tasks, it doesn't sleep. Instead, it becomes a **thief** and looks at other threads' queues.

**Step 4 — FIFO stealing (from the top):** The thief steals from the **top** of another thread's deque (oldest task). This is critical:
- The thief steals the **oldest** task (likely a large chunk of work, worth stealing)
- The victim continues working on the **newest** task (undisturbed, cache-hot)
- Minimal interference between thief and victim (they access opposite ends of the deque)

```
Core 3 is idle → steals T1 from Core 0's queue (top/oldest)

Core 0 queue:  [T2] [T3] [T4]  ← didn't notice, still working on T4
Core 3 queue:  [T1]             ← now has work!
```

**Why this is better than a central queue:**

| Approach | How it works | Problem |
|----------|-------------|---------|
| Central queue | All threads grab tasks from one shared queue | Contention: every thread fights for the lock |
| Static partition | Divide work evenly at start | Imbalance: some chunks finish faster than others |
| **Work-stealing** | Each thread has private queue; steal only when idle | Only incurs synchronization overhead when actually out of work |

**Why it matters:**
- **Dynamic load balancing:** Handles irregular workloads where some tasks take 10x longer than others (tree traversals, sparse matrix, uneven image regions)
- **Scalability:** Programs automatically scale to however many cores the machine has — no code changes needed
- **Efficiency:** Near-zero overhead when all cores are busy; only the idle core pays the cost of stealing

**Connection to the stack:**
- **L4 (Firmware):** Your AI chip's command processor needs a task scheduler for DMA, compute, and I/O tasks — work-stealing is one approach
- **L3 (Runtime):** CUDA's stream scheduler and TensorRT's engine executor solve similar problems on GPU

---

#### OpenMP vs oneTBB — When to Use Which

| | OpenMP | oneTBB |
|---|---|---|
| **Model** | Pragma annotations on existing code | C++ task scheduler with templates |
| **Ease of use** | Very easy (one `#pragma` line) | Moderate (C++ lambda + template patterns) |
| **Control** | Low (schedule policy, num threads) | High (grain size, task graphs, flow graphs, custom partitioners) |
| **Load balancing** | Static or dynamic (loop-level) | Work-stealing (automatic, task-level) |
| **Nested parallelism** | Awkward (nested `parallel for` creates too many threads) | Natural (tasks spawn sub-tasks, scheduler handles it) |
| **Concurrent containers** | None (use `critical` sections) | Built-in (`concurrent_hash_map`, `concurrent_vector`) |
| **Best for** | Regular loops, quick parallelization | Irregular work, graph algorithms, production C++ libraries |
| **Compiler support** | GCC, Clang, MSVC (built-in) | Separate library (install via package manager or oneAPI) |

**Rule of thumb:** Start with OpenMP for simple loops. Switch to oneTBB when you need irregular parallelism, concurrent containers, or fine-grained control.

---

#### Key Concepts (Both OpenMP and oneTBB)

- **Data races:** Two threads writing to the same memory → undefined behavior. Use `#pragma omp critical`, `std::mutex`, or oneTBB's concurrent containers.
- **False sharing:** Two threads writing to adjacent cache lines → cache thrashing even though they access different data. Fix: pad data structures to cache line boundaries (`alignas(64)`).
- **Amdahl's Law:** If 10% of your code is sequential, maximum speedup is 10x — no matter how many cores you add. Always parallelize the bottleneck first.
- **Grain size:** The minimum chunk of work per task. Too small → scheduling overhead dominates. Too large → poor load balancing. oneTBB's `auto_partitioner` tunes this automatically.

**Why it matters for AI hardware:**
- Phase 4B Jetson: Cortex-A78AE cores use OpenMP for CPU-side preprocessing
- Phase 2: FreeRTOS tasks are a form of multi-core parallelism on embedded SoCs
- L4 (Firmware): Your AI chip's command processor uses multi-core scheduling — work-stealing is directly applicable

**Projects:**
1. **OpenMP matmul** — Parallelize matrix multiplication with `#pragma omp parallel for`. Measure speedup from 1 to N cores. Plot the scaling curve.
2. **oneTBB parallel_reduce** — Sum 100M floats using `tbb::parallel_reduce`. Compare with OpenMP `reduction(+:sum)`. Measure both.
3. **oneTBB parallel_sort** — Sort 10M integers with `tbb::parallel_sort`. Benchmark against `std::sort` (single-threaded). Measure speedup.
4. **Work-stealing visualization** — Implement parallel merge sort with `tbb::parallel_invoke`. Use `tbb::task_scheduler_observer` to log which core runs which task. Observe work-stealing in action.
5. **False sharing demo** — Write a program where N threads increment N counters in a shared array. First version: counters are adjacent. Second version: counters are padded to 64 bytes. Measure the performance difference.

**Bridge to GPU:** CPU parallelism scales to 8–96 cores. For 10,000+ parallel operations (like a 4096-element vector times a 4096x4096 matrix), you need a GPU.

---

### Sub-Track 3: CUDA and SIMT (Main Focus)

> *Thousands of threads, one program — the execution model behind every modern AI workload.*

This is the **most important sub-track** in Phase 1. CUDA is the programming model for NVIDIA GPUs, which run the vast majority of AI training and inference worldwide. Understanding CUDA is essential for every role in the AI hardware stack.

#### GPU Architecture (The Hardware You're Programming)

**Streaming Multiprocessor (SM):**
- The basic compute unit on a GPU
- Contains: CUDA cores (ALUs), tensor cores, shared memory, register file, warp schedulers
- NVIDIA A100: 108 SMs. H100: 132 SMs.

**CUDA core:** A simple ALU that can do one multiply-add per clock. An SM has 64-128 CUDA cores.

**Tensor core:** A specialized unit that does a 4x4 matrix multiply-accumulate in one instruction. This is what makes GPUs fast for AI — and what you'll eventually design your own version of (Phase 5F — AI Chip Design).

#### CUDA Execution Model

**Hierarchy:** Thread → Warp → Block → Grid

```
Grid (the entire job)
├── Block 0
│   ├── Warp 0 (threads 0-31)
│   ├── Warp 1 (threads 32-63)
│   └── ...
├── Block 1
│   ├── Warp 0
│   └── ...
└── ...
```

**Key rules:**
- A **warp** = 32 threads that execute the same instruction simultaneously (SIMT)
- A **block** = group of threads that share memory and can synchronize
- A **grid** = all blocks for one kernel launch
- You choose block size (typically 128 or 256 threads) and grid size (total work / block size)

**Warp divergence:** If threads in a warp take different `if/else` branches, both branches execute serially. Avoid branching in hot paths.

#### Memory Hierarchy

This is the **most important concept** for GPU performance. The difference between a fast and slow CUDA kernel is almost always memory access.

| Memory | Speed | Size | Scope | Managed by |
|--------|-------|------|-------|-----------|
| **Registers** | Fastest (~0 cycles) | 64K per SM | Per-thread | Compiler |
| **Shared memory** | Fast (~5 cycles) | 48-164 KB per SM | Per-block | Programmer |
| **L1 cache** | Fast | 128 KB per SM | Per-SM | Hardware |
| **L2 cache** | Medium | 4-50 MB | All SMs | Hardware |
| **Global memory (HBM)** | Slow (~400 cycles) | 16-80 GB | All threads | Programmer |

**Golden rule:** Minimize global memory access. Keep data in shared memory or registers.

**Coalesced memory access:** When 32 threads in a warp access 32 consecutive addresses, the hardware combines them into one transaction. Random access = 32 separate transactions = 32x slower.

```cpp
// GOOD: coalesced — thread i accesses element i
int val = data[threadIdx.x];

// BAD: strided — threads access every 32nd element
int val = data[threadIdx.x * 32];
```

#### CUDA Programming

**A minimal kernel:**
```cpp
// Runs on GPU — one instance per thread
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Host code — launches the kernel
int main() {
    int n = 1000000;
    float *d_a, *d_b, *d_c;

    // Allocate GPU memory
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    // Copy data from CPU to GPU
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel: 256 threads per block, enough blocks to cover n elements
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, n);

    // Copy result back to CPU
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}
```

**Key qualifiers:**
- `__global__` — kernel function, callable from CPU, runs on GPU
- `__device__` — callable only from GPU code
- `__host__` — runs on CPU (default)

**Streams and async:** Overlap data transfer with computation:
```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c);
cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, stream);

cudaStreamSynchronize(stream);
```

**The real bottleneck:** CPU ↔ GPU data transfer over PCIe. A well-optimized program minimizes transfers and keeps data on the GPU.

#### Matrix Multiplication — The Core of AI

Every neural network layer is a matrix multiply. This is the operation you'll optimize in Phase 4C and eventually design hardware for in Phase 5F.

**Naive CUDA matmul:**
```cpp
__global__ void matmul_naive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

**Why this is slow:** Each thread reads an entire row of A and column of B from global memory. N=4096 → each thread does 4096 global memory reads.

**Tiled matmul with shared memory:**
```cpp
__global__ void matmul_tiled(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < N / TILE; t++) {
        // Cooperatively load one tile into shared memory
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();

        // Compute partial product from shared memory (fast!)
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}
```

**Why this is faster:** Each global memory element is loaded once into shared memory and reused TILE times. For TILE=32, that's 32x fewer global memory accesses.

**Connection to the stack:**
- **L2 (Compiler):** This tiling is exactly what TVM, MLIR, and tinygrad do automatically
- **L5 (Architecture):** Systolic arrays do this in hardware — data flows between PEs, each value is reused
- **L6 (RTL):** In Phase 4A HLS, you'll implement tiled matmul as an FPGA accelerator

**Projects:**
1. **Vector add** — write a CUDA kernel, benchmark vs CPU.
2. **SAXPY** — `y = a*x + y` on GPU. Measure bandwidth (GB/s) and compare to theoretical peak.
3. **Naive matmul** — implement the naive version, measure GFLOPS.
4. **Tiled matmul** — add shared memory tiling. Measure the speedup vs naive.
5. **CPU golden** — for every GPU kernel, write a CPU reference and verify correctness.

---

### Sub-Track 4: ROCm and HIP

> *AMD's answer to CUDA — write GPU code that runs on both NVIDIA and AMD hardware.*

**What ROCm is:** AMD's open-source GPU compute platform (Radeon Open Compute). It includes the HIP programming language, kernel driver (`amdgpu`), runtime, math libraries (rocBLAS, MIOpen), and profiling tools.

**What HIP is:** Heterogeneous-computing Interface for Portability — a C++ API almost identical to CUDA. HIP code compiles to both AMD GPUs (via ROCm) and NVIDIA GPUs (via CUDA backend). This means you can write one kernel and run it on both vendors.

**Why learn this now (not just in Phase 5A):**
- AMD Instinct GPUs (MI300X, MI350) are deployed at scale by Microsoft Azure, Meta, Oracle
- Understanding both ecosystems makes you more valuable for any GPU role
- HIP is the fastest path from CUDA to portable GPU code
- Phase 5A (GPU Infrastructure) goes deeper; this sub-track gives you the programming foundation

#### CUDA vs HIP — Almost Identical

| CUDA | HIP | Notes |
|------|-----|-------|
| `cudaMalloc()` | `hipMalloc()` | Same signature |
| `cudaMemcpy()` | `hipMemcpy()` | Same signature |
| `cudaStream_t` | `hipStream_t` | Same concept |
| `__shared__` | `__shared__` | Identical |
| `__syncthreads()` | `__syncthreads()` | Identical |
| `threadIdx.x` | `threadIdx.x` | Identical |
| `cudaDeviceSynchronize()` | `hipDeviceSynchronize()` | Same |
| Warp size: 32 | **Wavefront size: 64** | Key difference |

**HIP kernel example:**
```cpp
#include <hip/hip_runtime.h>

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, n * sizeof(float));
    hipMalloc(&d_b, n * sizeof(float));
    hipMalloc(&d_c, n * sizeof(float));

    hipMemcpy(d_a, h_a, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, n * sizeof(float), hipMemcpyHostToDevice);

    vector_add<<<(n+255)/256, 256>>>(d_a, d_b, d_c, n);

    hipMemcpy(h_c, d_c, n * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_a); hipFree(d_b); hipFree(d_c);
}
```

If you know CUDA, you already know 95% of HIP. The main differences:
- **Wavefront = 64 threads** (vs CUDA warp = 32). Affects reduction, ballot, shuffle operations.
- **Shared memory banks:** 64 (vs 32 on NVIDIA). Different bank conflict patterns.
- **No tensor cores** — AMD uses **Matrix Cores** via `rocWMMA` or Composable Kernel (CK).

#### HIPIFY — Automatic CUDA → HIP Conversion

```bash
# Convert CUDA source to HIP (automated)
hipify-clang my_kernel.cu -o my_kernel.hip.cpp

# Or use perl-based converter (simpler, less accurate)
hipify-perl my_kernel.cu > my_kernel.hip.cpp
```

HIPIFY handles ~90% of conversions automatically. Manual work is needed for:
- Inline PTX assembly
- CUDA-specific intrinsics (`__ballot_sync` with warp size assumptions)
- Vendor-specific libraries (cuBLAS → rocBLAS API differences)

#### AMD GPU Architecture (Brief)

| Component | AMD (CDNA) | NVIDIA (Ampere/Hopper) |
|-----------|-----------|----------------------|
| Compute unit | CU (Compute Unit) | SM (Streaming Multiprocessor) |
| Thread group | Wavefront (64 threads) | Warp (32 threads) |
| Shared memory | LDS (Local Data Share) | Shared memory |
| Vector ALU | SIMD unit (64-wide) | CUDA cores |
| Matrix unit | Matrix Cores | Tensor Cores |
| Interconnect | Infinity Fabric | NVLink |

**Projects:**
1. **HIPIFY your CUDA kernel** — Take your CUDA vector add and matmul from Sub-Track 3. Convert to HIP using `hipify-clang`. Run on AMD GPU (or ROCm Docker on NVIDIA). Verify identical output.
2. **Wavefront vs warp** — Write a reduction kernel. Run on both AMD (wavefront=64) and NVIDIA (warp=32). Measure how the size difference affects performance.
3. **Profile with rocProf** — Profile your HIP matmul with `rocprof`. Compare the output format with NVIDIA's `nsys`.

---

### Sub-Track 5: OpenCL and SYCL

> *Write once, run anywhere — portable compute across GPU, FPGA, and CPU.*

#### OpenCL — The Multi-Vendor Standard

**What OpenCL is:** An open standard for parallel programming across heterogeneous devices. Unlike CUDA (NVIDIA only) or HIP (AMD+NVIDIA), OpenCL runs on NVIDIA, AMD, Intel, ARM, and FPGAs.

**When to use OpenCL:**
- Targeting non-NVIDIA/non-AMD hardware (Intel GPUs, ARM Mali, Xilinx FPGAs)
- Phase 4 Track A: Xilinx Vitis uses OpenCL as the host API for FPGA kernels
- Embedded/mobile GPUs that only support OpenCL

**OpenCL kernel:**
```c
__kernel void vector_add(__global float* a, __global float* b, __global float* c) {
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}
```

**CUDA vs HIP vs OpenCL comparison:**

| Concept | CUDA | HIP | OpenCL |
|---------|------|-----|--------|
| Kernel qualifier | `__global__` | `__global__` | `__kernel` |
| Thread ID | `threadIdx.x + blockIdx.x * blockDim.x` | Same | `get_global_id(0)` |
| Shared memory | `__shared__` | `__shared__` | `__local` |
| Launch | `kernel<<<grid, block>>>()` | Same | `clEnqueueNDRangeKernel()` |
| Vendor | NVIDIA | AMD + NVIDIA | All vendors |
| Performance | Best on NVIDIA | Best on AMD | ~80-90% of native |
| Host API | C/C++ | C/C++ | C (verbose) |

**OpenCL's weakness:** The host API is extremely verbose — creating contexts, command queues, building programs, setting arguments requires dozens of lines of boilerplate. This is what SYCL solves.

#### SYCL — Modern C++ for Heterogeneous Computing

**What SYCL is:** A Khronos standard that provides a **single-source C++ programming model** for heterogeneous computing. Write host and device code in the same C++ file, using standard C++ features (lambdas, templates, RAII).

**Why SYCL matters:**
- **Single source:** No separate kernel files or string-based kernels
- **Modern C++:** Lambdas, templates, type safety — unlike OpenCL's C99 kernels
- **Multi-target:** Compile the same code for CPU, NVIDIA GPU, AMD GPU, Intel GPU, or FPGA
- **Growing ecosystem:** Intel oneAPI (DPC++), AdaptiveCpp (hipSYCL), Codeplay ComputeCpp

**SYCL example:**
```cpp
#include <sycl/sycl.hpp>

int main() {
    sycl::queue q;  // Auto-selects best device (GPU if available)

    float* a = sycl::malloc_shared<float>(N, q);
    float* b = sycl::malloc_shared<float>(N, q);
    float* c = sycl::malloc_shared<float>(N, q);

    // Initialize a, b on host...

    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        c[i] = a[i] + b[i];
    }).wait();

    sycl::free(a, q); sycl::free(b, q); sycl::free(c, q);
}
```

Compare with the equivalent CUDA code — SYCL is significantly less boilerplate while being portable across vendors.

**CUDA vs HIP vs SYCL side-by-side:**

| Feature | CUDA | HIP | SYCL |
|---------|------|-----|------|
| Language | CUDA C++ | HIP C++ | Standard C++ |
| Source model | Separate `.cu` files | Separate `.hip.cpp` files | **Single source** |
| Memory model | Manual `cudaMalloc/Memcpy` | Manual `hipMalloc/Memcpy` | **Unified shared memory** (USM) or buffers |
| Kernel syntax | `<<<grid, block>>>` | Same | `parallel_for` lambda |
| Targets | NVIDIA only | AMD + NVIDIA | **CPU, NVIDIA, AMD, Intel, FPGA** |
| Maturity | 17 years, dominant | 7 years, growing | 5 years, emerging |
| Best for | Peak NVIDIA perf | AMD perf + portability | **Maximum portability** |

**SYCL implementations:**

| Implementation | Vendor | Targets |
|---------------|--------|---------|
| **Intel oneAPI (DPC++)** | Intel | Intel GPU, CPU, FPGA, NVIDIA (plugin), AMD (plugin) |
| **AdaptiveCpp (hipSYCL)** | Open source | NVIDIA (CUDA), AMD (ROCm), Intel, CPU |
| **Codeplay ComputeCpp** | Codeplay | Multiple backends |

**Connection to the stack:**
- **L2 (Compiler):** SYCL compilers use LLVM/SPIR-V — the same IR infrastructure as MLIR (Phase 4C)
- **L3 (Runtime):** SYCL runtimes manage device selection, memory, and scheduling — same concepts as XRT (Phase 4A) and CUDA runtime (Phase 4B)
- **Future direction:** As AI inference moves beyond NVIDIA-only (AMD, Intel, custom NPUs), portable APIs like SYCL become essential

**Projects:**
1. **Port CUDA vector add to OpenCL** — experience the verbose host API. Run on CPU and GPU.
2. **Same kernel in SYCL** — rewrite the same vector add using SYCL `parallel_for`. Compare code size and readability.
3. **SYCL matmul** — implement tiled matrix multiplication in SYCL with local memory (`sycl::local_accessor`). Run on CPU and GPU, compare performance with your CUDA tiled matmul.
4. **Multi-backend test** — compile the same SYCL kernel for CPU (OpenMP backend), NVIDIA GPU (CUDA backend via AdaptiveCpp), and Intel GPU (Level Zero). Compare performance across all three.

---

## Part 3 — How This Connects to the Rest of the Roadmap

| What you learn here | Where it leads |
|--------------------|---------------|
| SIMD / vectorization | Phase 4C: MLIR `vector` dialect, compiler auto-vectorization |
| OpenMP / multi-core | Phase 2: FreeRTOS multi-core on Jetson SPE, Zynq PS |
| CUDA kernels | Phase 4B: Jetson inference, Phase 4C: Triton/CUTLASS kernel engineering |
| ROCm / HIP | Phase 5A: AMD GPU infrastructure (MI300X), portable kernel engineering |
| OpenCL | Phase 4A: Xilinx Vitis FPGA host API, embedded GPU compute |
| SYCL | Future portable compute — CPU, GPU, FPGA, custom NPU from one source |
| Memory hierarchy thinking | Phase 4A: FPGA BRAM/URAM tiling, Phase 5F: scratchpad design for AI chip |
| Tiled matmul | Phase 4A: HLS matmul accelerator, Phase 5F: systolic array architecture |
| GPU architecture model | Phase 5B: CUDA-X libraries, Phase 5F: design something better |

**The big picture:**
- Phase 1 §4 teaches you to **program** parallel hardware
- Phase 4 teaches you to **optimize and deploy** on real parallel hardware
- Phase 5F teaches you to **design** new parallel hardware

You're learning the workload first. Then you'll build the machine that runs it.

---

## Key Takeaways

1. **Parallelism exists at multiple levels** — instruction (SIMD), thread (OpenMP), massive (CUDA/HIP)
2. **CPU vs GPU is latency vs throughput** — different tools for different jobs
3. **Memory is the real bottleneck** — not compute. This is true for CUDA kernels, FPGA accelerators, and custom AI chips.
4. **CUDA is the industry standard** for GPU compute and AI inference — learn it first
5. **HIP makes your CUDA skills portable** — same code runs on AMD and NVIDIA
6. **Tiling is the universal optimization** — from shared memory in CUDA to systolic arrays in silicon
7. **Future = heterogeneous + portable** — SYCL targets CPU, GPU, FPGA, and custom accelerators from one codebase

---

## Next

→ [**Phase 3 — Neural Networks**](../../Phase%203%20-%20Artificial%20Intelligence/Neural%20Networks/Guide.md) — the workloads that make all this parallelism necessary.
