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

## Part 2 — The Four Sub-Tracks

Study these **in order**. Each builds on the previous, and each maps to a layer of the chip stack.

| Order | Sub-track | What you learn | Layer | Guide |
|:-----:|-----------|---------------|:-----:|-------|
| 1 | **C++ and SIMD** | Data-level parallelism inside a single core | L1 | [Guide →](C%2B%2B%20and%20SIMD/Guide.md) |
| 2 | **OpenMP and oneTBB** | Thread-level parallelism across CPU cores | L1/L3 | [Guide →](OpenMP%20and%20OneTBB/Guide.md) |
| 3 | **CUDA and SIMT** | Massive parallelism on GPU (the main focus) | L1/L3 | [Guide →](CUDA%20and%20SIMT/Guide.md) |
| 4 | **OpenCL** | Portable GPU/FPGA/CPU compute (optional) | L1/L3 | [Guide →](OpenCL/Guide.md) |

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

**OpenMP (pragma-based, easiest start):**
```cpp
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
}
```
One line turns a sequential loop into a multi-core parallel loop. The OpenMP runtime divides iterations across cores automatically.

**oneTBB (task-based, more control):**
```cpp
#include <tbb/parallel_for.h>

tbb::parallel_for(0, N, [&](int i) {
    C[i] = A[i] + B[i];
});
```
oneTBB uses a **work-stealing scheduler** — idle cores steal work from busy ones. Better load balancing for irregular workloads.

**Comparison:**

| | OpenMP | oneTBB |
|---|---|---|
| Model | Pragma annotations | C++ task scheduler |
| Ease of use | Very easy | Moderate |
| Control | Low | High (task graphs, flow graphs) |
| Load balancing | Static or dynamic | Work-stealing (automatic) |
| Best for | Regular loops | Irregular/nested parallelism |

**Key concepts:**
- **Data races:** Two threads writing to the same memory → undefined behavior. Use `#pragma omp critical` or `std::mutex`.
- **False sharing:** Two threads writing to adjacent cache lines → cache thrashing. Pad data structures.
- **Amdahl's Law:** If 10% of your code is sequential, max speedup is 10x no matter how many cores. Parallelize the bottleneck.
- **Reduction:** Summing across threads safely: `#pragma omp parallel for reduction(+:sum)`

**Why it matters for AI hardware:**
- Phase 4B Jetson: Cortex-A78AE cores use OpenMP for CPU-side preprocessing
- Phase 2: FreeRTOS tasks are a form of multi-core parallelism on embedded SoCs
- L4 (Firmware): Your AI chip's command processor uses multi-core scheduling

**Projects:**
- Parallelize matrix multiplication with OpenMP. Measure speedup from 1 to N cores.
- Implement parallel merge sort with oneTBB's `parallel_invoke`. Compare with `std::sort`.

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

### Sub-Track 4: OpenCL (Optional)

> *Write once, run on CPU, GPU, or FPGA — the portable compute API.*

**What OpenCL is:** An open standard for parallel programming across heterogeneous devices. Unlike CUDA (NVIDIA only), OpenCL runs on NVIDIA, AMD, Intel, ARM, and FPGAs.

**When to use OpenCL over CUDA:**
- Targeting non-NVIDIA hardware (AMD GPUs, Intel GPUs, Xilinx FPGAs)
- Portability is more important than peak performance
- Phase 4 Track A: Xilinx Vitis uses OpenCL as the host API for FPGA kernels

**Minimal example:**
```cpp
// OpenCL kernel (string or .cl file)
const char* kernel_src = R"(
    __kernel void vector_add(__global float* a, __global float* b, __global float* c) {
        int i = get_global_id(0);
        c[i] = a[i] + b[i];
    }
)";
```

**CUDA vs OpenCL comparison:**

| Concept | CUDA | OpenCL |
|---------|------|--------|
| Kernel qualifier | `__global__` | `__kernel` |
| Thread ID | `threadIdx.x + blockIdx.x * blockDim.x` | `get_global_id(0)` |
| Shared memory | `__shared__` | `__local` |
| Global memory | `__device__` | `__global` |
| Launch syntax | `kernel<<<grid, block>>>()` | `clEnqueueNDRangeKernel()` |
| Vendor | NVIDIA only | Multi-vendor |
| Performance | Best on NVIDIA | ~80-90% of CUDA on NVIDIA |

**Projects:**
- Port your CUDA vector add to OpenCL. Run on CPU and GPU. Compare performance.

---

## Part 3 — How This Connects to the Rest of the Roadmap

| What you learn here | Where it leads |
|--------------------|---------------|
| SIMD / vectorization | Phase 4C: MLIR `vector` dialect, compiler auto-vectorization |
| OpenMP / multi-core | Phase 2: FreeRTOS multi-core on Jetson SPE, Zynq PS |
| CUDA kernels | Phase 4B: Jetson inference, Phase 4C: Triton/CUTLASS kernel engineering |
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

1. **Parallelism exists at multiple levels** — instruction (SIMD), thread (OpenMP), massive (CUDA)
2. **CPU vs GPU is latency vs throughput** — different tools for different jobs
3. **Memory is the real bottleneck** — not compute. This is true for CUDA kernels, FPGA accelerators, and custom AI chips.
4. **CUDA is the industry standard** for GPU compute and AI inference
5. **Tiling is the universal optimization** — from shared memory in CUDA to systolic arrays in silicon
6. **Future = heterogeneous + portable** — SYCL, OpenCL, and custom accelerators all coexist

---

## Next

→ [**Phase 3 — Neural Networks**](../../Phase%203%20-%20Artificial%20Intelligence/Neural%20Networks/Guide.md) — the workloads that make all this parallelism necessary.
