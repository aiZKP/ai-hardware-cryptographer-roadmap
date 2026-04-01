# ROCm and HIP (Phase 1 §4 — Sub-Track 4)

**Parent:** [C++ and Parallel Computing](../Guide.md)

> *AMD's answer to CUDA — write GPU code that runs on both NVIDIA and AMD hardware.*

**Prerequisites:** Sub-Track 3 (CUDA and SIMT). You need working CUDA knowledge first — HIP is learned by comparison.

**Layer mapping:** **L1** (application — you write HIP kernels), **L3** (runtime — HIP runtime, ROCm driver stack).

---

## What ROCm Is

**ROCm** (Radeon Open Compute) is AMD's open-source GPU compute platform. It includes the HIP programming language, kernel driver (`amdgpu`), runtime, compiler (`amd-clang`), math libraries (rocBLAS, MIOpen, rocFFT), and profiling tools (Omniperf, Omnitrace). ROCm is to AMD what CUDA is to NVIDIA.

## What HIP Is

**HIP** (Heterogeneous-computing Interface for Portability) is a C++ API almost identical to CUDA. HIP code compiles to both AMD GPUs (via ROCm) and NVIDIA GPUs (via CUDA backend). This means you can write one kernel and run it on both vendors.

**Why learn this now (not just in Phase 5A):**
- AMD Instinct GPUs (MI300X, MI350) are deployed at scale by Microsoft Azure, Meta, Oracle
- Understanding both ecosystems makes you more valuable for any GPU role
- HIP is the fastest path from CUDA to portable GPU code
- Phase 5A (GPU Infrastructure) goes deeper; this sub-track gives you the programming foundation

---

## 1. CDNA vs RDNA — AMD's Two GPU Architectures

AMD makes two completely different GPU architectures for different markets. Understanding the difference is essential before writing any AMD GPU code.

### RDNA (Radeon DNA) — Gaming and Consumer

RDNA powers Radeon RX consumer GPUs. Optimized for graphics workloads: high clock speeds, rasterization, ray tracing, display output.

- **Products:** Radeon RX 7900 XTX, RX 7800 XT, Steam Deck APU, PlayStation 5, Xbox Series X
- **Design focus:** High single-thread performance, graphics pipeline, low power
- **Compute units:** Dual-issue SIMD, 32-wide wavefronts (RDNA 3+ supports wave32 *and* wave64)
- **Memory:** GDDR6 (up to 24 GB), no HBM
- **Use case:** Gaming, content creation, lightweight ML inference

### CDNA (Compute DNA) — Data Center and AI

CDNA powers AMD Instinct data-center GPUs. Optimized for matrix math and HPC — no graphics pipeline at all.

- **Products:** MI300X, MI300A, MI250X, MI210
- **Design focus:** Maximum compute throughput, matrix operations, multi-GPU scaling
- **Compute units:** 64-wide wavefronts, optimized for FP64/FP32/FP16/INT8 matrix operations
- **Matrix Cores:** Hardware matrix multiply-accumulate (equivalent to NVIDIA Tensor Cores)
- **Memory:** HBM3 (up to 192 GB on MI300X with 5.3 TB/s bandwidth)
- **Interconnect:** Infinity Fabric for GPU-to-GPU (like NVLink)
- **Use case:** AI training, AI inference, HPC, scientific computing

### CDNA vs RDNA Comparison

| | RDNA (Consumer) | CDNA (Data Center) |
|---|---|---|
| **Target** | Gaming, desktop | AI training/inference, HPC |
| **Products** | Radeon RX 7900 XTX | Instinct MI300X |
| **Graphics pipeline** | Yes (rasterization, ray tracing) | **No** — compute only |
| **Wavefront size** | 32 (native) + 64 (compatibility) | **64** (native) |
| **Matrix Cores** | Limited | Full matrix core array (FP16, BF16, FP8, INT8) |
| **Memory** | GDDR6 (up to 24 GB) | **HBM3** (up to 192 GB, 5.3 TB/s) |
| **Multi-GPU** | CrossFire (consumer) | **Infinity Fabric** (data center) |
| **ECC** | No | Yes |
| **FP64** | 1/16 rate | **Full rate** (important for scientific computing) |
| **ROCm support** | Limited | **Full** |
| **Price** | $500–1,000 | $10,000–25,000 |

**Why this matters for AI hardware engineers:**
- When you read "AMD GPU for AI," it means **CDNA / Instinct**, not RDNA / Radeon
- CDNA's matrix cores are AMD's answer to NVIDIA tensor cores — the hardware you'd study (or compete with) in Phase 5F (AI Chip Design)
- MI300X's chiplet design (8 XCDs on one package) is a reference for advanced packaging (L8)

### MI300X Architecture (Current Flagship)

```
┌─────────────────────────────────────────────────┐
│              MI300X Package                      │
│                                                  │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐              │
│  │ XCD │ │ XCD │ │ XCD │ │ XCD │  ← 8 Compute │
│  │  0  │ │  1  │ │  2  │ │  3  │    Chiplets   │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘    (CDNA 3)  │
│     │       │       │       │                    │
│  ┌──┴───────┴───────┴───────┴──┐                │
│  │      Infinity Fabric         │                │
│  │    (inter-chiplet network)   │                │
│  └──┬───────┬───────┬───────┬──┘                │
│  ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐              │
│  │ XCD │ │ XCD │ │ XCD │ │ XCD │               │
│  │  4  │ │  5  │ │  6  │ │  7  │               │
│  └─────┘ └─────┘ └─────┘ └─────┘               │
│                                                  │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐          │
│  │ HBM3 │ │ HBM3 │ │ HBM3 │ │ HBM3 │  192 GB │
│  │stack │ │stack │ │stack │ │stack │  5.3TB/s │
│  └──────┘ └──────┘ └──────┘ └──────┘          │
└─────────────────────────────────────────────────┘
```

Each XCD contains 38 Compute Units (CUs), each CU has 64 stream processors + matrix cores. Total: 304 CUs, 19,456 stream processors.

---

## 2. CUDA vs HIP — Almost Identical API

| CUDA | HIP | Notes |
|------|-----|-------|
| `cudaMalloc()` | `hipMalloc()` | Same signature |
| `cudaMemcpy()` | `hipMemcpy()` | Same signature |
| `cudaStream_t` | `hipStream_t` | Same concept |
| `__shared__` | `__shared__` | Identical |
| `__syncthreads()` | `__syncthreads()` | Identical |
| `threadIdx.x` | `threadIdx.x` | Identical |
| `cudaDeviceSynchronize()` | `hipDeviceSynchronize()` | Same |
| `cudaLaunchKernel()` | `hipLaunchKernelGGL()` | Slightly different |
| Warp size: 32 | **Wavefront size: 64** | **Key architectural difference** |

### HIP Kernel Example

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

If you know CUDA, you already know 95% of HIP. The critical differences:

**1. Wavefront = 64 threads (vs CUDA warp = 32):**
This affects any code that uses warp-level primitives:
```cpp
// CUDA: assumes warp = 32
unsigned mask = __ballot_sync(0xFFFFFFFF, predicate);  // 32-bit mask

// HIP on AMD: wavefront = 64
unsigned long long mask = __ballot(predicate);          // 64-bit mask
```
Reductions, shuffles, and vote operations all need adjustment for the wider wavefront.

**2. Shared memory (LDS) banks:** 64 banks on AMD (vs 32 on NVIDIA). Different bank conflict patterns — code that avoids conflicts on NVIDIA may still conflict on AMD, and vice versa.

**3. Matrix Cores (not Tensor Cores):** AMD uses `rocWMMA` (Wavefront Matrix Multiply-Accumulate) or the Composable Kernel (CK) library instead of NVIDIA's `mma.sync` / CUTLASS.

---

## 3. HIPIFY — Automatic CUDA → HIP Conversion

HIPIFY is AMD's tool suite for converting CUDA code to HIP. It's the fastest way to port existing CUDA projects to run on AMD GPUs.

### Two Conversion Tools

| Tool | How it works | Accuracy | Speed |
|------|-------------|----------|-------|
| **hipify-clang** | Uses Clang's AST parser to understand CUDA code semantically | ~95% accurate | Slower (full compilation) |
| **hipify-perl** | Regex-based find-and-replace (`cuda` → `hip`) | ~85% accurate | Very fast |

### hipify-clang (Recommended)

```bash
# Install (comes with ROCm)
sudo apt install hipify-clang

# Convert a single CUDA file
hipify-clang my_kernel.cu -o my_kernel.hip.cpp

# Convert an entire project (recursive)
hipify-clang --project-dir ./cuda_project --output-dir ./hip_project

# Show what would change without writing (dry run)
hipify-clang my_kernel.cu --print-stats
```

### hipify-perl (Quick and Dirty)

```bash
# Simple text replacement — fast but misses context-dependent conversions
hipify-perl my_kernel.cu > my_kernel.hip.cpp
```

### What HIPIFY Converts Automatically

| CUDA | Converted to HIP | Status |
|------|------------------|--------|
| `cuda*.h` headers | `hip/hip_runtime.h` | Automatic |
| `cudaMalloc/Free/Memcpy` | `hipMalloc/Free/Memcpy` | Automatic |
| `cudaStream_t`, `cudaEvent_t` | `hipStream_t`, `hipEvent_t` | Automatic |
| `__syncthreads()` | `__syncthreads()` | No change needed |
| `atomicAdd()` | `atomicAdd()` | No change needed |
| `cuBLAS` calls | `rocBLAS` calls | **Partial** — API differs slightly |
| `cuDNN` calls | `MIOpen` calls | **Manual** — different API design |
| `cuFFT` calls | `rocFFT` calls | **Partial** |

### What HIPIFY Cannot Convert (Manual Work Required)

| CUDA feature | Why it can't auto-convert | Manual fix |
|-------------|--------------------------|-----------|
| **Inline PTX assembly** | PTX is NVIDIA-specific ISA | Rewrite using HIP intrinsics or GCN assembly |
| **`__ballot_sync(0xFFFFFFFF, ...)`** | Assumes 32-bit warp mask | Use `__ballot()` (64-bit on AMD) |
| **`__shfl_sync(mask, val, lane)`** | Warp size dependent | Use `__shfl(val, lane)` with wavefront awareness |
| **Cooperative groups** | NVIDIA-specific extension | Use HIP cooperative groups (subset supported) |
| **cuDNN → MIOpen** | Completely different API | Rewrite using MIOpen API |
| **Thrust** | NVIDIA template library | Use rocThrust (mostly compatible) |
| **CUTLASS** | NVIDIA template library | Use AMD Composable Kernel (CK) |

### Typical HIPIFY Workflow for a Real Project

```bash
# Step 1: Run hipify-clang on the entire project
hipify-clang --project-dir ./my_cuda_project --output-dir ./my_hip_project

# Step 2: Try to build
cd my_hip_project
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=hipcc
make -j$(nproc)

# Step 3: Fix compilation errors (usually 5-15% of files need manual fixes)
# Common issues:
#   - warp size assumptions (32 → 64)
#   - library API differences (cuBLAS → rocBLAS)
#   - missing HIP equivalents for NVIDIA extensions

# Step 4: Run and validate
./my_program
# Compare output with CUDA version for correctness

# Step 5: Profile and optimize
rocprof --stats ./my_program
omniperf analyze -p ./profile_output
```

---

## 4. AMD GPU Architecture — CDNA Compute Unit Deep Dive

Understanding the CU (Compute Unit) is essential for writing fast HIP kernels — it's AMD's equivalent of NVIDIA's SM.

```
┌───────────────────────────────────────────┐
│           CDNA 3 Compute Unit (CU)        │
│                                           │
│  ┌─────────────────────────────────────┐  │
│  │  4x SIMD Units (16-wide each)      │  │
│  │  = 64 stream processors total      │  │
│  │  Execute one wavefront (64 threads) │  │
│  └─────────────────────────────────────┘  │
│                                           │
│  ┌─────────────────────────────────────┐  │
│  │  Matrix Cores                       │  │
│  │  FP16, BF16, FP8, INT8 MMA         │  │
│  │  (like NVIDIA Tensor Cores)         │  │
│  └─────────────────────────────────────┘  │
│                                           │
│  ┌──────────────┐  ┌──────────────────┐  │
│  │  Scalar Unit │  │  LDS (64 KB)     │  │
│  │  (control)   │  │  (shared memory) │  │
│  └──────────────┘  └──────────────────┘  │
│                                           │
│  ┌─────────────────────────────────────┐  │
│  │  Vector Register File               │  │
│  │  256 KB (vs ~256 KB per SM)         │  │
│  └─────────────────────────────────────┘  │
│                                           │
│  ┌──────────────┐  ┌──────────────────┐  │
│  │  L1 Cache    │  │  Scheduler       │  │
│  │  (16 KB)     │  │  (wavefront mgr) │  │
│  └──────────────┘  └──────────────────┘  │
└───────────────────────────────────────────┘
```

### NVIDIA SM vs AMD CU Comparison

| | NVIDIA SM (Hopper) | AMD CU (CDNA 3) |
|---|---|---|
| **ALUs per unit** | 128 CUDA cores | 64 stream processors |
| **Thread group** | Warp (32 threads) | Wavefront (64 threads) |
| **Shared memory** | 0–228 KB (configurable with L1) | 64 KB LDS (fixed) |
| **Register file** | 256 KB | 256 KB |
| **Matrix unit** | Tensor Cores (4th gen) | Matrix Cores |
| **L1 cache** | Shared with SMEM (configurable) | 16 KB (separate from LDS) |
| **Max wavefronts/warps** | 64 warps per SM | 32 wavefronts per CU |
| **Occupancy model** | Warps hide latency | Wavefronts hide latency (fewer but wider) |

### Key Optimization Differences When Porting from CUDA

- **Block size:** On NVIDIA, 256 threads = 8 warps. On AMD, 256 threads = 4 wavefronts. Fewer wavefronts means less latency hiding — consider using 512 or 1024 threads per block on AMD.
- **LDS vs shared memory:** AMD's LDS is fixed at 64 KB per CU (not configurable). On NVIDIA you can trade shared memory for L1 cache. Plan your tiling accordingly.
- **Bank conflicts:** 64 LDS banks (vs 32 on NVIDIA). A stride of 2 that's conflict-free on NVIDIA causes 2-way conflicts on AMD.

---

## 5. ROCm Software Stack

```
┌──────────────────────────────────────┐
│  Your HIP Application                │
├──────────────────────────────────────┤
│  Libraries: rocBLAS, MIOpen, rocFFT  │
│             Composable Kernel (CK)   │
├──────────────────────────────────────┤
│  HIP Runtime (hiprt)                 │
├──────────────────────────────────────┤
│  ROCm Compiler (amd-clang / hipcc)   │
│  LLVM AMDGPU backend                 │
├──────────────────────────────────────┤
│  ROCr (Runtime) + ROCt (Thunk)       │
├──────────────────────────────────────┤
│  amdgpu kernel driver (Linux)        │
├──────────────────────────────────────┤
│  AMD GPU Hardware (CDNA / RDNA)      │
└──────────────────────────────────────┘
```

### ROCm Libraries (Equivalent to CUDA-X)

| CUDA-X Library | ROCm Equivalent | Notes |
|---------------|-----------------|-------|
| cuBLAS | **rocBLAS** | GEMM, BLAS routines |
| cuDNN | **MIOpen** | Different API — not a drop-in replacement |
| cuFFT | **rocFFT** | FFT routines |
| cuSPARSE | **rocSPARSE** | Sparse matrix operations |
| cuRAND | **rocRAND** | Random number generation |
| NCCL | **RCCL** | Multi-GPU collectives |
| CUTLASS | **Composable Kernel (CK)** | Custom GEMM/attention kernels |
| Thrust | **rocThrust** | Parallel algorithms (mostly compatible) |
| CUB | **hipCUB** | Block/warp primitives |
| Nsight Systems | **Omnitrace** | Timeline profiling |
| Nsight Compute | **Omniperf** | Kernel-level analysis, roofline |

### Profiling Tools

```bash
# Timeline profiling (like nsys)
omnitrace-run -- ./my_hip_program

# Kernel-level roofline analysis (like ncu)
omniperf profile -n my_run -- ./my_hip_program
omniperf analyze -p workloads/my_run
```

---

## 6. Resources

| Resource | URL |
|----------|-----|
| ROCm Documentation | https://rocm.docs.amd.com/ |
| HIP Programming Guide | https://rocm.docs.amd.com/projects/HIP/en/latest/ |
| HIPIFY | https://rocm.docs.amd.com/projects/HIPIFY/en/latest/ |
| Composable Kernel (CK) | https://github.com/ROCm/composable_kernel |
| Omniperf | https://rocm.docs.amd.com/projects/omniperf/en/latest/ |
| AMD Instinct MI300X | https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html |

---

## 7. Projects

1. **HIPIFY your CUDA kernel** — Take your CUDA vector add and matmul from Sub-Track 3. Convert to HIP using `hipify-clang`. Build with `hipcc`. Run on AMD GPU (or ROCm Docker on NVIDIA). Verify identical output.
2. **Wavefront vs warp** — Write a parallel reduction kernel. Run on both AMD (wavefront=64) and NVIDIA (warp=32). Measure how the wavefront width difference affects performance and code structure.
3. **Profile with Omniperf** — Profile your HIP tiled matmul with `omniperf`. Generate a roofline plot. Compare with NVIDIA `ncu` roofline for the same kernel.
4. **rocBLAS vs cuBLAS** — Call rocBLAS `rocblas_sgemm` for matrix multiply. Compare performance and API with cuBLAS `cublasSgemm` for the same matrix sizes.
5. **HIPIFY a real project** — Pick a small open-source CUDA project (e.g., a convolution kernel or sorting algorithm). Run the full HIPIFY workflow: convert → build → fix errors → validate → profile.

---

## Next

→ [**Sub-Track 5 — OpenCL and SYCL**](../OpenCL%20and%20SYCL/Guide.md) — portable compute across GPU, FPGA, and CPU.
