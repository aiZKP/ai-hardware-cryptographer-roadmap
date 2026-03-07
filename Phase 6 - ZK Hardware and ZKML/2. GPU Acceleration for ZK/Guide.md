# GPU Acceleration for Zero-Knowledge Proofs

> **Goal:** Master the mapping of ZK proof computation onto GPU architectures — from field arithmetic at the PTX instruction level, through MSM and NTT kernel design, to full prover frameworks like ICICLE and production systems like zkSync Airbender. By the end, you will understand why GPUs deliver 800x MSM speedup over CPUs, how to handle register pressure for 254-bit field elements, why NTT becomes the bottleneck at scale 2^26, and how real-world proving clusters achieve sub-12-second Ethereum block proofs. This guide provides the concrete numbers, kernel patterns, and architectural reasoning a hardware engineer needs to evaluate and optimize GPU-based ZK provers.

**Prerequisite:** Phase 6.1 (MSM and NTT algorithms — Pippenger's bucket method, NTT butterfly, memory access patterns). Phase 5 complete (field arithmetic, elliptic curves, proof systems). Basic CUDA familiarity (grid/block/thread model) is helpful but not required — we build it from scratch.

---

## Table of Contents

1. [GPU Architecture Fundamentals for ZK](#1-gpu-architecture-fundamentals-for-zk)
2. [Field Arithmetic on GPU — The Foundation](#2-field-arithmetic-on-gpu--the-foundation)
3. [MSM on GPU — Pippenger Meets CUDA](#3-msm-on-gpu--pippenger-meets-cuda)
4. [NTT on GPU — Butterfly Parallelism](#4-ntt-on-gpu--butterfly-parallelism)
5. [Poseidon Hash on GPU](#5-poseidon-hash-on-gpu)
6. [GPU Prover Frameworks](#6-gpu-prover-frameworks)
7. [Multi-GPU and Distributed Proving](#7-multi-gpu-and-distributed-proving)
8. [Advanced GPU Optimization Techniques](#8-advanced-gpu-optimization-techniques)
9. [GPU vs FPGA vs ASIC — Quantitative Comparison](#9-gpu-vs-fpga-vs-asic--quantitative-comparison)
10. [Real-World GPU Proving Systems](#10-real-world-gpu-proving-systems)
11. [Projects](#11-projects)
12. [Resources](#12-resources)

---

## 1. GPU Architecture Fundamentals for ZK

### 1.1 Why GPUs for ZK?

ZK provers spend 80-95% of their time on MSM and NTT (Phase 6.1). Both operations decompose into millions of independent field arithmetic operations — exactly the workload GPUs are designed for. A single NVIDIA A100 has 6,912 CUDA cores executing in parallel, compared to 64-128 threads on a high-end CPU.

But there's a critical mismatch: **GPUs are designed for 32-bit floating-point, not 254-bit integer arithmetic.** Every ZK field multiplication must be built from dozens of 32-bit integer multiply-add instructions. Understanding this mismatch — and how to work around it — is the core of GPU ZK optimization.

### 1.2 GPU Specifications for ZK Workloads

```
GPU Specifications Relevant to ZK Provers
═══════════════════════════════════════════════════════════════════════════════
Specification          │ A100 (Ampere)    │ RTX 4090 (Ada)   │ H100 (Hopper)
───────────────────────┼──────────────────┼──────────────────┼──────────────────
SM Count               │ 108              │ 128              │ 132 (SXM: 144)
CUDA Cores (FP32)      │ 6,912            │ 16,384           │ 16,896
Warp Size              │ 32 threads       │ 32 threads       │ 32 threads
Shared Memory/SM       │ 164 KB (config)  │ ~128 KB          │ 228 KB (config)
L1 + SMEM Combined     │ 192 KB           │ ~128 KB          │ 256 KB
L2 Cache               │ 40 MB            │ 72 MB            │ 50 MB
HBM Capacity           │ 40/80 GB HBM2e   │ 24 GB GDDR6X     │ 80 GB HBM3
Memory Bandwidth       │ 2,039 GB/s       │ 1,008 GB/s       │ 3,352 GB/s (SXM)
FP32 Throughput        │ 19.5 TFLOPS      │ 82.6 TFLOPS      │ 67 TFLOPS (SXM)
Registers/SM           │ 256 KB (65,536)  │ 256 KB           │ 256 KB
Max Registers/Thread   │ 255              │ 255              │ 255
Max Threads/SM         │ 2,048            │ 1,536            │ 2,048
TDP                    │ 250-400W         │ 450W             │ 350-700W
───────────────────────┴──────────────────┴──────────────────┴──────────────────

Key insight for ZK: FP32 and Tensor Cores sit entirely idle.
ZK uses only integer execution units (IMAD, IADD3).
```

### 1.3 Memory Hierarchy — Bandwidth at Every Level

```
GPU Memory Hierarchy for ZK
═══════════════════════════════════════════════════════════════════
Level             │ Capacity (per SM) │ Bandwidth (aggregate) │ Latency
──────────────────┼───────────────────┼───────────────────────┼────────────
Registers         │ 256 KB            │ ~8 TB/s               │ 0 cycles
Shared Memory     │ 48-228 KB (conf.) │ ~15-20 TB/s           │ 20-30 cycles
L1 Cache          │ combined w/ SMEM  │ similar to SMEM       │ ~30 cycles
L2 Cache          │ 40-72 MB (total)  │ ~2-3 TB/s             │ ~200 cycles
HBM/Global Memory │ 24-80 GB          │ 1-3.35 TB/s           │ 400-600 cycles
──────────────────┴───────────────────┴───────────────────────┴────────────

For ZK workloads:
  MSM:  compute-bound (field multiply chains dominate)
  NTT:  memory-bound at large scales (data shuffling between stages)
  Hash: compute-bound (Poseidon MDS matrix multiply)
```

### 1.4 CUDA Execution Model Mapped to ZK

```
CUDA Hierarchy → ZK Workload Mapping
═════════════════════════════════════════════════════════════════════

Grid (all blocks)          │ All buckets in MSM / all NTT butterflies
  └─ Block (up to 1024     │ One bucket group / one sub-NTT of size
     threads, shared SMEM) │ ≤ 2048 elements
       └─ Warp (32 threads │ 32 parallel field operations or
          in lockstep)     │ 32 parallel bucket accumulations
            └─ Thread      │ One field multiply chain /
               (registers) │ one butterfly operation

Typical ZK kernel configurations:
  MSM bucket accumulation:  blocks of 32-256 threads, grid = num_buckets / block_size
  NTT butterfly (in-SMEM):  blocks of 1024 threads, grid = N / (2 * block_size)
  Poseidon batch hash:      blocks of 256 threads, grid = batch_size / block_size
```

### 1.5 The ZK-GPU Mismatch: Concrete Numbers

Per-operation latency comparison (NVIDIA A40 GPU vs AMD EPYC 7742 CPU, BN254 field):

```
Operation   │ CPU Cycles │ GPU Cycles │ GPU/CPU Ratio
────────────┼────────────┼────────────┼──────────────
FF_add      │ 29         │ 244        │ 8.4x slower
FF_sub      │ 27         │ 217        │ 8.0x slower
FF_dbl      │ 19         │ 121        │ 6.4x slower
FF_mul      │ 402        │ 2,656      │ 6.6x slower
FF_sqr      │ 402        │ 2,633      │ 6.6x slower

Each GPU field multiply is 6.6x SLOWER than on CPU!

But the GPU has thousands of execution units running in parallel:
  A100: 108 SMs × 64 CUDA cores = 6,912 parallel multiply pipelines
  CPU:  128 threads maximum

Effective throughput: GPU wins by 100x-800x depending on scale.
```

This is the fundamental GPU trade-off for ZK: **high latency per operation, massive throughput through parallelism.** The crossover point where GPU beats CPU depends on the number of independent operations available — and MSM at scale 2^20+ provides millions of them.

---

## 2. Field Arithmetic on GPU — The Foundation

### 2.1 Montgomery Multiplication on GPU

Every ZK operation on GPU ultimately reduces to Montgomery multiplication of multi-limb integers using 32-bit GPU instructions.

**Why Montgomery form?** Standard modular multiplication requires division by the prime p, which is extremely expensive. Montgomery form stores elements as `aR mod p` (where R = 2^256 for BN254), replacing division with shifts and adds.

```
BN254 Field Element Layout on GPU (254-bit prime, stored as 8 × 32-bit limbs):

  Limb:     [L₇]  [L₆]  [L₅]  [L₄]  [L₃]  [L₂]  [L₁]  [L₀]
  Bits:    255-224 223-192 191-160 159-128 127-96  95-64  63-32  31-0
  Registers:  r7     r6     r5     r4     r3     r2     r1     r0

  Total: 8 registers per field element

BLS12-381 Field Element (381-bit prime, stored as 12 × 32-bit limbs):
  Total: 12 registers per field element
```

### 2.2 PTX Instructions — The Real Building Blocks

GPU field arithmetic is built from PTX (Parallel Thread Execution) integer multiply-add instructions:

```
Core PTX instructions for field multiplication:

  mad.lo.cc.u32  r, a, b, c    // r = low32(a*b) + c, sets carry flag
  madc.hi.cc.u32 r, a, b, c    // r = high32(a*b) + c + carry_in, sets carry

  mul.wide.u32   r, a, b       // r = a*b (32×32 → 64-bit result)

Instruction timing:
  IMAD (integer multiply-add):  4 cycle latency
  IADD3 (integer 3-input add):  2 cycle latency

IMAD constitutes 70.8% of the instruction mix for FF_mul.
```

### 2.3 Montgomery Multiplication — Instruction-Level Detail

A full BN254 Montgomery multiply requires a schoolbook 8×8 limb multiplication with interleaved reduction (CIOS — Coarsely Integrated Operand Scanning):

```
Montgomery Multiply: a × b mod p  (BN254, 8 limbs)
════════════════════════════════════════════════════════════════

Algorithm (CIOS — one outer loop):
  for i = 0 to 7:
    // Multiply: accumulate a[0..7] × b[i] into temp[i..i+8]
    carry = 0
    for j = 0 to 7:
      (carry, temp[i+j]) = a[j] × b[i] + temp[i+j] + carry    // mad.lo.cc + madc.hi.cc
    temp[i+8] = carry

    // Montgomery reduction: cancel lowest limb
    m = temp[i] × N_PRIME_0    // N_PRIME_0 = -p⁻¹ mod 2³²
    carry = 0
    for j = 0 to 7:
      (carry, temp[i+j]) = m × MODULUS[j] + temp[i+j] + carry  // mad.lo.cc + madc.hi.cc
    temp[i+8] += carry

  // Final: conditional subtraction if result ≥ p
  result = temp[8..15]
  if result ≥ p: result -= p

Instruction count per multiply:
  Schoolbook: 8 × 8 = 64 multiply pairs          → 128 IMAD instructions
  Reduction:  8 × 8 = 64 multiply pairs           → 128 IMAD instructions
  Carry propagation + conditional subtract         → ~30 IADD3 instructions
  ─────────────────────────────────────────────────────────────────────
  Total:  ~130-150 PTX instructions per BN254 field multiply

At 4 cycles per IMAD: ~600 cycles compute + pipeline stalls → ~2,656 measured cycles
```

### 2.4 PTX Assembly for Inner Loop

Production ZK GPU code uses inline PTX for the inner multiply loop:

```c
// Inner product: accumulate a[j] × b[i] with carry chain
asm volatile(
    "mad.lo.cc.u32  %0, %2, %3, %1;\n\t"  // temp_lo = a[j]*b[i] + temp[i+j]
    "madc.hi.cc.u32 %4, %2, %3, %5;\n\t"  // carry   = high(a[j]*b[i]) + carry_in
    : "=r"(temp[j]), "=r"(carry)
    : "r"(a[j]), "r"(b_i), "r"(temp[j]), "r"(carry)
);

// Why PTX instead of C++?
// NVCC compiler output for multi-limb arithmetic is 10-30% slower
// than hand-tuned PTX due to suboptimal carry chain management.
// Every ZK library (ICICLE, sppark, cuZK) uses PTX for field ops.
```

### 2.5 Field Element Sizes and Register Pressure

```
Register Pressure Analysis by Field Size
═══════════════════════════════════════════════════════════════════════════
Field           │ Bits │ Limbs │ Regs/Element │ Regs/Multiply │ Occupancy
────────────────┼──────┼───────┼──────────────┼───────────────┼──────────
BabyBear        │ 31   │ 1     │ 1            │ ~5-8          │ High (>75%)
Goldilocks      │ 64   │ 2     │ 2            │ ~10-15        │ High (~60%)
BN254           │ 254  │ 8     │ 8            │ ~30-40        │ Low (~25%)
BLS12-381       │ 381  │ 12    │ 12           │ ~45-60        │ Very low
────────────────┴──────┴───────┴──────────────┴───────────────┴──────────

Why occupancy matters:
  Max registers per SM:   65,536 (= 256 KB)
  Max threads per SM:     2,048

  BN254 MSM kernel: 216-244 registers/thread
    → 65,536 / 244 = 268 threads per SM
    → 268 / 2,048 = 13% occupancy (very low!)

  BabyBear NTT kernel: ~30 registers/thread
    → 65,536 / 30 = 2,184 → capped at 2,048 threads
    → 100% theoretical occupancy

  Counterpoint: occupancy > 25% rarely helps for ZK MSM kernels
  because the workload is compute-bound with long dependency chains.
  Extra warps just compete for the same integer ALUs.
```

### 2.6 Small Fields: The GPU Sweet Spot

Small fields exploit native GPU word sizes, delivering massive performance gains:

```
BabyBear (p = 15 × 2²⁷ + 1 = 2,013,265,921)
═══════════════════════════════════════════════
  Fits in a single 32-bit word (< 2³¹)
  Multiplication: one mul.wide.u32 + single reduction step
  Reduction: result mod p using shift/subtract (p has special structure)

  Performance: up to 40x faster than BN254 field ops on GPU
  Used by: RISC Zero, SP1/Succinct
  Supported by: ICICLE (native BabyBear backend)

Goldilocks (p = 2⁶⁴ - 2³² + 1)
════════════════════════════════
  Fits in 2 × 32-bit limbs (or 1 × 64-bit word)
  Special structure: reduction uses only shifts and adds
  Has 2³²-th root of unity → NTTs up to degree 2³²

  Used by: Polygon zkEVM
  X Layer reimplemented Goldilocks modular ops in CUDA

Performance comparison for field multiply throughput:
  BabyBear:    ~40x faster than BN254 per operation
  Goldilocks:  ~10x faster than BN254 per operation
  BN254:       baseline (130-150 PTX instructions)
  BLS12-381:   ~2.3x slower than BN254 (12² vs 8² schoolbook)

This is why modern proof systems (STARKs, Plonky3) prefer small fields:
the GPU performance advantage is enormous.
```

### 2.7 CPU vs GPU Field Multiply Throughput

```
Aggregate throughput comparison (BN254, estimated):

  CPU (AMD EPYC 7742, 128 threads):
    402 cycles/multiply × 1 multiply/thread × 128 threads
    = 51,456 cycles per batch of 128 multiplies
    At 3.4 GHz: ~8.5 million multiplies/second

  GPU (NVIDIA A100, 6,912 CUDA cores):
    2,656 cycles/multiply BUT can issue ~1,000 concurrent warps
    ~200 million multiplies/second (estimated)

  GPU/CPU throughput ratio: ~23x for raw field multiplies
  For MSM (structured parallelism): up to 800x
```

---

## 3. MSM on GPU — Pippenger Meets CUDA

### 3.1 Recap: Pippenger's Four Phases (from Phase 6.1)

```
Pippenger's Bucket Method for MSM = Σ sᵢ·Pᵢ
══════════════════════════════════════════════

Phase 1: Scalar Decomposition
  Split each λ-bit scalar into ⌈λ/c⌉ windows of c bits
  → Generates bucket indices (0 to 2ᶜ-1) per window

Phase 2: Bucket Assignment
  Assign point Pᵢ to bucket[window_value] for each window

Phase 3: Bucket Accumulation
  Sum all points assigned to each bucket: bucket[j] = Σ Pᵢ

Phase 4: Bucket Reduction + Window Combination
  Within each window: running sum or multiply-by-index
  Across windows: double-and-add (shift by c bits)
```

### 3.2 GPU Mapping — The Four-Phase CUDA Pipeline

```
Pippenger on GPU: Phase → Kernel Mapping
═══════════════════════════════════════════════════════════════════════

Phase 1: SCALAR DECOMPOSITION KERNEL
  ┌─────────────────────────────────────────────────────────┐
  │ Grid: N/256 blocks × 256 threads                        │
  │ Each thread: decompose 1 scalar into ⌈λ/c⌉ chunks      │
  │ Output: (bucket_index, point_index) pairs                │
  │ Memory: coalesced read of scalar array, coalesced write  │
  │ Time: <1% of total (fully parallel, lightweight)         │
  └─────────────────────────────────────────────────────────┘
                          │
                          ▼
Phase 1.5: SORT BY BUCKET INDEX (critical for GPU efficiency)
  ┌─────────────────────────────────────────────────────────┐
  │ Use CUB/Thrust radix sort on (bucket_index, point_index)│
  │ Result: points grouped by bucket, largest buckets first  │
  │ Why: ensures coalesced memory access in Phase 3          │
  │       and uniform warp workloads (no divergence)         │
  │ Time: 5-15% of total                                     │
  └─────────────────────────────────────────────────────────┘
                          │
                          ▼
Phase 3: BUCKET ACCUMULATION KERNEL (the dominant cost)
  ┌─────────────────────────────────────────────────────────┐
  │ Small buckets: 1 thread per bucket                       │
  │ Large buckets: multiple threads per bucket               │
  │   (ICICLE: large_bucket_factor controls threshold)       │
  │                                                          │
  │ Each thread:                                             │
  │   acc = POINT_AT_INFINITY                                │
  │   for each point in my bucket:                           │
  │     acc = point_add_mixed(acc, points[sorted_index])     │
  │   buckets[my_id] = acc                                   │
  │                                                          │
  │ Mixed addition: Jacobian + Affine (saves 1 field mul)    │
  │ Time: 60-80% of total MSM time                           │
  └─────────────────────────────────────────────────────────┘
                          │
                          ▼
Phase 4: BUCKET REDUCTION KERNEL
  ┌─────────────────────────────────────────────────────────┐
  │ Per window: running sum from bucket[2ᶜ-1] down to [1]   │
  │   running_sum += bucket[i]                               │
  │   result += running_sum                                  │
  │ → Computes Σ j·bucket[j] without scalar multiplies       │
  │                                                          │
  │ Then combine windows: double-and-add across windows      │
  │ Time: 5-15% of total (sequential per window)             │
  └─────────────────────────────────────────────────────────┘
```

### 3.3 Point Representation on GPU

Choosing the right coordinate system is critical for GPU register pressure:

```
Point Representation Comparison (cost per point operation)
═══════════════════════════════════════════════════════════════════════
Representation  │ Coords │ Regs (BN254) │ Add Cost    │ Double Cost │ Notes
────────────────┼────────┼──────────────┼─────────────┼─────────────┼──────────
Affine (x,y)    │ 2      │ 16           │ 3 mul       │ 2 mul       │ Needs inv!
Jacobian (X:Y:Z)│ 3      │ 24           │ 7 mul       │ 2 mul       │ Standard
XYZZ (X:Y:ZZ:  │ 4      │ 32           │ 8 mul*      │ 6 mul       │ 17 total ops
  ZZZ)          │        │              │             │             │ per add (best)
────────────────┴────────┴──────────────┴─────────────┴─────────────┴──────────

  *XYZZ mixed addition (one input affine): cheaper than full XYZZ+XYZZ

GPU preference:
  sppark:     XYZZ — best balance of operations vs register usage
  ICICLE:     Jacobian with mixed-add (affine base points)
  CycloneMSM: Affine with batch inversion (Montgomery's trick)

Register budget for MSM kernel (BN254, XYZZ):
  Accumulator point:  32 registers (4 × 8 limbs)
  Input point:        16 registers (2 × 8 limbs, affine)
  Temporaries:        ~20 registers (for multiply)
  ─────────────────────────────────────────────
  Total:              ~68 registers minimum per active multiply
  With pipeline state: 216-244 registers per thread (measured)
```

### 3.4 The Sorting Optimization — Why It Matters

The ZPrize 2022 winning implementations (Yrrid/Matter Labs) showed that **sorting scalars by bucket index** before accumulation is critical for GPU performance:

```
Without sorting:
  Thread 0 reads bucket 5271 → point at address 0x7A000
  Thread 1 reads bucket 12   → point at address 0x00C00
  Thread 2 reads bucket 9844 → point at address 0xF2000
  → Scattered memory access, no coalescing, L2 thrashing

With sorting (by bucket index):
  Thread 0 reads bucket 0    → point at address 0x00000  ─┐
  Thread 1 reads bucket 0    → point at address 0x00040   ├─ Coalesced!
  Thread 2 reads bucket 0    → point at address 0x00080  ─┘
  Thread 3 reads bucket 1    → point at address 0x000C0
  → Sequential memory access, full cache line utilization

Additional benefit: workload balancing
  Sort by bucket SIZE (descending) → largest buckets processed first
  → Warps execute balanced workloads → minimal tail effect
  → <50% branch efficiency → >90% branch efficiency
```

### 3.5 MSM Benchmarks — GPU Speedup Over CPU

From ZKProphet (BN254, NVIDIA A40 GPU vs AMD EPYC 7742 CPU):

```
MSM GPU Speedup by Scale and Library
═══════════════════════════════════════════════════════
Scale  │ Fastest Library │ GPU Time      │ Speedup vs CPU
───────┼─────────────────┼───────────────┼────────────────
2^15   │ sppark          │ ~1.2 ms       │ 34.1x
2^18   │ sppark          │ ~4.5 ms       │ 78.1x
2^20   │ sppark          │ ~12 ms        │ 176.1x
2^22   │ ymc             │ ~35 ms        │ 408.1x
2^24   │ ymc             │ ~120 ms       │ 693.2x
2^26   │ ymc             │ ~800 ms       │ 799.5x
───────┴─────────────────┴───────────────┴────────────────

Key observations:
  - Speedup grows with scale (more parallelism to exploit)
  - Different libraries win at different scales:
      sppark: fastest at 2^15-2^20 (low overhead, XYZZ coords)
      ymc:    fastest at 2^22-2^26 (signed-digit endomorphism)
  - ymc has ~30% preprocessing overhead at small scales
  - At 2^26: GPU is almost 800x faster than CPU
```

### 3.6 ZPrize Competition Results

```
ZPrize MSM Competition Results
═══════════════════════════════════════════════════════════════════

ZPrize 2022 (Prize 1a: GPU MSM, $650,000):
  Task:     MSM of size 2^26 on BLS12-377
  Baseline: 5.86 seconds
  Winners:  Yrrid Software & Matter Labs (tied)
  Result:   2.52 seconds (131.85% improvement)
  Technique: 23-bit window Pippenger, sorted bucket lists,
             custom FF/EC routines in PTX

ZPrize 2023 (Beat the Best: GPU MSM):
  Winner:         Marco Zhou (StorSwift): 430 ms (17.5% improvement)
  Special mention: Yrrid/Snarkify: 367 ms (28.6% improvement, late)
  Challenge:      Optimize for TWO separate curves simultaneously

Progression: 5.86s → 2.52s → 0.43s in two years
  → 13.6x improvement through pure software optimization on same hardware
```

### 3.7 ICICLE MSM Benchmarks

ICICLE MSM performance on RTX 3090 Ti (BLS12-377):

```
ICICLE MSM Performance (BLS12-377, RTX 3090 Ti)
════════════════════════════════════════════════════════════════
MSM Size │ Batch │ Precompute │ c  │ Memory (GB) │ Time (ms)
─────────┼───────┼────────────┼────┼─────────────┼──────────
2^10     │ 1     │ 23         │ 11 │ 0.003       │ 1.76
2^16     │ 1     │ 1          │ 13 │ 0.02        │ 3.4
2^20     │ 1     │ 1          │ 16 │ 0.25        │ 14.5
2^22     │ 1     │ 1          │ 17 │ 1.64        │ 68
2^24     │ 1     │ 7          │ 21 │ 12.4        │ 199
─────────┴───────┴────────────┴────┴─────────────┴──────────

With precompute_factor=7 at 2^24:
  Precompute 2^7 - 1 = 127 additional points per base point
  Memory cost: 12.4 GB (trades memory for compute)
  Window size c=21 becomes efficient (fewer buckets to reduce)
```

---

## 4. NTT on GPU — Butterfly Parallelism

### 4.1 NTT Structure Recap (from Phase 6.1)

```
NTT of size N = 2^n has n stages, each with N/2 butterfly operations:

  Stage s (0-indexed): N/2 butterflies with stride 2^s

  Each butterfly (Cooley-Tukey DIT):
    a' = a + ω·b     (twiddle multiply + field add)
    b' = a - ω·b     (field subtract, reuses ω·b)

  Total: N/2 × log₂(N) butterflies
  For N = 2^24: 12,582,912 × 24 = 201,326,592 butterflies
```

### 4.2 Stage-by-Stage GPU Analysis

The key insight for NTT on GPU: **early stages and late stages have completely different characteristics.**

```
NTT Stage Analysis (N = 2^24, 24 stages)
═══════════════════════════════════════════════════════════════════════

Early stages (s = 0..11):
  Stride: 1, 2, 4, ..., 2048
  ┌─────────────────────────────────────────────────────┐
  │ Butterflies access elements within 4096-element     │
  │ blocks → fit entirely in shared memory              │
  │ N/2 = 8,388,608 independent butterflies per stage   │
  │ → Perfect GPU parallelism                           │
  │ → Coalesced global memory access (sequential load)  │
  │ → No bank conflicts in shared memory (small stride) │
  └─────────────────────────────────────────────────────┘
  GPU mapping: one thread block per sub-NTT, all stages in SMEM

Middle stages (s = 12..17):
  Stride: 4096, 8192, ..., 131072
  ┌─────────────────────────────────────────────────────┐
  │ Butterfly pairs span multiple thread blocks          │
  │ → Requires global memory synchronization             │
  │ → Scattered access patterns → L2 cache misses        │
  │ → This is where NTT becomes memory-bound on GPU      │
  └─────────────────────────────────────────────────────┘
  GPU mapping: must read/write global memory between stages

Late stages (s = 18..23):
  Stride: 262144, ..., 8388608
  ┌─────────────────────────────────────────────────────┐
  │ Each butterfly accesses elements far apart in memory │
  │ → Severe cache thrashing (stride > L2 cache)         │
  │ → Very low memory bandwidth utilization              │
  │ → But only N/2 butterflies per stage (still parallel)│
  └─────────────────────────────────────────────────────┘
  GPU mapping: bandwidth-limited, each access is a cache miss

Result: NTT is compute-bound at small scales, memory-bound at large scales.
```

### 4.3 Shared Memory NTT Kernel Pattern

For sub-NTTs that fit in shared memory (up to ~4096 elements per block with BN254):

```c
// Radix-2 NTT butterfly kernel (shared memory stages)
__global__ void ntt_butterfly_kernel(
    FieldElement* data,              // In-place NTT
    const FieldElement* twiddles,    // Precomputed twiddle factors
    int log_n,                       // log2(N)
    int stage                        // Current stage
) {
    extern __shared__ FieldElement smem[];
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x * 2;

    // Coalesced load into shared memory
    smem[2*tid]     = data[block_offset + 2*tid];
    smem[2*tid + 1] = data[block_offset + 2*tid + 1];
    __syncthreads();

    // Butterfly for current stage
    int half = 1 << stage;
    int group = tid / half;
    int pos = tid % half;
    int i = group * 2 * half + pos;
    int j = i + half;

    FieldElement w = twiddles[group];       // Or generate on-the-fly
    FieldElement t = field_mul(w, smem[j]); // The expensive step

    smem[j] = field_sub(smem[i], t);
    smem[i] = field_add(smem[i], t);
    __syncthreads();

    // Coalesced write back
    data[block_offset + 2*tid]     = smem[2*tid];
    data[block_offset + 2*tid + 1] = smem[2*tid + 1];
}
```

### 4.4 Radix-256: Fusing 8 Stages

Bellperson (Filecoin's prover) uses a radix-256 Cooley-Tukey approach — processing 8 NTT stages in a single kernel launch:

```
Radix-256 Optimization
══════════════════════════════════════════════════════════════

Standard approach (radix-2): 24 kernel launches for N = 2^24
  Each launch: read from global → compute in SMEM → write to global
  Global memory round-trips: 24

Radix-256 approach: 3 kernel launches (24/8 = 3 passes)
  Each launch: process 8 stages entirely within shared memory
  Global memory round-trips: 3

Memory traffic reduction: 8x fewer global memory accesses
  → This is why bellperson wins at large NTT scales

Configuration: blocks of 256 threads processing 256-element sub-NTTs
  Shared memory: 256 × 32 bytes (BN254) = 8 KB per block (fits easily)
  Kernel 1: stages 0-7   (local, fully in SMEM)
  Kernel 2: stages 8-15  (requires global shuffle between kernels)
  Kernel 3: stages 16-23 (requires global shuffle between kernels)
```

### 4.5 On-the-Fly Twiddle Factor Generation

Precomputing all twiddle factors requires significant memory and bandwidth. An alternative:

```
Twiddle Factor Strategy Comparison
═══════════════════════════════════════════════════════════════

Precomputed table (standard):
  Memory: N/2 field elements = 2^23 × 32 bytes = 256 MB for N=2^24 on BN254
  Bandwidth: must stream twiddle table from global memory each stage
  Latency: one global memory read per butterfly

On-the-fly generation (4.2x speedup from IACR 2023/1410):
  Store only "seed" twiddles: ω, ω², ω⁴, ..., ω^{2^{n-1}}
  Memory: n field elements × 32 bytes = 768 bytes for N=2^24
  Compute: 1 field multiply per butterfly (multiply by seed)
  → Trades compute for bandwidth — wins when memory-bound

  For N = 2^24 on GPU:
    Precomputed: 256 MB of twiddle reads per stage
    On-the-fly:  768 bytes BRAM + 1 extra multiply/butterfly
    → At large scales, the multiply is free (hidden behind memory latency)
    → 4.2x measured NTT speedup
```

### 4.6 NTT GPU Speedup Over CPU

From ZKProphet (BN254, NVIDIA A40):

```
NTT GPU Speedup by Scale and Library
══════════════════════════════════════════════════════════════
Scale  │ Fastest Library │ GPU Time      │ Speedup vs CPU
───────┼─────────────────┼───────────────┼────────────────
2^15   │ bellperson      │ ~0.5 ms       │ 12.5x
2^18   │ cuZK            │ ~3 ms         │ 20.4x
2^20   │ cuZK            │ ~12 ms        │ 35.4x
2^22   │ cuZK            │ ~50 ms        │ 50.6x
2^24   │ bellperson      │ ~250 ms       │ 40.5x
2^26   │ bellperson      │ ~2 sec        │ 24.3x
───────┴─────────────────┴───────────────┴────────────────

Critical observation:
  NTT speedup PEAKS at 50.6x (scale 2^22) then DECREASES.
  At 2^26: only 24.3x (vs 800x for MSM at same scale).

Why? NTT becomes memory-bandwidth-bound at large scales.
  2^26 field elements × 32 bytes = 2 GB of data
  Each stage shuffles the entire dataset
  24 stages × 2 GB = 48 GB of memory traffic
  At 1 TB/s bandwidth: 48 ms compute, but actual: ~2 sec
  → Memory access patterns (scattered at large strides) waste bandwidth

At scale 2^26, NTT constitutes 91% of total prover runtime.
This makes NTT — not MSM — the primary optimization target at scale.
```

### 4.7 Four-Step NTT for Multi-GPU

When NTT exceeds single-GPU memory or needs to span multiple GPUs:

```
Four-Step NTT for Large Transforms
═══════════════════════════════════════════════════════════════

View N = N₁ × N₂ elements as an N₁ × N₂ matrix:

Step 1: Perform N₂ independent NTTs on rows (size N₁ each)
  → Rows are independent → distribute across GPUs
  → Each row NTT fits in shared memory

Step 2: Multiply by twiddle factors (element-wise)
  → Embarrassingly parallel

Step 3: Transpose the matrix
  → THE BOTTLENECK for multi-GPU
  → Requires all-to-all communication between GPUs
  → PCIe: 32 GB/s, NVLink: 600-900 GB/s

Step 4: Perform N₁ independent NTTs on rows (size N₂ each)
  → Same as Step 1

Multi-GPU communication analysis (N = 2^26, BN254):
  Data size: 2^26 × 32 bytes = 2 GB
  Transpose: each GPU must send/receive 2 GB / K data (K = num GPUs)
  On 4 GPUs via PCIe 4.0: 500 MB / 32 GB/s = 15.6 ms per GPU
  On 4 GPUs via NVLink 3.0: 500 MB / 600 GB/s = 0.83 ms per GPU
  → NVLink is 19x faster for the transpose step
```

---

## 5. Poseidon Hash on GPU

### 5.1 Why Poseidon on GPU?

Poseidon is the ZK-friendly hash function (Phase 5.3). STARK provers and Merkle tree builders call Poseidon millions of times. Unlike SHA-256 (25,000+ R1CS constraints), Poseidon has ~300 constraints — but its algebraic structure maps well to GPU parallel field arithmetic.

### 5.2 Poseidon Structure for GPU

```
Poseidon Hash (width t, R_f full rounds, R_p partial rounds)
═══════════════════════════════════════════════════════════════

State: t field elements [s₀, s₁, ..., s_{t-1}]

Full round (applied R_f/2 times at start and end):
  1. Add round constants:  sᵢ += rcᵢ           (t field adds)
  2. S-box on ALL elements: sᵢ = sᵢ^α          (t field exponentiations)
     α = 5: x² = x·x, x⁴ = x²·x², x⁵ = x⁴·x  (3 field muls per element)
  3. MDS matrix multiply: s' = M · s            (t² field muls)

Partial round (applied R_p times in the middle):
  1. Add round constants:  sᵢ += rcᵢ
  2. S-box on s₀ ONLY:    s₀ = s₀^α            (1 exponentiation)
  3. Sparse MDS multiply                        (fewer muls than full MDS)

Typical parameters (BN254, t=3):
  R_f = 8 full rounds, R_p = 57 partial rounds
  Per hash: 8 × 3 × (3 + 9) + 57 × (3 + ~3) = 288 + 342 = 630 field muls

GPU parallelism:
  Batch N hashes → N × t independent state elements
  Full rounds: N × t parallel S-box computations
  MDS multiply: N parallel matrix-vector products
  → Map each hash to one thread, or each state element to one thread
```

### 5.3 GPU Poseidon Kernel Pattern

```c
__global__ void poseidon_hash_kernel(
    const FieldElement* inputs,     // Batch of preimages
    FieldElement* outputs,          // Hash outputs
    const FieldElement* rc,         // Round constants
    const FieldElement* mds,        // t × t MDS matrix
    int batch_size, int t,
    int R_f, int R_p
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    FieldElement state[MAX_WIDTH]; // t elements in REGISTERS
    for (int i = 0; i < t; i++)
        state[i] = inputs[idx * t + i];

    int rc_offset = 0;

    // First R_f/2 full rounds
    for (int r = 0; r < R_f / 2; r++) {
        for (int i = 0; i < t; i++)
            state[i] = field_add(state[i], rc[rc_offset++]);
        for (int i = 0; i < t; i++)
            state[i] = sbox(state[i]);      // x^5
        mds_multiply(state, mds, t);         // M × state
    }

    // R_p partial rounds (S-box only on state[0])
    for (int r = 0; r < R_p; r++) {
        for (int i = 0; i < t; i++)
            state[i] = field_add(state[i], rc[rc_offset++]);
        state[0] = sbox(state[0]);
        sparse_mds_multiply(state, r, t);
    }

    // Last R_f/2 full rounds (same as first)
    for (int r = 0; r < R_f / 2; r++) { /* ... */ }

    outputs[idx] = state[1]; // Output element
}
```

### 5.4 Poseidon GPU Performance

```
Poseidon Merkle Tree Performance
═══════════════════════════════════════════════════════════════

ICICLE on RTX 3090 Ti (BN254 Poseidon):
  Tree height 30 (~2^29 leaf elements):
    Keep 10 rows: 9.4 seconds
    Keep 29 rows: 13.7 seconds

Neptune (Filecoin) on RTX 2080 Ti:
  8-ary Merkle tree for 4 GiB input: 16 seconds
  Pure CUDA/OpenCL implementation

For context: CPU Poseidon hash for same tree: ~5-10 minutes
GPU speedup: ~30-50x for batch Poseidon hashing
```

---

## 6. GPU Prover Frameworks

### 6.1 ICICLE (Ingonyama) — The Standard GPU ZK Library

ICICLE is the most widely adopted GPU acceleration library for ZK. It provides CUDA-optimized primitives with Rust/Go/C++ bindings.

```
ICICLE Architecture Overview
═══════════════════════════════════════════════════════════════

Language:    CUDA C++ core, Rust/Go bindings (v3: also CPU backend)
License:     Apache 2.0

Supported Curves:
  BN254, BLS12-377, BLS12-381, BW6-761, Grumpkin

Supported Fields:
  BabyBear, Stark252, Mersenne31 (M31), KoalaBear

Supported Operations:
  ┌────────────────────────────┬──────────────────────────────┐
  │ Operation                  │ Notes                        │
  ├────────────────────────────┼──────────────────────────────┤
  │ MSM (Multi-Scalar Multiply)│ Batch, precompute, multi-GPU │
  │ NTT / INTT / ECNTT        │ Forward, inverse, on-curve   │
  │ Poseidon Hash              │ Arities 2, 4, 8, 11         │
  │ Merkle Tree                │ GPU-parallel tree building   │
  │ Polynomial Arithmetic      │ Eval, interpolate, divide    │
  │ Vector Field Operations    │ Add, mul, reduce (batched)   │
  └────────────────────────────┴──────────────────────────────┘
```

### 6.2 ICICLE Rust API Example

```rust
use icicle_runtime::{self, Device};
use icicle_bn254::curve::{CurveCfg, ScalarCfg, G1Projective};
use icicle_core::msm::{msm, MSMConfig};
use icicle_core::ntt::{ntt, NTTConfig, NTTDir};

fn main() {
    // Select CUDA device
    icicle_runtime::load_backend("path/to/cuda_backend");
    let device = Device::new("CUDA", 0); // GPU 0
    icicle_runtime::set_device(&device);

    // ----- MSM -----
    let n = 1 << 20; // 2^20 points
    let scalars: Vec<ScalarCfg> = generate_random_scalars(n);
    let points: Vec<G1Projective> = generate_random_points(n);

    let mut msm_config = MSMConfig::default();
    msm_config.c = 16;                 // Window size
    msm_config.precompute_factor = 1;  // No precomputation
    // msm_config.large_bucket_factor = 10; // Threshold for multi-thread buckets

    let mut result = G1Projective::zero();
    msm(&scalars, &points, &msm_config, &mut result).unwrap();

    // ----- NTT -----
    let ntt_size = 1 << 20;
    let mut data: Vec<ScalarCfg> = generate_random_field_elements(ntt_size);
    let ntt_config = NTTConfig::default();

    ntt(&mut data, NTTDir::kForward, &ntt_config).unwrap();
    // data now contains NTT(data)
}
```

### 6.3 ICICLE Performance Summary

```
ICICLE Performance vs CPU Baselines
═══════════════════════════════════════════════════════════════

General:
  MSM: ~50x faster than CPU baseline on average
  NTT: ~3-5x improvement over CPU (memory-bound at large scale)
  Poseidon: ~30-50x faster than CPU

ICICLE-Snark (Groth16 with ICICLE backend, size 2^22):
  MSM:     63x speedup (individual MSMs)
  FFT/NTT: 320x speedup
  VecOps:  200x speedup (with caching)
  Full Groth16: ~3x improvement (amortized over all phases)

Integration with proof systems:
  gnark:    Groth16 on BN254, BLS12-377, BLS12-381, BW6-761
  Halo 2:   ICICLE-Halo2 v2 backend
  arkworks: via Sparkworks adapter
```

### 6.4 cuZK (Zhang et al., 2022)

```
cuZK: Academic GPU ZK Implementation
═══════════════════════════════════════════════════════════════

Paper:   IACR 2022/1321 (published at IEEE TPDS)
GPU:     Evaluated on RTX 3090, A100

Three key innovations:
  1. Near-linear-speedup parallel MSM algorithm
     → Sparse-matrix representation for bucket accumulation
     → Eliminates atomic operations entirely

  2. Pipelined modular multiplication
     → Overlaps schoolbook multiply with Montgomery reduction
     → Saves ~10% on field multiply latency

  3. Reduced CPU-GPU data transfer
     → Overlaps data transfer with computation
     → Eliminates redundant transfers between MSM and NTT phases

Performance:
  MSM: 2.08x - 2.94x speedup over prior GPU state-of-the-art
  End-to-end prover: 2.65x - 4.86x speedup
  Filecoin prover: 2.18x speedup

NTT: optimal at scales 2^18-2^23
  (memory allocation failures beyond 2^23)
```

### 6.5 Other GPU Libraries

```
GPU ZK Library Landscape
═══════════════════════════════════════════════════════════════

sppark (Supranational):
  - XYZZ point representation, sorted bucket processing
  - Fastest at scales 2^15-2^20 (low overhead)
  - Peak: 176x speedup over CPU at 2^20
  - Used by Filecoin ecosystem

ymc:
  - Signed-digit endomorphism, pre-computed window weights
  - Fastest at scales 2^22-2^26 (best for large MSMs)
  - Peak: 799.5x speedup over CPU at 2^26
  - ~30% preprocessing overhead at small scales

bellperson (Filecoin):
  - Fork of bellman with GPU-accelerated FFT/multiexp
  - Supports CUDA and OpenCL via rust-gpu-tools
  - Radix-256 NTT (8 stages fused per kernel)
  - Best NTT at large scales (2^24-2^26)

era-bellman-cuda (Matter Labs):
  - GPU crypto library for zkSync prover (Boojum system)
  - CUDA-specific, optimized for zkSync's circuit structure

Rapidsnark:
  - Fast Groth16 prover for Circom circuits (C++)
  - GPU variant: ~4x speedup over CPU rapidsnark
  - CPU rapidsnark already 4-10x faster than snarkjs
```

---

## 7. Multi-GPU and Distributed Proving

### 7.1 MSM Distribution Across GPUs

MSM distributes naturally because scalar-point pairs are independent:

```
MSM Multi-GPU Strategy
═══════════════════════════════════════════════════════════════

Scalar vector: [s₀, s₁, ..., s_{N-1}]
Point vector:  [P₀, P₁, ..., P_{N-1}]

Split into K chunks (K = number of GPUs):
  GPU 0: computes Q₀ = Σ_{i=0}^{N/K-1} sᵢ · Pᵢ
  GPU 1: computes Q₁ = Σ_{i=N/K}^{2N/K-1} sᵢ · Pᵢ
  ...
  GPU K-1: computes Q_{K-1} = Σ_{i=(K-1)N/K}^{N-1} sᵢ · Pᵢ

Final: Q = Q₀ + Q₁ + ... + Q_{K-1}  (K-1 point additions, negligible)

Communication overhead: MINIMAL
  Each GPU receives its chunk of scalars + points at kernel launch
  Only K-1 point additions at the end (< 1 microsecond)
  → MSM scales nearly linearly with GPU count

ICICLE multi-GPU:
  Simply set device_id per MSM call
  "Doubling GPUs roughly doubles MSM speed"
  Auto-splits work exceeding single GPU memory
```

### 7.2 NTT Distribution Across GPUs

NTT distribution is harder due to inter-stage data dependencies:

```
NTT Multi-GPU Strategy
═══════════════════════════════════════════════════════════════

Option 1: Column-parallel (when matrix has multiple columns)
  Polygon zkEVM: 751 columns × 2^23 rows per column
  → 8 GPUs, each processes ~94 columns independently
  → No inter-GPU communication needed!
  → Total matrix: 94 GB (751 × 2^23 × 2 × 8 bytes, Goldilocks)

Option 2: Four-step NTT across GPUs (single large NTT)
  N = N₁ × N₂, split columns across GPUs

  Step 1: Row NTTs — each GPU handles N₂/K rows → PARALLEL
  Step 2: Twiddle multiply — element-wise → PARALLEL
  Step 3: Matrix transpose → ALL-TO-ALL COMMUNICATION (bottleneck!)
  Step 4: Row NTTs — each GPU handles N₁/K rows → PARALLEL

  Transpose communication cost (N = 2^26, BN254, 4 GPUs):
    Data size: 2 GB total → each GPU sends/receives ~1.5 GB
    PCIe 4.0 (32 GB/s):  ~47 ms
    NVLink 3.0 (600 GB/s): ~2.5 ms
    NVLink 4.0 (900 GB/s): ~1.7 ms
    → NVLink is 19-28x faster for the transpose step

Interconnect Impact on NTT:
═══════════════════════════════════════════════════════
Interconnect     │ Bandwidth     │ NTT Transpose (2GB)
─────────────────┼───────────────┼─────────────────────
PCIe 4.0 x16     │ 32 GB/s       │ ~47 ms (bottleneck)
PCIe 5.0 x16     │ 64 GB/s       │ ~23 ms
NVLink 3.0 (A100)│ 600 GB/s      │ ~2.5 ms
NVLink 4.0 (H100)│ 900 GB/s      │ ~1.7 ms
─────────────────┴───────────────┴─────────────────────
```

### 7.3 Real-World Proving Clusters

```
Production GPU Proving Clusters
═══════════════════════════════════════════════════════════════

SP1 Hypercube (Succinct):
  Hardware:  16 × NVIDIA RTX 5090 GPUs
  Result:    99.7% of Ethereum blocks proven in <12 seconds
  Cost:      ~$300-400K cluster (160 × RTX 4090 version)
             ~$100K optimized configuration

ZKsync Airbender:
  Hardware:  Single NVIDIA H100 GPU
  Result:    Ethereum blocks in <35 seconds (17s without recursion)
  Throughput: 21.8 million cycles/second
  Cost:      $0.0001 per transfer

RISC Zero R0VM 2.0:
  Hardware:  ~$120K GPU rig
  Target:    Real-time proving (<12s) by mid-2025
  Current:   <30 seconds per Ethereum block

Polygon zkEVM (X Layer):
  Hardware:  8 × GPUs (94 GB total GPU memory)
  Workload:  751 columns × 2^23 rows (Goldilocks field)
  Approach:  Column-parallel NTT + GPU Merkle tree
```

---

## 8. Advanced GPU Optimization Techniques

### 8.1 Kernel Fusion

```
Kernel Fusion: Reducing Global Memory Round-Trips
═══════════════════════════════════════════════════════════════

Problem: each CUDA kernel reads input from global memory and writes
output back to global memory. At 400-600 cycle latency per access,
this dominates execution time for lightweight operations.

Without fusion (NTT):
  Kernel 1: read data → compute stage 0 butterflies → write data
  Kernel 2: read data → compute stage 1 butterflies → write data
  ...
  24 kernels for N = 2^24 → 48 global memory passes

With fusion (bellperson radix-256):
  Kernel 1: read data → compute stages 0-7 in SMEM → write data
  Kernel 2: read data → compute stages 8-15 in SMEM → write data
  Kernel 3: read data → compute stages 16-23 in SMEM → write data
  → 6 global memory passes (8x reduction)

Other fusion opportunities:
  - Fuse twiddle multiply into butterfly (on-the-fly generation)
  - Fuse field add chain after field multiply (no SMEM round-trip)
  - Fuse Poseidon round constant addition with S-box computation
```

### 8.2 Occupancy vs Performance

```
Occupancy Paradox for ZK Kernels
═══════════════════════════════════════════════════════════════

Conventional GPU wisdom: maximize occupancy (threads per SM)
ZK reality: MSM kernels achieve ~25% occupancy but are OPTIMAL

Why? MSM is compute-bound with long dependency chains (field multiply).
  - Each field multiply: ~150 instructions, 4-cycle IMAD latency
  - ILP (instruction-level parallelism) within one multiply is low
  - Adding more warps doesn't help: they compete for the same ALUs
  - Reducing registers (to increase occupancy) causes register spills
    → Spills go to L1/SMEM (20-30 cycle latency) or worse, global memory
    → Net performance DECREASES

Measured (ZKProphet):
  MSM kernel: 216 registers/thread, 25% occupancy → OPTIMAL
  Forcing 50% occupancy: register spills → 15% slowdown

  NTT kernel: 56 registers/thread, 60% occupancy → room to optimize
  NTT is memory-bound, so more warps help hide memory latency

Rule of thumb:
  Compute-bound (MSM, Poseidon): minimize registers/thread without spills
  Memory-bound (NTT at large scale): maximize warps to hide latency
```

### 8.3 Branch Efficiency

```
Branch Divergence Analysis (from ZKProphet)
═══════════════════════════════════════════════════════════════

ZK field arithmetic has conditional branches (e.g., carry propagation,
conditional subtraction after Montgomery multiply). When threads in a
warp take different branches, the warp serializes both paths.

Measured branch efficiency:
  FF_add:  52.5% (worst — carry propagation depends on operand values)
  FF_sub:  similar (~55%)
  FF_mul:  ~70% (conditional subtraction at end)
  FF_sqr:  96.9% (optimized: fewer branches in squaring)

Impact: wasted execution slots, lower effective throughput
Mitigation: branchless algorithms (predicated execution via PTX)
  Replace: if (result >= p) result -= p;
  With:    mask = -(result >= p); result -= (p & mask);
  → No branch divergence, deterministic execution time
```

### 8.4 Memory Coalescing for ZK

```
Memory Coalescing Patterns
═══════════════════════════════════════════════════════════════

Coalesced access: 32 threads in a warp access consecutive 32-bit words
  → Single memory transaction (128 bytes)
  → Full bandwidth utilization

MSM point loading:
  Coalesced:   points[tid] where tid = 0,1,2,...,31
    → 32 consecutive points loaded in minimal transactions
  Scattered:   points[bucket_index[tid]] (random bucket assignments)
    → 32 random addresses → up to 32 separate transactions
    → Solution: sort by bucket index first (Section 3.4)

NTT butterfly access:
  Stage 0: stride = 1 → access pattern: a[2*tid], a[2*tid+1]
    → Nearly coalesced (2-stride)
  Stage 20: stride = 2^20 → access pattern: a[tid], a[tid + 2^20]
    → Completely scattered, each access is a cache miss
    → Solution: four-step NTT or data layout transformation

BN254 field element: 32 bytes = 8 × 32-bit limbs
  To coalesce: store elements in structure-of-arrays (SoA) layout
    Limb0[0], Limb0[1], ..., Limb0[31], Limb1[0], Limb1[1], ...
  vs array-of-structures (AoS):
    Elem0[Limb0..7], Elem1[Limb0..7], ...
  SoA enables perfect coalescing for each limb access
```

### 8.5 Data Transfer Optimization

```
CPU ↔ GPU Data Transfer
═══════════════════════════════════════════════════════════════

MSM data sizes (BN254):
  Scalars: N × 32 bytes  (2^24: 512 MB)
  Points:  N × 64 bytes  (2^24: 1 GB, affine)
  Total:   N × 96 bytes  (2^24: 1.5 GB)

PCIe 4.0 x16: 32 GB/s → 1.5 GB takes 47 ms
PCIe 5.0 x16: 64 GB/s → 1.5 GB takes 23 ms

Optimization strategies:
  1. Overlap transfer with compute (cuZK technique):
     While GPU processes batch K, CPU prepares batch K+1
     → Hides transfer latency behind computation

  2. Keep SRS (points) resident on GPU:
     SRS is the same across all proofs for a given circuit
     Upload once, reuse across all proofs
     Only transfer scalars per proof: N × 32 bytes

  3. Pinned (page-locked) host memory:
     Avoids OS page faults during DMA transfer
     Up to 2x faster than pageable memory transfers

  4. Compression:
     Scalars: transmit in canonical form, expand on GPU
     Points:  compressed form (x + sign bit) → 33 bytes
     Decompress on GPU: field sqrt is 1 exponentiation
```

---

## 9. GPU vs FPGA vs ASIC — Quantitative Comparison

```
Comprehensive Hardware Comparison for ZK Acceleration
═══════════════════════════════════════════════════════════════════════════════

Metric              │ GPU                  │ FPGA                 │ ASIC/ZPU
────────────────────┼──────────────────────┼──────────────────────┼─────────────────
MSM Speedup         │ Up to 800x over CPU  │ 15-29% over baseline │ 77.7x (PipeZK)
  (vs CPU)          │ (ymc, 2^26)          │ (ZPrize 2022 FPGA)  │
                    │                      │ 914ms for 2^24       │
                    │                      │ (single Xilinx U55C) │
────────────────────┼──────────────────────┼──────────────────────┼─────────────────
NTT Speedup         │ Up to 50x over CPU   │ Score 5.01           │ 197.5x (PipeZK)
  (vs CPU)          │ (cuZK, 2^22)         │ (ZPrize 2022 FPGA)  │
────────────────────┼──────────────────────┼──────────────────────┼─────────────────
Power               │ 250-700W             │ 75-225W              │ Lowest
                    │                      │                      │
Energy Efficiency   │ 398x more efficient  │ Better per-watt      │ Best per-watt
  (MSM, vs CPU)     │ than CPU at 2^26     │ than GPU             │
────────────────────┼──────────────────────┼──────────────────────┼─────────────────
Cost per Unit       │ $1,500 (RTX 4090)    │ $5,000-$15,000       │ $10M+ tape-out
                    │ $10,000 (A100)       │ (Xilinx U250/U55C)  │ Low marginal cost
                    │ $25,000 (H100)       │                      │ at volume
────────────────────┼──────────────────────┼──────────────────────┼─────────────────
Development Time    │ Days-Weeks           │ Months               │ Years
                    │ (CUDA mature)        │ (RTL/HLS expertise)  │ (tape-out cycle)
────────────────────┼──────────────────────┼──────────────────────┼─────────────────
Flexibility         │ High                 │ Moderate             │ None
                    │ (reprogram any       │ (reconfigurable      │ (fixed-function
                    │ proof system)        │ but complex)         │ post-fabrication)
────────────────────┼──────────────────────┼──────────────────────┼─────────────────
Availability        │ Commodity            │ Expensive,           │ Not commercially
                    │ (readily available)  │ limited supply       │ available (2025)
────────────────────┼──────────────────────┼──────────────────────┼─────────────────
Best For            │ Rapid prototyping,   │ Power-constrained    │ High-volume
                    │ production proving,  │ environments,        │ production with
                    │ proof system R&D     │ edge deployment      │ fixed proof system
────────────────────┴──────────────────────┴──────────────────────┴─────────────────
```

### Energy Efficiency Deep Dive

```
Energy Efficiency (CPU normalized to GPU, from ZKProphet)
═══════════════════════════════════════════════════════════════

Scale  │ CPU NTT Energy / GPU NTT Energy │ CPU MSM Energy / GPU MSM Energy
───────┼─────────────────────────────────┼────────────────────────────────
2^20   │ 3.21x                           │ 27.59x
2^22   │ 2.87x                           │ 100.2x
2^24   │ 2.93x                           │ 236.9x
2^26   │ 3.62x                           │ 398.4x
───────┴─────────────────────────────────┴────────────────────────────────

Interpretation:
  MSM energy efficiency scales dramatically with problem size
    → At 2^26, CPU uses 398x more energy than GPU for the same MSM
    → GPU's parallel execution amortizes fixed power overhead

  NTT energy efficiency stays relatively flat (~3x)
    → Memory-bound nature limits both throughput and efficiency gains
    → Memory system power dominates regardless of compute utilization
```

---

## 10. Real-World GPU Proving Systems

### 10.1 System Comparison (2025)

```
Production GPU Proving Systems
═══════════════════════════════════════════════════════════════════════════

System            │ Eth Block Time │ Hardware          │ Cost/Proof
──────────────────┼────────────────┼───────────────────┼────────────────
ZKsync Airbender  │ <35s           │ 1 × H100          │ $0.0001/transfer
                  │ (17s w/o rec.) │                   │
SP1 Hypercube     │ <12s (99.7%)   │ 16 × RTX 5090     │ ~$0.001/tx
RISC Zero R0VM 2  │ 44s → <30s     │ ~$120K GPU cluster │ 5x reduction
Scroll OpenVM     │ ~5 min (2-layer)│ GPU cluster       │ --
Polygon zkEVM     │ varies         │ 8 GPUs (94 GB)    │ --
──────────────────┴────────────────┴───────────────────┴────────────────

Trend: sub-12-second proving is the target for real-time Ethereum.
```

### 10.2 ZKsync Airbender (Single-GPU Champion)

```
ZKsync Airbender Architecture
═══════════════════════════════════════════════════════════════

Hardware:   Single NVIDIA H100 (SXM, 80 GB HBM3)
Throughput: 21.8 million cycles/second
Latency:    <35 seconds per Ethereum block (17s without recursion)
Cost:       $0.0001 per transfer
Status:     Live on mainnet (2025)

Key design decisions:
  - Custom proof system (Boojum evolution)
  - Shivini GPU library (era-shivini) with era-boojum-cuda backend
  - Optimized for H100's 3.35 TB/s memory bandwidth
  - Single-GPU design avoids multi-GPU communication overhead

Performance breakdown (on NVIDIA L4, 40 GB RAM):
  Component        │ 8 CPUs   │ 12 CPUs
  ─────────────────┼──────────┼─────────
  Witness LDE      │ 29s      │ 21s
  Quotient work    │ 82s      │ 60s
  Total            │ 371s     │ 273s

  On H100: 4-11x faster than SP1 and RISC Zero benchmarks
```

### 10.3 SP1 Hypercube (Multi-GPU Speed Record)

```
SP1 Hypercube Architecture
═══════════════════════════════════════════════════════════════

Hardware:    16 × NVIDIA RTX 5090 GPUs
Architecture: RISC-V zkVM with precompiles + multilinear
              polynomials + LogUp GKR protocol

Performance:
  99.7% of Ethereum blocks: <12 seconds
  95.4% of Ethereum blocks: <10 seconds
  Average:                   10.3 seconds
  Compute-heavy workloads:   up to 5x improvement over SP1 Turbo

Cost analysis:
  Cluster (160 × RTX 4090):   $300-400K
  Optimized configuration:     ~$100K
  Per transaction:             ~$0.001

Key techniques:
  - Precompiles reduce cycle counts 5-10x (Keccak, SHA-256, secp256k1)
  - Multilinear polynomials (no NTT needed → avoids NTT bottleneck!)
  - LogUp GKR protocol for efficient lookup arguments
  - BabyBear field → native 32-bit GPU arithmetic

GPU requirements:
  CUDA Compute Capability >= 8.6
  24 GB VRAM recommended per GPU
```

### 10.4 RISC Zero R0VM 2.0

```
RISC Zero R0VM 2.0
═══════════════════════════════════════════════════════════════

Architecture:  RISC-V zkVM with STARK proofs, BabyBear field

Evolution:
  Before R0VM 2.0: 35 minutes per Ethereum block
  After R0VM 2.0:  44 seconds (48x improvement!)
  Current target:  <30 seconds, aiming for <12s by mid-2025

Improvements:
  - 5x cost reduction
  - User memory expanded to 3 GB
  - CUDA GPU prover (fully open-source)

Hardware target: ~$120K GPU rig for real-time proving
```

### 10.5 The Proving Cost Trajectory

```
GPU Proving Cost Evolution (Ethereum block)
═══════════════════════════════════════════════════════════════

2023: ~$10-50 per proof (early GPU provers)
2024: ~$0.01-1 per proof (optimized GPU frameworks)
2025: ~$0.0001-0.001 per proof (Airbender, SP1 Hypercube)

3-4 orders of magnitude cost reduction in 2 years!

Drivers:
  1. Algorithm improvements (Pippenger optimization, kernel fusion)
  2. Better proof systems (multilinear → no NTT, small fields)
  3. GPU hardware improvements (H100 → 3.35 TB/s bandwidth)
  4. Software maturity (ICICLE, cuZK, custom PTX)
```

---

## 11. Projects

### Project 1: BabyBear Field Arithmetic Benchmark

Implement BabyBear (p = 2,013,265,921) field multiplication on GPU and compare throughput against BN254.

```
Tasks:
  1. Write BabyBear field_mul CUDA kernel (single 32-bit multiply + reduction)
  2. Write BN254 field_mul CUDA kernel (8-limb Montgomery multiply)
  3. Benchmark both: throughput (ops/second) on your GPU
  4. Measure: register usage, occupancy, achieved bandwidth
  5. Verify the ~40x performance ratio between small and large fields

Expected results:
  BabyBear: ~5 registers/thread, >75% occupancy, throughput limited by ALU
  BN254:    ~40 registers/thread, ~25% occupancy, throughput limited by ILP
```

### Project 2: GPU MSM with Pippenger's Method

Implement a basic Pippenger MSM on GPU using the four-phase CUDA pipeline.

```
Tasks:
  1. Scalar decomposition kernel: split 256-bit scalars into c-bit chunks
  2. Use CUB radix sort to sort (bucket_index, point_index) pairs
  3. Bucket accumulation kernel: one thread per bucket, mixed addition
  4. Bucket reduction kernel: running sum per window
  5. Compare against CPU implementation from Phase 6.1

Target: BN254 curve, n = 2^16 to 2^20
  At 2^20: should achieve ~50-100x speedup over single-thread CPU

Optimization experiments:
  - Vary window size c from 10 to 20, measure impact
  - Try XYZZ vs Jacobian coordinates, measure register pressure
  - Profile with NVIDIA Nsight Compute: identify bottleneck (compute vs memory)
```

### Project 3: NTT Kernel with Shared Memory Optimization

Implement a radix-2 NTT butterfly kernel with shared memory staging.

```
Tasks:
  1. In-shared-memory NTT for sub-problems up to 4096 elements
  2. Global-memory butterfly kernel for cross-block stages
  3. Implement on-the-fly twiddle factor generation
  4. Measure: stages 0-11 (SMEM) vs stages 12-23 (global) timing
  5. Profile memory bandwidth utilization per stage

Target: BN254 field, N = 2^16 to 2^24
  Expected: 30-50x speedup over CPU at 2^20, degrading at 2^24+

Advanced:
  - Implement radix-4 butterfly (2 stages fused per iteration)
  - Add padding to avoid shared memory bank conflicts
  - Compare precomputed twiddle table vs on-the-fly generation
```

### Project 4: ICICLE Integration Project

Use ICICLE to accelerate a Groth16 proof for a Circom circuit.

```
Tasks:
  1. Install ICICLE with CUDA backend (Rust)
  2. Define a simple Circom circuit (e.g., Sudoku verifier)
  3. Export R1CS and witness from Circom/snarkjs
  4. Use ICICLE MSM and NTT APIs to build a basic Groth16 prover
  5. Benchmark: end-to-end proving time vs snarkjs and rapidsnark

Measurement points:
  - MSM time (individual calls)
  - NTT time
  - Witness generation time (CPU-bound)
  - Data transfer time (CPU ↔ GPU)
  - Total proof generation time

Expected: 50-100x faster than snarkjs, 3-5x faster than CPU rapidsnark
```

### Project 5: Multi-GPU NTT Communication Analysis

Analyze and benchmark the communication overhead of multi-GPU NTT.

```
Tasks:
  1. Implement four-step NTT splitting across 2 GPUs
  2. Measure: row-NTT time (independent, parallel)
  3. Measure: matrix transpose time (all-to-all communication)
  4. Measure: total time vs single-GPU NTT
  5. Analyze: at what problem size does multi-GPU become worthwhile?

If you have NVLink: compare PCIe vs NVLink transpose overhead
If single GPU: simulate by measuring cudaMemcpy between host and device

Expected findings:
  - Row NTTs scale linearly with GPU count
  - Transpose dominates at small N (communication overhead)
  - Crossover point: N > 2^22 for PCIe, N > 2^20 for NVLink
```

---

## 12. Resources

### Papers

- **ZKProphet** — "Understanding Performance of ZK Proofs on GPUs" (2025) — comprehensive GPU benchmark paper with microarchitectural analysis of MSM/NTT kernels, register pressure, branch efficiency, and energy analysis
- **cuZK** — "Accelerating Zero-Knowledge Proof with A Faster Parallel MSM Algorithm on GPUs" (IACR 2022/1321) — pipelined MSM, parallel modular multiplication, CPU-GPU overlap
- **Two Algorithms for Fast GPU Implementation of NTT** (IACR 2023/1410) — on-the-fly twiddle factor generation achieving 4.2x NTT speedup
- **BatchZK** — "Fully Pipelined GPU-Accelerated ZK System" (IACR 2024/1862) — 259.5x throughput improvement, 9.52 proofs/sec for ML workloads
- **GZKP** — "GPU Accelerated Zero-Knowledge Proof System" (ACM HPDC 2023) — multi-GPU analysis, 2.1x scaling with additional GPUs
- **PipeZK** — "Pipelined Architecture for Zero-Knowledge Proofs" (ISCA 2021) — ASIC design achieving 77.7x MSM and 197.5x NTT speedup

### GPU ZK Libraries

- **ICICLE** (Ingonyama) — github.com/ingonyama-zk/icicle — primary GPU ZK library, CUDA C++ with Rust/Go bindings, supports BN254/BLS12-381/BabyBear/Goldilocks
- **sppark** (Supranational) — github.com/supranational/sppark — XYZZ coordinates, fastest at small MSM scales
- **bellperson** (Filecoin) — github.com/filecoin-project/bellperson — GPU Groth16 with radix-256 NTT
- **era-bellman-cuda** (Matter Labs) — github.com/matter-labs/era-bellman-cuda — zkSync GPU crypto library
- **Neptune** (Filecoin) — Poseidon GPU implementation for Merkle trees

### Competition Results

- **ZPrize 2022** — zprize.io — MSM on GPU: baseline 5.86s → winner 2.52s on BLS12-377 (2^26), $4.4M total prizes
- **ZPrize 2023** — zprize.io — GPU MSM: 430ms winner, 367ms late submission, dual-curve challenge

### Production Systems

- **ZKsync Airbender** — single H100, <35s Ethereum blocks, $0.0001/transfer, live on mainnet
- **SP1 Hypercube** (Succinct) — 16× RTX 5090, 99.7% of Ethereum blocks in <12s
- **RISC Zero R0VM 2.0** — 35 min → 44s Ethereum blocks, targeting real-time
- **Scroll OpenVM** — GPU prover: EVM circuit 30s (9x over CPU), aggregation 149s (15x over CPU)
- **X Layer/Polygon zkEVM** — 8 GPUs, Goldilocks NTT, 94 GB matrix partitioning

### Optimization Guides

- **Ingonyama Hardware Review** — "GPUs, FPGAs and Zero Knowledge Proofs" — GPU vs FPGA tradeoff analysis
- **Ingonyama ZK Benchmark Toolkit** — standardized benchmarking methodology for ZK operations
- **Paradigm** — "Hardware Acceleration for Zero-Knowledge Proofs" (2022) — industry overview of acceleration approaches
- **NVIDIA PTX ISA Documentation** — reference for integer multiply-add instructions used in field arithmetic

### Recommended Study Order

```
Day 1: Sections 1-2 (GPU architecture + field arithmetic)
  → Understand why BN254 needs 130+ PTX instructions per multiply
  → Grasp register pressure and occupancy tradeoffs

Day 2: Section 3 (MSM on GPU)
  → Map Pippenger phases to CUDA kernels
  → Understand sorting optimization and ZPrize progression

Day 3: Section 4 (NTT on GPU)
  → Why NTT becomes the bottleneck at 2^26
  → Shared memory staging, radix-256 fusion, four-step NTT

Day 4: Sections 5-6 (Poseidon + frameworks)
  → ICICLE API and integration patterns
  → cuZK, bellperson, sppark landscape

Day 5: Sections 7-10 (optimization + real-world systems)
  → Multi-GPU strategies, kernel fusion, energy efficiency
  → Production provers: Airbender vs SP1 vs RISC Zero

Day 6-7: Projects
  → Start with Project 1 (BabyBear vs BN254 benchmark)
  → Then Project 4 (ICICLE integration) for practical experience
```
