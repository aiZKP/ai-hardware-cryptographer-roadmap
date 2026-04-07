# Computer Architecture for AI Hardware Engineers

Phase 1 · Section 2 — How processors and memory systems work, and why that shapes every AI accelerator ever built.

> **Goal:** By the end of this section you can read a GPU or NPU architecture paper, understand why a transformer is memory-bandwidth-bound, and reason about the trade-offs in any custom accelerator design.

---

## 1. The Two Computing Paradigms

Every chip designer makes the same fundamental choice: optimise for **latency** or **throughput**. CPUs and GPUs represent the two extremes.

```
CPU — latency optimised                 GPU — throughput optimised
────────────────────────────────────    ────────────────────────────────────
Few, powerful cores (4–128)             Thousands of simple cores (1,000s–10,000s)
Deep OoO execution engine               In-order, warp-scheduled execution
Large private caches (L1: 32–64 KB)     Small per-SM caches, huge shared pool
Branch prediction, speculative exec     No branch prediction (divergence penalty)
Single-thread latency: 1–4 ns          Single-thread latency: 50–100 ns
Total throughput: 1–10 TFLOPS           Total throughput: 100–1,000 TFLOPS
Power: 5–350 W                          Power: 70–1,000 W

Use-case: operating system, compilers,  Use-case: matrix multiply, convolution,
  database queries, anything with         attention — embarrassingly parallel,
  complex control flow                    regular data access patterns
```

**The transistor budget question:** A modern chip has ~50–100 billion transistors. How do you spend them?

```
CPU (Apple M4 Pro die)               GPU (NVIDIA H100 SXM)
┌──────────────────────────┐         ┌──────────────────────────┐
│ OoO engines (P-cores) 45%│         │ CUDA/Tensor cores    60% │
│ Cache hierarchy       30%│         │ HBM memory interface 20% │
│ Memory controller      8%│         │ L2 cache             10% │
│ I/O, NPU, other       17%│         │ Control / scheduler  10% │
└──────────────────────────┘         └──────────────────────────┘
→ Optimises for the 1 hard task       → Optimises for 10,000 easy tasks
```

**AI connection:** Transformer attention is `O(n²)` parallel multiply-accumulate — exactly what the GPU transistor budget is built for. The CPU's OoO engine is wasted on this workload; the GPU's thousands of Tensor Cores are perfect.

---

## 2. Instruction Set Architecture (ISA)

The ISA is the **contract between software and hardware**. Hardware can be redesigned completely (new microarchitecture) as long as the ISA stays compatible — software keeps running.

### RISC vs CISC

| Property | RISC (ARM64, RISC-V) | CISC (x86-64) |
|----------|----------------------|---------------|
| Instruction length | Fixed 32 bits | Variable 1–15 bytes |
| Memory access | Load-store only | Any instruction can access memory |
| Decoder complexity | Simple | Complex (internal RISC micro-ops) |
| Code density | Lower | Higher |
| Power efficiency | Excellent | Good |
| AI hardware usage | Edge, mobile, Apple Silicon, Jetson | Data-centre training servers |

> In practice, modern x86 CPUs decode CISC instructions into RISC-like micro-ops internally — the distinction blurs at the microarchitecture level.

### The Three ISAs You Will Encounter

**ARM64 (AArch64)**

```
31 general-purpose 64-bit registers (X0–X30) + zero register (XZR)
128-bit SIMD/FP registers V0–V31 (also addressable as S/D/Q)

ADD  X0, X1, X2        // X0 = X1 + X2
LDR  X0, [X1, #8]      // X0 = mem[X1 + 8]
FMLA V0.4S, V1.4S, V2.4S  // 4-wide FP32 fused multiply-add (NEON)
```

Used in: Apple Silicon (M-series), NVIDIA Jetson, Qualcomm Snapdragon, AWS Graviton.

**x86-64**

```
16 general-purpose 64-bit registers (RAX–R15)
SIMD: XMM (128b), YMM (256b/AVX2), ZMM (512b/AVX-512)

mov  rax, [rbx + 8]    // load from memory
vmulps ymm0, ymm1, ymm2 // 8-wide FP32 multiply (AVX2)
vfmadd231ps zmm0, zmm1, zmm2  // 16-wide FP32 FMA (AVX-512)
```

Used in: Intel/AMD data-centre servers — the dominant platform for model training.

**RISC-V**

```
Modular open ISA: RV64I base + extensions (M, F, D, A, V, ...)
32 general-purpose 64-bit registers

add  x1, x2, x3        // x1 = x2 + x3
vsetvli t0, a0, e32,m4 // V extension: set vector length, FP32, 4x LMUL
vfmacc.vv v0, v4, v8   // vector FMA (scalable width)
```

Used in: Emerging edge AI chips (Tenstorrent, SiFive), custom accelerator control cores.

---

## 3. CPU Microarchitecture — The Pipeline

Understanding the CPU pipeline gives you the vocabulary to reason about any processor's performance: GPUs, NPUs, and custom accelerators all face the same fundamental constraints.

### The Classic 5-Stage Pipeline

```
Cycle:   1    2    3    4    5    6    7    8
Instr 1: IF   ID   EX   MEM  WB
Instr 2:      IF   ID   EX   MEM  WB
Instr 3:           IF   ID   EX   MEM  WB
Instr 4:                IF   ID   EX   MEM  WB

IF  = Instruction Fetch (read from instruction cache)
ID  = Instruction Decode + register file read
EX  = Execute (ALU, FP unit, address calculation)
MEM = Memory access (load/store)
WB  = Write-Back (result → register file)
```

One instruction finishes per cycle at steady state — ideal throughput is 1 IPC (instructions per cycle).

### Hazards — When the Pipeline Stalls

**Data hazard:** instruction needs a result not yet written back.

```
ADD X1, X2, X3    // produces X1 in WB (cycle 5)
SUB X4, X1, X5    // needs X1 in ID (cycle 3) — too early!
```

Solution: **forwarding** (bypass) routes the EX result directly to the next instruction's input, eliminating the stall in most cases.

```
        ADD  EX stage ──forward──► SUB  EX stage
                         ↑
               result available here, not after WB
```

**Control hazard:** a branch changes the PC; the instructions already in the pipeline may be wrong.

```
BEQ X1, X2, Label    // branch resolved in EX (cycle 3)
  →  already fetched 2 wrong instructions behind it
  →  must flush them: 2-cycle penalty for a 5-stage pipe
```

**AI connection:** GPU warps execute in lockstep — there is no branch prediction. If threads in a warp take different branches, they execute **both paths serially** (warp divergence). Good kernel design eliminates branches inside hot loops.

### Out-of-Order Execution

Modern high-performance CPUs don't wait for slow instructions — they look ahead and execute whatever is ready.

```
Program order:              Execution order:
1. ADD X1, X2, X3           1. SUB X6, X7, X8   (no dependencies, runs first)
2. MUL X4, X1, X5           2. ADD X1, X2, X3   (data ready)
3. SUB X6, X7, X8           3. MUL X4, X1, X5   (waits for ADD result)
4. DIV X9, X4, X10          4. DIV X9, X4, X10  (waits for MUL)
```

Key hardware structures:

| Structure | Role |
|-----------|------|
| **Reorder Buffer (ROB)** | Holds all in-flight instructions; enforces in-order commit |
| **Reservation Stations** | Instructions wait here until operands are ready |
| **Register Renaming** | Eliminates false WAR/WAW dependencies by mapping to physical registers |
| **Common Data Bus (CDB)** | Broadcasts results to all waiting reservation stations |

**Superscalar:** fetch and issue *multiple* instructions per cycle. Apple M4 P-cores are ~6-wide; Intel/AMD are 4–6-wide. Beyond 6-wide, the dispatch logic complexity dominates.

### Branch Prediction

Modern predictors achieve 94–98% accuracy on typical code using:

- **2-bit saturating counter** per branch: tracks recent taken/not-taken history
- **Pattern History Table (PHT):** indexes by branch address XOR global history
- **Branch Target Buffer (BTB):** caches `branch address → target address`
- **Return Stack Buffer (RSB):** stack of return addresses, one per `CALL`

Misprediction penalty: 10–20 cycles on a modern deep pipeline. Every wrong prediction flushes the pipe and wastes those cycles.

---

## 4. Memory Hierarchy — The Real Bottleneck

Memory access time is the dominant constraint for AI workloads. A GPU with 1,000 TFLOPS of compute can be throttled to 10 TFLOPS effective throughput by insufficient memory bandwidth.

### The Memory Mountain

```
Level          | Latency    | Bandwidth     | Capacity  | Location
───────────────────────────────────────────────────────────────────
Registers      | 0 cycles   | unlimited     | ~KB       | on-core
L1 cache       | 4 cycles   | ~1 TB/s       | 32–64 KB  | per-core
L2 cache       | 12 cycles  | ~500 GB/s     | 256 KB–4 MB  per-core/cluster
L3 cache       | 40 cycles  | ~200 GB/s     | 8–64 MB   | shared (CPU)
               |            |               |           |
DRAM (DDR5)    | 70 ns      | 50–100 GB/s   | 16–512 GB | off-chip
HBM3 (GPU)     | 100 ns     | 3.35 TB/s     | 80–192 GB | off-chip stacked
NVMe SSD       | ~100 µs    | 7 GB/s        | TB        | storage
```

**The gap is enormous.** Registers are 10,000× faster than DRAM. Cache hierarchy exists entirely to bridge this gap.

### Cache Organisation

A cache is divided into **sets** of **ways**. An address maps to exactly one set; within the set, it can go into any way.

```
Address bits:  [  Tag  ] [  Index  ] [ Offset ]
                   ↓          ↓
               Compare    Select set   Byte within line

8-way set-associative example (64-byte lines, 8 MB L3):
  Sets = 8 MB / (8 ways × 64 bytes) = 16,384 sets
  Index bits  = log2(16,384) = 14
  Offset bits = log2(64)     = 6
  Tag bits    = 64 - 14 - 6  = 44
```

**Associativity trade-off:**

| Type | Miss rate | Hardware cost | Used where |
|------|-----------|---------------|------------|
| Direct-mapped (1-way) | High (conflict misses) | Minimal | Not common today |
| 4–8 way set-associative | Low | Moderate | L1/L2/L3 caches |
| Fully associative | Lowest | Very high | TLB, small caches |

**Three kinds of misses:**

1. **Compulsory (cold):** first touch — unavoidable
2. **Capacity:** working set larger than cache — solution: tile/block your algorithm
3. **Conflict:** same index but different tags evict each other — solution: increase associativity or change data layout

### Cache Coherence (Why Multi-Core Is Hard)

When multiple cores have private caches, they can hold **stale copies** of the same memory location. Coherence protocols enforce a consistent view.

**MESI protocol** — each cache line has one of four states:

```
M (Modified)   — this core wrote it; dirty; no other core has it
E (Exclusive)  — clean; only this core has it
S (Shared)     — multiple cores have valid read-only copies
I (Invalid)    — stale or not present

Transitions:
  Core 0 reads X:  I → E (if only core)  or  I → S (if others have it)
  Core 0 writes X: any → M; all other cores' copies → I (invalidate)
  Core 1 reads X while Core 0 has M: Core 0 flushes, both → S
```

**Why this matters for AI:** In multi-GPU systems, each GPU has its own HBM. Coherence is handled by NVLink/NVSwitch (NVIDIA) or Infinity Fabric (AMD). Understanding MESI is the mental model for understanding why all-reduce operations are expensive and why NCCL ring-allreduce is designed the way it is.

### DRAM and HBM

**DDR5 (CPU memory):**
- Bandwidth: ~50–100 GB/s per channel (8 channels on EPYC = 800 GB/s)
- Latency: ~70–80 ns
- Capacity: up to 512 GB per socket

**HBM3 (GPU memory) — the key innovation for AI:**

```
Traditional GDDR:            HBM (High Bandwidth Memory):
  Package ─── PCB ──── GPU     GPU die
                               ━━━━━━━━━
                               ▓▓▓▓▓▓▓▓  ← HBM stack (stacked DRAM dies)
                               ━━━━━━━━━
                               Silicon interposer (CoWoS)

Bus width: 32 bits            Bus width: 1,024 bits per stack
Bandwidth: ~600 GB/s          Bandwidth: 3.35 TB/s (H100 SXM)
```

HBM achieves 5× higher bandwidth than GDDR because it uses a **wide, short bus** (1,024 bits × multiple stacks) instead of a narrow, long one. The GPU die and HBM stacks sit on the same silicon interposer — this is CoWoS (Chip-on-Wafer-on-Substrate) packaging.

**AI connection:** LLM inference is almost entirely memory-bandwidth-bound. The rate at which you can load model weights from HBM determines tokens/second — compute is secondary.

---

## 5. SIMD — One Instruction, Many Data

SIMD is the bridge between CPU vector units and GPU Tensor Cores. The same idea — execute one operation on a wide register of packed data — scales from 128-bit NEON to 512-bit AVX-512 to a 16-wide GPU warp executing 32 CUDA cores in lock-step.

### How SIMD Works

```
Scalar (no SIMD):                   SIMD (4-wide FP32):
  a[0] = b[0] * c[0]   4 cycles       [a0 a1 a2 a3] = [b0 b1 b2 b3]
  a[1] = b[1] * c[1]                                 × [c0 c1 c2 c3]
  a[2] = b[2] * c[2]                  1 instruction, 4 results
  a[3] = b[3] * c[3]
```

**SIMD widths across ISAs:**

| ISA | Extension | Width | FP32 lanes | Fused-multiply-add? |
|-----|-----------|-------|------------|---------------------|
| ARM64 | NEON | 128 bits | 4 | Yes (FMLA) |
| ARM64 | SVE/SVE2 | 128–2048 bits (scalable) | 4–64 | Yes |
| x86-64 | SSE4.2 | 128 bits | 4 | No |
| x86-64 | AVX2 | 256 bits | 8 | Yes (FMA3) |
| x86-64 | AVX-512 | 512 bits | 16 | Yes |
| RISC-V | RVV | 128–65,536 bits (scalable) | variable | Yes |

**Example — vectorised dot product (AVX2):**

```cpp
// Scalar: N multiplications + N additions
float dot_scalar(const float* a, const float* b, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) sum += a[i] * b[i];
    return sum;
}

// AVX2: processes 8 FP32 per iteration
#include <immintrin.h>
float dot_avx2(const float* a, const float* b, int N) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < N; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        acc = _mm256_fmadd_ps(va, vb, acc);   // acc += va * vb (8-wide FMA)
    }
    // horizontal reduction of 8 lanes → scalar
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}
// Speedup vs scalar: ~6–7× (limited by memory bandwidth for large N)
```

### SIMD → GPU SIMT

A GPU warp (32 threads) executing the same instruction at the same time is SIMD taken to 32-wide, with the additional twist that each "lane" is an independent thread with its own register state.

```
CPU SIMD (AVX-512):              GPU SIMT (warp of 32):
  1 instruction                    1 instruction
  16 FP32 lanes                    32 "lanes" = 32 threads
  all same operation                all same operation (PC)
  same data register               each thread has own registers
  no divergence possible           divergence possible but costly
```

Tensor Cores go further — a single `wmma::mma_sync` instruction operates on an entire 16×16×16 matrix fragment, effectively 4,096-wide for FP16.

---

## 6. GPU Architecture (Conceptual)

The CUDA programming model is covered in Section 4. Here we look at the **hardware** — what the silicon actually does.

### SM: The Streaming Multiprocessor

The GPU is a collection of SMs. Every kernel launch distributes thread blocks across available SMs.

```
One SM (NVIDIA Ampere A100):
┌─────────────────────────────────────────────────────────┐
│  4 × Warp Schedulers (issue 1 warp/cycle each)          │
│                                                         │
│  4 × Dispatch Units                                     │
│  ───────────────────────────────────────────────────    │
│  64 × CUDA Cores (FP32)   │  32 × INT32 cores          │
│   4 × Tensor Core units   │   4 × FP64 cores           │
│  16 × Load/Store units    │   4 × SFU (sin, cos, etc.) │
│  ───────────────────────────────────────────────────    │
│  192 KB unified shared memory / L1 cache (configurable) │
│  256 KB register file (32-bit registers per SM)         │
│                                                         │
│  Max 1,536 resident threads (48 warps × 32 threads)     │
└─────────────────────────────────────────────────────────┘
```

A100 has 108 SMs → 108 × 64 = 6,912 CUDA cores; 108 × 4 = 432 Tensor Core units.

### SIMT Execution and Warp Scheduling

All 32 threads in a warp execute the **same instruction** every cycle. When a warp stalls (e.g., waiting for a global memory load that takes 300+ cycles), the warp scheduler **instantly switches** to another ready warp — zero overhead context switch because each warp has its own dedicated register file.

```
Cycle 100: Warp 0 issues LOAD (takes 200+ cycles to return)
Cycle 101: Warp scheduler picks Warp 1 (compute-bound, no stall)
Cycle 102: Warp 2
Cycle 103: Warp 3
...
Cycle 300: Warp 0's data arrives; it becomes eligible again
```

This **latency hiding through massive parallelism** is the GPU's secret: it tolerates 300-cycle memory latency by having 40+ other warps to run while waiting. The CPU hides latency with 10 MB of cache; the GPU hides latency with thousands of threads.

### GPU Memory Hierarchy

```
Thread-private:
  Registers (fastest)      — local variables, per-thread, ~255 registers each
  Local memory (slowest)   — register spill → goes to global memory

Block-shared:
  Shared memory (fast)     — 16–164 KB per SM, programmer-managed, ~4–10 cycles
  L1 cache (fast)          — same SRAM bank as shared memory, automatic

Device-wide:
  L2 cache                 — 40–50 MB on H100, shared across all SMs, ~100 cycles
  Global memory (HBM)      — 80–192 GB, 3.35 TB/s, ~300–400 cycles
  Constant memory          — 64 KB, cached in L1, read-only
  Texture memory           — spatial locality cache, hardware interpolation
```

**Shared memory is the programmer's L1.** Tiled matrix multiplication explicitly loads a tile from global memory into shared memory, so each value is read from HBM once but used many times by threads in the block.

### Tensor Cores

Tensor Cores are hardwired matrix-multiply-accumulate (MMA) units that compute a 16×16×16 matrix product in a single "instruction":

```
D = A × B + C
  A: 16×16 matrix (FP16 or BF16)
  B: 16×16 matrix (FP16 or BF16)
  C: 16×16 accumulator (FP32)
  D: 16×16 result (FP32)

Throughput on H100 SXM:
  FP16  Tensor Core: 989  TFLOPS
  BF16  Tensor Core: 989  TFLOPS
  FP8   Tensor Core: 1,979 TFLOPS
  FP32  CUDA Core:   67   TFLOPS  ← 14.7× slower!
```

Every major AI training and inference framework ultimately generates `cublasSgemm` / `cublasHgemm` calls that map to Tensor Core wmma instructions.

---

## 7. AI Accelerator Design Patterns

### Systolic Arrays

A systolic array is a grid of identical processing elements (PEs) where data flows rhythmically through neighbours — like a heart pumping data instead of blood. Google's TPU uses a 256×256 systolic array.

```
Input matrix A rows flow →→→→→→→→
Weight matrix B cols flow ↓↓↓↓↓↓↓

     PE(0,0) → PE(0,1) → PE(0,2) → ...
       ↓         ↓         ↓
     PE(1,0) → PE(1,1) → PE(1,2) → ...
       ↓         ↓         ↓
     PE(2,0) → PE(2,1) → PE(2,2) → ...

Each PE: accumulator += A_val × B_val
Result flows out the bottom

Advantage: weights loaded once, reused across entire array → minimal memory traffic
```

**NVIDIA Tensor Cores are essentially a 16×16 systolic array** implemented in SRAM-adjacent logic.

### Dataflow Architectures

Traditional CPUs/GPUs: **von Neumann** — load operands → execute → store result → repeat. Memory traffic dominates.

Dataflow: operations fire as soon as operands arrive. No central memory round-trip.

```
Von Neumann:                     Dataflow:
  LOAD A from HBM                  A ──► multiply ──► add ──► result
  LOAD B from HBM                  B ──►    ↑
  MUL C = A × B                    C ──────►┘
  STORE C to HBM
  LOAD C from HBM
  LOAD D from HBM
  ADD E = C + D
  STORE E to HBM                 ← no HBM round-trips for intermediate values
```

SambaNova SN40L, Cerebras WSE-3, and Graphcore IPU are commercial dataflow architectures for AI.

### NPUs — Neural Processing Units

Dedicated inference accelerators in mobile/edge SoCs:

| Chip | NPU | Peak INT8 | Architecture |
|------|-----|-----------|--------------|
| Apple M4 | 16-core Neural Engine | 38 TOPS | Apple proprietary |
| Qualcomm Snapdragon X Elite | Hexagon NPU | 45 TOPS | Qualcomm proprietary |
| Intel Core Ultra 200V | AI Boost NPU | 48 TOPS | Intel proprietary |
| Google Tensor G4 | TPU v5 lite | ~30 TOPS | Systolic array |
| NVIDIA Jetson Orin NX | DLA v2 | 57 TOPS (DLA+GPU) | Mixed |

NPUs trade flexibility for efficiency: they are optimised for a fixed set of layer types (conv, matmul, elementwise) at reduced precision (INT8/FP16), consuming 10–50× less power than a discrete GPU for the same throughput.

---

## 8. The Roofline Model — Your Most Important Analysis Tool

The roofline model answers the single most important question about any kernel or workload: **are you memory-bound or compute-bound?**

### Arithmetic Intensity

```
Arithmetic Intensity (AI) = FLOPs executed / Bytes transferred from memory

Examples:
  Vector addition  y = a + b:        1 FLOP / 12 bytes = 0.08 FLOP/byte  (memory-bound)
  Matrix multiply  C = A × B (N=1K): 2N³ FLOPs / 3N² × 4 bytes          (compute-bound)
    = 2×10⁹ / 12×10⁶ ≈ 167 FLOP/byte
  Transformer attention (seq 2048):  ~10 FLOP/byte                        (memory-bound)
  LLM weight loading (batch=1):      ~0.5 FLOP/byte                       (severely memory-bound)
```

### The Roofline

```
TFLOPS (log)
 │                          ───────────────── Peak compute (989 TFLOPS FP16)
 │                      ╱
 │                  ╱  ╲── compute-bound region
 │              ╱        (more compute = more speed)
 │          ╱
 │      ╱ ← memory-bandwidth roof slope = HBM BW × AI
 │  ╱       (memory-bound region: more bandwidth = more speed)
 └──────────────────────────────────────────── Arithmetic Intensity (FLOP/byte)
         ↑
    Ridge point ≈ 989 TFLOPS / 3.35 TB/s ≈ 295 FLOP/byte (H100)

Any kernel with AI < 295 FLOP/byte is memory-bandwidth-limited on H100.
Transformer attention (~10 FLOP/byte) is 29× below the ridge point.
```

### Where AI Workloads Live

```
AI (FLOP/byte)      Workload                    Bound
─────────────────────────────────────────────────────────
0.5                 LLM decoding (batch=1)       Severely memory-bound
2–5                 Batch norm, layer norm        Memory-bound
10–30               Transformer attention         Memory-bound
50–100              Small batch matmul            Borderline
>295                Large matmul (batch≥64)       Compute-bound
```

**Practical implication:** Improving LLM inference token rate requires:
1. Larger batches (increase arithmetic intensity)
2. More HBM bandwidth (H100 → H200: 3.35 → 4.8 TB/s)
3. Weight quantisation (INT4 = 2× bandwidth; INT8 = same compute but half the bytes)

Not a faster GPU clock.

---

## 9. ISA and Architecture Across the AI Stack

| Layer in your roadmap | Relevant ISA/arch |
|----------------------|-------------------|
| PyTorch model training | x86-64 (AVX-512) + CUDA (H100/A100) |
| ML compiler backends | ARM64, x86-64, RISC-V, GPU PTX |
| FPGA accelerator control | ARM Cortex-M/A (Zynq PS) |
| Jetson edge deployment | ARM64 (Cortex-A78AE) + NVIDIA Ampere |
| Custom AI chip design | RISC-V (control core) + custom dataflow |
| Mobile inference | ARM64 (Neural Engine, Hexagon) |

**Key takeaway:** The ISA you write code for is secondary. The memory hierarchy and arithmetic intensity of your workload determines performance. A 10 FLOP/byte kernel runs at the same ~33 TFLOPS on H100 regardless of whether you write it in CUDA PTX or high-level Python — it's hitting the HBM bandwidth ceiling.

---

## 10. Labs

### Lab 1 — Cache Miss Profiling

**Goal:** measure how access pattern affects cache performance; tie latency numbers to the memory mountain table.

```bash
# Install perf and valgrind
sudo apt install linux-perf valgrind

# L1/L2/L3 miss rates for matrix transpose (row-major vs column-major)
perf stat -e cache-references,cache-misses,L1-dcache-misses \
          ./matrix_transpose 1024

# Cache simulation
valgrind --tool=cachegrind --cache-sim=yes ./matrix_transpose 1024
cg_annotate cachegrind.out.*
```

Write a 1024×1024 float matrix. Implement:
- Row-major transpose (good spatial locality for reads, bad for writes)
- Column-major transpose (bad for reads)
- Tiled transpose (32×32 blocks) — should be fastest

**Deliverable:** table of L1/L2/L3 miss rates for each version; explain the difference.

### Lab 2 — SIMD Vectorisation

**Goal:** measure scalar vs SIMD vs compiler-auto-vectorised throughput on a dot product kernel.

```cpp
// Compile and compare:
// gcc -O2 -march=native -o dot dot.cpp        (auto-vectorise)
// gcc -O0 -no-vec -o dot_scalar dot.cpp       (scalar)
// Manual intrinsics version (see section 5 example)
```

Benchmark with N = 10M floats. Use `perf stat` to verify SIMD instructions appear (`avx2` or `sse4.2` in the `cycles` vs `instructions` ratio).

**Deliverable:** bandwidth (GB/s) for each version; compare to DRAM bandwidth ceiling.

### Lab 3 — Roofline on a Real GPU

**Goal:** compute arithmetic intensity of three kernels and place them on the roofline.

```python
# Use NVIDIA Nsight Compute CLI
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
              l1tex__t_bytes.sum,\
              sm__sass_thread_inst_executed_op_fadd_pred_on.sum \
    python train_step.py

# Kernels to measure:
# 1. Element-wise ReLU (low AI — memory-bound)
# 2. Batch matrix multiply (high AI — compute-bound)
# 3. Softmax (medium AI)
```

**Deliverable:** roofline plot with three kernels marked; conclude which optimisations would help each.

### Lab 4 — Architecture ISA Comparison

**Goal:** compare code generation and performance across ISAs.

```bash
# Cross-compile the same matrix multiply kernel for three targets:
# x86-64 with AVX2
gcc -O3 -mavx2 -mfma -o matmul_x86 matmul.c

# ARM64 with NEON
aarch64-linux-gnu-gcc -O3 -mfpu=neon -o matmul_arm matmul.c

# RISC-V 64
riscv64-linux-gnu-gcc -O3 -march=rv64gcv -o matmul_riscv matmul.c

# Inspect generated assembly:
objdump -d matmul_x86  | grep -E "vmulps|vfmadd"
objdump -d matmul_arm  | grep -E "fmla|fmul"
```

**Deliverable:** instruction count, code size, and measured GFLOPS for each ISA (use QEMU for emulation if no native hardware).

---

## 11. Where This Takes You

```
This section
      │
      ├── Phase 1 §3 — Operating Systems
      │     Memory management, virtual memory, MMU (ties to TLB section above)
      │     Device drivers (how OS talks to your GPU/FPGA)
      │
      ├── Phase 1 §4 — C++ and Parallel Computing
      │     SIMD intrinsics in practice, OpenMP, CUDA kernels
      │     The CUDA memory model directly mirrors the GPU hierarchy in §6
      │
      ├── Phase 4A — Xilinx FPGA
      │     Implement your own systolic array in SystemVerilog
      │     Understand timing and pipeline stages at the RTL level
      │
      ├── Phase 4B — NVIDIA Jetson
      │     ARM64 host CPU + Ampere GPU — both covered here
      │
      └── Phase 5F — AI Chip Design
            Design a custom accelerator using systolic array / dataflow patterns
            Roofline model guides your PE count and memory bandwidth decisions
```

---

## Resources

| Resource | What for |
|----------|----------|
| **Patterson & Hennessy — Computer Organization and Design (ARM edition)** | Pipeline, cache, memory — the textbook standard |
| **Hennessy & Patterson — Computer Architecture: A Quantitative Approach** | Deep OoO, superscalar, memory system design |
| **NVIDIA H100 Architecture Whitepaper** | SM internals, Tensor Core specs, NVLink |
| **"What Every Programmer Should Know About Memory" — Ulrich Drepper** | Cache hierarchy, NUMA, prefetching (free PDF) |
| **Roofline: An Insightful Visual Performance Model — Williams et al. (2009)** | Original roofline paper — 10 pages, essential |
| **Chips and Cheese (blog)** | Reverse-engineered microarchitecture analysis (AMD, Intel, Apple) |
| **Wikichip** | Die shots, cache sizes, core counts, process nodes |
| **NVIDIA Nsight Compute** | GPU roofline and memory hierarchy profiling |
