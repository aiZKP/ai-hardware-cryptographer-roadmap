# MSM and NTT Algorithms — From Theory to Hardware Implementation

> **Goal:** Master the two computational kernels that dominate ZK proof generation — Multi-Scalar Multiplication (MSM) and Number Theoretic Transform (NTT) — at the algorithmic level required to design FPGA, GPU, and ASIC accelerators. By the end, you will understand Pippenger's bucket method down to exact operation counts, NTT butterfly architectures with memory access patterns, and the parallelism and bandwidth profiles that drive hardware design decisions. This guide bridges Phase 5 (the math) to the hardware implementations in Phase 6.2–6.3.

**Prerequisite:** Phase 5 complete — finite fields, elliptic curve arithmetic (point addition, doubling, projective coordinates), polynomial commitment schemes (KZG, FRI), proof systems (Groth16, PLONK, STARKs). You must be comfortable with field multiplication cost models and the concept of MSM and NTT from Phase 5.2–5.3.

---

## Table of Contents

1. [MSM — Why It Dominates ZK Proving](#1-msm--why-it-dominates-zk-proving)
2. [Naive MSM and Why It Fails](#2-naive-msm-and-why-it-fails)
3. [Pippenger's Bucket Method — The Standard Algorithm](#3-pippengerss-bucket-method--the-standard-algorithm)
4. [Pippenger Optimizations](#4-pippenger-optimizations)
5. [MSM Parallelism Analysis — For Hardware Designers](#5-msm-parallelism-analysis--for-hardware-designers)
6. [NTT Algorithm Deep Dive — Beyond the Basics](#6-ntt-algorithm-deep-dive--beyond-the-basics)
7. [NTT Optimizations for Hardware](#7-ntt-optimizations-for-hardware)
8. [NTT Memory Architecture — The Real Bottleneck](#8-ntt-memory-architecture--the-real-bottleneck)
9. [Batch MSM and Batch NTT](#9-batch-msm-and-batch-ntt)
10. [Operation Count Summary](#10-operation-count-summary)
11. [Projects](#11-projects)
12. [Resources](#12-resources)

---

## 1. MSM — Why It Dominates ZK Proving

### Formal Definition

Multi-Scalar Multiplication computes a single elliptic curve point from n scalar-point pairs:

```
Q = s₀·P₀ + s₁·P₁ + s₂·P₂ + ... + s_{n-1}·P_{n-1}

where:
  s_i ∈ F_r   (scalars, typically 253-255 bits for BN254/BLS12-381)
  P_i ∈ G₁    (elliptic curve points, from the SRS or witness)
  Q ∈ G₁      (the result — a single curve point)
```

### Where MSM Appears in ZK Provers

```
MSM occurrences in proof systems:
──────────────────────────────────
  KZG Polynomial Commitment:
    C = c₀·[1]₁ + c₁·[τ]₁ + c₂·[τ²]₁ + ... + c_d·[τ^d]₁
    → 1 MSM of size d+1 per polynomial commitment

  Groth16 Prover:
    [A]₁ = [α]₁ + Σ z_k·[A_k(τ)]₁ + r·[δ]₁      → MSM of size n in G₁
    [B]₂ = [β]₂ + Σ z_k·[B_k(τ)]₂ + s·[δ]₂       → MSM of size n in G₂
    [B]₁ = [β]₁ + Σ z_k·[B_k(τ)]₁ + s·[δ]₁       → MSM of size n in G₁
    [C]₁ = Σ z_j·[...] + Σ h_i·[τⁱ·t(τ)/δ]₁      → MSM of size ~2n in G₁
    Total: 4 MSMs in G₁, 1 MSM in G₂ (G₂ costs ~2-3x G₁)

  PLONK Prover:
    [a(τ)]₁, [b(τ)]₁, [c(τ)]₁                     → 3 MSMs of size n
    [z(τ)]₁                                          → 1 MSM of size n
    [t_lo]₁, [t_mid]₁, [t_hi]₁                      → 3 MSMs of size n
    [W_ζ]₁, [W_{ζω}]₁                                → 2 MSMs of size n
    Total: 9-11 MSMs in G₁

  IPA (Halo 2):
    Pedersen commitment = MSM of size n
    Each IPA round: MSM of shrinking size (n/2, n/4, ...)

  Nova Folding:
    Cross-term commitment: 1 MSM of size |circuit| per step
```

### MSM as Percentage of Total Prover Time

```
Proof system      MSM %        NTT %        Other %
─────────────────────────────────────────────────────
Groth16           60-80%       20-30%       5-10%
PLONK             40-50%       35-45%       5-10%
STARK             0%           50-60%       40-50% (hashing)
Nova (per step)   90%+         ~0%          ~10%

Key insight: MSM dominates SNARK provers.
            NTT dominates STARK provers.
            Hardware accelerators must target the right operation.
```

### Practical MSM Sizes

```
Application                         Circuit Size    MSM Size (n)
──────────────────────────────────────────────────────────────────
Simple token transfer (Zcash)       ~2^16 gates     2^16 (65K)
DeFi proof (Tornado Cash)           ~2^18 gates     2^18 (262K)
zkEVM block validity                ~2^22-2^24      2^22-2^24
Large ML inference proof            ~2^24-2^26      2^24-2^26
Filecoin sector proof               ~2^27           2^27 (134M)
```

---

## 2. Naive MSM and Why It Fails

### The Naive Algorithm

```
Algorithm: Naive MSM
────────────────────
Input: scalars s₀,...,s_{n-1} (each λ bits), points P₀,...,P_{n-1}
Output: Q = Σ s_i · P_i

Q = O  (point at infinity)
for i = 0 to n-1:
    T_i = ScalarMul(s_i, P_i)     // double-and-add: λ-1 doublings + ~λ/2 additions
    Q = Q + T_i                    // accumulate

Total cost per scalar multiplication (λ = 253 for BN254):
  Doublings:  252 (one per bit after MSB)
  Additions:  ~127 (average Hamming weight of random 253-bit scalar)
  Total:      ~379 point operations per scalar mul
```

### Cost of a Single Point Operation

```
Elliptic curve operation costs (BN254, Jacobian projective, a=0):
──────────────────────────────────────────────────────────────────
  Operation                    Field Muls (M)  Squarings (S)  M-equivalent
  ─────────────────────────────────────────────────────────────────────────
  Full addition (J + J)         11M + 5S                       ~15M
  Mixed addition (J + Aff)      7M + 4S                        ~10.2M
  Point doubling (J, a=0)       1M + 5S (dbl-2009-l)           ~5M
  Point negation                0 (negate y-coordinate)         free

Each field multiplication (BN254, 254-bit):
  4-limb Montgomery (CIOS method): 36 word multiplications + reductions
  Time on CPU: ~30-50 ns per field mul

So one mixed addition ≈ 10.2 × 30ns = ~306 ns
   one doubling ≈ 5 × 30ns = ~150 ns
```

### Naive MSM Total Cost

```
For n = 2^20 (1,048,576 points), λ = 253:

  Per scalar mul: ~252 doublings + ~127 additions = 379 point ops
  n scalar muls: 2^20 × 379 = ~397,556,224 point operations
  Accumulation: 2^20 - 1 additions

  Total: ~398 MILLION point operations
  At ~10M per mixed addition: ~3.98 BILLION field multiplications
  At 30 ns per mul: ~120 seconds on a single CPU core

  This is completely impractical. Pippenger reduces this by ~20-30x.
```

---

## 3. Pippenger's Bucket Method — The Standard Algorithm

### The Key Insight

Instead of computing each s_i·P_i independently, decompose all scalars into small windows and group points by their window values into "buckets." This converts n independent scalar multiplications into shared-structure accumulations.

### Algorithm Overview

```
PIPPENGER'S BUCKET METHOD
══════════════════════════

Input: n points P₀,...,P_{n-1}, n scalars s₀,...,s_{n-1} (each λ bits)
Parameters:
  c = window width (bits), to be optimized
  w = ⌈λ/c⌉ = number of windows
  B = 2^c - 1 = number of buckets per window (bucket 0 is identity, skipped)

────────────────────────────────────────────────────────────
Phase 1: SCALAR DECOMPOSITION
────────────────────────────────────────────────────────────
Decompose each scalar into w windows of c bits:

  s_i = s_{i,0} + s_{i,1}·2^c + s_{i,2}·2^{2c} + ... + s_{i,w-1}·2^{(w-1)c}

  where s_{i,j} ∈ [0, 2^c - 1] is the j-th window of scalar i.

The MSM becomes:
  Q = Σ_i s_i·P_i = Σ_{j=0}^{w-1} 2^{jc} · W_j

  where W_j = Σ_i s_{i,j}·P_i  is the "window sum" for window j.

────────────────────────────────────────────────────────────
Phase 2: BUCKET ACCUMULATION (for each window j)
────────────────────────────────────────────────────────────
Create 2^c - 1 buckets: B₁, B₂, ..., B_{2^c-1}, all = O (identity)

for i = 0 to n-1:
    v = s_{i,j}              // read c-bit window value
    if v ≠ 0:
        B_v = B_v + P_i       // add point to its bucket

Cost: at most n point additions per window
      (points with v=0 are skipped)

────────────────────────────────────────────────────────────
Phase 3: BUCKET REDUCTION (for each window j)
────────────────────────────────────────────────────────────
Convert bucket sums to window sum using the running-sum trick:

  W_j = 1·B₁ + 2·B₂ + 3·B₃ + ... + (2^c-1)·B_{2^c-1}

Computed WITHOUT any scalar multiplications:

  running_sum = O
  W_j = O
  for t = 2^c - 1 down to 1:
      running_sum = running_sum + B_t
      W_j = W_j + running_sum

Cost: 2·(2^c - 1) point additions per window

Why this works:
  After t=3: running = B₃,           W = B₃
  After t=2: running = B₃+B₂,        W = 2B₃+B₂
  After t=1: running = B₃+B₂+B₁,     W = 3B₃+2B₂+B₁  ✓

────────────────────────────────────────────────────────────
Phase 4: WINDOW COMBINATION
────────────────────────────────────────────────────────────
Combine window sums from MSB to LSB:

  Q = W_{w-1}
  for j = w-2 down to 0:
      Q = 2^c · Q           // c doublings
      Q = Q + W_j            // 1 addition

Cost: (w-1)·c doublings + (w-1) additions ≈ λ doublings + w additions
```

### Concrete Worked Example: n = 8, c = 2

```
Example: 8 points with 8-bit scalars, window width c = 2
─────────────────────────────────────────────────────────
  w = ⌈8/2⌉ = 4 windows,  3 buckets per window (B₁, B₂, B₃)

Scalars decomposed into 2-bit windows [win0, win1, win2, win3]:
  s₀ = 11 = 00_00_10_11  →  [3, 2, 0, 0]
  s₁ = 5  = 00_00_01_01  →  [1, 1, 0, 0]
  s₂ = 14 = 00_00_11_10  →  [2, 3, 0, 0]
  s₃ = 9  = 00_00_10_01  →  [1, 2, 0, 0]
  s₄ = 7  = 00_00_01_11  →  [3, 1, 0, 0]
  s₅ = 3  = 00_00_00_11  →  [3, 0, 0, 0]
  s₆ = 12 = 00_00_11_00  →  [0, 3, 0, 0]
  s₇ = 2  = 00_00_00_10  →  [2, 0, 0, 0]

═══ Window 0 (bits [1:0]) ═══

  Bucket assignments:
    B₁ ← P₁, P₃          (s₁,₀=1, s₃,₀=1)
    B₂ ← P₂, P₇          (s₂,₀=2, s₇,₀=2)
    B₃ ← P₀, P₄, P₅     (s₀,₀=3, s₄,₀=3, s₅,₀=3)
    Skipped: P₆ (s₆,₀=0)

  Bucket accumulation:
    B₁ = P₁ + P₃                    (1 addition)
    B₂ = P₂ + P₇                    (1 addition)
    B₃ = P₀ + P₄ + P₅              (2 additions)
    Total: 4 additions

  Bucket reduction (running sum from t=3 to t=1):
    t=3: running = B₃               W₀ = B₃
    t=2: running = B₃ + B₂          W₀ = 2B₃ + B₂
    t=1: running = B₃ + B₂ + B₁    W₀ = 3B₃ + 2B₂ + B₁
    = 3(P₀+P₄+P₅) + 2(P₂+P₇) + 1(P₁+P₃)
    Total: 4 additions

═══ Window 1 (bits [3:2]) ═══

  Bucket assignments:
    B₁ ← P₁, P₄          (s₁,₁=1, s₄,₁=1)
    B₂ ← P₀, P₃          (s₀,₁=2, s₃,₁=2)
    B₃ ← P₂, P₆          (s₂,₁=3, s₆,₁=3)
    Skipped: P₅, P₇ (window value = 0)

  Same cost: 3 accumulation adds + 4 reduction adds

═══ Windows 2, 3: all zeros → produce identity ═══

═══ Window Combination ═══

  Q = W₃                           (identity)
  Q = 4·Q + W₂ = W₂               (identity, 2 doublings + 1 add)
  Q = 4·Q + W₁ = W₁               (2 doublings + 1 add)
  Q = 4·Q + W₀ = 4·W₁ + W₀       (2 doublings + 1 add)

  = Σ_i (4·s_{i,1} + s_{i,0}) · P_i = Σ_i s_i · P_i  ✓

Total: 4+4 + 3+4 = 15 additions + 6 doublings = ~21 point operations
Naive: 8 × (8 doublings + 4 additions) + 7 accumulations = ~103 point ops
Speedup: ~5x (even for this tiny example!)
```

### Optimal Window Width

```
Total operations:
  T(n, c) = w · (n + 2^{c+1})    where w = ⌈λ/c⌉

  = (λ/c) · (n + 2^{c+1})

To minimize, set dT/dc = 0:
  When n >> 2^c: n/c ≈ 2^c · ln(2)  →  2^c ≈ n/(c · ln 2)
  Taking log:    c ≈ log₂(n) - log₂(c) - log₂(ln 2)

  Practical rule: c ≈ log₂(n) - 2  to  log₂(n)

  For n = 2^20:  optimal c ≈ 15-16
  For n = 2^24:  optimal c ≈ 19-20
  For n = 2^16:  optimal c ≈ 12-13
```

### Concrete Operation Counts for n = 2^20

```
n = 2^20, λ = 253 (BN254), various window widths:
──────────────────────────────────────────────────────────
  c    w     Accum (w·n)    Reduce (w·2^{c+1})  Comb    Total
  ─── ──── ─────────────── ──────────────────── ─────── ──────────
  12   22   23,068,672      180,224              263     23,249,159
  14   19   19,922,944      622,592              265     20,545,801
  15   17   17,825,792      1,048,576            256     18,874,624
  16   16   16,777,216      2,097,152            255     18,874,623  ←
  18   15   15,728,640      7,864,320            267     23,593,227
  20   13   13,631,488      27,262,976           252     40,894,716

  Optimum: c = 15-16, giving ~18.9M point operations
  Compare naive: ~398M point operations → 21x speedup
```

---

## 4. Pippenger Optimizations

### 4.1 Signed Digit Representation

```
SIGNED BUCKET INDEX
═══════════════════
Core idea: treat window values as SIGNED integers in [-(2^{c-1}), ..., -1, 1, ..., 2^{c-1}]
           instead of unsigned [0, ..., 2^c - 1].

For each window value v:
  If v < 2^{c-1}:   use bucket +v, add P_i normally
  If v ≥ 2^{c-1}:   use bucket (2^c - v), add -P_i, carry +1 to next window

Why this works:
  Point negation is FREE on elliptic curves (just negate y-coordinate).
  We halve the number of buckets with zero additional cost.

Impact:
  Bucket count:   2^c - 1  →  2^{c-1}  (halved!)
  Reduce cost:    2·(2^c - 1) →  2·(2^{c-1} - 1)  (halved!)
  Memory:         halved (critical for on-chip SRAM)

With signed digits, c = 16, n = 2^20:
  Buckets per window: 2^15 = 32,768 (vs 65,535 unsigned)
  Accumulation: 16 × 2^20 = 16,777,216
  Reduction: 16 × 2 × 32,767 = 1,048,544
  Total: ~17.8M ops (vs 18.9M unsigned) → 6% improvement
  Memory: 16 windows × 32,768 buckets × 144 bytes = 75 MB (vs 151 MB)
```

### 4.2 GLV/GLS Endomorphism

```
GLV ENDOMORPHISM FOR BN254 / BLS12-381
═══════════════════════════════════════
These curves have an efficient endomorphism φ: (x, y) → (β·x, y)
where β is a cube root of unity in F_p.

Computing φ(P) costs: 1 field multiplication (multiply x by β).
The endomorphism has eigenvalue λ in the scalar field (cube root of unity in F_r).

Scalar decomposition:
  For each scalar s, find s₁, s₂ (each ~λ/2 bits) such that:
    s·P = s₁·P + s₂·φ(P)

  This converts n-point MSM into 2n-point MSM with half-length scalars:
    Σ s_i·P_i = Σ [s_{i,1}·P_i + s_{i,2}·φ(P_i)]

Impact on Pippenger:
  Before: w = ⌈253/c⌉ windows over n points
  After:  w = ⌈127/c⌉ windows over 2n points

  Cost comparison (c = 15):
  ┌──────────────┬─────────────────────┬─────────────────────┐
  │              │ Without GLV         │ With GLV            │
  ├──────────────┼─────────────────────┼─────────────────────┤
  │ Points       │ n = 2^20            │ 2n = 2^21           │
  │ Scalar bits  │ 253                 │ ~127                │
  │ Windows      │ ⌈253/15⌉ = 17       │ ⌈127/15⌉ = 9        │
  │ Accumulation │ 17 × 2^20 = 17.8M  │ 9 × 2^21 = 18.9M   │
  │ Reduction    │ 17 × 32,768 = 557K │ 9 × 32,768 = 295K  │
  │ Total ops    │ ~18.4M              │ ~19.2M              │
  │ But: each op │ 253-bit scalar proc │ 127-bit scalar proc │
  │ Net speedup  │ baseline            │ ~33-42% faster      │
  └──────────────┴─────────────────────┴─────────────────────┘

  The 33-42% speedup comes from: fewer windows reduces overhead,
  and shorter scalars mean simpler decomposition and fewer carries.
```

### 4.3 Mixed Addition Optimization

```
MIXED AFFINE-PROJECTIVE ADDITION
═════════════════════════════════
Store base points P_i in AFFINE coordinates: (x, y) — 2 field elements
Keep bucket accumulators in JACOBIAN projective: (X, Y, Z) — 3 field elements

  Mixed addition (J + Aff): 7M + 4S ≈ 10.2M-equivalent
  Full Jacobian (J + J):    11M + 5S ≈ 15M-equivalent

  Saving: 4M + 1S per addition ≈ 35% cheaper

Since bucket accumulation dominates (n additions per window),
this optimization applies to the BULK of Pippenger's cost.

SRS point storage (BN254 affine):
  Per point: 2 × 32 bytes = 64 bytes
  n = 2^20: 64 MB of SRS data (streamed from DDR)
  n = 2^24: 1 GB of SRS data
```

### 4.4 Batch Affine Conversion (Montgomery's Trick)

```
MONTGOMERY'S BATCH INVERSION
═════════════════════════════
Problem: converting k projective points to affine requires k field inversions.
         One inversion costs ~100-300 field multiplications.

Montgomery's trick: compute k inversions using 1 inversion + 3(k-1) multiplications.

  Forward pass: compute prefix products
    prod[0] = Z₀
    prod[i] = prod[i-1] × Z_i    for i = 1,...,k-1

  Single inversion:
    inv_all = 1 / prod[k-1]

  Backward pass: extract individual inverses
    for i = k-1 down to 1:
        Z_i⁻¹ = inv_all × prod[i-1]
        inv_all = inv_all × Z_i
    Z₀⁻¹ = inv_all

Cost: 3(k-1) multiplications + 1 inversion
For k = 32,768 buckets: 1 inversion + ~98K multiplications
  vs. 32,768 inversions (each ~200 muls) = ~6.5M equivalent muls
  Savings: ~98% of inversion cost

Application: after bucket accumulation, convert all bucket accumulators
to affine using batch inversion. This enables the batch affine addition
technique used in ZPrize-winning MSM implementations.

Batch affine point addition (amortized):
  6M per addition (vs 10.2M mixed addition)
  Used in batches of 256-512 points for maximum amortization.
```

### 4.5 Pre-sorting Points by Bucket Index

```
BUCKET-SORTED ACCESS PATTERN
═════════════════════════════
Default: process points sequentially, each goes to a random bucket
  → random read-modify-write to bucket memory (cache-unfriendly)

Optimization: pre-sort points by bucket index for each window
  → sequential scan through each bucket's point list (cache-friendly)

Trade-off:
  Sorting cost: O(n·w) work for all windows (counting sort, O(n) per window)
  Benefit: eliminates random bucket access in accumulation phase
  Memory: O(n) extra for index arrays

For hardware: sorting enables STREAMING access to bucket accumulators.
Instead of random-access SRAM, you can use a FIFO-based architecture.
```

---

## 5. MSM Parallelism Analysis — For Hardware Designers

### Data Dependency Graph

```
PIPPENGER PARALLELISM MAP
═════════════════════════

Phase 2 (Bucket Accumulation):
  ┌─────────────────────────────────────────────────────────┐
  │ Inter-window: ALL w windows are FULLY INDEPENDENT       │
  │   → w-way parallelism (w = 16 for c=16, λ=253)        │
  │   → Each window can run on a separate hardware unit     │
  │                                                         │
  │ Intra-window: Points within a window are MOSTLY independent │
  │   → Points mapping to DIFFERENT buckets: fully parallel │
  │   → Points mapping to the SAME bucket: must serialize   │
  │   → Collision probability: ~1/2^c per pair             │
  │     For c=16: ~0.0015% chance two consecutive points   │
  │     hit the same bucket                                 │
  └─────────────────────────────────────────────────────────┘

Phase 3 (Bucket Reduction):
  ┌─────────────────────────────────────────────────────────┐
  │ Inter-window: ALL w windows INDEPENDENT (same as above) │
  │                                                         │
  │ Intra-window: SEQUENTIAL (running sum depends on        │
  │   previous bucket)                                      │
  │   → Sequential depth: 2^{c-1} - 1 additions            │
  │   → For c=16 (signed): 32,767 sequential additions     │
  │   → CANNOT be parallelized within a single window       │
  └─────────────────────────────────────────────────────────┘

Phase 4 (Window Combination):
  ┌─────────────────────────────────────────────────────────┐
  │ FULLY SEQUENTIAL: MSB to LSB, c doublings + 1 add each │
  │   → Sequential depth: (w-1)·c + (w-1) ≈ λ operations  │
  │   → For c=16, w=16: 240 doublings + 15 additions       │
  │   → Negligible compared to Phases 2-3                   │
  └─────────────────────────────────────────────────────────┘
```

### Hardware Collision Handling

```
BUCKET COLLISION IN PIPELINED HARDWARE
═══════════════════════════════════════
Problem: a hardware point adder has pipeline depth d cycles.
If point i maps to bucket B_v, the result B_v + P_i is not
available until d cycles later. If point i+k (where k < d)
also maps to B_v, we have a read-after-write hazard.

Collision probability per point: d / 2^c
  For d = 200 cycles (typical FPGA EC adder), c = 16:
    collision rate ≈ 200/65,536 ≈ 0.3%

Solution (from ZPrize 2023 winner):
  ┌──────────────────────────────────────────────────────────┐
  │  Point stream  →  [Collision Detector]  →  [Point Adder Pipeline]  │
  │                        │                         │                  │
  │                        │ collision?              │ result           │
  │                        ↓                         ↓                  │
  │                   [Pending Buffer]         [Bucket Memory]          │
  │                   (~35 entries)            (on-chip SRAM)           │
  │                        │                                            │
  │                        └─ retry when bucket unlocked ──→            │
  └──────────────────────────────────────────────────────────┘

  Pending buffer stabilizes at ~35 entries for 2^24 MSM
  Pipeline utilization: >99%
```

### Parallelism Profile for n = 2^20, c = 16

```
┌──────────────────┬────────────────┬───────────────────┬────────────────────┐
│ Phase            │ Operations     │ Available          │ Sequential Depth   │
│                  │                │ Parallelism        │                    │
├──────────────────┼────────────────┼───────────────────┼────────────────────┤
│ Bucket accum.    │ 16.8M adds     │ 16 windows ×       │ ~2^20/16 = 65K    │
│                  │                │ ~65K indep buckets │ per window          │
├──────────────────┼────────────────┼───────────────────┼────────────────────┤
│ Bucket reduction │ 2.1M adds      │ 16 windows         │ ~131K sequential   │
│                  │                │ (parallel)         │ per window          │
├──────────────────┼────────────────┼───────────────────┼────────────────────┤
│ Window combine   │ 240 dbl + 15   │ 1 (sequential)    │ 255 operations     │
│                  │ add            │                    │                    │
└──────────────────┴────────────────┴───────────────────┴────────────────────┘

Bottleneck: Bucket accumulation (89% of operations)
With 16 parallel EC adders (one per window):
  Accumulation: ~2^20 cycles × 200 ns/add = ~210 ms
  Reduction:    ~131K × 200 ns = ~26 ms
  Combination:  negligible

Total: ~240 ms at 200 ns per point addition
This matches real hardware performance (ZPrize FPGA: ~500 ms for 4× larger MSM)
```

### Memory Bandwidth Requirements

```
MSM MEMORY PROFILE
═══════════════════
Input streaming (SRS points):
  Each point (BN254 affine): 64 bytes
  Rate: 1 point per EC addition cycle (~200 ns at 250 MHz)
  Bandwidth: 64 bytes / 200 ns = 320 MB/s per window
  16 windows: 5.1 GB/s total → achievable with DDR4/HBM

Bucket memory (random access):
  Each bucket (Jacobian): 3 × 32 bytes = 96 bytes (BN254)
  Total buckets (c=16, signed, 16 windows): 16 × 32K = 524K buckets
  Total memory: 524K × 96 bytes = 50 MB → must be on-chip SRAM

  For c=13 (smaller windows): 16 × 4K = 65K buckets × 96 = 6.2 MB
    → fits in FPGA URAM (Alveo U250 has ~54 MB URAM)

Memory sizing drives the choice of c:
  c=16: 50 MB bucket memory (tight for FPGA)
  c=13: 6 MB bucket memory (comfortable for FPGA)
  c=20: 800 MB bucket memory (GPU HBM only)
```

---

## 6. NTT Algorithm Deep Dive — Beyond the Basics

### Recap: The Radix-2 Butterfly

```
COOLEY-TUKEY BUTTERFLY (Radix-2 DIT)
═════════════════════════════════════
Input:  a, b (two field elements)
Twiddle factor: ω^t (precomputed power of root of unity)

  temp = ω^t × b              // 1 field multiplication
  a' = a + temp                // 1 field addition
  b' = a - temp                // 1 field subtraction (reuse temp)

Cost: 1 MUL + 2 ADD per butterfly
Total for n-point NTT: (n/2)·log₂(n) MUL + n·log₂(n) ADD
```

### DIT vs DIF — Which Is Better for Hardware?

```
DECIMATION-IN-TIME (DIT) — Cooley-Tukey:
  Butterfly: MULTIPLY first, then ADD/SUB
    b_tw = ω^t × b
    a' = a + b_tw
    b' = a - b_tw

  Input order: bit-reversed
  Output order: natural

DECIMATION-IN-FREQUENCY (DIF) — Gentleman-Sande:
  Butterfly: ADD/SUB first, then MULTIPLY
    a' = a + b
    b' = (a - b) × ω^t

  Input order: natural
  Output order: bit-reversed

HARDWARE BEST PRACTICE:
  Use DIT for forward NTT:   bit-reversed input → natural output
  Use DIF for inverse NTT:   bit-reversed input → natural output

  Chaining: forward(DIT) → pointwise multiply → inverse(DIF)
    Output of DIT is natural order
    But DIF expects natural input for its standard form...

  Actually, the optimal combination:
    Forward NTT (DIF): natural input → bit-reversed output
    Pointwise multiply: bit-reversed order (both operands same order) ✓
    Inverse NTT (DIT): bit-reversed input → natural output
    → NO explicit bit-reversal permutation needed!
```

### Bit-Reversal Permutation

```
BIT-REVERSAL FOR n = 16 (4 bits)
═════════════════════════════════
  Index (binary)    Bit-reversed    Decimal → Decimal
  0000 (0)     →    0000 (0)       0 ↔ 0
  0001 (1)     →    1000 (8)       1 ↔ 8
  0010 (2)     →    0100 (4)       2 ↔ 4
  0011 (3)     →    1100 (12)      3 ↔ 12
  0100 (4)     →    0010 (2)       4 ↔ 2  (already swapped)
  0101 (5)     →    1010 (10)      5 ↔ 10
  0110 (6)     →    0110 (6)       6 ↔ 6  (self-swap)
  0111 (7)     →    1110 (14)      7 ↔ 14
  ...

Cost of explicit bit-reversal:
  ~n/2 swaps, each reading and writing 2 elements
  For n = 2^20, element = 48 bytes (BLS12-381):
    Data movement: 2^20 × 48 = 48 MB
    At 25 GB/s DDR4: ~2 ms  (non-negligible vs NTT compute)

Avoidance: Use DIF forward + DIT inverse to eliminate the permutation entirely.
```

### Butterfly Diagram for n = 16

```
16-POINT RADIX-2 DIT NTT (4 stages, 8 butterflies per stage)
═══════════════════════════════════════════════════════════════

Input                Stage 0        Stage 1        Stage 2        Stage 3       Output
(bit-rev)           stride=1       stride=2       stride=4       stride=8      (natural)

x[0] ──────────┬──(+)────────┬──(+)────────┬──(+)────────┬──(+)────── X[0]
               │              │              │              │
x[8] ──[×ω⁰₂]┴──(-)────┬──│──(+)────┬──│──(+)────┬──│──(+)────── X[1]
                         │  │          │  │          │  │
x[4] ──────────┬──(+)───│──┴──(-)───│──│──(+)───│──│──(+)────── X[2]
               │         │           │  │         │  │
x[12]──[×ω⁰₂]┴──(-)───┴──[×ω¹₄]──┴──(-)───┬──│──(+)────── X[3]
                                              │  │
x[2] ──────────┬──(+)────────┬──(+)─────────│──┴──(-)────── X[4]
               │              │               │
x[10]──[×ω⁰₂]┴──(-)────┬──│──(+)────────│──[×ω¹₈]──── X[5]
                         │  │               │
x[6] ──────────┬──(+)───│──┴──(-)─────────│────────────── X[6]
               │         │                  │
x[14]──[×ω⁰₂]┴──(-)───┴──[×ω¹₄]────────┴──[×ω³₈]──── X[7]

x[1] ──────────┬──(+)────────┬──(+)────────┬──(+)────── X[8]
               │              │              │
x[9] ──[×ω⁰₂]┴──(-)────┬──│──(+)────┬──│──(-)────── X[9]
                         │  │          │  │  [×ω¹₁₆]
x[5] ──────────┬──(+)───│──┴──(-)───│──│────────────── X[10]
               │         │           │  │
x[13]──[×ω⁰₂]┴──(-)───┴──[×ω¹₄]──┴──(-)────────── X[11]
                                        [×ω³₁₆]
x[3] ──────────┬──(+)────────┬──(+)──────────────────── X[12]
               │              │
x[11]──[×ω⁰₂]┴──(-)────┬──│──(+)──────────────────── X[13]
                         │  │  [×ω²₁₆]
x[7] ──────────┬──(+)───│──┴──(-)──────────────────── X[14]
               │         │
x[15]──[×ω⁰₂]┴──(-)───┴──[×ω¹₄]──────────────────── X[15]
                                   [×ω⁵₁₆]

  (Simplified diagram — full twiddle factors shown at stride boundaries)

Key observation:
  Stage 0: stride 1  → elements 0,1 paired → adjacent access (CACHE-FRIENDLY)
  Stage 1: stride 2  → elements 0,2 paired → still local
  Stage 2: stride 4  → elements 0,4 paired → getting distant
  Stage 3: stride 8  → elements 0,8 paired → far apart

  For n = 2^20:
    Stage 19: stride 2^19 = 524,288 → elements 25 MB apart!
```

---

## 7. NTT Optimizations for Hardware

### 7.1 Mixed-Radix NTT

```
MIXED-RADIX BUTTERFLY COMPARISON
═════════════════════════════════

Radix-2:  processes 2 elements per butterfly
  Cost: 1 MUL + 2 ADD
  Stages: log₂(n)
  Total MUL: (n/2) · log₂(n)

Radix-4:  processes 4 elements per butterfly
  Cost: 3 MUL + 8 ADD
  Stages: log₂(n) / 2
  Total MUL: (3n/8) · log₂(n)
  Saving vs radix-2: 25% fewer multiplications

Radix-8:  processes 8 elements per butterfly
  Cost: 7 MUL + 24 ADD
  Stages: log₂(n) / 3
  Total MUL: (7n/24) · log₂(n)
  Saving vs radix-2: 42% fewer multiplications

For n = 2^20:
  ┌──────────┬─────────────┬──────────────┬─────────┐
  │ Radix    │ Total MUL   │ Total ADD    │ MUL Saving │
  ├──────────┼─────────────┼──────────────┼─────────┤
  │ Radix-2  │ 10,485,760  │ 20,971,520   │ baseline │
  │ Radix-4  │  7,864,320  │ 20,971,520   │ 25%      │
  │ Radix-8  │  6,106,710  │ ~21,000,000  │ 42%      │
  └──────────┴─────────────┴──────────────┴─────────┘

Hardware trade-off:
  Higher radix = fewer multiplications, but more complex routing
  Radix-4 butterfly: 4 input ports, 3 multipliers, 8 adders
  Radix-8 butterfly: 8 input ports, 7 multipliers, 24 adders

  Sweet spot for FPGA/ASIC: radix-4 or radix-8
  (routing complexity grows quadratically, may limit clock frequency)
```

### 7.2 Twiddle Factor Generation

```
TWIDDLE FACTOR STRATEGIES
═════════════════════════

Option 1: Full precomputed table
  Store: ω⁰, ω¹, ω², ..., ω^{n/2-1}
  Memory: n/2 field elements
  For n=2^20, BLS12-381 (48 bytes): 24 MB
  Access: O(1) per lookup
  Problem: exceeds FPGA BRAM (typical ~4.5 MB)

Option 2: Per-stage smaller tables
  Stage s uses factors: ω^{k·2^{log n - 1 - s}} for k = 0..n/2^{s+1}-1
  Each stage table: n/2^{s+1} entries
  Early stages: large tables; late stages: small tables

Option 3: On-the-fly with seed table
  Store: 2^{k/2} seed values (for n = 2^k)
  Compose: any twiddle = seed[high_bits] × seed[low_bits]
  For n = 2^20: 2^10 = 1024 entries = 48 KB (BLS12-381)
  Cost: 1 extra multiplication per twiddle factor
  Perfect for FPGA: tiny BRAM usage, 1 multiplier overhead

Option 4: Sequential generation
  Compute ω^{k+1} = ω^k × ω (1 multiplication per step)
  Memory: O(1) — just store current value
  Only works for stages with sequential twiddle access
  Ideal for streaming architectures

FPGA recommendation: Option 3 (seed table) — 48 KB BRAM + 1 MUL
GPU recommendation:  Option 1 (full table) — fits in shared memory for small fields
```

### 7.3 Four-Step NTT (Cache-Oblivious Decomposition)

```
FOUR-STEP NTT FOR LARGE TRANSFORMS
═══════════════════════════════════
When n exceeds on-chip memory, decompose n = n₁ × n₂:

  Step 1: Arrange n elements as an n₂ × n₁ matrix (column-major)
  Step 2: Perform n₂ NTTs of size n₁ (column NTTs)
  Step 3: Multiply element (i,j) by twiddle factor ω_n^{i·j}
  Step 4: Perform n₁ NTTs of size n₂ (row NTTs)

Example: n = 2^24 = 2^12 × 2^12:
  Step 2: 4,096 NTTs of size 4,096 (each fits in ~192 KB BRAM for BN254)
  Step 3: 2^24 = 16.8M twiddle multiplications
  Step 4: 4,096 NTTs of size 4,096

Operation overhead vs monolithic NTT:
  ┌────────┬────────────────────┬───────────────────────┬──────────┐
  │ n      │ Monolithic MUL     │ Four-step MUL         │ Overhead │
  ├────────┼────────────────────┼───────────────────────┼──────────┤
  │ 2^20   │ 10,485,760         │ 10,485,760 + 2^20     │ +10%     │
  │ 2^24   │ 201,326,592        │ 201,326,592 + 2^24    │ +8.3%    │
  │ 2^28   │ 3,758,096,384      │ 3,758,096,384 + 2^28  │ +7.1%    │
  └────────┴────────────────────┴───────────────────────┴──────────┘

  The ~8-10% extra multiplications are a small price for:
    → Each sub-NTT fits in on-chip memory (no cache misses!)
    → Sequential access to DDR for loading/storing columns and rows
    → Enabling hardware implementations for arbitrarily large n

FPGA architecture:
  [DDR/HBM] ←──→ [Column/Row Buffer] ←──→ [On-chip NTT Engine]
                        ↕
                 [Matrix Transpose Buffer]

  The NTT engine processes one sub-NTT at a time,
  reading a column from DDR, computing NTT on-chip,
  writing the result back. Then transposes and repeats.
```

### 7.4 NTT Over Different Fields

```
FIELD SIZE DETERMINES HARDWARE COST PER BUTTERFLY
═══════════════════════════════════════════════════

┌──────────────┬────────┬───────────────┬───────────────┬────────────┐
│ Field        │ Bits   │ Word-level    │ ~CPU Cycles   │ Relative   │
│              │        │ MULs per      │ per field MUL │ Cost       │
│              │        │ field MUL     │               │            │
├──────────────┼────────┼───────────────┼───────────────┼────────────┤
│ BN254 scalar │ 254    │ 36 (CIOS,     │ 50-80         │ 1.0x       │
│              │        │  4×64 limbs)  │               │            │
├──────────────┼────────┼───────────────┼───────────────┼────────────┤
│ BLS12-381    │ 381    │ 78 (CIOS,     │ 100-150       │ ~2.0x      │
│ base         │        │  6×64 limbs)  │               │            │
├──────────────┼────────┼───────────────┼───────────────┼────────────┤
│ Goldilocks   │ 64     │ 1 native      │ 5-8           │ ~0.1x      │
│ (2^64-2^32+1)│        │ 64-bit        │               │ (10x faster)│
├──────────────┼────────┼───────────────┼───────────────┼────────────┤
│ BabyBear     │ 31     │ 1 native      │ 3-5           │ ~0.06x     │
│ (15·2^27+1)  │        │ 32-bit        │               │ (16x faster)│
├──────────────┼────────┼───────────────┼───────────────┼────────────┤
│ Mersenne31   │ 31     │ 1 native      │ 3-4           │ ~0.05x     │
│ (2^31-1)     │        │ + shift/add   │               │ (20x faster)│
└──────────────┴────────┴───────────────┴───────────────┴────────────┘

Hardware implications:
  BN254/BLS12-381: each NTT butterfly needs a multi-limb Montgomery
    multiplier using 16-36 DSP slices on FPGA
  BabyBear/M31: each butterfly needs ONE 32-bit multiplier = 1 DSP slice
    → Can instantiate 16-36x MORE butterflies in the same FPGA area!

This is WHY STARK provers (using small fields) achieve higher
throughput than SNARK provers (using large fields) on equivalent hardware.
```

---

## 8. NTT Memory Architecture — The Real Bottleneck

### Why NTT Is Memory-Bound

```
NTT COMPUTE vs MEMORY
═════════════════════
Per butterfly (BN254):
  Compute: 1 field MUL (~50 ns) + 2 ADD (~5 ns) = ~55 ns
  Memory:  2 reads + 2 writes × 48 bytes = 192 bytes

  For n = 2^20 (48 MB data), L2 cache = 1 MB:
    Only 2^{14.3} elements fit in L2 cache

  Compute time (all butterflies): 10.5M × 55 ns = 577 ms
  Memory time (if fully cache-resident): 0 (overlapped with compute)
  Memory time (with cache misses): ???

  At late NTT stages (stride > cache size):
    Each butterfly causes 2 cache misses × 100 ns DRAM latency = 200 ns
    This EXCEEDS the 55 ns compute time → NTT is MEMORY-BOUND
```

### Stage-by-Stage Cache Analysis

```
CACHE MISS PROFILE (n = 2^20, elem = 48 bytes, L2 = 1 MB)
═══════════════════════════════════════════════════════════

Elements in L2: 1 MB / 48 bytes ≈ 21,845 ≈ 2^{14.4}

  Stage    Stride    Stride (bytes)   Fits in L2?    Cache Misses
  ────── ─────────── ──────────────── ────────────── ─────────────
  0        1          48              ✓ (adjacent)   ~0
  1        2          96              ✓              ~0
  ...      ...        ...             ✓              ~0
  14       16,384     786 KB          ✓ (barely)     some
  15       32,768     1.5 MB          ✗              ~2^19 = 524K
  16       65,536     3 MB            ✗              ~524K
  17       131,072    6 MB            ✗              ~524K
  18       262,144    12 MB           ✗              ~524K
  19       524,288    24 MB           ✗              ~524K

  Stages 0-14: ~0 cache misses (data fits in L2)
  Stages 15-19: ~524K misses per stage × 5 stages = ~2.6M cache misses
  At 100 ns per miss: 260 ms EXTRA latency (nearly doubling NTT time!)

  THIS is why the four-step NTT exists.
```

### Bank Conflict Analysis for FPGA Multi-Bank SRAM

```
BANK CONFLICTS IN MULTI-BANK BRAM
═══════════════════════════════════
FPGA BRAM is organized in banks. For P parallel butterfly units
accessing B memory banks:

Address mapping: element at address a → bank (a mod B)
Butterfly at stage s pairs: (a, a + 2^s)
Banks accessed: (a mod B) and ((a + 2^s) mod B)

CONFLICT: when 2^s ≡ 0 (mod B), EVERY butterfly hits the same bank pair!

Example: B = 4 banks, stage s = 2 (stride = 4):
  Pairs (0,4): banks (0, 0) → CONFLICT!
  Pairs (1,5): banks (1, 1) → CONFLICT!
  All butterflies in this stage conflict!

Solutions:
  1. CONFLICT-FREE MEMORY MAPPING (CFNTT scheme):
     σ(a, stage) maps logical address to physical bank
     such that no two accesses in the same butterfly
     hit the same bank. Implemented as XOR-based address
     scrambling in hardware.

  2. PERMUTATION NETWORK:
     Crossbar switch between processing elements and
     memory banks. Adds ~1 cycle latency but eliminates
     all conflicts.

  3. DUAL-PORT BRAM:
     Each bank has 2 read/write ports. Halves conflict
     rate but doesn't eliminate it.
```

---

## 9. Batch MSM and Batch NTT

### 9.1 Batch MSM

```
BATCH MSM — SAME BASE POINTS, DIFFERENT SCALARS
═════════════════════════════════════════════════
Scenario: PLONK prover commits multiple polynomials to the same SRS.
  MSM₁ = Σ a_i · [τⁱ]₁    (wire polynomial a)
  MSM₂ = Σ b_i · [τⁱ]₁    (wire polynomial b)
  MSM₃ = Σ c_i · [τⁱ]₁    (wire polynomial c)

Optimizations:
  1. Read SRS points ONCE from DDR, reuse for all MSMs
     → Reduces memory bandwidth by factor of k (number of MSMs)
  2. Pre-sort bucket indices once per SRS permutation
  3. Share twiddle/endomorphism precomputation

  Amortized SRS read cost: O(n) instead of O(k·n)
  For PLONK with 9 MSMs: ~9x reduction in SRS bandwidth

BATCH MSM — SAME SCALARS, DIFFERENT BASE POINTS
═════════════════════════════════════════════════
Scenario: Groth16 computes MSMs with the same witness z_k
but different SRS polynomials [A_k(τ)]₁, [B_k(τ)]₁, etc.

Optimizations:
  1. Scalar decomposition (window splitting) computed ONCE
  2. Bucket index for each point is the SAME across MSMs
  3. Hardware scheduler processes all MSMs with shared indices
```

### 9.2 Batch NTT

```
BATCH NTT — SHARED TWIDDLE FACTORS
═══════════════════════════════════
Scenario: Groth16 needs 3 IFFTs + 3 FFTs, all of size n.
         PLONK needs 10-15 NTTs, all over the same domain.

Optimizations:
  1. Load twiddle factor table ONCE, reuse for all NTTs
     → For on-chip table: load once at startup
     → For DDR table: amortize bandwidth across k NTTs

  2. Interleaved processing:
     Process same butterfly position across all k NTTs:
       Read a₁[i], b₁[i], a₂[i], b₂[i], ..., aₖ[i], bₖ[i]
       Apply SAME twiddle factor ω^t to all k butterfly pairs
       Write results for all k NTTs

     Benefits:
       - Amortize twiddle factor read across k multiplications
       - Better memory bandwidth utilization (larger burst reads)
       - More arithmetic operations per twiddle factor fetch

  3. ICICLE GPU batch NTT:
     Single kernel launch for k NTTs
     Shared twiddle factor memory
     k-way data parallelism within each warp
```

---

## 10. Operation Count Summary

### MSM Comparison (n = 2^20, λ = 253, BN254)

```
┌─────────────────────────────────┬──────────────┬──────────────┬───────────┐
│ Algorithm                       │ Point Ops    │ Field MULs   │ Relative  │
│                                 │ (additions)  │ (approx)     │           │
├─────────────────────────────────┼──────────────┼──────────────┼───────────┤
│ Naive (double-and-add)          │ 398M         │ 3,980M       │ 1.00x     │
│ Pippenger (c=16, unsigned)      │ 18.9M        │ 189M         │ 0.048x    │
│ Pippenger (c=16, signed)        │ 17.8M        │ 178M         │ 0.045x    │
│ Pippenger + mixed addition      │ 17.8M        │ 127M*        │ 0.032x    │
│ Pippenger + signed + GLV        │ ~13M         │ ~93M*        │ 0.023x    │
│ Pippenger + all optimizations   │ ~13M         │ ~78M**       │ 0.020x    │
│   (signed + GLV + batch affine) │              │              │           │
└─────────────────────────────────┴──────────────┴──────────────┴───────────┘
* using mixed addition (7M+4S per add instead of 11M+5S)
** using batch affine (~6M per add)

Speedup: ~50x over naive with all optimizations
```

### NTT Comparison (n = 2^20, BN254 scalar field)

```
┌───────────────────────────┬──────────────┬──────────────┬───────────┐
│ Algorithm                 │ Field MULs   │ Field ADDs   │ MUL Saving│
├───────────────────────────┼──────────────┼──────────────┼───────────┤
│ Radix-2 (baseline)        │ 10,485,760   │ 20,971,520   │ —         │
│ Radix-4                   │ 7,864,320    │ 20,971,520   │ 25%       │
│ Radix-8                   │ 6,106,710    │ ~21,000,000  │ 42%       │
│ Four-step (radix-2 sub)   │ 11,534,336   │ 20,971,520   │ -10% *    │
└───────────────────────────┴──────────────┴──────────────┴───────────┘
* Four-step has MORE multiplications but FAR fewer cache misses

Four-step advantage (wall-clock, n = 2^24):
  Monolithic: ~6 sec (compute) + ~2 sec (cache misses) = ~8 sec
  Four-step:  ~6.5 sec (compute) + ~0.1 sec (cache-friendly) = ~6.6 sec
```

### Hardware Performance Benchmarks

```
┌──────────────────┬──────────────┬──────────────┬──────────────┬───────────┐
│ Operation        │ CPU (1 core) │ GPU (A100)   │ FPGA (U250)  │ ASIC est. │
├──────────────────┼──────────────┼──────────────┼──────────────┼───────────┤
│ 2^20 MSM (BN254) │ ~10-15 s     │ ~0.5-1 s     │ ~0.3-0.5 s   │ ~0.06 s   │
│ 2^20 NTT (BN254) │ ~1-3 s       │ ~50-100 ms   │ ~20-50 ms    │ ~10 ms    │
│ 2^24 MSM (BLS377)│ ~200 s       │ ~8-15 s      │ ~2-4 s (×4)  │ —         │
│ 2^24 NTT (Baby)  │ ~0.5 s       │ ~5-20 ms     │ ~2-5 ms      │ ~1 ms     │
└──────────────────┴──────────────┴──────────────┴──────────────┴───────────┘

ZPrize 2023 FPGA winner (AMD Alveo U250, 250 MHz):
  4× 2^24 MSMs on BLS12-377: 2.04 seconds total
  ~750 million EC-operations per second
  Power: ~120W

PipeZK (ASIC simulation, 28nm, 300 MHz):
  2^20 MSM (BN128): 61 ms  (77.7x over CPU)
  2^20 NTT (256-bit): 11 ms  (24.3x over CPU)

ICICLE GPU library (A100):
  MSM: 8-10x over CPU for 2^20
  NTT: 3-5x over CPU
  Small-field NTT (BabyBear): 50-100x over large-field NTT
```

---

## 11. Projects

### Project 1: Naive MSM vs Pippenger

```
Implement both algorithms in Rust or C using an existing curve library
(arkworks ark-ec, or blst for BLS12-381).

Requirements:
  1. Naive MSM: double-and-add per scalar, then accumulate
  2. Pippenger with configurable window width c
  3. Signed digit optimization
  4. Benchmark for n = 2^10, 2^12, 2^14, 2^16, 2^18, 2^20
  5. Sweep c = 4 to 24 for n = 2^20, plot total operations vs c
  6. Find empirical optimal c and compare with theoretical log₂(n)

Expected: Pippenger ~20x faster than naive at n = 2^20.
Plot should show a clear minimum around c ≈ 15-16.
```

### Project 2: Radix-2 vs Radix-4 NTT

```
Implement both NTT variants over BabyBear field (p = 2013265921).

Requirements:
  1. Radix-2 Cooley-Tukey (DIT) with in-place butterfly
  2. Radix-4 NTT with in-place 4-element butterfly
  3. Count exact multiplications and additions for each
  4. Benchmark for n = 2^10 through 2^24
  5. Verify: radix-4 uses 25% fewer multiplications
  6. Implement both DIT and DIF; measure bit-reversal cost

Stretch: implement radix-8 and verify 42% multiplication saving.
```

### Project 3: NTT Cache Miss Profiler

```
Instrument an NTT implementation to measure cache behavior per stage.

Requirements:
  1. Use perf stat or PAPI counters (L1-dcache-load-misses, LLC-load-misses)
  2. For each stage s = 0,...,log₂(n)-1, run ONLY that stage's butterflies
     and measure cache miss count
  3. Plot cache misses vs stage number for n = 2^16, 2^20, 2^24
  4. Identify the transition point where stride exceeds L2 cache
  5. Implement four-step NTT and show reduced total cache misses

Expected: sharp cache miss increase at stage ≈ log₂(L2_size / elem_size).
Four-step NTT should reduce total misses by 50-80% for large n.
```

### Project 4: Pippenger Parallelism Analyzer

```
Build a simulation tool for Pippenger bucket assignment analysis.

Requirements:
  1. Generate n random 253-bit scalars, decompose into c-bit windows
  2. For each window, compute bucket assignments
  3. Measure bucket collision rate for consecutive points
  4. For configurable pipeline depth d, count stall cycles
  5. Compute pipeline utilization percentage
  6. Generate histogram: number of points per bucket
  7. Vary c and plot: parallelism vs bucket count trade-off

Expected: collision rate < 1% for typical c. Utilization > 99%.
```

### Project 5: Four-Step NTT with Performance Comparison

```
Implement and benchmark monolithic vs four-step NTT.

Requirements:
  1. Monolithic radix-2 NTT for n = 2^20 through 2^26
  2. Four-step NTT with configurable n₁ × n₂ decomposition
  3. Measure wall-clock time, cache miss rates, memory bandwidth
  4. Implement explicit matrix transpose and measure its cost
  5. Find optimal n₁, n₂ for different cache sizes
  6. Plot: time vs n for monolithic and four-step

Expected: four-step wins for n > L2_capacity / element_size (~2^14 for BN254).
```

---

## 12. Resources

### Original Papers

* **Pippenger (1980):** N. Pippenger, "On the evaluation of powers and monomials," SIAM J. Comput., 9(2):230-250. The foundational multi-exponentiation algorithm.
* **Bernstein (2002):** D.J. Bernstein, "Pippenger's exponentiation algorithm." Clearest modern exposition. [cr.yp.to/papers/pippenger.pdf](https://cr.yp.to/papers/pippenger.pdf)
* **Cooley-Tukey (1965):** J.W. Cooley, J.W. Tukey, "An algorithm for the machine calculation of complex Fourier series," Math. Comput., 19(90):297-301. The original FFT paper.
* **GLV (2001):** R. Gallant, R. Lambert, S. Vanstone, "Faster Point Multiplication on Elliptic Curves with Efficient Endomorphisms," CRYPTO 2001.

### Modern MSM Papers

* **PipeMSM (2022):** "PipeMSM: Hardware Acceleration for Multi-Scalar Multiplication." [eprint.iacr.org/2022/999](https://eprint.iacr.org/2022/999)
* **cuZK (2022):** "cuZK: Accelerating Zero-Knowledge Proof with A Faster Parallel Multi-Scalar Multiplication Algorithm on GPUs." [eprint.iacr.org/2022/1321](https://eprint.iacr.org/2022/1321)
* **CycloneMSM (2022):** "FPGA Acceleration of Multi-Scalar Multiplication: CycloneMSM." [eprint.iacr.org/2022/1396](https://eprint.iacr.org/2022/1396)
* **EdMSM (2022):** "EdMSM: Multi-Scalar-Multiplication for SNARKs and Faster Montgomery Multiplication." [eprint.iacr.org/2022/1400](https://eprint.iacr.org/2022/1400)
* **OPTIMSM (2024):** "OPTIMSM: FPGA hardware accelerator for Zero-Knowledge MSM." [eprint.iacr.org/2024/1827](https://eprint.iacr.org/2024/1827)

### Hardware-Focused NTT Papers

* **PipeZK (ISCA 2021):** "PipeZK: Accelerating Zero-Knowledge Proof with a Pipelined Architecture." [people.iiis.tsinghua.edu.cn/~gaomy/pubs/pipezk.isca21.pdf](https://people.iiis.tsinghua.edu.cn/~gaomy/pubs/pipezk.isca21.pdf)
* **CFNTT (TCHES 2022):** "CFNTT: Scalable Radix-2/4 NTT Multiplication Architecture with Conflict-free Memory Mapping." [tches.iacr.org/index.php/TCHES/article/view/9291](https://tches.iacr.org/index.php/TCHES/article/view/9291)

### Benchmarks and Implementations

* **ICICLE (Ingonyama):** GPU acceleration library for MSM, NTT, Poseidon. [github.com/ingonyama-zk/icicle](https://github.com/ingonyama-zk/icicle)
* **ZPrize Competition:** Annual ZK hardware acceleration competition. [zprize.io](https://www.zprize.io/)
* **Ingonyama Hardware Review:** GPU vs FPGA vs ASIC comparison. [ingonyama.com/post/hardware-review-gpus-fpgas-and-zero-knowledge-proofs](https://www.ingonyama.com/post/hardware-review-gpus-fpgas-and-zero-knowledge-proofs)
* **Jane Street FPGA Blog:** MSM and NTT on FPGAs with Hardcaml. [blog.janestreet.com/zero-knowledge-fpgas-hardcaml/](https://blog.janestreet.com/zero-knowledge-fpgas-hardcaml/)
* **arkmsm implementation notes:** Detailed Pippenger optimization walkthrough. [hackmd.io/@drouyang/msm](https://hackmd.io/@drouyang/msm)
* **Explicit Formulas Database:** Elliptic curve operation costs in all coordinate systems. [hyperelliptic.org/EFD/](https://www.hyperelliptic.org/EFD/)

### Deep Dive Resources

* **Ingonyama ZK Hardware Table Stakes (MSM):** [ingonyama.com/post/zk-hardware-table-stakes-part-1-msm](https://www.ingonyama.com/post/zk-hardware-table-stakes-part-1-msm)
* **NTT Deep Dive (Math & Engineering):** Comprehensive NTT algorithms including Good-Thomas, Bluestein. [xn--2-umb.com/23/ntt/](https://xn--2-umb.com/23/ntt/)
* **Paradigm — Hardware Acceleration for ZK Proofs:** [paradigm.xyz/2022/04/zk-hardware](https://www.paradigm.xyz/2022/04/zk-hardware)
* **Small Fields for ZK (ICME):** BabyBear, Goldilocks, Mersenne31 comparison. [blog.icme.io/small-fields-for-zero-knowledge/](https://blog.icme.io/small-fields-for-zero-knowledge/)
