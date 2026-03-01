# Polynomial Arithmetic and Commitment Schemes

> **Goal:** Master polynomials as the universal language of ZK proof systems — from basic evaluation and interpolation through the Number Theoretic Transform (NTT), to the three major commitment schemes (KZG, IPA, FRI) that bind provers to their computations. By the end, you will understand why NTT is the #2 hardware bottleneck (after MSM), how each commitment scheme creates different hardware requirements, and why the choice of commitment scheme determines whether your system needs pairings, large proofs, or hash-heavy computation.

**Prerequisite:** Phase 5.1 (Mathematical Foundations) — finite fields, modular arithmetic, polynomial rings, roots of unity. Phase 5.2 (Elliptic Curve Cryptography) — curve groups, scalar multiplication, bilinear pairings, MSM. You must be comfortable with F_p arithmetic and the concept of polynomial evaluation.

---

## Table of Contents

1. [Polynomials as the Language of ZK](#1-polynomials-as-the-language-of-zk)
2. [Polynomial Evaluation and Interpolation](#2-polynomial-evaluation-and-interpolation)
3. [The Number Theoretic Transform (NTT)](#3-the-number-theoretic-transform-ntt)
4. [Inverse NTT and Domain Conversion](#4-inverse-ntt-and-domain-conversion)
5. [NTT in Hardware — The #2 Bottleneck](#5-ntt-in-hardware--the-2-bottleneck)
6. [Polynomial Commitment Schemes — Overview](#6-polynomial-commitment-schemes--overview)
7. [KZG Commitments — The Pairing-Based Approach](#7-kzg-commitments--the-pairing-based-approach)
8. [IPA — The Inner Product Argument](#8-ipa--the-inner-product-argument)
9. [FRI — Fast Reed-Solomon Interactive Oracle Proof](#9-fri--fast-reed-solomon-interactive-oracle-proof)
10. [Comparing Commitment Schemes — Hardware Implications](#10-comparing-commitment-schemes--hardware-implications)
11. [Projects](#11-projects)
12. [Resources](#12-resources)

---

## 1. Polynomials as the Language of ZK

### Why Polynomials?

Every modern ZK proof system reduces the statement "I know a valid computation" to the statement "I know a polynomial that satisfies certain constraints." This is not a design accident — polynomials have unique algebraic properties that make them the perfect encoding:

```
The fundamental insight:
───────────────────────
  A polynomial of degree d is completely determined by d+1 points.
  Two distinct polynomials of degree d can agree on at most d points.

  If you check equality at a random point z ∈ F_p:
    Pr[f(z) = g(z) | f ≠ g] ≤ d/p

  For d = 2^20 and p ≈ 2^254:
    Pr[false equality] ≤ 2^20 / 2^254 = 2^{-234}   ← negligible!
```

This is the **Schwartz-Zippel lemma** — the reason ZK works. Instead of checking that two polynomials are equal at all points (impossible — infinitely many), the verifier checks at ONE random point and gets overwhelming statistical confidence.

### How Computation Becomes Polynomials

```
ZK pipeline (every proof system):
──────────────────────────────────
  Computation (program/circuit)
        ↓
  Constraint system (R1CS, AIR, PLONKish gates)
        ↓
  Polynomial equations over F_p
        ↓
  Prover commits to polynomials        ← THIS is what this chapter covers
        ↓
  Verifier checks polynomial identities at random points
        ↓
  Accept / Reject
```

The constraint system (covered in Phase 5.4) determines HOW computation maps to polynomials. The commitment scheme (this chapter) determines HOW the prover convinces the verifier that the polynomials are correct — without revealing them.

### Polynomial Representations

A polynomial of degree d over F_p can be stored in two equivalent forms:

```
Coefficient form:  f(x) = c₀ + c₁x + c₂x² + ... + c_d x^d
                   → store [c₀, c₁, c₂, ..., c_d]   (d+1 field elements)

Evaluation form:   f(ω⁰), f(ω¹), f(ω²), ..., f(ω^{n-1})
                   → store [f(ω⁰), f(ω¹), ..., f(ω^{n-1})]   (n evaluations)
                   where ω is a primitive n-th root of unity, n ≥ d+1
```

```
Operation costs in each representation:
──────────────────────────────────────────────
                    Coefficient      Evaluation
  Addition          O(n)             O(n)        ← same
  Multiplication    O(n²) naive      O(n)        ← evaluation wins!
  Evaluation at z   O(n) Horner      O(n) barycentric
  Commitment (KZG)  O(n) via MSM     must convert to coefficient first
  Commitment (FRI)  must evaluate    O(n) via Merkle tree

Convert between:    NTT: coeff → eval in O(n log n)
                    INTT: eval → coeff in O(n log n)
```

**The NTT is the bridge between representations.** ZK provers constantly switch between coefficient and evaluation forms, performing whichever operations are cheaper in each domain.

---

## 2. Polynomial Evaluation and Interpolation

### Horner's Method (Efficient Evaluation)

To evaluate f(x) = c₀ + c₁x + c₂x² + c₃x³ at a point z:

**Naive:** compute z², z³, multiply each, add — requires d multiplications + d additions + storage for powers of z.

**Horner's method:** rewrite as nested multiplication:

```
f(z) = c₀ + z·(c₁ + z·(c₂ + z·c₃))

Step-by-step for f(x) = 3 + 2x + 5x² + 7x³  at z = 4  over F₁₃:

  Start with c₃ = 7
  Multiply by z:    7 × 4 = 28 ≡ 2 (mod 13)
  Add c₂:           2 + 5 = 7
  Multiply by z:    7 × 4 = 28 ≡ 2 (mod 13)
  Add c₁:           2 + 2 = 4
  Multiply by z:    4 × 4 = 16 ≡ 3 (mod 13)
  Add c₀:           3 + 3 = 6

  Result: f(4) = 6 in F₁₃
```

**Cost: d multiplications + d additions.** Optimal — you cannot do better without preprocessing. In hardware, this is a simple sequential pipeline: one multiplier and one adder, d clock cycles.

### Lagrange Interpolation

Given n points (x₀, y₀), (x₁, y₁), ..., (x_{n-1}, y_{n-1}), find the unique polynomial of degree < n that passes through all of them:

```
                  n-1
f(x) = Σ   y_i · L_i(x)
                  i=0

where the Lagrange basis polynomials are:

              n-1    (x - x_j)
L_i(x) = ∏   ─────────────
              j=0    (x_i - x_j)
              j≠i
```

**Example over F₁₃:**

```
Given points: (1, 3), (2, 5), (4, 11)  — find f(x) of degree ≤ 2

L₀(x) = (x-2)(x-4) / ((1-2)(1-4)) = (x-2)(x-4) / ((-1)(-3))
       = (x-2)(x-4) / 3
       = (x-2)(x-4) · 3⁻¹
       In F₁₃: 3⁻¹ = 9  (since 3×9 = 27 ≡ 1 mod 13)
       L₀(x) = 9·(x-2)(x-4)

L₁(x) = (x-1)(x-4) / ((2-1)(2-4)) = (x-1)(x-4) / (1·(-2))
       In F₁₃: (-2)⁻¹ = 11⁻¹ = 6  (since 11×6 = 66 ≡ 1 mod 13)
       L₁(x) = 6·(x-1)(x-4)

L₂(x) = (x-1)(x-2) / ((4-1)(4-2)) = (x-1)(x-2) / (3·2)
       In F₁₃: 6⁻¹ = 11  (since 6×11 = 66 ≡ 1 mod 13)
       L₂(x) = 11·(x-1)(x-2)

f(x) = 3·L₀(x) + 5·L₁(x) + 11·L₂(x)

Verify: f(1) = 3·1 + 5·0 + 11·0 = 3 ✓
        f(2) = 3·0 + 5·1 + 11·0 = 5 ✓
        f(4) = 3·0 + 5·0 + 11·1 = 11 ✓
```

**Naive cost: O(n²) field operations.** For ZK-scale polynomials (n = 2^20 to 2^26), this is too slow. The NTT solves this.

### Vanishing Polynomials

A crucial construct in ZK: the polynomial that vanishes (equals zero) on the entire evaluation domain.

```
For domain H = {ω⁰, ω¹, ω², ..., ω^{n-1}} where ω is a primitive n-th root of unity:

Z_H(x) = (x - ω⁰)(x - ω¹)···(x - ω^{n-1}) = x^n - 1

This is because ω^n = 1, so every ω^i is a root of x^n - 1.
```

**Why this matters:** In every proof system, the prover must show that constraint polynomials equal zero on the evaluation domain H. This means showing that Z_H(x) divides the constraint polynomial — the prover computes a **quotient polynomial** q(x) = f(x) / Z_H(x) and commits to it. The verifier checks f(z) = q(z) · Z_H(z) at a random point z.

```
The "divide-and-check" pattern:
────────────────────────────────
  Prover's claim: "f(x) = 0 for all x ∈ H"

  Equivalent to: Z_H(x) | f(x)   (Z_H divides f)

  Prover computes: q(x) = f(x) / Z_H(x)
  Prover commits to: q(x)

  Verifier checks at random z:
    f(z) ?= q(z) · Z_H(z)
    where Z_H(z) = z^n - 1   ← O(log n) to compute!

  If f is not zero on H, then q doesn't exist as a polynomial,
  and with overwhelming probability f(z) ≠ q(z)·Z_H(z).
```

This pattern is used in Groth16, PLONK, STARKs, and every other modern proof system.

---

## 3. The Number Theoretic Transform (NTT)

### What Is NTT?

The NTT is the **finite-field analog of the FFT** (Fast Fourier Transform). Where FFT uses complex roots of unity (e^{2πi/n}), NTT uses roots of unity in a finite field (ω where ω^n ≡ 1 mod p).

```
NTT converts between polynomial representations:
─────────────────────────────────────────────────
  Coefficient form: [c₀, c₁, c₂, ..., c_{n-1}]
                              ↓ NTT
  Evaluation form:  [f(ω⁰), f(ω¹), f(ω²), ..., f(ω^{n-1})]

  This IS polynomial evaluation at all n-th roots of unity simultaneously.
  Naive: O(n²).  NTT: O(n log n).
```

### Why NTT Is Critical for ZK

```
Where NTT appears in ZK proof generation:
──────────────────────────────────────────
  1. Polynomial multiplication:    coeff → eval (NTT), pointwise multiply, eval → coeff (INTT)
  2. Quotient polynomial:          divide f(x) by Z_H(x) in evaluation domain
  3. PLONK's permutation argument: evaluate gate polynomials at all n roots
  4. Coset evaluation:             evaluate on a shifted domain for DEEP-FRI
  5. STARK constraints:            evaluate AIR transition constraints at all steps

  In typical proof generation:
    Groth16:  4-8 NTTs of size n
    PLONK:   10-15 NTTs of size n or 2n
    STARKs:  20-40 NTTs (more NTTs, but over smaller fields)
```

### The Butterfly Operation

The NTT works by recursively splitting the polynomial into even-indexed and odd-indexed coefficients:

```
f(x) = f_even(x²) + x · f_odd(x²)

where:
  f_even(x) = c₀ + c₂x + c₄x² + ...
  f_odd(x)  = c₁ + c₃x + c₅x² + ...
```

Each recursive step performs the **butterfly operation**:

```
Radix-2 Butterfly (Cooley-Tukey):
──────────────────────────────────
  Input:  a, b  (two field elements)
  Twiddle factor: ω^k  (a power of the root of unity)

     a ───────────(+)──── a + ω^k · b  = a'
              ╲   ╱
               ╲ ╱
                ╳
               ╱ ╲
              ╱   ╲
     b ──[×ω^k]──(-)──── a - ω^k · b  = b'

  Cost per butterfly: 1 multiplication + 2 additions
```

### NTT Algorithm (Radix-2 Decimation-in-Frequency)

For n = 8 (the smallest non-trivial example), with ω = primitive 8th root of unity:

```
Stage 0 (input)        Stage 1              Stage 2              Stage 3 (output)

c₀ ─────────(+)───────────(+)───────────(+)─── f(ω⁰)
             │              │              │
c₁ ─────────│──(+)─────────│──(+)─────(-)─── f(ω⁴)
             │   │          │   │
c₂ ─────(+)─│───│──────(+)─│───│─────────── f(ω²)
          │  │   │       │  │   │
c₃ ─────(+)─│───│──(+)──│──│───│──(-)───── f(ω⁶)
          │  │   │   │   │  │   │
c₄ ──(+)─│──│───│───│───│──(-)─│────────── f(ω¹)
       │  │  │   │   │      │   │
c₅ ──(+)─│──│───│───│──(-)──│───│──(-)──── f(ω⁵)
       │  │  │   │              │   │
c₆ ──(+)─│──(-)─│──────────(-)─│────────── f(ω³)
       │  │      │               │
c₇ ──(+)──(-)───│──(-)─────(-)──────────── f(ω⁷)

Each (×) includes multiplication by the appropriate twiddle factor ω^k.

Total operations for n = 8:
  Stages: log₂(8) = 3
  Butterflies per stage: n/2 = 4
  Total butterflies: 3 × 4 = 12
  Total multiplications: 12  (some are by ω⁰ = 1, which can be skipped)
  Total additions: 24
```

### Operation Count

For a general size-n NTT where n = 2^k:

```
NTT operation count:
────────────────────
  Stages:                 log₂(n)
  Butterflies per stage:  n/2
  Total butterflies:      (n/2) · log₂(n)

  Multiplications:  (n/2) · log₂(n)    (field multiplications by twiddle factors)
  Additions:        n · log₂(n)         (2 additions per butterfly)

  Practical counts for ZK-scale sizes:
  ┌───────────┬─────────────────┬──────────────────┬───────────────────┐
  │ n         │ Multiplications │ Additions        │ Compare naive O(n²) │
  ├───────────┼─────────────────┼──────────────────┼───────────────────┤
  │ 2^16      │    524,288      │  1,048,576       │ 4,294,967,296     │
  │ 2^20      │  10,485,760     │ 20,971,520       │ ~10^12            │
  │ 2^24      │ 201,326,592     │ 402,653,184      │ ~2.8 × 10^14     │
  │ 2^26      │ 872,415,232     │ 1,744,830,464    │ ~4.5 × 10^15     │
  └───────────┴─────────────────┴──────────────────┴───────────────────┘

  For n = 2^24 on BLS12-381 (381-bit field):
    ~200M field multiplications + ~400M field additions
    At 30ns per multiplication: ~6 seconds per NTT
    PLONK needs ~12 NTTs: ~72 seconds just for NTTs!
    This is why NTT hardware acceleration matters.
```

### Twiddle Factors

The twiddle factors are precomputed powers of the root of unity:

```
For size-n NTT over F_p:
  ω = generator of the n-element subgroup of F_p*
  ω^n = 1 (mod p)

  Stage s, butterfly j uses twiddle factor: ω^{j · 2^{k-1-s}}

  Total unique twiddle factors: n - 1
  Storage: n - 1 field elements

  For BN254 (254-bit field), n = 2^24:
    Twiddle table size: 2^24 × 32 bytes = 512 MB

  For BabyBear (31-bit field), n = 2^24:
    Twiddle table size: 2^24 × 4 bytes = 64 MB
```

The twiddle table size is a significant memory constraint for hardware implementations. Phase 6 covers techniques like on-the-fly twiddle generation and mixed-radix NTT to reduce memory requirements.

### Requirements for NTT to Work

```
For NTT of size n over F_p, you need:
──────────────────────────────────────
  1. n must be a power of 2 (for radix-2; or a smooth number for mixed-radix)
  2. n must divide p - 1  (so that n-th roots of unity exist in F_p)
  3. You must know a primitive n-th root of unity ω

  This is why STARK-friendly primes are chosen carefully:
  ┌─────────────┬────────────────────────────┬───────────────┐
  │ Field       │ p - 1 factorization        │ Max NTT size  │
  ├─────────────┼────────────────────────────┼───────────────┤
  │ BabyBear    │ 2^27 × 15                  │ 2^27          │
  │ Goldilocks  │ 2^32 × (2^32 - 1)          │ 2^32          │
  │ Mersenne31  │ 2 × 3 × ... (no large 2^k) │ limited!      │
  │ BN254 scalar│ 2^28 × ...                  │ 2^28          │
  │ BLS12-381   │ 2^32 × ...                  │ 2^32          │
  └─────────────┴────────────────────────────┴───────────────┘

  Mersenne31 (p = 2^31 - 1) has p - 1 = 2 × 3 × ..., with only one factor
  of 2, so NTT cannot be done natively. Plonky3 uses a circle-group technique
  (circle STARKs) or extension fields to work around this.
```

---

## 4. Inverse NTT and Domain Conversion

### INTT (Inverse NTT)

The INTT converts from evaluation form back to coefficient form:

```
INTT: [f(ω⁰), f(ω¹), ..., f(ω^{n-1})] → [c₀, c₁, ..., c_{n-1}]

Algorithm: Run NTT with ω⁻¹ instead of ω, then multiply each output by n⁻¹.

  INTT(v) = (1/n) · NTT_{ω⁻¹}(v)

Cost: Same as NTT — (n/2)·log₂(n) multiplications + n·log₂(n) additions
      plus n multiplications by n⁻¹.
```

### Polynomial Multiplication via NTT

This is the primary use case: multiply two polynomials of degree < n/2 in O(n log n):

```
To compute h(x) = f(x) · g(x):
────────────────────────────────
  1. NTT: [f coefficients] → [f evaluations]     O(n log n)
  2. NTT: [g coefficients] → [g evaluations]     O(n log n)
  3. Pointwise: h_eval[i] = f_eval[i] × g_eval[i]   O(n)
  4. INTT: [h evaluations] → [h coefficients]     O(n log n)

  Total: 3 NTTs + n multiplications = O(n log n)
  Compare: schoolbook multiplication = O(n²)

  For n = 2^20:
    NTT-based:  ~31.5M multiplications (3 × 10.5M)
    Schoolbook: ~10^12 multiplications
    Speedup:    ~30,000×
```

### Coset Evaluation

Sometimes you need to evaluate a polynomial on a **coset** of the root-of-unity domain (a shifted copy of the domain):

```
Standard domain:  H = {1, ω, ω², ..., ω^{n-1}}
Coset:           kH = {k, kω, kω², ..., kω^{n-1}}    where k is a shift factor

To evaluate on a coset:
  1. Multiply coefficient i by k^i:  c'_i = c_i · k^i
  2. Perform standard NTT on [c'₀, c'₁, ..., c'_{n-1}]

Cost: n extra multiplications + 1 NTT = O(n log n)
```

Coset evaluation is critical for:
- **PLONK:** evaluating quotient polynomials on cosets to avoid division by zero
- **FRI:** the "fold" operation evaluates on progressively smaller cosets
- **STARKs:** the DEEP technique evaluates on extended domains

### The Prover's NTT Workflow

A typical PLONK prover performs:

```
PLONK prover NTT operations (simplified):
──────────────────────────────────────────
  1. INTT on witness polynomials (convert evaluations → coefficients)     × 3
  2. NTT on quotient numerator (evaluate constraint polynomial)          × 1
  3. Division in evaluation domain                                        -
  4. INTT on quotient polynomial (back to coefficients for KZG)           × 1
  5. NTT on opening evaluation polynomial                                × 1
  6. Coset NTTs for degree checks                                        × 4-6

  Total: ~10-15 NTTs of size n or 2n

  For n = 2^20 (a moderate circuit):
    Each NTT: ~10.5M multiplications × 30ns = ~315ms on CPU
    15 NTTs: ~4.7 seconds just in NTTs
    Plus MSM: ~seconds more
    This is why hardware acceleration targets both NTT and MSM.
```

---

## 5. NTT in Hardware — The #2 Bottleneck

### Why NTT Is a Hardware Challenge

```
NTT computational profile:
──────────────────────────
  Arithmetic:  Regular, predictable, embarrassingly structured
               (n/2)·log₂(n) butterflies, each: 1 MUL + 2 ADD

  Memory:      IRREGULAR — this is the real problem
               Butterfly at stage s connects elements distance 2^s apart
               Stage 0: neighbors (stride 1)
               Stage 10: elements 1024 apart
               Stage 20: elements 1,048,576 apart!

  Cache behavior:
    Early stages:  local access → cache-friendly
    Late stages:   global access → cache thrashing, memory-bandwidth bound

  This is the opposite of MSM:
    MSM:  irregular computation, regular memory (sequential point access)
    NTT:  regular computation, irregular memory (butterfly stride patterns)
```

### Memory Access Patterns

```
NTT butterfly stride pattern for n = 16:
─────────────────────────────────────────
  Stage 0: stride = 1    [0,1] [2,3] [4,5] [6,7] [8,9] [10,11] [12,13] [14,15]
  Stage 1: stride = 2    [0,2] [1,3] [4,6] [5,7] [8,10] [9,11] [12,14] [13,15]
  Stage 2: stride = 4    [0,4] [1,5] [2,6] [3,7] [8,12] [9,13] [10,14] [11,15]
  Stage 3: stride = 8    [0,8] [1,9] [2,10] [3,11] [4,12] [5,13] [6,14] [7,15]

  At stage s, butterfly connects elements at distance 2^s.
  For n = 2^24, stage 23 connects element 0 with element 8,388,608.
  If each element is 32 bytes (BN254): stride = 256 MB in memory!
```

### Hardware Approaches (Preview of Phase 6)

```
Three strategies for NTT acceleration:
───────────────────────────────────────
  1. GPU:  Massive parallelism, but memory bandwidth limited
           - All n/2 butterflies per stage execute in parallel
           - Global memory access for large strides → bottleneck
           - Best for large NTTs (n ≥ 2^20)

  2. FPGA: Custom memory architecture
           - Multi-bank SRAM eliminates stride conflicts
           - Pipeline butterflies with on-chip twiddle factors
           - Limited by BRAM/URAM capacity (~10-50 MB on-chip)
           - Best when polynomial fits in on-chip memory

  3. ASIC (ZPU): Dedicated NTT engine
           - Hardwired butterfly units with optimized routing
           - Custom memory controllers for stride patterns
           - Highest throughput, highest NRE cost

  Performance targets (Phase 6 will detail):
  ┌────────────┬──────────────┬────────────────┐
  │ Platform   │ NTT 2^24     │ Speedup vs CPU │
  ├────────────┼──────────────┼────────────────┤
  │ CPU (AVX)  │ ~3-6 sec     │ 1×             │
  │ GPU (A100) │ ~50-200 ms   │ 15-60×         │
  │ FPGA (U250)│ ~30-100 ms   │ 30-100×        │
  │ ASIC       │ ~5-20 ms     │ 150-600×       │
  └────────────┴──────────────┴────────────────┘
```

### Mixed-Radix NTT

Real implementations rarely use pure radix-2. Mixed-radix approaches reduce the number of stages:

```
Radix-2: log₂(n) stages, each with n/2 butterflies
Radix-4: (1/2)·log₂(n) stages, each with n/4 radix-4 butterflies
          Each radix-4 butterfly: 3 MUL + 8 ADD (vs 4 MUL + 8 ADD for two radix-2)
          Saves 25% multiplications!

Radix-8, Radix-16: Further reduction in multiplications, at the cost of more
                    complex butterfly circuits.

In hardware (FPGA/ASIC): Radix-4 or radix-8 is the sweet spot.
  - Fewer pipeline stages
  - Better utilization of multiplier resources
  - More complex control logic, but well worth it
```

---

## 6. Polynomial Commitment Schemes — Overview

### What Is a Polynomial Commitment?

A polynomial commitment scheme lets a prover convince a verifier about polynomial evaluations without revealing the polynomial:

```
Polynomial Commitment Scheme API:
─────────────────────────────────
  Setup(d)      → params            Generate system parameters for degree ≤ d
  Commit(f)     → C                 Commit to polynomial f, producing a short commitment C
  Open(f, z)    → (v, π)           Prove that f(z) = v, producing proof π
  Verify(C, z, v, π) → accept/reject   Check the proof

  Security properties:
    Binding:     Prover cannot open the same commitment to two different values
    Hiding:      Commitment reveals nothing about f (optional, depends on scheme)
    Succinctness: Commitment C and proof π are small (constant or polylog in degree)
```

### Why This Is the Heart of ZK

```
Without polynomial commitments, we have no ZK proofs:
──────────────────────────────────────────────────────
  "I know a satisfying assignment to this circuit"
      ↓ (arithmetize)
  "I know polynomials f₁,...,f_k satisfying constraint polynomial C(f₁,...,f_k) = 0 on H"
      ↓ (commit)
  "Here are commitments C₁,...,C_k.  At random challenge z, I can prove f_i(z) = v_i"
      ↓ (open)
  "Here are opening proofs π₁,...,π_k"
      ↓ (verify)
  Verifier checks: C(v₁,...,v_k) = 0 at z, and each π_i is valid

  The commitment scheme determines:
    - Trusted setup or not?
    - Proof size (affects on-chain cost)
    - Verification time (affects L1 gas cost)
    - Prover time (affects latency and hardware requirements)
```

### The Three Major Schemes

```
                    KZG              IPA              FRI
                    ─────────        ─────────        ─────────
Used by:            Groth16          Halo 2           STARKs
                    PLONK (vanilla)  Bulletproofs     PLONK+FRI
                    EigenDA          Zcash (Orchard)  STARK provers

Setup:              Trusted setup    Transparent      Transparent
Commitment:         1 G₁ point       1 G point        Merkle root
                    (48 bytes BLS)   (32-48 bytes)    (32 bytes)
Proof size:         1 G₁ point       O(log n) G pts   O(log² n) hashes
                    (48 bytes)       (32·log₂(n) B)   (polylog kB)
Verify time:        2 pairings       O(n)             O(log² n) hashes
                    (~1-2 ms)        (too slow alone!) (~1-5 ms)
Prover bottleneck:  MSM (O(n))       MSM (O(n))       NTT + hashing
```

---

## 7. KZG Commitments — The Pairing-Based Approach

### Structured Reference String (SRS)

KZG requires a one-time **trusted setup** that generates a structured reference string:

```
Trusted Setup Ceremony:
───────────────────────
  1. Choose a secret random τ ∈ F_p   (the "toxic waste")
  2. Compute SRS in G₁: [G, τG, τ²G, τ³G, ..., τ^d G]
  3. Compute SRS in G₂: [H, τH]  (only need degree 1 in G₂)
  4. DESTROY τ!  If anyone knows τ, they can forge proofs.

  SRS size for degree d:
    G₁ points: d + 1   (for BLS12-381: 48 bytes each)
    G₂ points: 2       (for BLS12-381: 96 bytes each)

  Examples:
    Ethereum's KZG ceremony (2023): d = 2^12 = 4096
      SRS: 4097 G₁ points + 2 G₂ points ≈ 197 KB
    Large circuits (d = 2^24):
      SRS: 16,777,217 G₁ points ≈ 768 MB
```

### Multi-Party Computation (MPC) for Setup

The toxic waste problem is solved by distributing the ceremony:

```
MPC Ceremony (Powers of Tau):
─────────────────────────────
  Participant 1: chooses τ₁, computes [τ₁^k · G] for k = 0,...,d
  Participant 2: receives [τ₁^k · G], chooses τ₂, computes [(τ₁τ₂)^k · G]
  ...
  Participant m: receives [s^k · G], chooses τ_m, computes [(s·τ_m)^k · G]

  The effective secret is τ = τ₁ · τ₂ · ... · τ_m.
  Security: as long as ANY ONE participant deletes their τ_i,
            nobody knows τ.

  Ethereum's ceremony had 141,416 participants — breaking it
  would require ALL of them to collude.
```

### KZG Commitment

To commit to polynomial f(x) = c₀ + c₁x + ... + c_d x^d:

```
Commit(f) = c₀·G + c₁·(τG) + c₂·(τ²G) + ... + c_d·(τ^d G)
          = f(τ)·G

This is a MULTI-SCALAR MULTIPLICATION (MSM)!
────────────────────────────────────────────
  Scalars:      [c₀, c₁, c₂, ..., c_d]     (polynomial coefficients)
  Base points:  [G, τG, τ²G, ..., τ^d G]    (from SRS)
  Result:       one elliptic curve point      (the commitment)

  The prover never knows τ — they only know [τ^k · G] from the SRS.
  But the commitment equals f(τ)·G as if they had computed f(τ) directly.

  Cost: MSM of size d+1
    For d = 2^20: ~2^20 scalar multiplications → Pippenger: ~seconds on CPU
    This is THE bottleneck in KZG-based proof systems.
```

### KZG Opening Proof

To prove that f(z) = v:

```
The key identity:
─────────────────
  If f(z) = v, then (x - z) divides f(x) - v.

  Quotient polynomial: q(x) = (f(x) - v) / (x - z)

  This is a polynomial of degree d - 1 (one less than f).

  Opening proof: π = q(τ)·G = Commit(q)
                 → another MSM of size d

Step-by-step:
  1. Compute quotient q(x) = (f(x) - v) / (x - z)    via polynomial long division
  2. Compute π = Commit(q) using the SRS                via MSM
  3. Send (v, π) to verifier
```

### KZG Verification

The verifier checks using a **bilinear pairing**:

```
Verification equation:
──────────────────────
  Check: e(C - v·G, H) = e(π, τH - z·H)

  Expanding:
    Left:  e(f(τ)·G - v·G, H) = e((f(τ) - v)·G, H)
    Right: e(q(τ)·G, (τ - z)·H)

  By bilinearity:
    Left  = (f(τ) - v) · e(G, H)
    Right = q(τ)(τ - z) · e(G, H)

  So the check becomes:
    f(τ) - v ?= q(τ) · (τ - z)

  Which is true iff q(x) = (f(x) - v)/(x - z), i.e., f(z) = v.

  Verification cost: 2 pairings + 2 G₁ scalar multiplications + 1 G₁ addition
  This is CONSTANT TIME — independent of polynomial degree!
```

### Batch Opening

KZG supports efficient batch openings — prove multiple evaluations with fewer pairings:

```
Batch opening at the same point z:
──────────────────────────────────
  Polynomials: f₁, f₂, ..., f_m
  Claims: f_i(z) = v_i for all i

  Verifier sends random γ (after seeing commitments).
  Aggregate: F(x) = f₁(x) + γ·f₂(x) + γ²·f₃(x) + ...
  Aggregate claim: V = v₁ + γ·v₂ + γ²·v₃ + ...

  Single opening proof for F(z) = V.

  Cost: 1 pairing verification instead of m.
  Used in PLONK to batch-open all wire polynomials at once.

Batch opening at different points z₁, z₂, ...:
  More complex (uses the KZG multi-open technique).
  Still sublinear in the number of openings.
```

---

## 8. IPA — The Inner Product Argument

### Motivation

IPA (Inner Product Argument) achieves polynomial commitments **without** a trusted setup and **without** pairings. The tradeoff: verification is O(n) instead of O(1), and proofs are O(log n) instead of O(1).

```
IPA vs KZG — design philosophy:
────────────────────────────────
  KZG:  "Give me a trusted setup and pairings, and I'll give you
         constant-size proofs with constant-time verification."

  IPA:  "No trusted setup needed. Any elliptic curve works.
         Proofs are O(log n), verification is O(n) — but I can make
         verification amortizable using recursion."
```

### Pedersen Vector Commitment

IPA starts with a **Pedersen vector commitment** — committing to a vector instead of a single value:

```
Setup: Choose n random, independent group generators G₁, G₂, ..., G_n ∈ G
       plus a blinding generator H ∈ G.
       (No trusted setup — these can be hash-derived.)

Commit(v⃗, r):
  v⃗ = (v₁, v₂, ..., v_n)  ← the vector to commit
  r ← random blinding factor

  C = v₁·G₁ + v₂·G₂ + ... + v_n·G_n + r·H
    = ⟨v⃗, G⃗⟩ + r·H

  This is an MSM of size n + 1.

Properties:
  - Computationally binding (under DLP)
  - Perfectly hiding (r randomizes the commitment)
  - Homomorphic: Commit(v⃗) + Commit(w⃗) = Commit(v⃗ + w⃗)
```

### The Inner Product Argument Protocol

The prover wants to convince the verifier that ⟨a⃗, b⃗⟩ = c where a⃗ is the committed vector and b⃗ is public. The key insight is a **recursive halving** protocol:

```
IPA Protocol (Bulletproofs-style):
──────────────────────────────────
  Goal: Prove ⟨a⃗, b⃗⟩ = c  where |a⃗| = |b⃗| = n

  Round 1 (n → n/2):
    Split: a⃗ = (a⃗_L, a⃗_R),  b⃗ = (b⃗_L, b⃗_R),  G⃗ = (G⃗_L, G⃗_R)

    Compute cross-terms:
      L₁ = ⟨a⃗_L, G⃗_R⟩ + ⟨a⃗_L, b⃗_R⟩·Q    (left "proof" element)
      R₁ = ⟨a⃗_R, G⃗_L⟩ + ⟨a⃗_R, b⃗_L⟩·Q    (right "proof" element)

    Send L₁, R₁ to verifier.
    Verifier sends random challenge u₁.

    Fold:
      a⃗' = a⃗_L + u₁·a⃗_R       (n/2 elements)
      b⃗' = b⃗_L + u₁⁻¹·b⃗_R     (n/2 elements)
      G⃗' = G⃗_L + u₁⁻¹·G⃗_R     (n/2 generators)

  Round 2 (n/2 → n/4):
    Same procedure with a⃗', b⃗', G⃗'

  ...

  Round log₂(n) (2 → 1):
    Final scalar a', b', generator G'
    Verifier checks: a'·G' + a'·b'·Q = folded commitment

  Proof: {L₁, R₁, L₂, R₂, ..., L_{log n}, R_{log n}, a'}
  Size: 2·log₂(n) group elements + 1 scalar
```

### From Inner Products to Polynomial Commitments

A polynomial evaluation f(z) = v can be expressed as an inner product:

```
f(x) = c₀ + c₁x + c₂x² + ... + c_{n-1}x^{n-1}

f(z) = ⟨(c₀, c₁, ..., c_{n-1}), (1, z, z², ..., z^{n-1})⟩

So:
  a⃗ = (c₀, c₁, ..., c_{n-1})   ← polynomial coefficients (committed)
  b⃗ = (1, z, z², ..., z^{n-1})  ← powers of evaluation point (public)
  c = f(z) = v                    ← claimed evaluation

This reduces polynomial commitment opening to an inner product argument!
```

### IPA Costs

```
IPA cost analysis:
──────────────────
  Prover:
    Each round: 2 MSMs of size n/2, then n/4, then n/8, ...
    Total: 2 × (n/2 + n/4 + ... + 1) = 2(n - 1) scalar multiplications
    ≈ 2n scalar multiplications total
    Plus: O(n) field operations for vector folding

  Proof size:
    2·log₂(n) group elements + 1 scalar
    For n = 2^20: 2 × 20 = 40 group elements + 1 scalar
    With BLS12-381 (48-byte points): 40 × 48 + 32 = 1,952 bytes

  Verifier:
    Must reconstruct folded generators: O(n) group operations
    This is the problem — verification is LINEAR in n!

  ┌────────────┬──────────────────┬──────────────┬───────────────┐
  │            │ Prover cost      │ Proof size   │ Verify cost   │
  ├────────────┼──────────────────┼──────────────┼───────────────┤
  │ KZG        │ O(n) MSM         │ O(1) = 48 B  │ O(1) pairings │
  │ IPA        │ O(n) scalar muls │ O(log n)     │ O(n) !!       │
  └────────────┴──────────────────┴──────────────┴───────────────┘
```

### Recursive Verification (Halo Technique)

The O(n) verification cost seems fatal — but Halo (2019) showed how to defer it:

```
The Halo trick:
───────────────
  Instead of verifying the IPA inside the current proof, defer it
  to the NEXT proof. The next prover absorbs the previous verification
  as part of its circuit.

  Proof 1: prove statement S₁, defer IPA verification V₁
  Proof 2: prove statement S₂ AND verify V₁, defer IPA verification V₂
  Proof 3: prove statement S₃ AND verify V₂, defer IPA verification V₃
  ...

  This is INCREMENTALLY VERIFIABLE COMPUTATION (IVC).
  Each proof "carries forward" the cost of verifying the previous one.
  The final verifier only checks the last proof's IPA — O(n) once.

  This is the foundation of Halo 2 (used by Zcash) and Nova/SuperNova.
```

---

## 9. FRI — Fast Reed-Solomon Interactive Oracle Proof

### Motivation

FRI works entirely with **hashes** — no elliptic curves, no pairings. This makes it:
- Transparent (no trusted setup)
- Plausibly post-quantum secure (no DLP assumption)
- Extremely fast prover (field arithmetic + hashing, no MSM)
- Larger proofs (tens of KB instead of <1 KB)

```
FRI design philosophy:
──────────────────────
  KZG:  algebraic security (DLP + pairings)    → small proofs
  IPA:  algebraic security (DLP, no pairings)   → medium proofs
  FRI:  hash-based security (collision resistance) → larger proofs, fast prover

  FRI is not itself a polynomial commitment — it proves that a function
  is "close to" a low-degree polynomial (a Reed-Solomon proximity test).
  Combined with Merkle trees, it becomes a polynomial commitment scheme.
```

### Reed-Solomon Codes and Low-Degree Testing

The core question FRI answers:

```
Given evaluations of a function f on a domain D:
  { f(d) : d ∈ D }

Is f (close to) a polynomial of degree < k?

If |D| = n and deg(f) < k where k << n, then f is a
REED-SOLOMON codeword with rate ρ = k/n.

FRI proves this with O(log² n) queries and hash operations.
```

### The FRI Protocol (Split-and-Fold)

```
FRI Protocol overview:
──────────────────────
  Input: evaluations of f₀(x) on domain D₀ (size n), claimed degree < d

  Commit Phase:
  ─────────────
  Round 0:
    f₀(x) has evaluations on D₀ = {ω⁰, ω¹, ..., ω^{n-1}}
    Prover sends: Merkle root of all f₀ evaluations
    Verifier sends: random α₀

    FOLD:
      f₁(x) = f₀_even(x) + α₀ · f₀_odd(x)

      where f₀(x) = f₀_even(x²) + x · f₀_odd(x²)

    f₁ has degree < d/2, evaluated on D₁ (size n/2)

  Round 1:
    Prover sends: Merkle root of all f₁ evaluations
    Verifier sends: random α₁
    FOLD: f₂(x) = f₁_even(x) + α₁ · f₁_odd(x)
    f₂ has degree < d/4, on D₂ (size n/4)

  ...

  Round log₂(d):
    f_{log d} is a CONSTANT (degree 0).
    Prover sends the constant directly.
```

```
Visualization of FRI folding:
─────────────────────────────
  f₀: degree < 2^20, domain size 2^24
    │ fold with α₀
    ▼
  f₁: degree < 2^19, domain size 2^23
    │ fold with α₁
    ▼
  f₂: degree < 2^18, domain size 2^22
    │ ...
    ▼
  f₂₀: degree 0 (constant), domain size 2^4
    → prover sends this constant

  Each fold: split polynomial into even/odd, combine with random α.
  Each step halves the degree and halves the domain.
```

### FRI Query Phase

After the commit phase, the verifier checks consistency:

```
Query Phase (repeated λ times for soundness):
──────────────────────────────────────────────
  For each query:
    1. Verifier picks random index i₀ ∈ D₀
    2. Asks for f₀(x₀) and f₀(-x₀)  (x₀ = ω^{i₀})
       → Prover opens Merkle proof for both
    3. Verifier computes what f₁(x₀²) should be:
       f₁(x₀²) = (f₀(x₀) + f₀(-x₀))/2 + α₀ · (f₀(x₀) - f₀(-x₀))/(2x₀)
    4. Checks this against the committed f₁ Merkle tree
    5. Repeats for f₁ → f₂ → ... → f_{log d}
    6. Checks final value equals the claimed constant

  Why this works:
    If f₀ is NOT low-degree, then with high probability, the
    folded polynomials will be inconsistent at random query points.

  Soundness: each query gives ~log(|D|/d) bits of security.
  For 128-bit security: λ ≈ 128 / log₂(n/d) queries.
  Typical: λ = 30-80 queries.
```

### FRI as a Polynomial Commitment

To commit to f and prove f(z) = v:

```
FRI-based polynomial commitment:
────────────────────────────────
  Commit:
    Evaluate f on domain D.
    Merkle-hash all evaluations.
    Commitment = Merkle root (32 bytes).

  Open at z:
    Compute quotient: q(x) = (f(x) - v) / (x - z)
    If f(z) = v, then q is a polynomial of degree d - 1.
    Run FRI on q to prove it is low-degree.

  The verifier checks:
    1. q is low-degree (via FRI)
    2. q(x) · (x - z) + v = f(x) at queried points (via Merkle openings)

  Proof size: O(log² n) hashes + O(log n) field elements
  Typical: 50-200 KB for large circuits (much larger than KZG's 48 bytes)
```

### ZK-Friendly Hash Functions

FRI's performance depends heavily on the hash function used for Merkle trees:

```
Hash functions for FRI/STARK:
─────────────────────────────
  Traditional hashes (SHA-256, Keccak):
    - Fast on CPU (~500 MB/s)
    - Well-studied security
    - NOT designed for finite fields → expensive to prove in ZK recursion

  Algebraic hashes (field-native):
  ┌────────────┬───────────────┬─────────────────┬──────────────────────┐
  │ Hash       │ Field         │ Speed (native)  │ ZK circuit cost      │
  ├────────────┼───────────────┼─────────────────┼──────────────────────┤
  │ Poseidon   │ any prime     │ ~10× slower     │ ~250 constraints     │
  │ Rescue     │ any prime     │ ~20× slower     │ ~300 constraints     │
  │ Poseidon2  │ any prime     │ ~5× slower      │ ~150 constraints     │
  │ RPO        │ Goldilocks    │ ~3× slower      │ ~200 constraints     │
  │ SHA-256    │ native binary │ 1× (baseline)   │ ~25,000 constraints! │
  └────────────┴───────────────┴─────────────────┴──────────────────────┘

  For recursive proofs (proof of proof), the hash is verified INSIDE the circuit.
  Poseidon: 250 constraints per hash ← cheap
  SHA-256: 25,000 constraints per hash ← 100× more expensive!

  This is why STARKs/FRI systems use algebraic hashes like Poseidon.
```

### Hardware Implications of FRI

```
FRI hardware profile:
─────────────────────
  Prover computation:
    1. NTT (multiple rounds of evaluation on shrinking domains)
    2. Merkle tree construction (hashing all evaluations)
    3. Field arithmetic (folding operations)

  Key difference from KZG:
    KZG prover: dominated by MSM (elliptic curve operations)
    FRI prover: dominated by NTT + hashing (field + hash operations)

  Hardware acceleration for FRI:
    - NTT accelerator (same as Phase 6 topics)
    - Hash accelerator (Poseidon hardware: ~10× speedup over CPU)
    - NO elliptic curve hardware needed!
    - Smaller field elements (BabyBear = 31 bits vs BN254 = 254 bits)
      → more operations per clock cycle, simpler multipliers

  This is why STARK/FRI provers can achieve higher throughput
  than SNARK/KZG provers on equivalent hardware.
```

---

## 10. Comparing Commitment Schemes — Hardware Implications

### Side-by-Side Comparison

```
┌────────────────────┬──────────────────┬──────────────────┬──────────────────┐
│                    │ KZG              │ IPA              │ FRI              │
├────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Setup              │ Trusted (MPC)    │ Transparent      │ Transparent      │
│ Assumption         │ DLP + pairings   │ DLP              │ Collision-resist │
│ Post-quantum       │ No               │ No               │ Plausibly yes    │
│ Commitment size    │ 1 G₁ pt (48 B)  │ 1 G pt (32-48 B)│ Hash (32 B)      │
│ Proof size         │ 1 G₁ pt (48 B)  │ O(log n) G pts  │ O(log² n) hashes │
│ Proof (n=2^20)     │ 48 bytes         │ ~1.9 KB          │ ~50-200 KB       │
│ Verify time        │ O(1) pairings    │ O(n)             │ O(log² n) hashes │
│ Verify (n=2^20)    │ ~1-2 ms          │ ~seconds (alone) │ ~5-20 ms         │
│ Prover bottleneck  │ MSM              │ MSM              │ NTT + hashing    │
│ Prover field size  │ 254-381 bit      │ 254-381 bit      │ 31-64 bit        │
│ HW: curve ops      │ Yes (MSM)        │ Yes (scalar mul) │ No               │
│ HW: NTT            │ Yes              │ Minimal          │ Yes (dominant)   │
│ HW: hash accel     │ No               │ No               │ Yes (Poseidon)   │
│ Used by            │ PLONK, Groth16   │ Halo2, Zcash     │ STARKs, Plonky3  │
│                    │ EigenDA          │ Bulletproofs     │ RISC Zero        │
└────────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

### How Commitment Choice Drives Hardware Design

```
The hardware architect's decision tree:
────────────────────────────────────────

  "Which proof system are we accelerating?"
                    │
         ┌──────────┼──────────┐
         │          │          │
      KZG-based   IPA-based  FRI-based
      (PLONK,     (Halo 2,   (STARKs,
       Groth16)   Zcash)      Plonky3)
         │          │          │
  ┌──────┴──────┐   │   ┌──────┴──────┐
  │ MSM engine  │   │   │ NTT engine  │
  │ (dominant)  │   │   │ (dominant)  │
  │ NTT engine  │   │   │ Hash engine │
  │ (secondary) │   │   │ (Poseidon)  │
  └─────────────┘   │   └─────────────┘
                    │
             ┌──────┴──────┐
             │ MSM engine  │
             │ + recursive │
             │   circuit   │
             └─────────────┘

  Key insight: MSM hardware (Phase 6.1-6.3) benefits KZG and IPA.
               NTT hardware (Phase 6.1-6.3) benefits KZG and FRI.
               Hash hardware (Poseidon) benefits FRI only.
               A general-purpose ZK accelerator needs all three.
```

### Proof Aggregation and Recursion

```
Modern systems combine schemes for the best of both worlds:
────────────────────────────────────────────────────────────
  Layer 1 (fast prover): STARK with FRI
    → Fast proof generation, large proof (~200 KB)

  Layer 2 (proof compression): SNARK with KZG wrapping the STARK
    → Verifies the STARK proof inside a SNARK circuit
    → Output: small proof (~128 bytes)

  This is what RISC Zero, Polygon zkEVM, and others do:
    STARK prover (hardware-friendly, fast) → KZG wrapper (small, cheap to verify on-chain)

  Hardware implication: you may need BOTH NTT acceleration (for STARKs)
  AND MSM acceleration (for the KZG wrapper).
```

---

## 11. Projects

### Project 1: Implement NTT from Scratch

Build a complete NTT library in Rust, C, or Python:

```
Requirements:
  1. Implement forward NTT (radix-2, Cooley-Tukey)
  2. Implement inverse NTT (INTT)
  3. Implement polynomial multiplication via NTT
  4. Support configurable prime field (test with BabyBear p = 2^31 - 2^27 + 1)

Tests:
  - NTT then INTT recovers original coefficients
  - Polynomial multiplication matches naive schoolbook result
  - Evaluate f at each root of unity manually, compare with NTT output
  - Benchmark: NTT of size 2^16, 2^20 — measure time, compare with O(n²) naive

Stretch: Implement radix-4 NTT and compare multiplier count with radix-2.
```

### Project 2: Build a KZG Commitment Scheme

Using an existing elliptic curve library (e.g., `arkworks` in Rust, `py_ecc` in Python):

```
Requirements:
  1. Generate a mock SRS (using a known τ for testing — NOT secure!)
  2. Commit to a polynomial (implement as MSM)
  3. Create opening proofs (compute quotient polynomial, commit)
  4. Verify opening proofs (implement pairing check)
  5. Implement batch opening at a single point

Tests:
  - Commit to f(x) = x³ + 2x + 5, open at z = 3, verify
  - Verify that opening proof for wrong value is rejected
  - Batch open 5 polynomials, compare verification time vs. individual
  - Benchmark: commitment time for degree 2^10, 2^14, 2^16 polynomials

Stretch: Use Ethereum's actual KZG SRS (download from the ceremony).
```

### Project 3: Implement FRI Low-Degree Testing

Build a simplified FRI protocol:

```
Requirements:
  1. Evaluate a polynomial on a domain (using NTT from Project 1)
  2. Build Merkle trees over evaluations (use SHA-256 or Blake3)
  3. Implement the FRI commit phase (fold with random challenges)
  4. Implement the FRI query phase (verify consistency across layers)
  5. Verify that a degree-d polynomial passes, and a random function fails

Tests:
  - Honest prover (degree-d polynomial) — verifier accepts
  - Cheating prover (degree > d) — verifier rejects with high probability
  - Measure proof sizes for d = 2^10, 2^14, 2^16
  - Compare proof size with KZG (should be ~100-1000× larger)

Stretch: Replace SHA-256 with a Poseidon hash implementation.
```

### Project 4: Polynomial Arithmetic Benchmark Suite

Create a comprehensive benchmark comparing polynomial operations:

```
Requirements:
  1. Implement schoolbook polynomial multiplication O(n²)
  2. Implement NTT-based polynomial multiplication O(n log n)
  3. Implement Horner evaluation
  4. Implement Lagrange interpolation
  5. Benchmark all operations for n = 2^8, 2^12, 2^16, 2^20

Output: A table showing:
  - Time per operation
  - Operations per second
  - Crossover point where NTT beats schoolbook
  - Memory usage per operation

Stretch: Profile memory access patterns (cache misses) for NTT at each stage.
         Identify the stage where performance degrades due to stride.
```

### Project 5: Commitment Scheme Comparison

Compare KZG, IPA (simplified), and FRI on identical polynomials:

```
Requirements:
  1. Use the same polynomial f(x) of degree d for all three schemes
  2. Measure: setup time, commit time, prove time, verify time, proof size
  3. Test for d = 2^10, 2^12, 2^14

Expected observations:
  - KZG: smallest proofs, fastest verification, slowest setup (SRS generation)
  - IPA: medium proofs, slow verification (O(n)), no setup
  - FRI: largest proofs, medium verification, fastest prover (small field)

Stretch: Implement proof aggregation — wrap a FRI proof inside a KZG proof.
         Measure the combined proof size and verification time.
```

---

## 12. Resources

### Textbooks and Courses

* **"Proofs, Arguments, and Zero-Knowledge" by Justin Thaler:** The definitive graduate-level reference. Chapter 14 covers polynomial commitments (KZG, IPA, FRI) with full proofs. Free PDF available at [people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.html](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.html)
* **"The MoonMath Manual":** A comprehensive resource for ZK mathematics and polynomial commitments, with step-by-step examples over small fields. Free at [leastauthority.com/community-matters/moonmath-manual/](https://leastauthority.com/community-matters/moonmath-manual/)
* **RareSkills ZK Book — Polynomials and NTT chapters:** Practical, code-focused treatment of polynomial arithmetic for ZK. [rareskills.io/zk-book](https://www.rareskills.io/zk-book)

### Original Papers

* **KZG (Kate, Zaverucha, Goldberg, 2010):** "Constant-Size Commitments to Polynomials and Their Applications." The original paper introducing the KZG commitment. [iacr.org/archive/asiacrypt2010/6477178/6477178.pdf](https://www.iacr.org/archive/asiacrypt2010/6477178/6477178.pdf)
* **Bulletproofs (Bünz et al., 2018):** "Bulletproofs: Short Proofs for Confidential Transactions and More." Introduces the IPA protocol. [eprint.iacr.org/2017/1066](https://eprint.iacr.org/2017/1066)
* **FRI (Ben-Sasson et al., 2018):** "Fast Reed-Solomon Interactive Oracle Proofs of Proximity." The original FRI paper. [eccc.weizmann.ac.il/report/2017/134/](https://eccc.weizmann.ac.il/report/2017/134/)
* **Halo (Bowe et al., 2019):** "Recursive Proof Composition without a Trusted Setup." Shows how IPA's O(n) verification can be deferred using recursion. [eprint.iacr.org/2019/1021](https://eprint.iacr.org/2019/1021)

### Technical Blogs and Tutorials

* **Vitalik Buterin — "KZG Commitments":** Accessible introduction to KZG with Python code examples. [vitalik.eth.limo/general/2024/07/23/kzg.html](https://vitalik.eth.limo/general/2024/07/23/kzg.html)
* **Vitalik Buterin — "STARKs, Part II: Thank Goodness It's FRI-day":** Excellent explanation of FRI with diagrams. [vitalik.eth.limo/general/2017/11/22/starks_part_2.html](https://vitalik.eth.limo/general/2017/11/22/starks_part_2.html)
* **Dankrad Feist — "KZG Polynomial Commitments":** Deep dive into KZG with batch openings and implementation details. [dankradfeist.de/ethereum/2020/06/16/kate-polynomial-commitments.html](https://dankradfeist.de/ethereum/2020/06/16/kate-polynomial-commitments.html)
* **RiscZero — "Understanding FRI":** Clear walkthrough of FRI protocol for implementers. [dev.risczero.com/proof-system/stark-by-hand](https://dev.risczero.com/proof-system/stark-by-hand)

### NTT and Hardware

* **"NTT Implementation Guide" (Longa and Naehrig, 2016):** "Speeding up the Number Theoretic Transform for Faster Ideal Lattice-Based Cryptography." Hardware-oriented NTT optimization. [eprint.iacr.org/2016/504](https://eprint.iacr.org/2016/504)
* **Ingonyama — "Hardware Friendliness of NTT":** Analysis of NTT memory access patterns and hardware architectures. [ingonyama.com/blog](https://www.ingonyama.com/blog)
* **PipeZK (2021) and CuZK (2022):** Research papers on FPGA and GPU acceleration of NTT for ZK proving (covered in depth in Phase 6).

### Interactive Tools

* **"STARK 101" by StarkWare:** A hands-on Python tutorial that builds a STARK prover from scratch, including NTT and FRI. [starkware.co/stark-101/](https://starkware.co/stark-101/)
* **Ethereum's KZG Ceremony Documentation:** Understand how real-world trusted setup works. [ceremony.ethereum.org](https://ceremony.ethereum.org)
* **arkworks-rs:** Rust library with reference implementations of KZG (ark-poly-commit), NTT (ark-poly), and field arithmetic (ark-ff). Essential for Projects 2 and 5. [github.com/arkworks-rs](https://github.com/arkworks-rs)
