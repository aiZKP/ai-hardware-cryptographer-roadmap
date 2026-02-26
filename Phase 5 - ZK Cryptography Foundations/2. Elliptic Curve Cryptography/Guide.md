# Elliptic Curve Cryptography for Zero-Knowledge Proofs

> **Goal:** Build a ground-up understanding of elliptic curves — from the group law to bilinear pairings — with constant focus on what operations become hardware bottlenecks in ZK proof generation. By the end, you will understand why MSM dominates prover cost, why specific curves were chosen for ZK, and what hardware must accelerate.

**Prerequisite:** Phase 5.1 (Mathematical Foundations) — finite fields, modular arithmetic, group theory, polynomial rings. You must be comfortable with F_p arithmetic and the discrete logarithm problem.

---

## Table of Contents

1. [Elliptic Curves Over Finite Fields](#1-elliptic-curves-over-finite-fields)
2. [Point Arithmetic — Affine Coordinates](#2-point-arithmetic--affine-coordinates)
3. [Projective Coordinates — Eliminating Inversions](#3-projective-coordinates--eliminating-inversions)
4. [Curve Forms and Their Hardware Implications](#4-curve-forms-and-their-hardware-implications)
5. [Scalar Multiplication — The Core Operation](#5-scalar-multiplication--the-core-operation)
6. [ZK-Specific Curves — Parameters and Design Choices](#6-zk-specific-curves--parameters-and-design-choices)
7. [Small Fields for STARKs](#7-small-fields-for-starks)
8. [Bilinear Pairings — The One-Time Multiplication](#8-bilinear-pairings--the-one-time-multiplication)
9. [Multi-Scalar Multiplication (MSM) — The Hardware Bottleneck](#9-multi-scalar-multiplication-msm--the-hardware-bottleneck)
10. [Projects](#10-projects)
11. [Resources](#11-resources)

---

## 1. Elliptic Curves Over Finite Fields

### What Is an Elliptic Curve?

An elliptic curve over a finite field F_p is the set of points (x, y) satisfying:

```
y² = x³ + ax + b   (mod p)
```

plus a special "point at infinity" O that acts as the identity element. The condition `4a³ + 27b² ≠ 0` ensures the curve has no self-intersections (non-singular).

### Why Elliptic Curves for ZK?

```
                What you need for ZK:
                ─────────────────────
                1. A group where DLP is hard        → elliptic curve points
                2. Homomorphic commitment schemes    → Pedersen on the curve
                3. Polynomial commitment verification → bilinear pairings (some curves)
                4. Efficient multi-scalar multiplication → Pippenger on the curve
```

Compared to multiplicative groups of F_p (used in classical Diffie-Hellman):
- **Same security, smaller keys:** 256-bit curve ≈ 3072-bit RSA/DH
- **No index calculus:** the best attack on elliptic curve DLP is Pollard's rho at O(√r), giving r ≈ 2^256 for 128-bit security
- **Rich structure:** pairing-friendly curves enable KZG commitments and Groth16

### Visualizing the Group Law

On a real-number curve, point addition has geometric meaning:

```
To add P + Q:
  1. Draw a line through P and Q
  2. The line intersects the curve at a third point R'
  3. Reflect R' over the x-axis to get R = P + Q

To double P (compute 2P):
  1. Draw the tangent line at P
  2. The tangent intersects the curve at R'
  3. Reflect to get R = 2P

Special cases:
  - P + O = P         (identity)
  - P + (-P) = O      (inverse: -P is the reflection of P)
```

Over a finite field, there is no geometry — but the algebraic formulas are the same. The "line through P and Q" is computed with modular arithmetic.

### The Group Structure

The set of curve points forms an **abelian group** under point addition:
- **Closure:** P + Q is always a curve point
- **Associativity:** (P + Q) + R = P + (Q + R)
- **Identity:** the point at infinity O
- **Inverse:** for P = (x, y), the inverse is -P = (x, -y mod p)
- **Commutativity:** P + Q = Q + P

The group order `r` (number of points) is close to `p` by Hasse's theorem:

```
|r - (p + 1)| ≤ 2√p
```

For ZK curves, `r` is chosen to be a large prime so the group is cyclic.

---

## 2. Point Arithmetic — Affine Coordinates

### Point Addition (P₁ ≠ P₂)

Given P₁ = (x₁, y₁) and P₂ = (x₂, y₂):

```
λ = (y₂ - y₁) / (x₂ - x₁)  mod p        ← "slope of the line"
x₃ = λ² - x₁ - x₂            mod p
y₃ = λ · (x₁ - x₃) - y₁      mod p
```

**Cost: 1I + 2M + 1S** (1 field inversion, 2 multiplications, 1 squaring)

### Point Doubling (2P₁)

Given P₁ = (x₁, y₁):

```
λ = (3x₁² + a) / (2y₁)       mod p        ← "slope of the tangent"
x₃ = λ² - 2x₁                 mod p
y₃ = λ · (x₁ - x₃) - y₁      mod p
```

**Cost: 1I + 2M + 2S**

### The Inversion Problem

The division `/` in the slope computation is actually a **field inversion** followed by a multiplication:

```
(y₂ - y₁) / (x₂ - x₁) = (y₂ - y₁) · (x₂ - x₁)^{-1} mod p
```

Computing `z^{-1} mod p` costs:
- **Extended Euclidean Algorithm:** O(log² p) — about 20-100× more than a single multiplication
- **Fermat's method:** z^{-1} = z^{p-2} mod p — about 254 multiplications for a 254-bit prime

For a 256-bit scalar multiplication requiring ~256 doublings + ~128 additions, affine coordinates would need ~384 inversions. This is **catastrophically expensive**.

```
Cost comparison for BN254 (254-bit field):
─────────────────────────────────────────
  Field multiplication:  ~10-50 ns
  Field squaring:        ~8-40 ns  (~0.8× multiplication)
  Field inversion:       ~500-5000 ns  (20-100× multiplication!)

  Affine scalar mul:  ~384 inversions → dominated by inversions
  Projective scalar mul: 0 inversions during the loop, 1 at the end
```

**This is why projective coordinates exist.**

---

## 3. Projective Coordinates — Eliminating Inversions

### Jacobian Projective Coordinates

Represent the affine point (x, y) as a triple (X : Y : Z) where:

```
x = X / Z²
y = Y / Z³
```

The point at infinity is (1 : 1 : 0). Two triples represent the same point if one is a scalar multiple of the other.

### Why This Works

Every division in the affine formulas is absorbed into the Z coordinate. Instead of computing `a/b`, you multiply the numerator into X or Y and the denominator into Z. The accumulated Z denominators cancel when you convert back to affine at the very end.

**Result:** Replace ~384 inversions with 0 inversions during the main loop, plus 1 inversion at the end.

### Jacobian Operation Counts (for a = 0 curves: BN254, BLS12-381)

From the [Explicit-Formulas Database](https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html):

| Operation | Best Formula | Cost | M-equivalent (1S ≈ 0.8M) |
|-----------|-------------|------|---------------------------|
| **General addition** | add-2007-bl | 11M + 5S | **~15M** |
| **Mixed addition** (Z₂=1) | madd-2007-bl | 7M + 4S | **~10.2M** |
| **Affine-affine add** (Z₁=Z₂=1) | mmadd-2007-bl | 4M + 2S | **~5.6M** |
| **Doubling** | dbl-2009-l | 2M + 5S | **~6M** |
| **Doubling** (Z₁=1) | mdbl-2007-bl | 1M + 5S | **~5M** |

### Cost of a Full Scalar Multiplication (Jacobian, a=0)

For a 253-bit scalar (BLS12-381 group order):

```
~253 doublings + ~127 additions (average for random scalar)

Doublings:  253 × 6M = 1,518 M-equivalents
Additions:  127 × 15M = 1,905 M-equivalents  (general)
         or 127 × 10.2M = 1,295 M-equivalents (mixed, base point in affine)
Final inversion: ~254 M-equivalents (one-time)

Total (mixed): ~3,067 M-equivalents
```

Each M-equivalent is one field multiplication over a 381-bit prime. On CPU at ~30ns per multiplication: **~92 microseconds** for one scalar multiplication.

### Hardware Implication

```
For FPGA design, the key resource is the FIELD MULTIPLIER:

  BN254 (254-bit):  4 × 64-bit limbs → DSP slice count for one multiplier
  BLS12-381 (381-bit): 6 × 64-bit limbs → larger DSP count

  Scalar mul pipeline:
    [Doubling unit] ──→ [Addition unit] ──→ [Result register]
         6M/cycle          10-15M/cycle

  With pipelining: multiple scalar multiplications in flight simultaneously
  This is the foundation of MSM acceleration in Phase 6
```

---

## 4. Curve Forms and Their Hardware Implications

### 4.1 Short Weierstrass (y² = x³ + ax + b)

The standard form used by most ZK curves.

```
Pros:
  ✓ Most studied, most implementations available
  ✓ a=0 specialization gives fast doubling (BN254, BLS12-381)
  ✓ Direct pairing computation

Cons:
  ✗ Separate formulas for addition vs. doubling
  ✗ Must handle edge cases (P = Q, P = -Q, P = O)
  ✗ In hardware: needs multiplexing logic or separate circuits
```

**Used by:** BN254, BLS12-381, BLS12-377, BW6-761, Pallas, Vesta, secp256k1

### 4.2 Montgomery Form (By² = x³ + Ax² + x)

```
Key innovation: the Montgomery ladder
─────────────────────────────────────
For each bit of the scalar (from MSB to LSB):
  if bit == 0:  R₁ = 2R₁,  R₀ = R₀ + R₁
  if bit == 1:  R₀ = 2R₀,  R₁ = R₀ + R₁

Always: one doubling + one differential addition per bit
```

```
Pros:
  ✓ Constant-time by design (no branching on scalar bits)
  ✓ x-coordinate-only arithmetic (y is never needed during the ladder)
  ✓ Perfect side-channel resistance
  ✓ Extremely regular dataflow for FPGA/ASIC pipelines

Cons:
  ✗ Only supports single scalar multiplication (not MSM-friendly)
  ✗ Differential addition requires knowing P - Q (the base point)
  ✗ No direct pairing computation
```

**Used by:** Curve25519 (the canonical example). In ZK: some circuits internally use Montgomery form for ECDSA signature verification.

### 4.3 Twisted Edwards Form (ax² + y² = 1 + dx²y²)

```
The breakthrough: UNIFIED addition formula
───────────────────────────────────────────
The SAME formula works for:
  - P + Q (addition)
  - P + P (doubling)
  - P + O (identity)
  - P + (-P) (inverse)

No case distinctions. No edge-case handling. One circuit does everything.
```

**Extended Coordinates (X : Y : Z : T)** where x = X/Z, y = Y/Z, T = XY/Z:

| Operation | Cost (a = -1) | Cost (general) |
|-----------|--------------|----------------|
| Unified addition | **8M** | 9M + 1S |
| Doubling | **4M + 4S** | 4M + 4S |
| Mixed addition (Z₂=1) | **8M** | — |

Compare to Jacobian Weierstrass: addition 11M + 5S, doubling 2M + 5S.

```
Hardware advantage:
──────────────────
  Weierstrass: need MUX between add/double circuits + edge-case FSM
  Edwards:     ONE arithmetic unit handles ALL point operations

  → Simpler control logic
  → No timing side channels (constant operation)
  → Smaller silicon area for the control path
  → Ideal for pipelined MSM engines
```

**ZK curves in twisted Edwards form:**

| Curve | Defined Over | a | Used In |
|-------|-------------|---|---------|
| **Baby Jubjub** | BN254 scalar field | 168700 | circom/snarkjs, Ethereum ZK circuits |
| **Jubjub** | BLS12-381 scalar field | -1 | Zcash Sapling |
| **Bandersnatch** | BLS12-381 scalar field | -5 | Ethereum Verkle trees (proposed) |

**The embedded curve pattern:** These curves are defined over the *scalar field* of the outer pairing-friendly curve. This means operations on Baby Jubjub points can be expressed as arithmetic constraints in a BN254 SNARK circuit. This is how ECDSA signature verification is done inside ZK proofs.

### 4.4 Birational Equivalence

Every Montgomery curve has a twisted Edwards equivalent, and vice versa:

```
Montgomery (A, B) ←→ Twisted Edwards (a, d)

  a = (A + 2) / B
  d = (A - 2) / B
```

This means an implementation can switch between forms depending on which operation is cheapest. Some ZK systems use Edwards form for in-circuit computations and Weierstrass form for the outer proof system.

---

## 5. Scalar Multiplication — The Core Operation

Scalar multiplication `k · P` (computing P added to itself k times) is the fundamental building block. Everything in ZK — commitments, proof generation, MSM — reduces to scalar multiplication.

### 5.1 Double-and-Add (Binary Method)

The simplest algorithm. Process the scalar bit-by-bit:

```python
def scalar_mul(k, P):
    """Left-to-right double-and-add"""
    Q = O                          # point at infinity
    for i in range(bits(k)-1, -1, -1):
        Q = double(Q)              # always double
        if bit(k, i) == 1:
            Q = add(Q, P)          # conditionally add
    return Q
```

**Cost:** ~n doublings + ~n/2 additions for an n-bit scalar.

```
For BLS12-381 (253-bit scalar):
  253 doublings × 6M  = 1,518 M-eq
  ~127 additions × 15M = 1,905 M-eq
  Total: ~3,423 M-eq
```

**Problem:** The conditional addition leaks the scalar through timing/power side channels. Not suitable for secret-key operations without countermeasures.

### 5.2 Windowed Methods (w-NAF)

Process w bits at a time instead of 1:

```
w-NAF (width-w Non-Adjacent Form):
─────────────────────────────────
  1. Recode scalar k into signed digits from {0, ±1, ±3, ..., ±(2^{w-1}-1)}
     Property: at most 1 in every w consecutive digits is nonzero
  2. Precompute: {P, 3P, 5P, 7P, ...} (odd multiples up to (2^{w-1}-1)P)
  3. Scan digits: double w times for each nonzero, add the precomputed point

Additions reduced from ~n/2 to ~n/(w+1)
```

| Window w | Precomputed points | Additions (253-bit) | Speedup vs binary |
|----------|-------------------|---------------------|-------------------|
| 1 | 0 | ~127 | baseline |
| 4 | 7 | ~51 | 2.5× fewer adds |
| 5 | 15 | ~42 | 3× fewer adds |
| 6 | 31 | ~36 | 3.5× fewer adds |

**Hardware mapping:** The precomputed table stores in BRAM/URAM on FPGAs. The main loop is a pipeline: w consecutive doublings feed into a table lookup + addition. Window width trades table size against main-loop additions.

### 5.3 GLV Endomorphism — Halving the Doublings

Some curves have an efficiently computable **endomorphism** φ where φ(P) = λ · P for a known scalar λ.

```
GLV decomposition:
─────────────────
  1. Given scalar k, find k₁, k₂ each ~n/2 bits
     such that k ≡ k₁ + k₂·λ (mod r)
  2. Compute: k·P = k₁·P + k₂·φ(P)
     using Shamir's trick (simultaneous double-and-add)

Result:
  ~n/2 doublings + ~n/2 additions (instead of ~n doublings + ~n/2 additions)
  Speedup: ~33-40%
```

**Which curves support GLV:**

| Curve | Endomorphism φ | How φ works |
|-------|---------------|-------------|
| secp256k1 | φ(x, y) = (βx, y) | β = cube root of unity in F_p (exists because p ≡ 1 mod 3) |
| BN254 | Via curve automorphism | Built into the BN construction |
| BLS12-381 | Via curve automorphism | From the x-parameter |
| Bandersnatch | Explicit GLV | Specifically designed for this |

**Hardware mapping:** GLV replaces a full-width (253-bit) scalar-multiplication engine with two half-width (~127-bit) engines running simultaneously. The endomorphism φ is trivially cheap (one field multiplication by a constant). On FPGA, this halves pipeline depth.

---

## 6. ZK-Specific Curves — Parameters and Design Choices

### 6.1 BN254 (alt_bn128)

A Barreto-Naehrig curve: y² = x³ + 3

```
Base field prime (254 bits):
  p = 21888242871839275222246405745257275088696311157297823662689037894645226208583

Group order (254 bits):
  r = 21888242871839275222246405745257275088548364400416034343698204186575808495617

Embedding degree: 12
Generator G₁ = (1, 2)
```

**Why BN254 is everywhere in Ethereum:**
- EIP-196 and EIP-197 hardcoded precompiles at addresses 0x06, 0x07, 0x08
- ecAdd (150 gas), ecMul (6000 gas), ecPairing (34000 + 45000/pair gas)
- Groth16 verification on-chain costs ~230,000 gas (~$0.50 at 30 gwei)

**The security debate:**

```
Attack complexity:
──────────────────
  EC discrete log (Pollard rho):        ~2^127 operations ← still secure
  Extension field DLP (F_{p^12}, exTNFS): ~2^100-110 operations ← WEAKENED

  BN254 security is bounded by the weaker target: ~100-110 bits
  BLS12-381 was designed to fix this: ~128 bits for both targets
```

**Hardware datapath:**
- Field element: 254 bits → 4 × 64-bit limbs
- Montgomery multiplier: operates on 4-limb × 4-limb → 4-limb product
- DSP slice requirement (FPGA): ~16 DSP48 slices per multiplier stage

### 6.2 BLS12-381

A Barreto-Lynn-Scott curve: y² = x³ + 4

```
Curve parameter (very low Hamming weight — only 5 set bits):
  x = -0xd201000000010000 = -15132376222941642752

Base field prime (381 bits):
  p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf
        6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab

Group order (255 bits):
  r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001

Embedding degree: 12
```

**Group structure:**
```
G₁: points on E(F_p)          compressed size: 48 bytes
G₂: points on twist E'(F_{p²})  compressed size: 96 bytes
GT: elements of F_{p^{12}}     size: 576 bytes

Groth16 proof: 2 G₁ + 1 G₂ = 2×48 + 96 = 192 bytes (BN254: 128 bytes)
```

**Why the low Hamming weight matters:**
- The Miller loop iterates over bits of |x|
- Only 5 set bits → only 4 addition steps in the Miller loop (vs. ~32 for BN254)
- Result: BLS12-381 pairing is faster despite the larger field

**Hardware datapath:**
- Field element: 381 bits → 6 × 64-bit limbs
- Montgomery multiplier: 6-limb × 6-limb → 6-limb product
- ~50% more DSPs per multiplier vs BN254
- Extension field F_{p²}: elements are pairs (a + bu), multiply via Karatsuba: 3 base-field muls

### 6.3 BLS12-377 and BW6-761 (Recursive Proof Composition)

```
BLS12-377: y² = x³ + 1
─────────────────────────
  p: 377 bits    r: 253 bits
  Key property: 2^47 divides (r - 1)
  → NTT domains up to size 2^47 over the scalar field
  → Massive circuits can be proven efficiently

BW6-761: a companion curve
──────────────────────────
  p: 761 bits    embedding degree: 6
  Scalar field of BW6-761 = base field of BLS12-377
  → A BLS12-377 pairing check can be verified inside a BW6-761 SNARK
  → Enables recursive proof composition: prove a proof of a proof
```

**Used by:** Aleo (Varuna SNARK), Celo

### 6.4 Pasta Curves — Pallas and Vesta (The Cycle)

Both: y² = x³ + 5

```
Pallas base field: p = 28948022309329048855892746252171976963363056481941560715954676764349967630337
Vesta base field:  q = 28948022309329048855892746252171976963363056481941647379679742748393362948097

THE CYCLE:
──────────
  Pallas scalar field = q    (Vesta's base field)
  Vesta scalar field  = p    (Pallas's base field)

  → A proof over Pallas can be verified in a circuit over Vesta
  → A proof over Vesta can be verified in a circuit over Pallas
  → Infinite recursion without pairings!
```

```
    ┌─────────────────────────────────┐
    │ Prove over Pallas (base field p) │
    │  scalar field = q                │
    └──────────────┬──────────────────┘
                   │ verify inside
                   ▼
    ┌─────────────────────────────────┐
    │ Prove over Vesta (base field q)  │
    │  scalar field = p                │
    └──────────────┬──────────────────┘
                   │ verify inside
                   ▼
    ┌─────────────────────────────────┐
    │ Prove over Pallas again...       │
    └─────────────────────────────────┘
```

**Used by:** Zcash Halo 2 (IPA-based commitments, no trusted setup), Mina Protocol

**Hardware note:** Pasta curves use IPA (Inner Product Argument) for polynomial commitments, which requires **no pairings**. The prover cost is dominated by MSM — same as pairing-based systems. But the verifier is more expensive (logarithmic in degree, vs. constant for KZG).

### 6.5 Embedded Curves for In-Circuit Operations

These curves are defined over the *scalar field* of the outer curve, enabling elliptic curve operations inside ZK circuits:

| Embedded Curve | Outer Curve | Form | Used For |
|---------------|-------------|------|----------|
| **Baby Jubjub** | BN254 | Twisted Edwards | ECDSA/EdDSA verification in Ethereum ZK |
| **Jubjub** | BLS12-381 | Twisted Edwards | Zcash Sapling signatures |
| **Bandersnatch** | BLS12-381 | Twisted Edwards | Ethereum Verkle tree proofs |

**Why twisted Edwards for embedded curves?**
- Unified addition = fewer constraints in the SNARK circuit
- No branching = simpler circuit topology
- Baby Jubjub addition costs ~6 multiplication constraints in a BN254 SNARK

---

## 7. Small Fields for STARKs

STARKs (and FRI-based proof systems) don't use elliptic curves at all. They work in **small prime fields** where field arithmetic is dramatically cheaper:

| Field | Prime | Bits | NTT Domain | Used By |
|-------|-------|------|------------|---------|
| **BabyBear** | 2³¹ - 2²⁷ + 1 = 2013265921 | 31 | up to 2²⁷ | Risc0, Plonky3, SP1 |
| **KoalaBear** | 2³¹ - 2²⁴ + 1 | 31 | large | Plonky3 |
| **Mersenne31** | 2³¹ - 1 = 2147483647 | 31 | variable | Plonky3 |
| **Goldilocks** | 2⁶⁴ - 2³² + 1 | 64 | up to 2³² | Plonky2 |

**Why small fields are fast:**

```
Field multiplication cost (approximate):
──────────────────────────────────────
  BabyBear (31-bit):    1 native 32-bit multiply + 1 reduction
  Goldilocks (64-bit):  1 native 64-bit multiply + fast reduction
  BN254 (254-bit):      ~16 native 64-bit multiplies (4-limb schoolbook)
  BLS12-381 (381-bit):  ~36 native 64-bit multiplies (6-limb schoolbook)

  Ratio: BabyBear is ~50-100× cheaper per field multiply than BN254
```

**The security tradeoff:** 31-bit field → 31 bits of direct security (terrible). Solution: use a **degree-4 extension field** F_{p⁴} with BabyBear for the cryptographic commitment (FRI), giving ~124 bits of security. The prover mostly works in the base field (cheap NTT); only the commitment step uses the extension.

**Hardware implication:** Small-field provers are ideal for GPU acceleration — 32-bit multiplies map perfectly to GPU integer arithmetic. This is why STARK provers on GPUs can be extremely fast. SNARK provers on curves (256-381 bit) require multi-limb emulation on GPUs, which is much slower.

---

## 8. Bilinear Pairings — The One-Time Multiplication

### 8.1 What Is a Pairing?

A bilinear pairing is a map:

```
e : G₁ × G₂ → GT
```

where G₁, G₂ are elliptic curve groups and GT is a multiplicative subgroup of F_{p^k}*.

### 8.2 The Bilinearity Property

For all P ∈ G₁, Q ∈ G₂, and scalars a, b:

```
e(aP, bQ) = e(P, Q)^{ab}
```

**What this means in plain terms:**

```
On the curve:     you can ADD committed values    [a]·G + [b]·G = [a+b]·G
With pairings:    you can CHECK ONE MULTIPLICATION e([a]·G₁, [b]·G₂) = e(G₁,G₂)^{ab}

This is the ONLY known way to verify a product of two hidden scalars
in elliptic curve cryptography. You get exactly ONE level of multiplication.
```

### 8.3 Why Pairings Enable KZG Verification

KZG polynomial commitment scheme:

```
Setup:     Trusted setup produces [s^i]₁ for i = 0..d and [s]₂
Commit:    C = [f(s)]₁ = Σ cᵢ · [s^i]₁               (MSM!)
Prove:     For evaluation f(z) = y:
           q(x) = (f(x) - y) / (x - z)
           π = [q(s)]₁                                  (MSM!)
Verify:    e(π, [s - z]₂) = e(C - [y]₁, [1]₂)         (2 pairings)
```

The verifier checks a polynomial relation using 2 pairings — **constant time regardless of polynomial degree**. The prover computes 2 MSMs — this is where the cost is.

### 8.4 How a Pairing Is Computed

The **optimal Ate pairing** on BLS12-381:

```
Phase 1: Miller Loop
────────────────────
  Iterate over bits of |x| = 0xd201000000010000 (64 bits, Hamming weight 5)

  For each bit:
    - Doubling step: double a point on G₂, evaluate "line function" at Q
    - Multiply intermediate result in F_{p^{12}}
  For each SET bit:
    - Addition step: add a point on G₂, evaluate line function
    - Multiply again in F_{p^{12}}

  Cost: ~63 doubling steps + 4 addition steps
  Each step: multiple F_p multiplications through the tower F_p → F_{p²} → F_{p⁶} → F_{p^{12}}

Phase 2: Final Exponentiation
──────────────────────────────
  Raise Miller loop result to (p^{12} - 1) / r

  Easy part: (p⁶ - 1)(p² + 1) — uses Frobenius maps (cheap: just conjugation)
  Hard part: curve-specific exponentiation in F_{p^{12}}
```

### 8.5 Operation Cost

| Curve | Miller Loop | Final Exp | Total Pairing | Relative |
|-------|------------|-----------|---------------|----------|
| BN254 | ~4,500 M_p | ~5,000 M_p | ~9,500 M_p | 1.0× |
| BLS12-381 | ~7,050 M_p | ~8,339 M_p | ~15,389 M_p | 1.6× |

(M_p = base field multiplications)

A single BLS12-381 pairing ≈ **15,389 field multiplications** ≈ 3.4 million CPU cycles.

### 8.6 Why Pairings Are Only in the Verifier

```
PROVER workload:                      VERIFIER workload:
────────────────                      ─────────────────
  MSM over G₁ (millions of points)     2-3 pairings (fixed, small)
  MSM over G₂ (smaller)                A few scalar multiplications
  NTT (polynomial evaluation)

  Prover: O(n log n) where n = circuit size
  Verifier: O(1) — CONSTANT TIME!

  The prover never computes a pairing.
  The verifier never computes an MSM.
```

**Groth16 proof system:**
- Proof size: 2 G₁ + 1 G₂ = 192 bytes (BLS12-381) or 128 bytes (BN254)
- Verification: 1 multi-pairing of size 3
- Verification time: ~1.5 ms on CPU (BN254)

**Hardware implication:** ZK hardware accelerators target the **prover** (MSM + NTT). Pairing hardware is only needed for high-throughput verification (e.g., on-chain verification of thousands of proofs).

---

## 9. Multi-Scalar Multiplication (MSM) — The Hardware Bottleneck

### 9.1 Definition

Given n scalars k₁, ..., kₙ and n elliptic curve points P₁, ..., Pₙ, compute:

```
T = Σᵢ kᵢ · Pᵢ = k₁·P₁ + k₂·P₂ + ... + kₙ·Pₙ
```

### 9.2 Where MSM Appears

| Proof System | MSM Operations | Typical Size |
|-------------|---------------|--------------|
| Groth16 | 3 MSMs (one each for [A]₁, [B]₂, [C]₁) | n = circuit size |
| PLONK (KZG) | Multiple MSMs for polynomial commitments | n = circuit size |
| Marlin/Varuna | MSMs for commitments | n = circuit size |

**MSM as percentage of prover time:**

```
Groth16:     ~70% MSM + ~20% NTT + ~10% other
PLONK:       ~85-90% MSM
Marlin:      ~70-80% MSM

MSM + NTT combined: >90% of ALL proving time
```

### 9.3 Naive vs. Pippenger

```
Naive approach:
──────────────
  Compute each kᵢ·Pᵢ independently, then sum
  Cost: n × O(b) point operations (b = scalar bit-length)
  For n = 2^20, b = 253: ~265 million group operations

Pippenger's Bucket Method:
──────────────────────────
  1. Choose window width c ≈ log₂(n) bits
  2. Slice each scalar into ⌈b/c⌉ windows of c bits
  3. For each window position j:
     a. Create 2^c - 1 buckets
     b. For each point Pᵢ: read window value v → add Pᵢ to bucket[v]
     c. Aggregate buckets with running sum
  4. Combine windows with doublings

  Cost: O(n·b / log₂(n)) group operations

  For n = 2^20, c = 20:
    Windows: ⌈253/20⌉ = 13
    Per window: ~2^20 bucket adds + ~2^20 bucket aggregation
    Total: ~27 million group operations  (10× better than naive!)
```

### 9.4 MSM Size in Practice

| Circuit Size | MSM Size (n) | Pippenger Ops | CPU Time (est.) |
|-------------|-------------|---------------|-----------------|
| 2¹⁶ (64K gates) | 2¹⁶ | ~4M | ~120 ms |
| 2²⁰ (1M gates) | 2²⁰ | ~27M | ~810 ms |
| 2²² (4M gates) | 2²² | ~96M | ~2.9 s |
| 2²⁴ (16M gates) | 2²⁴ | ~350M | ~10.5 s |
| 2²⁶ (64M gates) | 2²⁶ | ~1.3B | ~39 s |

**This is why hardware acceleration matters.** A 2²⁴ MSM takes ~10 seconds on CPU. GPU can achieve >50× speedup. FPGA can offer better power efficiency. A dedicated ZPU could be 100×+.

### 9.5 Memory Access Pattern — The Hardware Challenge

```
Pippenger's bucket accumulation:
────────────────────────────────
  For each point Pᵢ:
    Read window value vᵢ (from scalar)
    Read bucket[vᵢ]  (current accumulator)
    Compute bucket[vᵢ] += Pᵢ  (point addition)
    Write bucket[vᵢ]

  Problem: vᵢ is RANDOM (data-dependent)
    → Random memory access to bucket array
    → Cache-unfriendly for CPU
    → Bank conflicts on GPU shared memory
    → On FPGA: requires multi-port BRAM or serialization

  With 2^20 buckets × 96 bytes per G₁ point (BLS12-381):
    Bucket table = 96 MB  → does NOT fit in L1/L2 cache
    → Memory bandwidth is the bottleneck, not compute!
```

**This is the fundamental hardware design challenge you will tackle in Phase 6.**

---

## 10. Projects

### Project 1: Elliptic Curve Point Arithmetic from Scratch

Implement point operations over a small prime field in C or Rust.

```
Goal: Understand the group law, affine vs projective, operation counts
Field: use a small prime first (p = 65537 or similar), then scale to BN254

Implement:
  1. Affine point addition and doubling (with inversion)
  2. Jacobian projective addition and doubling (no inversion)
  3. Conversion: affine ↔ Jacobian
  4. Verify: Jacobian result converted to affine matches affine result
  5. Benchmark: count field multiplications for each method

Deliverable: working library with property-based tests (P + Q = Q + P, etc.)
```

### Project 2: Scalar Multiplication — Three Algorithms

Implement and benchmark three scalar multiplication methods.

```
Goal: Understand algorithm-hardware tradeoffs
Curve: BN254 G₁ (use your field library from Phase 5.1 or a reference)

Implement:
  1. Double-and-add (binary, left-to-right)
  2. w-NAF with w = 4 and w = 5
  3. Montgomery ladder (x-coordinate only)

Benchmark:
  - Count field multiplications for each
  - Measure wall-clock time for 1000 random scalar multiplications
  - Verify all three produce the same result
  - Plot: additions per bit vs window width

Deliverable: comparison table + timing results
```

### Project 3: Pippenger MSM — Small Scale

Implement Pippenger's algorithm for small MSMs.

```
Goal: Understand the bucket method before hardware acceleration in Phase 6
Size: n = 2^10 to 2^16 (start small, scale up)
Curve: BN254 G₁

Implement:
  1. Naive MSM (independent scalar multiplications + sum)
  2. Pippenger's bucket method with configurable window width c
  3. Window width optimization: try c = 8, 10, 12, 14 and measure

Benchmark:
  - Naive vs Pippenger wall-clock time for n = 2^10, 2^12, 2^14, 2^16
  - Verify both produce the same result
  - Profile: what fraction of time is bucket accumulation vs bucket aggregation?
  - Plot: speedup vs n (should approach O(log n) improvement)

Deliverable: working MSM implementation + benchmark report
```

### Project 4: Pairing Computation (Use a Library)

Use an existing library (arkworks-rs, blst, or gnark-crypto) to compute pairings and verify KZG commitments.

```
Goal: Understand the verifier's perspective and pairing cost
Library: arkworks-rs (Rust) recommended

Tasks:
  1. Compute e(G₁, G₂) on BN254 and BLS12-381, print the GT element
  2. Verify bilinearity: e(aP, bQ) == e(P, Q)^{ab} for random a, b
  3. Implement a toy KZG commitment:
     - Commit to polynomial f(x) = 3x² + 2x + 1
     - Generate proof for evaluation at z = 5
     - Verify the proof using a pairing check
  4. Benchmark: measure pairing time on BN254 vs BLS12-381

Deliverable: working KZG toy example + pairing benchmark
```

### Project 5: Curve Comparison Report

Write a technical comparison of ZK curves.

```
Goal: Understand why different systems choose different curves

Compare for each curve (BN254, BLS12-381, BLS12-377, Pallas/Vesta):
  1. Field size → hardware datapath width (limbs × 64 bits)
  2. Group order → scalar multiplication cost
  3. Pairing support → yes/no, and pairing cost
  4. NTT-friendliness → largest power of 2 dividing (r - 1)
  5. Security level → EC DLP + pairing DLP (if applicable)
  6. Proof size for Groth16 (or equivalent)
  7. On-chain verification cost (gas for Ethereum, if applicable)

Deliverable: comparison table + 1-page analysis of which curve for which use case
```

---

## 11. Resources

### Textbooks and Manuals

* **"Pairings for Beginners" by Craig Costello:**  The best introduction to bilinear pairings with full mathematical detail. Free PDF from Microsoft Research.
* **MoonMath Manual (Least Authority):**  Free, comprehensive manual covering fields, curves, pairings, and ZK proofs. [github.com/LeastAuthority/moonmath-manual](https://github.com/LeastAuthority/moonmath-manual)
* **"Elliptic Curve Cryptography for Developers" by Michael Rosing (Manning, 2024):**  Practical guide with C implementations covering ECC, digital signatures, and pairings.
* **"Proofs, Arguments, and Zero-Knowledge" by Justin Thaler:**  The definitive academic textbook. Free PDF. Covers everything from interactive proofs to modern SNARKs.

### Online References

* **Explicit-Formulas Database (EFD):**  The definitive reference for point arithmetic operation counts. [hyperelliptic.org/EFD](https://www.hyperelliptic.org/EFD/)
* **"BN254 For The Rest Of Us":**  Detailed walkthrough of BN254 parameters. [hackmd.io/@jpw/bn254](https://hackmd.io/@jpw/bn254)
* **"BLS12-381 For The Rest Of Us" by Ben Edgington:**  Complete BLS12-381 reference. [hackmd.io/@benjaminion/bls12-381](https://hackmd.io/@benjaminion/bls12-381)
* **RareSkills ZK Book — Elliptic Curve chapters:**  [rareskills.io/zk-book](https://www.rareskills.io/zk-book)
* **0xPARC — Elliptic Curves In-Depth:**  Applied cryptography with circom exercises. [learn.0xparc.org](https://learn.0xparc.org/materials/circom/learning-group-1/elliptic-curves/)

### Key Papers

* **"A Survey of Elliptic Curves for Proof Systems" (ePrint 2022/586):**  Comprehensive survey of all ZK-relevant curves. [eprint.iacr.org/2022/586](https://eprint.iacr.org/2022/586.pdf)
* **"Twisted Edwards Curves Revisited" (Hisil et al., 2008):**  The reference for extended coordinates. [eprint.iacr.org/2008/522](https://eprint.iacr.org/2008/522.pdf)
* **"On the Evaluation of Powers and Monomials" (Pippenger, 1980):**  The original MSM algorithm.
* **"PipeMSM: Hardware Acceleration for Multi-Scalar Multiplication" (2022):**  [eprint.iacr.org/2022/999](https://eprint.iacr.org/2022/999.pdf)

### Implementation Libraries (Read the Source)

* **arkworks-rs (Rust):**  Production curves + MSM. [github.com/arkworks-rs/curves](https://github.com/arkworks-rs/curves)
* **gnark-crypto (Go):**  Optimized assembly implementations. [github.com/Consensys/gnark-crypto](https://github.com/Consensys/gnark-crypto)
* **blst (C/assembly):**  Fastest BLS12-381 library. [github.com/supranational/blst](https://github.com/supranational/blst)

---

*Next: [3. Polynomial Arithmetic & Commitments](../3.%20Polynomial%20Arithmetic%20and%20Commitments/Guide.md) — NTT/FFT, KZG commitments, FRI protocol, and the #2 hardware bottleneck*
