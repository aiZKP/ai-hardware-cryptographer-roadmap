# ZK Proof Systems — From Arithmetization to Hardware Acceleration

> **Goal:** Master the internals of modern ZK proof systems — Groth16, PLONK, STARKs, and folding schemes — with a sharp focus on how each system's mathematical structure determines its hardware acceleration requirements. By the end, you will understand why Groth16 provers are MSM-dominated, why STARK provers are NTT-and-hash-dominated, how PLONK sits in between, and what this means for FPGA/GPU/ASIC design decisions. Every section connects proof system internals to concrete computational profiles.

**Prerequisite:** Phase 5.1 (Mathematical Foundations) — finite fields, modular arithmetic, polynomial rings, roots of unity. Phase 5.2 (Elliptic Curve Cryptography) — curve groups, scalar multiplication, bilinear pairings, MSM. Phase 5.3 (Polynomial Arithmetic and Commitments) — NTT/FFT, KZG, IPA, FRI commitment schemes. You must be comfortable with polynomial commitment verification, pairing equations, and the Schwartz-Zippel lemma.

---

## Table of Contents

1. [Arithmetization — How Computation Becomes Constraints](#1-arithmetization--how-computation-becomes-constraints)
   - 1.1 [R1CS (Rank-1 Constraint System)](#11-r1cs-rank-1-constraint-system)
   - 1.2 [AIR (Algebraic Intermediate Representation)](#12-air-algebraic-intermediate-representation)
   - 1.3 [PLONKish Arithmetization](#13-plonkish-arithmetization)
   - 1.4 [Arithmetization and Hardware Implications](#14-arithmetization-and-hardware-implications)
2. [Groth16 — The Gold Standard SNARK](#2-groth16--the-gold-standard-snark)
3. [PLONK — The Universal SNARK](#3-plonk--the-universal-snark)
4. [STARKs — Transparent and Post-Quantum](#4-starks--transparent-and-post-quantum)
5. [Folding Schemes (Nova, SuperNova, HyperNova)](#5-folding-schemes-nova-supernova-hypernova)
6. [Proof System Comparison Table](#6-proof-system-comparison-table)
7. [The Prover Pipeline — What Hardware Accelerates](#7-the-prover-pipeline--what-hardware-accelerates)
8. [Resources](#8-resources)

---

## 1. Arithmetization — How Computation Becomes Constraints

Arithmetization is the process of transforming an arbitrary computation into a system of algebraic constraints over a finite field F_p. This is the foundational step in every ZK proof system: before anything can be proved, the computation must become math.

```
The universal pipeline:
──────────────────────
  Program / Statement
        |
        v
  Arithmetization   ← THIS section
  (R1CS, AIR, or PLONKish)
        |
        v
  Polynomial equations over F_p
        |
        v
  Polynomial commitment + proof generation
        |
        v
  Succinct proof π
```

The choice of arithmetization directly determines:
- Which polynomial commitment scheme is natural to use
- Which mathematical operations dominate the prover
- What hardware architecture is optimal for acceleration

---

### 1.1 R1CS (Rank-1 Constraint System)

R1CS is the arithmetization used by Groth16, Marlin, Spartan, and Nova. It is the most widely deployed arithmetization in production SNARKs.

#### Format

An R1CS instance consists of three matrices A, B, C in F_p^{m x n} and a witness vector s in F_p^n such that:

```
(A · s) ⊙ (B · s) = (C · s)

where:
  ⊙ = element-wise (Hadamard) product
  A, B, C ∈ F_p^{m × n}   (m constraints, n variables)
  s ∈ F_p^n                (witness vector: [1, public_inputs..., private_inputs..., intermediate_wires...])

Each row i of the matrices defines one constraint:
  (∑_j A_{ij} · s_j) × (∑_j B_{ij} · s_j) = (∑_j C_{ij} · s_j)
```

Key insight: each constraint captures exactly ONE multiplication gate. Addition gates are "free" — they are absorbed into the linear combinations A·s, B·s, C·s. This is why R1CS is called "Rank-1": each constraint has rank 1 (one multiplication).

#### How Gates Map to Constraints

```
Multiplication gate:  x * y = z
  Row in A: coefficient 1 at position of x
  Row in B: coefficient 1 at position of y
  Row in C: coefficient 1 at position of z

Addition gate:  x + y = z
  NOT a separate constraint! Folded into the linear combination of the
  next multiplication constraint that uses z.

Constant multiplication:  5 * x = z
  Row in A: coefficient 5 at position of x (or at the "1" position)
  Row in B: coefficient 1 at position of x (or a trivial 1)
  Row in C: coefficient 1 at position of z
  (Can often be folded into surrounding constraints)
```

#### Concrete Example: Proving x^3 + x + 5 = 35

Claim: "I know x such that x^3 + x + 5 = 35." The solution is x = 3.

**Step 1: Flatten the computation into simple statements**

```
Original:   out = x^3 + x + 5

Flattened:
  sym_1 = x * x        (compute x^2)
  y     = sym_1 * x    (compute x^3)
  sym_2 = y + x        (compute x^3 + x)
  out   = sym_2 + 5    (compute x^3 + x + 5)
```

**Step 2: Define the witness vector**

```
s = [1, x, out, sym_1, y, sym_2]
  = [1, 3, 35,  9,     27, 30]

where:
  s[0] = 1     (constant)
  s[1] = 3     (x, the private input)
  s[2] = 35    (out, the public output)
  s[3] = 9     (sym_1 = x^2)
  s[4] = 27    (y = x^3)
  s[5] = 30    (sym_2 = x^3 + x)
```

**Step 3: Write the R1CS constraints**

We need one constraint per multiplication (additions are absorbed):

```
Constraint 1: sym_1 = x * x
  A: [0, 1, 0, 0, 0, 0]    (select x)
  B: [0, 1, 0, 0, 0, 0]    (select x)
  C: [0, 0, 0, 1, 0, 0]    (select sym_1)

  Check: (0·1 + 1·3 + 0·35 + 0·9 + 0·27 + 0·30) = 3
       × (0·1 + 1·3 + 0·35 + 0·9 + 0·27 + 0·30) = 3
       = 9
       = (0·1 + 0·3 + 0·35 + 1·9 + 0·27 + 0·30) = 9  ✓

Constraint 2: y = sym_1 * x
  A: [0, 0, 0, 1, 0, 0]    (select sym_1)
  B: [0, 1, 0, 0, 0, 0]    (select x)
  C: [0, 0, 0, 0, 1, 0]    (select y)

  Check: 9 × 3 = 27  ✓

Constraint 3: out = sym_2 + 5  →  rearrange as:  (sym_2 + 5) * 1 = out
  A: [5, 0, 0, 0, 0, 1]    (select 5·1 + sym_2)
  B: [1, 0, 0, 0, 0, 0]    (select 1, trivial multiplication)
  C: [0, 0, 1, 0, 0, 0]    (select out)

  Check: (5·1 + 30) × 1 = 35  ✓

Note: sym_2 = y + x is NOT a separate constraint.
It is absorbed: wherever sym_2 appears in A, B, or C,
we can substitute (y + x) directly. Alternatively, constraint 3's
A row could select y and x: [5, 1, 0, 0, 1, 0], eliminating sym_2.
```

**Constraint count:** 3 constraints for this computation (one per multiplication).

#### Number of Constraints for Common Operations

```
Operation                    # R1CS Constraints
─────────────────────────────────────────────────
Addition (a + b)             0 (free — absorbed into linear combinations)
Multiplication (a * b)       1
Squaring (a^2)               1
Exponentiation (a^n)         ~log₂(n) (via square-and-multiply)
Division (a / b)             1 (prove a = b * c for witness c)
Boolean check (b ∈ {0,1})   1 (b * (1-b) = 0)
Comparison (a < b), 254-bit ~254 (decompose into bits + check each)
SHA-256 hash                 ~25,000
Poseidon hash                ~250-300
ECDSA verification           ~10,000-15,000
Merkle proof (depth d)       ~d × hash_cost
```

The implication for hardware: R1CS-based systems like Groth16 convert these constraints into polynomial evaluations and then perform MSM over the structured reference string. The number of constraints directly determines the size of the MSMs and NTTs.

---

### 1.2 AIR (Algebraic Intermediate Representation)

AIR is the arithmetization used by STARKs (StarkWare, Winterfell, Plonky3). It is fundamentally different from R1CS: instead of modelling individual gates, AIR models the step-by-step execution trace of a computation.

#### Execution Trace

The execution trace is a matrix T of field elements with dimensions (T_rows x w), where each row represents one step of the computation, and each column (register) tracks a specific value over time.

```
Execution trace structure:
──────────────────────────

         Register 0    Register 1    Register 2    ...    Register w-1
Step 0:  t[0,0]        t[0,1]        t[0,2]        ...    t[0,w-1]
Step 1:  t[1,0]        t[1,1]        t[1,2]        ...    t[1,w-1]
Step 2:  t[2,0]        t[2,1]        t[2,2]        ...    t[2,w-1]
  ...
Step n:  t[n,0]        t[n,1]        t[n,2]        ...    t[n,w-1]

Each column is interpolated into a polynomial:
  f_j(ω^i) = t[i,j]   for all steps i = 0..n-1

where ω is a primitive n-th root of unity.
```

#### Transition Constraints

Transition constraints are polynomial equations that must hold between consecutive rows of the trace. They enforce that each step of the computation follows from the previous step correctly.

```
Transition constraint format:
────────────────────────────

  P(t[i,0], t[i,1], ..., t[i,w-1],  t[i+1,0], t[i+1,1], ..., t[i+1,w-1]) = 0

  for all i = 0, 1, ..., n-2

In polynomial form (using trace polynomials f_j):
  P(f_0(x), f_1(x), ..., f_{w-1}(x),  f_0(ω·x), f_1(ω·x), ..., f_{w-1}(ω·x)) = 0

  for all x in the trace domain {ω^0, ω^1, ..., ω^{n-2}}
```

The key insight: f_j(omega * x) gives the value of register j in the NEXT row, because omega shifts from step i to step i+1.

#### Boundary Constraints

Boundary constraints fix specific values at specific positions in the trace (typically the first or last row).

```
Boundary constraint format:
──────────────────────────

  f_j(ω^k) = v     (register j at step k must equal v)

Examples:
  f_0(ω^0) = 1     (register 0 at step 0 is 1)
  f_1(ω^{n-1}) = 34  (register 1 at the last step is 34)
```

#### Concrete Example: Fibonacci Sequence

Prove: "The 8th Fibonacci number is 21."

**Trace definition:** Two registers, 8 rows.

```
        Reg 0 (a)    Reg 1 (b)
Step 0:    1             1
Step 1:    1             2
Step 2:    2             3
Step 3:    3             5
Step 4:    5             8
Step 5:    8            13
Step 6:   13            21
Step 7:   21            34
```

**Transition constraints** (must hold for rows i = 0..6):

```
1.  f_1(ω·x) - f_0(ω·x) - f_1(x) = 0
    Meaning: b_{next_row} = a_{next_row} + b_{current_row} is wrong;
    Actually:
      f_0(ω·x) = f_1(x)           →  a_{i+1} = b_i
      f_1(ω·x) = f_0(x) + f_1(x)  →  b_{i+1} = a_i + b_i

Constraint polynomial 1:  f_0(ω·x) - f_1(x) = 0
Constraint polynomial 2:  f_1(ω·x) - f_0(x) - f_1(x) = 0

These must vanish on the set {ω^0, ω^1, ..., ω^6} (all steps except the last).
```

**Boundary constraints:**

```
f_0(ω^0) = 1     (a starts at 1)
f_1(ω^0) = 1     (b starts at 1)
f_1(ω^7) = 34    (b at the last step is the 9th Fibonacci number)
                  or equivalently: f_0(ω^7) = 21 (the 8th Fibonacci number)
```

**Constraint enforcement:** The prover constructs quotient polynomials by dividing out the vanishing polynomial:

```
For transition constraints:
  q_trans(x) = constraint_polynomial(x) / Z_T(x)

  where Z_T(x) = (x - ω^0)(x - ω^1)...(x - ω^6) = ∏_{i=0}^{6} (x - ω^i)

  If the constraint holds on all relevant rows, the division is exact
  and q_trans is a polynomial (not a rational function).

For boundary constraints:
  q_bound(x) = (f_j(x) - v) / (x - ω^k)
```

#### Comparison: AIR vs R1CS

```
Feature              R1CS                           AIR
─────────────────────────────────────────────────────────────────────────────
Unit of computation  Individual gates               Execution steps (rows)
Constraint type      One multiplication per row      Arbitrary polynomial
                                                     relations between
                                                     consecutive rows
Additions            Free (absorbed)                 Explicitly constrained
Trace structure      Flat witness vector             2D table (time × regs)
Natural fit          SNARKs (Groth16, Marlin)        STARKs (FRI-based)
Constraint degree    Always degree 2                 Can be higher degree
Repetitive structure Not exploited                   Heavily exploited
                                                     (same constraint at
                                                     every row)
```

---

### 1.3 PLONKish Arithmetization

PLONKish is the arithmetization used by PLONK, Halo2 (Zcash), and many modern SNARKs. It generalizes R1CS by supporting custom gates and lookup tables.

#### The Gate Equation

The basic PLONK gate is defined by selector polynomials over a circuit with n gates:

```
PLONK gate equation (for gate i):
──────────────────────────────────

  q_L(i) · a(i)  +  q_R(i) · b(i)  +  q_O(i) · c(i)  +  q_M(i) · a(i)·b(i)  +  q_C(i) = 0

where:
  a(i), b(i), c(i)    = left, right, output wire values at gate i
  q_L, q_R, q_O       = selector polynomials for linear terms
  q_M                  = selector for multiplication term
  q_C                  = constant selector

This single equation can encode:
  Addition:       q_L=1, q_R=1, q_O=-1, q_M=0, q_C=0    →   a + b - c = 0
  Multiplication: q_L=0, q_R=0, q_O=-1, q_M=1, q_C=0    →   a·b - c = 0
  Constant:       q_L=1, q_R=0, q_O=0,  q_M=0, q_C=-5   →   a - 5 = 0
  Bool check:     q_L=-1, q_R=0, q_O=0, q_M=1, q_C=0    →   a·b - a = 0
                  (with copy constraint a = b: a^2 - a = 0)
```

#### Copy Constraints via Permutation Argument

PLONK's key innovation is how it connects gates. Instead of repeating values in the witness, it uses a permutation argument:

```
Copy constraint mechanism:
─────────────────────────

Problem: Gate 3's output must equal Gate 7's left input.
  We need c(3) = a(7).

Solution: Define a permutation σ over all wire positions:
  Positions: {a(0), b(0), c(0), a(1), b(1), c(1), ..., a(n-1), b(n-1), c(n-1)}

  For each equality constraint, create a cycle in σ.
  Example: c(3) = a(7) means σ maps the position of c(3) to the position of a(7)
           and vice versa (forming a cycle).

Enforcement via grand product:
  Define accumulator polynomial Z(X) such that:

  Z(ω^0) = 1    (initial value)

  Z(ω^{i+1}) = Z(ω^i) · ∏_{j∈{a,b,c}} (f_j(ω^i) + β·id_j(ω^i) + γ)
                         ─────────────────────────────────────────────────
                         ∏_{j∈{a,b,c}} (f_j(ω^i) + β·σ_j(ω^i) + γ)

  Z(ω^{n-1}) = 1    (grand product equals 1 iff permutation is satisfied)

  where β, γ are random challenges, id_j assigns unique identifiers,
  and σ_j encodes the permutation.
```

If all copy constraints hold, the numerator and denominator products are equal (just reordered), so Z cycles back to 1.

#### Custom Gates (TurboPLONK)

TurboPLONK extends PLONK with higher-degree custom gates:

```
Custom gate example (range check — prove 0 ≤ a < 4):
──────────────────────────────────────────────────────
  a · (a - 1) · (a - 2) · (a - 3) = 0

This is a degree-4 constraint. In standard PLONK (degree 2), this would
require decomposition into multiple gates. With custom gates, it is ONE gate.

Custom gate for Poseidon S-box (x^5):
  q_5 · a^5 + q_O · c = 0     (degree-5 gate, one constraint)

Standard PLONK would need: t1 = a*a, t2 = t1*t1, c = t2*a → 3 constraints.
```

#### Lookup Tables (Plookup / UltraPLONK)

Lookup tables allow the prover to prove that a value belongs to a predefined table, without computing the relationship arithmetically:

```
Lookup argument (Plookup):
─────────────────────────

Given a table T = {t_0, t_1, ..., t_{d-1}} of allowed values,
prove that wire value f(ω^i) ∈ T.

Example: XOR table for 4-bit values
  Table: {(0,0,0), (0,1,1), (1,0,1), (1,1,0)}

  Instead of decomposing XOR into ~32 constraints per bit,
  a single lookup proves the XOR relationship.

Mechanism:
  1. Sort the combined (lookup_values ∪ table) vector
  2. Prove the sorted vector satisfies a "difference is in table" check
  3. Uses a grand product argument (similar to permutation checks)

Selector q_lu = 1 when a row uses a lookup gate.
```

#### Comparison: PLONKish vs R1CS

```
Feature              R1CS                    PLONKish
──────────────────────────────────────────────────────────────────
Gate type            Fixed: one mult per      Configurable: custom
                     constraint               gates of any degree
Wiring               Part of A, B, C         Separate permutation
                     matrices                 argument
Lookup support       No (must decompose)     Yes (Plookup)
Selector flexibility Fixed structure          Arbitrary selector
                                              polynomials
Constraint degree    Always 2                Configurable (2-8+)
SHA-256 cost         ~25,000 constraints     ~3,000-5,000 (with
                                              lookups + custom gates)
Poseidon cost        ~250-300                ~50-100 (with x^5 gate)
```

---

### 1.4 Arithmetization and Hardware Implications

The choice of arithmetization directly determines the dominant hardware operations:

```
Arithmetization → Proof System → Dominant Operations → Hardware Needs
─────────────────────────────────────────────────────────────────────

R1CS  →  Groth16     →  MSM (~70%), NTT (~20%), Field Arith (~10%)
                        → Needs: massive parallel scalar-point multiplication
                        → Hardware: wide MSM engines, elliptic curve units
                        → Memory: O(n) curve points for SRS

R1CS  →  Spartan     →  Sumcheck (~60%), MSM (~30%), other (~10%)
                        → Needs: efficient multilinear polynomial evaluation
                        → Hardware: field multipliers, MSM units

AIR   →  STARKs      →  NTT (~60%), Hashing (~30%), Field Arith (~10%)
                        → Needs: fast NTT butterflies, hash engines
                        → Hardware: NTT pipeline, Poseidon/Keccak cores
                        → Memory: O(n log n) for NTT, hash trees

PLONKish → PLONK/    →  MSM (~50%), NTT (~40%), Field Arith (~10%)
           Halo2        → Needs: both MSM and NTT engines
                        → Hardware: balanced MSM + NTT pipeline
                        → Memory: O(n) SRS + O(n) for NTT domains
```

---

## 2. Groth16 — The Gold Standard SNARK

Groth16 (Jens Groth, 2016) produces the smallest proofs and fastest verification of any general-purpose SNARK. It remains the most widely deployed system for on-chain verification (Zcash, Tornado Cash, Filecoin, many L2s).

### 2.1 Trusted Setup — The Ceremony

Groth16 requires a **circuit-specific trusted setup** that generates a Structured Reference String (SRS). This setup involves "toxic waste" — random values that must be destroyed.

```
Trusted setup parameters (toxic waste):
───────────────────────────────────────
  τ (tau)  — random field element for polynomial evaluation point
  α (alpha) — binds left/right/output wires
  β (beta)  — binds left/right/output wires
  γ (gamma) — separates public inputs from private
  δ (delta) — separates private witness from verification

These are sampled randomly, used to compute the SRS, then DESTROYED.
If any party knows τ, α, β, γ, δ, they can forge proofs.
```

**Two-phase setup:**

```
Phase 1: Powers of Tau (universal, reusable)
────────────────────────────────────────────
  Compute and publish:
    In G₁: [τ⁰]₁, [τ¹]₁, [τ²]₁, ..., [τⁿ]₁
    In G₂: [τ⁰]₂, [τ¹]₂, [τ²]₂, ..., [τⁿ]₂

  This is a multi-party ceremony (MPC):
    Participant 1: picks τ₁, computes [τ₁^i]
    Participant 2: picks τ₂, computes [(τ₁·τ₂)^i] from participant 1's output
    ...
    Security: only ONE participant needs to be honest (destroy their τ_k).

Phase 2: Circuit-specific (must redo for each circuit)
──────────────────────────────────────────────────────
  Using the R1CS matrices A, B, C for the specific circuit, compute:

  Proving key (pk):
    [α]₁, [β]₁, [β]₂, [δ]₁, [δ]₂
    {[τⁱ]₁}_{i=0}^{n-1}                          (powers of tau in G₁)
    {[(β·A_j(τ) + α·B_j(τ) + C_j(τ))/δ]₁}        (for private wires j)
    {[(β·A_j(τ) + α·B_j(τ) + C_j(τ))/γ]₁}        (for public wires j)
    {[τⁱ·t(τ)/δ]₁}_{i=0}^{n-2}                   (for quotient polynomial)

  Verification key (vk):
    [α]₁, [β]₂, [γ]₂, [δ]₂
    {[(β·A_j(τ) + α·B_j(τ) + C_j(τ))/γ]₁}        (for public inputs only)

  t(τ) = ∏(τ - ω^i) is the vanishing polynomial evaluated at τ.
```

**Limitation:** If the circuit changes (even by one constraint), Phase 2 must be completely redone. This is why Groth16 is NOT universal.

### 2.2 Prover Algorithm

The Groth16 prover takes the circuit (R1CS), the SRS, and the witness, and outputs a proof of 3 group elements.

```
Groth16 Prover — Step by Step:
──────────────────────────────

Input: R1CS (A, B, C), witness s, proving key pk

Step 1: Compute witness polynomial evaluations
  Evaluate A(x) = ∑_j s_j · A_j(x) on the domain
  Evaluate B(x) = ∑_j s_j · B_j(x) on the domain
  Evaluate C(x) = ∑_j s_j · C_j(x) on the domain

Step 2: Compute the quotient polynomial h(x)
  h(x) = (A(x) · B(x) - C(x)) / Z_H(x)

  where Z_H(x) = x^n - 1 (vanishing polynomial on domain H)

  This requires:
    a. IFFT to get coefficient form of A, B, C        → 3 IFFTs of size n
    b. Evaluate A, B, C on a coset (2n or 4n points)  → 3 FFTs of size 2n
    c. Multiply A·B pointwise on the coset
    d. Subtract C pointwise
    e. Divide by Z_H (pointwise on coset)
    f. IFFT to get h(x) in coefficient form            → 1 IFFT of size 2n

Step 3: Compute proof elements via MSM
  Choose random blinding factors r, s ∈ F_p

  [A]₁ = [α]₁ + ∑_j s_j · [A_j(τ)]₁ + r·[δ]₁         ← MSM in G₁ (size n)

  [B]₂ = [β]₂ + ∑_j s_j · [B_j(τ)]₂ + s·[δ]₂          ← MSM in G₂ (size n)
  [B]₁ = [β]₁ + ∑_j s_j · [B_j(τ)]₁ + s·[δ]₁          ← MSM in G₁ (size n)

  [C]₁ = ∑_{j=ℓ+1}^{m} s_j · [(β·A_j(τ) + α·B_j(τ) + C_j(τ))/δ]₁
         + ∑_i h_i · [τⁱ·t(τ)/δ]₁                       ← MSM in G₁ (size ~2n)
         + r·[B]₁ + s·[A]₁ - r·s·[δ]₁

Output: π = ([A]₁, [B]₂, [C]₁)

Total dominant operations:
  5 MSMs (4 in G₁, 1 in G₂) of size ≈ n each
  3 IFFTs + 3 FFTs of size n (or 2n for coset evaluation)
```

### 2.3 Proof Structure

```
Groth16 proof on BN254:
───────────────────────
  [A]₁ ∈ G₁     →  2 × 32 bytes = 64 bytes  (uncompressed)
                    1 × 32 bytes = 32 bytes  (compressed, point compression)
  [B]₂ ∈ G₂     →  2 × 64 bytes = 128 bytes (uncompressed)
                    1 × 64 bytes = 64 bytes  (compressed)
  [C]₁ ∈ G₁     →  2 × 32 bytes = 64 bytes  (uncompressed)
                    1 × 32 bytes = 32 bytes  (compressed)

  Total uncompressed: 256 bytes
  Total compressed:   128 bytes

  This is the SMALLEST possible proof for a general-purpose SNARK.
  No other pairing-based system achieves fewer than 3 group elements.
```

### 2.4 Verifier Algorithm

The Groth16 verifier is the fastest of any general-purpose SNARK:

```
Groth16 Verifier:
─────────────────

Input: proof π = ([A]₁, [B]₂, [C]₁), public inputs x₁...xₗ, verification key vk

Step 1: Compute public input contribution (MSM over public inputs)
  [I]₁ = ∑_{i=0}^{ℓ} x_i · [vk_i]₁      ← MSM in G₁ (size ℓ, typically small)

  where vk_i = [(β·A_i(τ) + α·B_i(τ) + C_i(τ))/γ]₁ from the verification key

Step 2: Check pairing equation
  e([A]₁, [B]₂) = e([α]₁, [β]₂) · e([I]₁, [γ]₂) · e([C]₁, [δ]₂)

  Equivalently (more efficient — single multi-pairing check):
  e([A]₁, [B]₂) · e(-[I]₁, [γ]₂) · e(-[C]₁, [δ]₂) = e([α]₁, [β]₂)

  The right side e([α]₁, [β]₂) can be precomputed and stored in vk.

Verification cost:
  1 MSM of size ℓ (number of public inputs, usually small: 1-10)
  3 pairing computations (or 1 multi-pairing)
  Total: ~1-3 ms on modern CPU
  On-chain (EVM): ~270,000 gas (BN254 precompile)
```

### 2.5 Why Groth16 Is Smallest and Fastest to Verify

```
Theoretical optimality:
──────────────────────
  Groth16 achieves the MINIMUM possible proof size for a pairing-based
  SNARK in the generic group model:
    - 2 G₁ elements + 1 G₂ element = 3 elements total
    - No other system achieves fewer while maintaining soundness

  The verification requires only:
    - 3 pairing operations (one can be precomputed)
    - 1 small MSM over public inputs
    - No polynomial evaluations, no hash computations

  Compare to:
    PLONK:  ~10-15 G₁ elements, requires KZG opening verification
    Marlin: ~18 G₁ elements, multiple pairing checks
    STARK:  No pairings but ~50-200 KB of hash-based proof data
```

### 2.6 Limitations

```
1. Circuit-specific setup: New ceremony required for EACH circuit.
   → Operational burden: Zcash Sapling ceremony took months, hundreds of participants
   → If you change the circuit (bug fix, upgrade), you need a new ceremony

2. Not universal: Cannot reuse SRS across different circuits.
   → PLONK solves this with universal SRS

3. Trusted setup security: If ALL ceremony participants collude (or a single
   participant in a non-MPC setup), they can forge proofs.
   → STARKs eliminate this with transparent setup

4. Not post-quantum: Relies on discrete log and pairing assumptions.
   → STARKs are post-quantum (hash-based)

5. Fixed circuit size: The circuit must be fully determined at setup time.
   → No conditional branching, loops must be unrolled to max bound
```

### 2.7 Hardware Profile

```
Groth16 prover computation breakdown (circuit size n):
──────────────────────────────────────────────────────

  Operation           % of Prover Time    Count              Hardware Need
  ─────────────────────────────────────────────────────────────────────────
  MSM (G₁)            ~50-60%             4 MSMs, size n     Parallel point
                                                              addition engines
  MSM (G₂)            ~10-15%             1 MSM, size n      G₂ point arith
                                                              (4x costlier)
  NTT/IFFT            ~20-30%             3 IFFT + 3 FFT     Butterfly units,
                                          (size n to 2n)     large memory BW
  Field arithmetic    ~5-10%              O(n) mults          Field multipliers
  ─────────────────────────────────────────────────────────────────────────

  At small n (< 2^18):  MSM dominates (~70-80%)
  At large n (> 2^24):  NTT grows to dominate (up to 91%)
  Reason: MSM is O(n), NTT is O(n log n)

  Memory: O(n) elliptic curve points for SRS (~96 bytes/point for BN254 G₁)
    n = 2^20: ~100 MB for SRS
    n = 2^24: ~1.6 GB for SRS
    n = 2^28: ~25 GB for SRS

  Key insight for hardware:
    → Groth16 ASICs/FPGAs need LARGE on-chip SRAM or HBM for SRS storage
    → MSM parallelism: each scalar-point multiplication is independent
    → NTT butterfly stages have data dependencies but high parallelism within stages
    → G₂ arithmetic uses extension field F_{p²}, roughly 4x cost of G₁
```

---

## 3. PLONK — The Universal SNARK

PLONK (Permutations over Lagrange-bases for Oecumenical Noninteractive arguments of Knowledge — Gabizon, Williamson, Ciobotaru, 2019) introduced the first practical universal SNARK: a single trusted setup supports ALL circuits up to a given size.

### 3.1 Universal and Updatable SRS

```
PLONK SRS (powers of tau only — NOT circuit-specific):
──────────────────────────────────────────────────────

  SRS = {[τ⁰]₁, [τ¹]₁, [τ²]₁, ..., [τⁿ]₁,  [τ]₂}

  - Only powers of τ in G₁ and [τ]₂ in G₂
  - NO circuit-specific information baked in
  - ANY circuit with ≤ n gates can use this SRS
  - Updatable: new participants can strengthen the ceremony
    without seeing previous τ values

  Compare to Groth16 SRS:
    Groth16: [α]₁, [β]₂, {[(β·A_j(τ) + α·B_j(τ) + C_j(τ))/δ]₁}_j, ...
    → Contains circuit-specific A_j, B_j, C_j baked into curve points
    → CANNOT be reused for a different circuit

  PLONK advantage: One ceremony → unlimited circuits
  PLONK cost:      Larger proofs, slower verification
```

### 3.2 The Gate Equation (Recap)

```
q_L(X)·a(X) + q_R(X)·b(X) + q_O(X)·c(X) + q_M(X)·a(X)·b(X) + q_C(X) = 0

All of q_L, q_R, q_O, q_M, q_C are preprocessed (fixed for a given circuit).
a(X), b(X), c(X) are the witness polynomials (provided by the prover).
```

### 3.3 Permutation Argument

```
The permutation argument enforces copy constraints:
───────────────────────────────────────────────────

Given 3n wire positions (n gates × 3 wires each), define:
  - id(X): identity permutation polynomial
  - σ(X): the actual permutation encoding copy constraints

Build the grand product polynomial Z(X) over domain H = {1, ω, ω², ..., ω^{n-1}}:

  Z(1) = 1

  Z(ω^{i+1}) = Z(ω^i) · (a_i + β·ω^i + γ)(b_i + β·k₁·ω^i + γ)(c_i + β·k₂·ω^i + γ)
                          ─────────────────────────────────────────────────────────────────
                          (a_i + β·σ_a(ω^i) + γ)(b_i + β·σ_b(ω^i) + γ)(c_i + β·σ_c(ω^i) + γ)

  Z(ω^{n-1}) should cycle back to 1.

If all copy constraints are satisfied:
  - The numerator and denominator products are permutations of each other
  - Their ratio telescopes to 1
  - Z remains well-defined and returns to 1

k₁, k₂ are distinct coset generators that separate the wire namespaces:
  Wire a positions: {ω^0, ω^1, ..., ω^{n-1}}
  Wire b positions: {k₁·ω^0, k₁·ω^1, ..., k₁·ω^{n-1}}
  Wire c positions: {k₂·ω^0, k₂·ω^1, ..., k₂·ω^{n-1}}
```

### 3.4 The PLONK Prover — Step by Step

The PLONK prover operates in 5 rounds, producing commitments and evaluations at each round:

```
Round 1: Commit to wire polynomials
──────────────────────────────────
  1. Compute witness values a(ω^i), b(ω^i), c(ω^i) for all gates i
  2. Add random blinding factors (for zero-knowledge)
  3. Interpolate to get polynomials a(X), b(X), c(X)
  4. Compute KZG commitments: [a(τ)]₁, [b(τ)]₁, [c(τ)]₁   ← 3 MSMs of size n

  Send: [a]₁, [b]₁, [c]₁ to verifier

  Verifier responds with challenges β, γ (via Fiat-Shamir in practice)

Round 2: Commit to permutation polynomial
─────────────────────────────────────────
  1. Compute Z(X) as the grand product accumulator (see permutation argument above)
  2. Compute KZG commitment [z(τ)]₁                       ← 1 MSM of size n

  Send: [z]₁ to verifier

  Verifier responds with challenge α

Round 3: Commit to quotient polynomial
──────────────────────────────────────
  1. Compute the full constraint polynomial t(X) that must vanish on H:

     t(X) · Z_H(X) = gate_constraint(X)
                    + α · permutation_constraint_1(X)
                    + α² · permutation_constraint_2(X)

     where:
       gate_constraint = q_L·a + q_R·b + q_O·c + q_M·a·b + q_C
       perm_constraint_1 = (a + β·X + γ)(b + β·k₁·X + γ)(c + β·k₂·X + γ)·z(X)
                         - (a + β·σ_a + γ)(b + β·σ_b + γ)(c + β·σ_c + γ)·z(ωX)
       perm_constraint_2 = (z(X) - 1) · L₁(X)    (Z starts at 1)

  2. Divide by Z_H(X) to get t(X)
     - Requires evaluation on a coset (NTT on coset of size 4n or 8n)
     - This is the most expensive step computationally

  3. Split t(X) = t_lo(X) + X^n · t_mid(X) + X^{2n} · t_hi(X)
  4. Commit: [t_lo]₁, [t_mid]₁, [t_hi]₁                  ← 3 MSMs of size n

  Send: [t_lo]₁, [t_mid]₁, [t_hi]₁

  Verifier responds with evaluation challenge ζ (zeta)

Round 4: Compute evaluations (opening)
──────────────────────────────────────
  Evaluate the following polynomials at ζ:
    ā = a(ζ),  b̄ = b(ζ),  c̄ = c(ζ)
    s̄_σ₁ = σ_a(ζ),  s̄_σ₂ = σ_b(ζ)
    z̄_ω = z(ζ·ω)     (shifted evaluation for permutation check)

  Send: ā, b̄, c̄, s̄_σ₁, s̄_σ₂, z̄_ω

  Verifier responds with challenge v

Round 5: Compute linearization and opening proofs
─────────────────────────────────────────────────
  1. Construct the linearization polynomial r(X):
     - Replace known evaluations (ā, b̄, c̄, etc.) with their values
     - Leave committed polynomials in symbolic form
     - This avoids the verifier needing to compute products of commitments

  2. Compute KZG opening proofs:
     W_ζ(X) = (combined_poly(X) - combined_eval) / (X - ζ)    ← 1 MSM
     W_{ζω}(X) = (z(X) - z̄_ω) / (X - ζ·ω)                   ← 1 MSM

  Send: [W_ζ]₁, [W_{ζω}]₁

Total proof:
  9 G₁ points: [a]₁, [b]₁, [c]₁, [z]₁, [t_lo]₁, [t_mid]₁, [t_hi]₁, [W_ζ]₁, [W_{ζω}]₁
  6 field elements: ā, b̄, c̄, s̄_σ₁, s̄_σ₂, z̄_ω
```

### 3.5 Proof Size and Verification

```
PLONK proof size (BN254):
─────────────────────────
  9 G₁ points × 64 bytes = 576 bytes (uncompressed)
                × 32 bytes = 288 bytes (compressed)
  6 field elements × 32 bytes = 192 bytes

  Total: ~768 bytes uncompressed, ~480 bytes compressed
  (Some variants: ~400-900 bytes depending on optimizations)

  Compare: Groth16 = 128-256 bytes (2-3x smaller)

PLONK verification:
──────────────────
  1. Reconstruct the linearization commitment from evaluations and commitments
  2. Two KZG pairing checks (batched into one):
     e([W_ζ]₁ + u·[W_{ζω}]₁, [τ]₂) = e(ζ·[W_ζ]₁ + u·ζ·ω·[W_{ζω}]₁ + [F]₁ - [E]₁, [1]₂)

  Cost:
    - 1 MSM in G₁ (size ~10-15 for combining commitments)
    - 2 pairings (or 1 multi-pairing with batching)
    - ~10-20 field multiplications

  Total: ~5-15 ms on modern CPU
  On-chain (EVM): ~300,000 gas

  Compare: Groth16 verification = ~1-3 ms, ~270,000 gas (faster but not dramatically)
```

### 3.6 PLONK Variants

```
Variant          Key Innovation                          Impact
────────────────────────────────────────────────────────────────────────────

TurboPLONK       Custom gates (degree > 2)               Fewer constraints for
                 e.g., q·a·b·c, Poseidon x^5 gate       non-linear operations

UltraPLONK       Lookup tables (Plookup)                 SHA-256: 25k → 3-5k
                 + custom gates from TurboPLONK          constraints. Range
                                                          checks become trivial.

HyperPLONK       Multilinear polynomial commitments      No FFT needed!
                 over boolean hypercube {0,1}^n           Linear-time prover.
                 Supports high-degree custom gates        Uses sumcheck protocol.
                 without affecting prover time.

Halo2            IPA commitments (no trusted setup)       Recursive proof
(Zcash)          UltraPLONK arithmetization               composition without
                                                          pairing cycles.

fflonk           Single evaluation point optimization     Proof = 1 G₁ point
                                                          + field elements.
                                                          Even smaller proofs.
```

### 3.7 Hardware Profile

```
PLONK prover computation breakdown (circuit size n):
───────────────────────────────────────────────────

  Operation           % of Prover Time    Count              Hardware Need
  ─────────────────────────────────────────────────────────────────────────
  MSM (G₁)            ~40-50%             ~8-10 MSMs,        Parallel point
                                          size n each        addition engines
  NTT/IFFT            ~35-45%             Multiple NTTs      Butterfly units,
                                          of size 4n-8n      very high memory BW
                                          (coset evals)
  Field arithmetic    ~5-10%              O(n) mults         Field multipliers
  Polynomial eval     ~3-5%               6 evaluations      Sequential, fast
  ─────────────────────────────────────────────────────────────────────────

  Key differences from Groth16:
    → More MSMs (8-10 vs 5) but each is simpler (only G₁, no G₂)
    → LARGER NTTs (4n or 8n vs 2n) due to quotient polynomial degree
    → NTT is a larger fraction because quotient poly computation
      requires multiple domain conversions

  Memory: O(n) for SRS + O(n) for witness polynomials + O(4n) for coset NTTs
    n = 2^20: ~150-200 MB
    n = 2^24: ~2-3 GB
    n = 2^28: ~40+ GB

  Key insight for hardware:
    → PLONK hardware needs BOTH strong MSM and strong NTT engines
    → Unlike Groth16, PLONK can trade MSM cost for NTT cost
      (e.g., computing commitments in Lagrange basis)
    → The quotient polynomial step is the bottleneck:
      it requires the largest NTT and produces the most MSM work
```

---

## 4. STARKs — Transparent and Post-Quantum

STARKs (Scalable Transparent Arguments of Knowledge — Ben-Sasson et al., 2018) eliminate trusted setup entirely and rely only on hash functions, making them post-quantum secure. The trade-off: much larger proofs.

### 4.1 Key Properties

```
STARK properties:
─────────────────
  Transparent:    No trusted setup. Anyone can verify the setup.
                  No "toxic waste" that could compromise soundness.

  Post-quantum:   Security based on collision-resistant hash functions,
                  not discrete log or pairings. Secure against quantum computers.

  Scalable:       Prover time O(n log n), verifier time O(log² n).
                  Quasi-linear prover, polylogarithmic verifier.

  Large proofs:   ~50-200 KB (vs 128-768 bytes for SNARKs).
                  This is the primary disadvantage.

  No elliptic curves: All arithmetic is over small prime fields + hashing.
                      No pairing-friendly curves needed.
```

### 4.2 AIR-Based Arithmetization (Recap)

STARKs use AIR (Section 1.2). The computation is represented as an execution trace with transition and boundary constraints.

### 4.3 Small Fields

STARKs gain massive performance advantages by working over small prime fields instead of the 254-bit fields required by pairing-based SNARKs:

```
Field comparison:
─────────────────

Field         Modulus            Bits   CPU Word Fit   Mul Cost (relative)
───────────────────────────────────────────────────────────────────────────
BN254 (Fr)    ~2^254             254    8 × 32-bit     1.0x (baseline)
BLS12-381     ~2^255             255    8 × 32-bit     ~1.1x
Goldilocks    2^64 - 2^32 + 1   64     1 × 64-bit     ~0.05x (20x faster)
BabyBear      15·2^27 + 1       31     1 × 32-bit     ~0.02x (50x faster)
Mersenne31    2^31 - 1           31     1 × 32-bit     ~0.015x (65x faster)

Why BabyBear (p = 2013265921):
  - Fits in 32-bit word → native CPU/GPU integer ops
  - p - 1 = 15 · 2^27 → has large 2-adic subgroup → efficient NTT
  - Modular reduction: after 32×32→64 bit multiply, reduction is
    just two 32-bit additions (no division)

Why Mersenne31 (p = 2^31 - 1 = 2147483647):
  - Fastest modular arithmetic of any prime
  - Reduction: (a mod p) = (a >> 31) + (a & p), then conditional subtract
  - Used in Circle STARKs (Plonky3, Stwo)
  - BUT: p-1 = 2 × 3 × ... no large power-of-2 factor
    → Cannot do traditional NTT! Need Circle FFT instead

Why Goldilocks (p = 2^64 - 2^32 + 1):
  - Fits in 64-bit word → native CPU ops
  - p - 1 = 2^32 · (2^32 - 1) → huge 2-adic subgroup → very efficient NTT
  - Used by Plonky2, Polygon zkEVM
```

### 4.4 STARK Prover Pipeline

```
STARK Prover — Step by Step:
───────────────────────────

Step 1: Generate execution trace
  Run the computation, recording all register values at each step.
  Result: matrix T of size (n_steps × n_registers)
  Each column becomes a polynomial via interpolation (NTT).

Step 2: Commit to trace polynomials
  For each column j, interpolate trace values to get polynomial f_j(X):
    f_j(ω^i) = T[i][j]   for i = 0..n-1

  Evaluate f_j on a larger domain (blowup factor ρ, typically 4-16):
    Domain: {g^0, g^1, ..., g^{ρ·n-1}} where g is a root of unity of order ρ·n

  Build Merkle tree over the evaluations:
    Leaf_i = (f_0(g^i), f_1(g^i), ..., f_{w-1}(g^i))    (all columns at row i)

  Commit: Send Merkle root R_trace                       ← NTT + hashing

Step 3: Compute constraint polynomials
  For each transition constraint P:
    c_P(X) = P(f_0(X), f_1(X), ..., f_0(ωX), f_1(ωX), ...)

  If the trace is valid, c_P vanishes on the constraint domain.

Step 4: Compute quotient polynomial (composition polynomial)
  q(X) = ∑_P α_P · c_P(X) / Z_C(X)

  where Z_C(X) = ∏_{i in constraint_domain} (X - ω^i) is the vanishing polynomial
  and α_P are random combiners from the verifier (Fiat-Shamir).

  If all constraints hold, q(X) is a polynomial (exact division).
  Degree of q: deg(constraint) · n - n = (d-1) · n   for degree-d constraints.

Step 5: Commit to quotient polynomial via FRI
  Evaluate q(X) on the extended domain.
  Build Merkle tree, send root R_quotient.

Step 6: DEEP-ALI (Domain Extending for Eliminating Pretenders)
  The verifier samples a random point z₀ OUTSIDE the evaluation domain.
  The prover sends evaluations:
    f_j(z₀), f_j(ω·z₀)   for all columns j
    q(z₀)

  This out-of-domain sampling dramatically improves soundness:
    → Without DEEP: soundness per query ≈ 1/8 (need many queries)
    → With DEEP: soundness per query ≈ 1 - √ρ (close to 1)
    → Allows fewer FRI queries → smaller proofs

Step 7: FRI commitment (low-degree testing)
  Prove that the committed polynomials actually have the claimed degree.

  FRI protocol (commit phase):
    Round 0: Start with polynomial p₀(X) of degree < D over domain D₀
    Round 1: Verifier sends random α₀
             Prover folds: p₁(X) = even(p₀)(X) + α₀ · odd(p₀)(X)
             → p₁ has degree < D/2 over domain D₁ = D₀² (halved domain)
             Commit: Merkle root of p₁ evaluations
    Round 2: Verifier sends random α₁
             Prover folds: p₂(X) = even(p₁)(X) + α₁ · odd(p₁)(X)
             → p₂ has degree < D/4 over domain D₂
             Commit: Merkle root of p₂ evaluations
    ...
    Round k: p_k is a constant (degree 0). Prover sends it directly.

  FRI protocol (query phase):
    Verifier picks random indices i₁, i₂, ..., i_q
    For each index, prover opens Merkle paths and shows
    consistency of folding between consecutive rounds.

  Total FRI rounds: log₂(D) rounds of folding
  Total queries: ~20-40 (depending on security parameter)
  Each query: log₂(D) Merkle path openings × hash size
```

### 4.5 Proof Size

```
STARK proof size breakdown:
──────────────────────────

Component                          Size Contribution
─────────────────────────────────────────────────────────────
FRI round commitments              log₂(n) Merkle roots
FRI query responses                q queries × log₂(n) layers ×
                                   Merkle path (log₂(ρ·n) hashes each)
Trace evaluations (DEEP)           O(w) field elements
Constraint evaluations             O(1) field elements
─────────────────────────────────────────────────────────────

Typical sizes:
  n = 2^20, blowup ρ = 4, security λ = 128 bits:
    FRI queries: ~30
    Per query: ~20 hash digests × 32 bytes = ~640 bytes
    Total: ~30 × 640 = ~19 KB for FRI queries alone
    + commitments + DEEP evaluations
    ≈ 50-100 KB total

  n = 2^24:
    ≈ 100-200 KB

  Compare:
    Groth16: 128-256 bytes      (400-1000x smaller)
    PLONK:   400-800 bytes      (100-500x smaller)
```

### 4.6 Verification

```
STARK verification:
──────────────────
  1. Recompute Fiat-Shamir challenges from commitments
  2. Check DEEP evaluations satisfy constraint polynomials at z₀
  3. Verify FRI queries:
     - For each query index i:
       a. Open Merkle paths in trace commitment and FRI round commitments
       b. Check folding consistency between consecutive FRI rounds
       c. Verify final constant matches
  4. Verify Merkle paths are valid (hash computations)

  Cost:
    - ~q × log₂(n) hash computations (Merkle path verification)
    - O(1) field arithmetic per query (folding checks)
    - Total: ~5-50 ms depending on parameters and hash function

  On-chain verification:
    - Expensive due to large proof size and many hash operations
    - Typical: ~500K-5M gas on EVM (without Poseidon precompile)
    - Often cheaper to wrap STARK in Groth16: prove "I verified a STARK"
      → Get STARK's transparency + Groth16's cheap verification
```

### 4.7 Hardware Profile

```
STARK prover computation breakdown (trace size n, w registers):
──────────────────────────────────────────────────────────────

  Operation           % of Prover Time    Count              Hardware Need
  ─────────────────────────────────────────────────────────────────────────
  NTT/IFFT            ~50-60%             O(w) NTTs of       Butterfly units,
                                          size ρ·n           streaming NTT
                                          (trace interp +    architecture
                                          coset evals)
  Hashing             ~25-35%             O(ρ·n·w) leaves    Poseidon/SHA-256
  (Merkle trees)                          + FRI trees         hash engines,
                                          + query paths       pipelined
  Field arithmetic    ~10-15%             O(n) per           Small field
                                          constraint          multipliers
                                                              (32-bit or 64-bit)
  ─────────────────────────────────────────────────────────────────────────

  Key differences from SNARKs:
    → NO MSM (no elliptic curves at all!)
    → NO pairings
    → NTT operates over SMALL fields (32-64 bit) → much faster per element
      but more elements needed (blowup factor ρ = 4-16)
    → Hashing is a major bottleneck (not present in SNARK provers)
    → Poseidon hash is ZK-friendly: ~300 field mults per hash
      SHA-256 would be ~1000x more expensive in-circuit

  Memory:
    Execution trace: n × w field elements (small fields → less memory)
    Extended evaluations: ρ · n × w field elements
    Merkle trees: O(ρ · n) hash digests

    n = 2^20, w = 50, ρ = 4, BabyBear (4 bytes/element):
      Trace: 2^20 × 50 × 4 = ~200 MB
      Extended: 4 × 200 = ~800 MB
      Trees: ~100 MB
      Total: ~1-1.5 GB

  Key insight for hardware:
    → STARK hardware is COMPLETELY DIFFERENT from SNARK hardware
    → No elliptic curve arithmetic needed at all
    → Need: fast NTT butterflies over small fields
    → Need: high-throughput Poseidon hash engines (algebraic hash)
    → Need: Merkle tree construction hardware (sequential dependency at top)
    → Small field arithmetic is trivially parallelized on GPUs (32-bit native)
    → FPGAs can pipeline Poseidon rounds extremely efficiently
```

---

## 5. Folding Schemes (Nova, SuperNova, HyperNova)

Folding schemes represent a fundamentally different approach to recursive proof composition: instead of proving each step and then aggregating proofs, they "fold" multiple instances into one, deferring all proving to the end.

### 5.1 IVC (Incrementally Verifiable Computation)

```
IVC concept:
───────────

  Given a long computation:  z₀ → F → z₁ → F → z₂ → ... → F → z_n

  Goal: produce a proof π_n that the entire chain is correct,
        with constant-size proof and constant verification time,
        updating π incrementally at each step.

  Traditional approach (recursive SNARKs):
    At step i, prove: "z_i = F(z_{i-1}) AND π_{i-1} is valid"
    → Each step requires a FULL SNARK proof (expensive!)
    → Prover cost per step: O(|F| · log|F|) at best

  Nova approach (folding):
    At step i, FOLD the current instance with the running instance.
    → No proof generated until the very end!
    → Prover cost per step: O(|F|) — just ONE MSM
    → Only at the end: one single SNARK proof over the folded instance
```

### 5.2 Nova: R1CS + Relaxed R1CS

```
Nova's key insight: Relaxed R1CS
────────────────────────────────

Standard R1CS:
  Az ⊙ Bz = Cz

Relaxed R1CS (introduces error term):
  Az ⊙ Bz = u · Cz + E

  where:
    u ∈ F_p       (scalar, u=1 for standard R1CS)
    E ∈ F_p^m     (error vector, E=0 for standard R1CS)

  Any standard R1CS instance is a special case with u=1, E=0.

Folding two instances:
  Given:
    Instance 1: (A·z₁ ⊙ B·z₁ = u₁·C·z₁ + E₁)  with committed witness (W₁, E₁)
    Instance 2: (A·z₂ ⊙ B·z₂ = u₂·C·z₂ + E₂)  with committed witness (W₂, E₂)

  The verifier sends random challenge r ∈ F_p.

  Folded instance:
    z  = z₁ + r · z₂
    u  = u₁ + r · u₂
    E  = E₁ + r · T + r² · E₂     (T is a "cross term" computed by prover)
    W  = W₁ + r · W₂

  The folded instance (z, u, E) satisfies relaxed R1CS:
    A·z ⊙ B·z = u · C·z + E

  The prover's work to fold:
    1. Compute cross term T = Az₁ ⊙ Bz₂ + Az₂ ⊙ Bz₁ - u₁·Cz₂ - u₂·Cz₁
    2. Commit to T: [T] = Commit(T)                    ← 1 MSM of size m
    3. Linear combinations of z, E, W                   ← O(n) field ops

  Total prover crypto cost per step: ONE MSM of size |circuit|
  Compare to recursive SNARK: full prover (many MSMs + NTTs) per step
```

### 5.3 Nova IVC Architecture

```
Nova IVC — step by step:
───────────────────────

Step 0:
  Running instance: (u₀, x₀, W₀, E₀) = initial instance (standard R1CS: u=1, E=0)

Step i (for i = 1, 2, ..., n):
  1. Prover computes F(z_{i-1}) = z_i (the actual computation)
  2. Create new R1CS instance for step i: Instance_i
  3. FOLD Instance_i into running instance:
     Running ← Fold(Running, Instance_i, challenge r_i)
     → Cost: 1 MSM for commitment to cross-term T

Step n (final):
  4. Prove the final folded instance using a standard SNARK (e.g., Spartan)
     → This is ONE proof, regardless of how many steps n were folded

Total cost:
  Per step:  1 MSM of size |circuit|  +  O(|circuit|) field arithmetic
  Final:     1 SNARK proof of size |circuit|

  Compare recursive SNARK IVC:
    Per step:  Full SNARK prover (~5-10 MSMs + NTTs)
    Savings:   Nova is ~10-100x cheaper per step
```

### 5.4 SuperNova

```
SuperNova extends Nova to non-uniform computation:
──────────────────────────────────────────────────

Nova limitation: Every step must execute the SAME function F.
  → If you have conditional logic (if/else), you must include
    ALL branches in F and select via multiplexing.

SuperNova: At each step i, the prover can choose a function F_j from a set {F_1, F_2, ..., F_k}.

  → Maintains k running instances (one per function type)
  → At step i, fold the new instance into the appropriate running instance
  → Only the "active" function F_j incurs the folding cost

  Use case: Virtual machines with multiple opcodes
    → Each opcode is a different F_j
    → Only fold the constraints for the executed opcode
    → Much more efficient than padding all opcodes into one giant circuit
```

### 5.5 HyperNova

```
HyperNova: Folding for CCS (Customizable Constraint Systems)
────────────────────────────────────────────────────────────

CCS generalizes R1CS, AIR, and PLONKish:
  CCS instance: (M₁, ..., M_t, S₁, ..., S_q, c₁, ..., c_q)

  Satisfied when:  ∑_{i=1}^{q} c_i · ⊙_{j ∈ S_i} M_j · z = 0

  R1CS is CCS with: q=1, S₁={1,2,3}, c₁=1, M₁=A, M₂=B, M₃=-C
    → A·z ⊙ B·z ⊙ (-C·z) = A·z ⊙ B·z - C·z = 0

  AIR is CCS with: appropriate shifted matrices for next-row access
  PLONKish is CCS with: selector matrices for gate types

HyperNova's folding:
  Uses a sum-check protocol to fold CCS instances.
  Prover cost: ONE MSM of size |variables|  (same as Nova)
  But supports any CCS constraint system (not just R1CS).

  → Can fold PLONK circuits, AIR circuits, or custom constraint systems
  → Strictly more general than Nova
  → Same asymptotic cost: O(n) prover work per step
```

### 5.6 Connection to Recursive SNARKs

```
Folding vs. Recursive SNARKs:
────────────────────────────

                    Recursive SNARK          Folding (Nova)
                    ──────────────          ──────────────
Per step cost       Full SNARK prover       1 MSM
                    (MSM + NTT + ...)       (~10-100x cheaper)

Proof at step i     Complete proof           Running commitment
                    (independently           (not a proof until
                     verifiable)              final SNARK step)

Final proof         Any intermediate         Must complete final
                    proof works              SNARK at the end

Verification        Any step is              Only final proof
                    verifiable               is verifiable

Latency             Higher per step          Lower per step,
                                             but delayed proof

Use cases           Blockchain (need         Long computations
                    proofs at each block)    (zkVMs, rollups
                                             with batching)

The trend: Most modern systems (SP1, RISC0) use folding internally
for accumulation, then produce a final SNARK/STARK proof.
```

---

## 6. Proof System Comparison Table

For a circuit of size n = 2^20 (~1 million constraints/gates), on BN254 or equivalent:

```
┌─────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│                     │   Groth16    │    PLONK     │    STARK     │  Nova (IVC)  │  HyperNova   │
├─────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Setup type          │ Circuit-     │ Universal    │ Transparent  │ Universal    │ Universal    │
│                     │ specific     │ (powers of   │ (no setup)   │ (same as     │ (same as     │
│                     │ trusted      │ tau)         │              │ underlying   │ underlying   │
│                     │ setup        │              │              │ SNARK)       │ SNARK)       │
├─────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Arithmetization     │ R1CS         │ PLONKish     │ AIR          │ R1CS +       │ CCS          │
│                     │              │              │              │ Relaxed R1CS │ (general)    │
├─────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Proof size          │ 128-256 B    │ 400-900 B    │ 50-200 KB    │ varies       │ varies       │
│                     │ (smallest)   │              │ (largest)    │ (final SNARK)│ (final SNARK)│
├─────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Prover time         │ ~5-15 s      │ ~10-30 s     │ ~5-20 s      │ ~0.5-2 s     │ ~0.5-2 s     │
│ (n=2^20, CPU)       │              │              │ (small field)│ per step     │ per step     │
├─────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Verification time   │ ~1-3 ms      │ ~5-15 ms     │ ~5-50 ms     │ N/A per step │ N/A per step │
│                     │ (fastest)    │              │              │ (final: as   │ (final: as   │
│                     │              │              │              │ underl. SNARK)│ underl. SNARK│
├─────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ On-chain cost       │ ~270K gas    │ ~300K gas    │ ~500K-5M gas │ depends on   │ depends on   │
│ (EVM)               │              │              │ (or wrap in  │ final proof  │ final proof  │
│                     │              │              │  Groth16)    │ system       │ system       │
├─────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Prover bottleneck   │ MSM (70%)    │ MSM+NTT      │ NTT (60%)   │ MSM (1 per   │ MSM (1 per   │
│                     │ NTT (20%)    │ (50%/40%)    │ Hash (30%)  │ step)        │ step)        │
├─────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Post-quantum?       │ No           │ No           │ Yes          │ No (unless   │ No (unless   │
│                     │ (pairings)   │ (pairings/   │ (hash-based) │ hash-based   │ hash-based   │
│                     │              │  DL)         │              │ PCS)         │ PCS)         │
├─────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Recursion           │ Difficult    │ Natural      │ Natural      │ Built-in     │ Built-in     │
│                     │ (need        │ (accumulate  │ (STARK-in-   │ (folding IS  │ (folding IS  │
│                     │ pairing      │ via IPA/KZG) │ STARK)       │ recursion)   │ recursion)   │
│                     │ cycles)      │              │              │              │              │
├─────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Field size          │ 254-bit      │ 254-bit      │ 31-64 bit    │ 254-bit      │ Flexible     │
│                     │ (BN254,      │ (BN254,      │ (BabyBear,   │ (same as     │              │
│                     │  BLS12-381)  │  BLS12-381)  │ Goldilocks,  │ underl.)     │              │
│                     │              │              │  Mersenne31) │              │              │
├─────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Memory (n=2^20)     │ ~100 MB      │ ~150-200 MB  │ ~1-1.5 GB    │ ~50 MB/step  │ ~50 MB/step  │
│                     │ (SRS-bound)  │              │ (trace+tree) │              │              │
├─────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Key implementations │ arkworks,    │ halo2,       │ winterfell,  │ Nova (MS),   │ HyperNova    │
│                     │ gnark,       │ gnark,       │ plonky3,     │ Arecibo      │ paper (2023) │
│                     │ snarkjs,     │ plonky2      │ stwo,        │ (Lurk)       │              │
│                     │ rapidsnark   │              │ stone        │              │              │
└─────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

---

## 7. The Prover Pipeline — What Hardware Accelerates

### 7.1 Groth16 Prover Pipeline

```
Groth16 Prover Execution Flow:
──────────────────────────────

┌───────────────────────────────────────────────────────────────────────┐
│ Phase 1: Witness Generation                          (~5% of time)   │
│  Input: circuit definition + private inputs                          │
│  Output: full witness vector s = [1, pub..., priv..., intermediates] │
│  Operations: sequential circuit evaluation (field arithmetic)        │
│  Parallelism: limited (topological order dependencies)               │
│  Hardware: CPU-bound, hard to accelerate                             │
└───────────────────────────────────────────────────────────────────────┘
                              │
                              v
┌───────────────────────────────────────────────────────────────────────┐
│ Phase 2: NTT / Domain Conversion                    (~20-30% of time)│
│  Operations:                                                         │
│    3 × IFFT(size n): A·s, B·s, C·s → coefficient form               │
│    3 × FFT(size 2n): evaluate on coset for h(x)                      │
│  Data: n field elements per transform (254-bit each)                 │
│  Memory: 2 × n × 32 bytes ≈ 64 MB for n=2^20                       │
│  Parallelism: HIGH within each butterfly stage                       │
│               LIMITED between stages (data dependencies)             │
│  Bandwidth: ~O(n log n) memory accesses (non-sequential at later     │
│             stages — "bit-reversal" access pattern)                   │
│  Hardware: NTT butterfly units, wide memory bus, on-chip SRAM        │
└───────────────────────────────────────────────────────────────────────┘
                              │
                              v
┌───────────────────────────────────────────────────────────────────────┐
│ Phase 3: Quotient Polynomial h(x)                    (~5% of time)   │
│  Operations: pointwise multiply A·B, subtract C, divide by Z_H      │
│  All in evaluation form (coset domain)                               │
│  Data: 3 × 2n field elements                                         │
│  Parallelism: PERFECT (each point independent)                       │
│  Hardware: field multiplier arrays                                    │
└───────────────────────────────────────────────────────────────────────┘
                              │
                              v
┌───────────────────────────────────────────────────────────────────────┐
│ Phase 4: MSM (Multi-Scalar Multiplication)          (~60-70% of time)│
│  Operations:                                                         │
│    4 MSMs in G₁: [A]₁, [B]₁, [C]₁, and h(τ) terms                  │
│    1 MSM in G₂: [B]₂                                                │
│    Each MSM: ∑_{i} s_i · [P_i]₁  (n scalars × n curve points)      │
│  Data: n scalars (32 bytes each) + n SRS points (64-96 bytes each)  │
│  Memory: ~100 MB SRS for n=2^20 on BN254                            │
│  Parallelism: VERY HIGH (Pippenger/bucket method)                    │
│    - Partition scalars into c-bit windows                             │
│    - For each window: accumulate points into 2^c buckets             │
│    - Reduce buckets via summation                                     │
│    - Combine windows via doubling                                     │
│    - Optimal c ≈ log₂(n) ≈ 20                                       │
│  Bandwidth: sequential read of SRS points (streaming)                │
│  Hardware: elliptic curve point addition/doubling units,              │
│            bucket accumulators, wide integer multipliers (254-bit)    │
└───────────────────────────────────────────────────────────────────────┘
                              │
                              v
┌───────────────────────────────────────────────────────────────────────┐
│ Output: proof π = ([A]₁, [B]₂, [C]₁)  — 128-256 bytes              │
└───────────────────────────────────────────────────────────────────────┘
```

### 7.2 PLONK Prover Pipeline

```
PLONK Prover Execution Flow:
────────────────────────────

┌───────────────────────────────────────────────────────────────────────┐
│ Phase 1: Witness Generation + Assignment              (~5% of time)  │
│  Fill in wire values a(ω^i), b(ω^i), c(ω^i) for all n gates        │
│  Sequential, CPU-bound                                               │
└───────────────────────────────────────────────────────────────────────┘
                              │
                              v
┌───────────────────────────────────────────────────────────────────────┐
│ Phase 2: Wire Polynomial Commitments (Round 1)       (~15% of time)  │
│  3 × IFFT(n): evaluation form → coefficient form                     │
│  3 × MSM(n): commit [a(τ)]₁, [b(τ)]₁, [c(τ)]₁                      │
│  Parallelism: IFFTs parallel to each other, MSMs parallel to each    │
└───────────────────────────────────────────────────────────────────────┘
                              │
                              v
┌───────────────────────────────────────────────────────────────────────┐
│ Phase 3: Permutation Polynomial Z(X) (Round 2)       (~10% of time)  │
│  Compute grand product accumulator Z(X) — sequential scan            │
│  1 × IFFT(n): Z evaluations → coefficients                           │
│  1 × MSM(n): commit [z(τ)]₁                                          │
└───────────────────────────────────────────────────────────────────────┘
                              │
                              v
┌───────────────────────────────────────────────────────────────────────┐
│ Phase 4: Quotient Polynomial t(X) (Round 3)          (~40% of time)  │
│  THIS IS THE BOTTLENECK                                              │
│                                                                       │
│  Sub-steps:                                                           │
│    a. Evaluate all polynomials on coset (NTT of size 4n or 8n)       │
│       → Multiple NTTs, each 4-8x larger than base n                   │
│    b. Compute constraint + permutation polynomials pointwise          │
│    c. Divide by Z_H(X) pointwise                                     │
│    d. IFFT back to coefficient form                                   │
│    e. Split t(X) into 3 degree-n pieces                               │
│    f. 3 × MSM(n): commit [t_lo]₁, [t_mid]₁, [t_hi]₁                │
│                                                                       │
│  NTT count: ~6-10 NTTs of size 4n-8n                                 │
│  MSM count: 3 MSMs of size n                                          │
│  Memory: 4-8 × n × 32 bytes for coset evaluations                    │
│  Parallelism: NTTs dominate, then MSMs                                │
└───────────────────────────────────────────────────────────────────────┘
                              │
                              v
┌───────────────────────────────────────────────────────────────────────┐
│ Phase 5: Evaluations + Opening (Rounds 4-5)          (~15% of time)  │
│  Evaluate polynomials at challenge point ζ: O(n) each, ~6 evals     │
│  Compute linearization polynomial r(X)                                │
│  2 × MSM(n): opening proofs [W_ζ]₁, [W_{ζω}]₁                       │
└───────────────────────────────────────────────────────────────────────┘
                              │
                              v
┌───────────────────────────────────────────────────────────────────────┐
│ Output: proof (9 G₁ points + 6 field elements) — ~400-900 bytes     │
└───────────────────────────────────────────────────────────────────────┘

PLONK operation summary:
  MSM total:  ~8-10 MSMs of size n in G₁        (~40-50% of time)
  NTT total:  ~8-14 NTTs of size n-8n            (~35-45% of time)
  Field ops:  O(n) multiplications               (~5-10% of time)
  Poly eval:  6 evaluations of degree-n polys    (~3-5% of time)
```

### 7.3 STARK Prover Pipeline

```
STARK Prover Execution Flow:
────────────────────────────

┌───────────────────────────────────────────────────────────────────────┐
│ Phase 1: Execution Trace Generation                   (~5% of time)  │
│  Run the computation, record all register values                     │
│  Output: n × w matrix of field elements (SMALL field: 32-64 bit)     │
│  Sequential, CPU-bound (VM execution)                                │
└───────────────────────────────────────────────────────────────────────┘
                              │
                              v
┌───────────────────────────────────────────────────────────────────────┐
│ Phase 2: Trace Polynomial Interpolation               (~15% of time) │
│  w × IFFT(n): interpolate each column to get trace polynomials       │
│  w × FFT(ρ·n): evaluate on extended domain (blowup factor ρ=4-16)   │
│  Field: 32-bit (BabyBear) or 64-bit (Goldilocks)                    │
│  Parallelism: all w columns independent                               │
│  Memory: w × ρ × n field elements                                    │
└───────────────────────────────────────────────────────────────────────┘
                              │
                              v
┌───────────────────────────────────────────────────────────────────────┐
│ Phase 3: Merkle Tree Commitment                      (~15-20% of time│
│  Build Merkle tree over extended trace evaluations                    │
│  Leaves: rows of (f_0(g^i), f_1(g^i), ..., f_{w-1}(g^i))           │
│  Number of leaves: ρ × n                                             │
│  Hash function: Poseidon (ZK-friendly) or Blake3/SHA-256             │
│  Poseidon: ~300 field multiplications per hash call                   │
│  Total hashes: ~2 × ρ × n (tree has ρ×n leaves → 2×ρ×n-1 nodes)    │
│  Parallelism: tree levels are parallel, but each level depends on    │
│               the previous (log₂(ρ·n) sequential steps)              │
│  Hardware: Poseidon hash cores (algebraic, pipelineable)              │
└───────────────────────────────────────────────────────────────────────┘
                              │
                              v
┌───────────────────────────────────────────────────────────────────────┐
│ Phase 4: Constraint Evaluation + Quotient Polynomial  (~20% of time) │
│  Evaluate constraint polynomials on extended domain                   │
│  Compute quotient polynomial: q(X) = ∑ α_i · C_i(X) / Z(X)         │
│  Multiple NTTs for domain conversion                                  │
│  Pointwise field arithmetic for constraint evaluation                 │
│  Commit quotient polynomial (another Merkle tree)                     │
└───────────────────────────────────────────────────────────────────────┘
                              │
                              v
┌───────────────────────────────────────────────────────────────────────┐
│ Phase 5: DEEP-ALI + FRI                              (~30-40% of time│
│  THIS IS THE BOTTLENECK                                              │
│                                                                       │
│  DEEP evaluations: evaluate trace polynomials at out-of-domain point  │
│                                                                       │
│  FRI folding rounds (log₂(D) rounds):                                │
│    Each round:                                                        │
│      a. Split polynomial into even/odd parts                          │
│      b. Combine with random challenge                                 │
│      c. Evaluate on halved domain                                     │
│      d. Build Merkle tree over new evaluations                        │
│    → NTT-like operation at each round                                 │
│    → Hash-heavy (Merkle tree per round)                               │
│                                                                       │
│  FRI query phase:                                                     │
│    Open ~20-40 Merkle paths across all FRI rounds                     │
│    Each path: log₂(ρ·n) hashes                                       │
│                                                                       │
│  Dominant operations: NTT (folding) + hashing (Merkle trees)          │
└───────────────────────────────────────────────────────────────────────┘
                              │
                              v
┌───────────────────────────────────────────────────────────────────────┐
│ Output: proof (commitments + FRI data + DEEP evals) — ~50-200 KB    │
└───────────────────────────────────────────────────────────────────────┘

STARK operation summary:
  NTT total:    ~w+log₂(D) NTTs over small fields    (~50-60% of time)
  Hashing:      ~O(ρ·n·log₂(ρ·n)) hash operations    (~25-35% of time)
  Field arith:  O(n) multiplications per constraint   (~10-15% of time)
  NO MSM. NO pairings. NO elliptic curves.
```

### 7.4 Parallelism Opportunities Summary

```
Operation      Parallelism Type           GPU Fit     FPGA Fit    ASIC Fit
──────────────────────────────────────────────────────────────────────────
MSM            Bucket-level parallel       Excellent   Good        Best
(Pippenger)    (2^c independent buckets)   (CUDA       (limited    (custom
               Window-level parallel       cores)      LUTs for    EC arith
               Point additions independent              EC arith)   units)

NTT            Intra-stage parallel        Good        Excellent   Excellent
(Butterfly)    (n/2 independent butterflies (memory     (pipelined  (streaming
               per stage)                   BW limited  butterfly   NTT)
               Inter-stage sequential       by stride)  stages)
               (log n stages)

Hashing        Fully parallel per leaf     Good        Excellent   Excellent
(Poseidon)     Sequential at tree top      (many       (pipelined  (hash
                                           SM cores)   rounds)     pipeline)

Field          Fully parallel              Excellent   Good        Excellent
Multiply       (SIMD for small fields)     (32-bit     (DSP        (custom
                                           native)     blocks)     ALUs)
──────────────────────────────────────────────────────────────────────────

Memory bandwidth requirements:
  MSM:     Stream through SRS points + random scalar access → HIGH BW
  NTT:     Butterfly pattern → non-sequential access at later stages → VERY HIGH BW
  Hashing: Sequential leaf processing → MODERATE BW
  Field:   Streaming → LOW BW

Key hardware design decision:
  For Groth16: Optimize MSM engine (70% of time, highly parallel)
  For PLONK:   Balance MSM + NTT (50%/40%, both need high bandwidth)
  For STARK:   Optimize NTT + hash (60%/30%, small field advantage)
  For Nova:    Optimize single large MSM (dominates per-step cost)
```

---

## 8. Resources

### 8.1 Original Papers

```
Groth16:
  "On the Size of Pairing-based Non-interactive Arguments"
  Jens Groth, 2016
  https://eprint.iacr.org/2016/260.pdf
  → The original paper defining the 3-element proof

PLONK:
  "Permutations over Lagrange-bases for Oecumenical Noninteractive
   arguments of Knowledge"
  Ariel Gabizon, Zachary J. Williamson, Oana Ciobotaru, 2019
  https://eprint.iacr.org/2019/953.pdf
  → Universal SNARK with permutation argument

STARK:
  "Scalable, transparent, and post-quantum secure computational integrity"
  Eli Ben-Sasson, Iddo Bentov, Yinon Horesh, Michael Riabzev, 2018
  https://eprint.iacr.org/2018/046.pdf
  → The foundational STARK paper

DEEP-FRI:
  "DEEP-FRI: Sampling Outside the Box Improves Soundness"
  Eli Ben-Sasson et al., 2019
  https://eprint.iacr.org/2019/336.pdf
  → Critical improvement to FRI soundness

Nova:
  "Nova: Recursive Zero-Knowledge Arguments from Folding Schemes"
  Abhiram Kothapalli, Srinath Setty, Ioanna Tzialla, 2021
  https://eprint.iacr.org/2021/370.pdf
  → Folding scheme for IVC

HyperNova:
  "HyperNova: Recursive arguments for customizable constraint systems"
  Abhiram Kothapalli, Srinath Setty, 2023
  https://eprint.iacr.org/2023/573.pdf
  → Generalized folding for CCS

Plookup:
  "plookup: A simplified polynomial protocol for lookup tables"
  Ariel Gabizon, Zachary J. Williamson, 2020
  https://eprint.iacr.org/2020/315.pdf
  → Lookup argument for PLONK

Circle STARKs:
  "Exploring circle STARKs"
  Vitalik Buterin, 2024
  https://vitalik.eth.limo/general/2024/07/23/circlestarks.html
  → Circle FFT for Mersenne31 field
```

### 8.2 Textbooks

```
"Proofs, Arguments, and Zero-Knowledge"
  Justin Thaler, 2022
  https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf
  → The comprehensive academic reference. Covers interactive proofs,
    sumcheck, GKR, polynomial commitments, SNARKs, and STARKs.
    Essential reading for understanding the theory.

"A Graduate Course in Applied Cryptography"
  Dan Boneh, Victor Shoup
  https://toc.cryptobook.us/
  → Chapters on zero-knowledge proofs and sigma protocols.
    Good foundation before diving into SNARKs.
```

### 8.3 Tutorials and Blog Series

```
RareSkills ZK Book:
  https://www.rareskills.io/zk-book
  → Excellent practical tutorials covering R1CS, Groth16, PLONK
  → Includes code examples and step-by-step constructions

Vitalik Buterin's Blog Posts:
  "Quadratic Arithmetic Programs: from Zero to Hero"
  https://medium.com/@VitalikButerin/quadratic-arithmetic-programs-from-zero-to-hero-f6d558cea649
  → Classic R1CS/QAP walkthrough with concrete x^3+x+5=35 example

  "An approximate introduction to how zk-SNARKs are possible"
  https://vitalik.eth.limo/general/2021/01/26/snarks.html

  "STARKs, Part I-III"
  https://vitalik.eth.limo/general/2017/11/09/starks_part_1.html

STARK 101 (StarkWare):
  https://starkware.co/stark-101/
  → Hands-on tutorial building a STARK from scratch in Python

Anatomy of a STARK (Alan Szepieniec):
  https://aszepieniec.github.io/stark-anatomy/
  → Deep mathematical walkthrough of STARK construction

Lambda Class Blog:
  https://blog.lambdaclass.com/groth16/
  https://blog.lambdaclass.com/arithmetization-schemes-for-zk-snarks/
  → Clear technical explanations of proof system internals

Paradigm "Hardware Acceleration for Zero Knowledge Proofs":
  https://www.paradigm.xyz/2022/04/zk-hardware
  → Essential reading for hardware engineers entering ZK

Ingonyama Hardware Reviews:
  https://www.ingonyama.com/post/hardware-review-gpus-fpgas-and-zero-knowledge-proofs
  → GPU vs FPGA analysis for ZK prover acceleration

How to PlonK (zkSecurity):
  https://plonk.zksecurity.xyz/
  → Detailed round-by-round PLONK walkthrough
```

### 8.4 Reference Implementations

```
arkworks (Rust) — Multi-proof-system library
  https://github.com/arkworks-rs
  → Implements: Groth16, Marlin, finite fields, elliptic curves
  → Used as backend by many other projects
  → Best for: learning Groth16 internals, field/curve arithmetic

halo2 (Rust) — PLONK/UltraPLONK implementation
  https://github.com/zcash/halo2
  → Implements: PLONKish arithmetization with lookup tables
  → Used by: Zcash, Scroll, Taiko, PSE
  → Best for: learning PLONKish circuit design and IPA commitments

winterfell (Rust) — STARK implementation
  https://github.com/facebook/winterfell
  → Implements: AIR-based STARKs with FRI
  → Used by: Polygon Miden
  → Best for: learning STARK/AIR/FRI from well-documented code

plonky3 (Rust) — Next-generation STARK toolkit
  https://github.com/Plonky3/Plonky3
  → Implements: AIR + FRI over small fields (BabyBear, Mersenne31)
  → Used by: SP1 (Succinct), multiple zkVMs
  → Best for: state-of-the-art STARK proving with Circle STARKs

gnark (Go) — Production-grade prover
  https://github.com/Consensys/gnark
  → Implements: Groth16, PLONK
  → Best for: high-performance proving in Go ecosystem

snarkjs (JavaScript) — Accessible SNARK tooling
  https://github.com/iden3/snarkjs
  → Implements: Groth16, PLONK (with Circom frontend)
  → Best for: learning and prototyping (slower but readable)

rapidsnark (C++) — Optimized Groth16 prover
  https://github.com/iden3/rapidsnark
  → Implements: highly optimized Groth16 prover for Circom circuits
  → Best for: production Groth16 proving performance

Nova (Rust) — Microsoft Research implementation
  https://github.com/microsoft/Nova
  → Implements: Nova folding scheme with Spartan backend
  → Best for: learning folding schemes and IVC
```

---

## Summary for Hardware Engineers

```
If you are designing hardware for ZK:
──────────────────────────────────────

1. First question: Which proof system?
   → Groth16:  Build MSM engines (Pippenger buckets, EC point adders)
   → PLONK:    Build balanced MSM + NTT (both are critical)
   → STARK:    Build NTT pipelines + Poseidon hash engines (no EC needed)
   → Nova/IVC: Build one very fast MSM engine

2. Second question: What field?
   → BN254/BLS12-381 (254-bit): Need wide integer multipliers (8x32-bit)
   → Goldilocks (64-bit):       Native 64-bit CPU/GPU multiplication
   → BabyBear (31-bit):         Native 32-bit, GPU-optimal
   → Mersenne31 (31-bit):       Fastest reduction, needs Circle FFT

3. Third question: What memory architecture?
   → MSM: needs random access to SRS points → HBM or large SRAM
   → NTT: stride-based access pattern → high bandwidth, cache-unfriendly
   → Hashing: sequential → streaming architecture works

4. The meta-trend:
   Modern systems (SP1, RISC0, Starknet) increasingly use:
     STARK (small field, fast prover) → folding/aggregation → Groth16 (small proof, cheap verify)
   This means hardware must eventually support BOTH paradigms.
```
