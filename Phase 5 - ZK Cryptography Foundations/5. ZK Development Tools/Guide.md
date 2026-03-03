# ZK Development Tools — From DSL to Proof to Hardware

> **Goal:** Master the entire ZK development tool ecosystem with a focus on how each tool maps to the computational pipeline you will accelerate in hardware. By the end, you will understand how a high-level circuit description in Circom, Halo 2, Cairo, or Noir compiles down to the NTT/MSM/hash operations that consume FPGA/GPU/ASIC resources. You will be able to choose the right tool for any ZK task, read and write circuits in multiple frameworks, and profile provers to identify hardware bottlenecks. This guide completes Phase 5 (ZK Cryptography Foundations) and bridges directly to Phase 6 (ZK Hardware Acceleration).

**Prerequisite:** Phase 5.1-5.4 complete. You must be comfortable with finite field arithmetic, elliptic curve operations (MSM), polynomial arithmetic (NTT/FFT), polynomial commitment schemes (KZG, IPA, FRI), and the proof systems (Groth16, PLONK, STARKs) that these tools implement.

---

## Table of Contents

1. [The ZK Development Stack — From DSL to Proof](#1-the-zk-development-stack--from-dsl-to-proof)
2. [Circom + snarkjs — The Entry-Level Stack](#2-circom--snarkjs--the-entry-level-stack)
3. [Halo 2 — Production PLONKish Framework](#3-halo-2--production-plonkish-framework)
4. [Cairo + STARK Provers — The STARK Ecosystem](#4-cairo--stark-provers--the-stark-ecosystem)
5. [Noir — The Simplicity-First DSL](#5-noir--the-simplicity-first-dsl)
6. [arkworks — The Rust Cryptographic Library](#6-arkworks--the-rust-cryptographic-library)
7. [zkVMs — General-Purpose ZK Virtual Machines](#7-zkvms--general-purpose-zk-virtual-machines)
8. [Choosing the Right Tool — Decision Framework](#8-choosing-the-right-tool--decision-framework)
9. [Projects](#9-projects)
10. [Resources](#10-resources)

---

## 1. The ZK Development Stack — From DSL to Proof

Every ZK application, no matter how complex, follows the same fundamental pipeline. Understanding this pipeline is critical for hardware engineers because each layer maps to distinct computational workloads.

### 1.1 The Full Pipeline

```
THE ZK DEVELOPMENT STACK — FULL PIPELINE
=========================================

 Layer 5: APPLICATION
 +----------------------------------------------------------+
 |  Smart Contract / dApp / Privacy Protocol / zkBridge     |
 |  (Tornado Cash, Semaphore, zkEmail, zkML, rollup state)  |
 +----------------------------------------------------------+
                          |
                          v
 Layer 4: HIGH-LEVEL LANGUAGE (DSL or zkVM input)
 +----------------------------------------------------------+
 |  Circom    | Cairo   | Noir    | Leo     | Rust/C (zkVM) |
 |  (.circom) | (.cairo)| (.nr)   | (.leo)  | (.rs/.c)      |
 +----------------------------------------------------------+
                          |
                          | Compilation
                          v
 Layer 3: INTERMEDIATE REPRESENTATION / CONSTRAINT SYSTEM
 +----------------------------------------------------------+
 |  R1CS          | AIR             | PLONKish / ACIR       |
 |  (Circom,      | (Cairo/STARK)   | (Halo2, Noir/BB,      |
 |   arkworks)    |                 |  PLONK variants)       |
 +----------------------------------------------------------+
 |                                                          |
 |  Contains:                                               |
 |  - Constraint matrices (A, B, C for R1CS)                |
 |  - Transition constraints (AIR)                          |
 |  - Custom gates + lookup tables (PLONKish)               |
 |  - Witness (private inputs + intermediate values)        |
 +----------------------------------------------------------+
                          |
                          | Polynomial encoding
                          v
 Layer 2: POLYNOMIAL COMMITMENT + PROVER BACKEND
 +----------------------------------------------------------+
 |  Groth16   | PLONK    | STARK/FRI | Marlin | HyperPlonk |
 |  (KZG+     | (KZG or  | (hash-    | (KZG)  | (multilin-  |
 |   pairings)| IPA)     |  based)   |        |  ear)       |
 +----------------------------------------------------------+
 |                                                          |
 |  This is where the HEAVY COMPUTATION happens:            |
 |                                                          |
 |  KZG-based:                                              |
 |    - MSM (multi-scalar multiplication) ~70% of time      |
 |    - NTT (number theoretic transform)  ~30% of time      |
 |                                                          |
 |  STARK/FRI-based:                                        |
 |    - NTT/FFT                           ~40% of time      |
 |    - Merkle hashing (Poseidon/Keccak)  ~40% of time      |
 |    - Polynomial evaluation             ~20% of time      |
 |                                                          |
 |  These are YOUR hardware acceleration targets.           |
 +----------------------------------------------------------+
                          |
                          v
 Layer 1: PROOF OUTPUT
 +----------------------------------------------------------+
 |  proof.json / proof.bin                                  |
 |  Groth16: ~128 bytes (2 G1 + 1 G2 point)                |
 |  PLONK:   ~400-800 bytes                                 |
 |  STARK:   ~50-200 KB (much larger, no trusted setup)     |
 +----------------------------------------------------------+
                          |
                          v
 Layer 0: VERIFIER
 +----------------------------------------------------------+
 |  On-chain Solidity verifier / Off-chain verifier         |
 |  Groth16: ~200K gas (cheapest on-chain)                  |
 |  PLONK:   ~300K gas                                      |
 |  STARK:   expensive on-chain (hence STARK->SNARK wrapper)|
 +----------------------------------------------------------+
```

### 1.2 How Each Layer Relates to Hardware

The key insight for hardware engineers: **you are accelerating Layer 2**. Everything above Layer 2 is software that generates the mathematical objects (polynomials, constraint matrices, witnesses). Everything below Layer 2 is a small, fast verification. Layer 2 is where provers spend 99%+ of their time.

```
HARDWARE ACCELERATION MAP
=========================

Layer 4-3 (DSL -> Constraints):
  CPU-bound, compilation-time only, NOT a proving bottleneck.
  Exception: witness generation can be slow for large circuits.
  snarkjs generates WASM witness generators; rapidsnark uses C++.

Layer 2 (Prover Backend) — YOUR TARGET:
  +-------------------+------------------+--------------------+
  | Operation         | Where it occurs  | Hardware target    |
  +-------------------+------------------+--------------------+
  | MSM               | KZG commit,      | GPU (parallelism), |
  | (multi-scalar     |  Groth16 proof,  | FPGA (pipelining), |
  |  multiplication)  |  PLONK commit    | ASIC (dedicated)   |
  +-------------------+------------------+--------------------+
  | NTT/INTT          | Polynomial mul,  | GPU (butterfly),   |
  | (number theoretic |  domain convert, | FPGA (streaming),  |
  |  transform)       |  STARK LDE,      | ASIC (fixed radix) |
  |                   |  coset eval      |                    |
  +-------------------+------------------+--------------------+
  | Hash functions    | STARK Merkle     | FPGA (Poseidon),   |
  | (Poseidon, Keccak |  trees, FRI      | ASIC (hash core),  |
  |  SHA-256, BLAKE)  |  commitments     | GPU (parallel)     |
  +-------------------+------------------+--------------------+
  | Field arithmetic  | Everything above | All targets; the   |
  | (modular mul/add) | uses this as     | foundation of all  |
  |                   | primitive        | ZK computation     |
  +-------------------+------------------+--------------------+

Layer 1-0 (Proof output + Verification):
  Tiny computation. Verification is O(1) or O(log n).
  Only relevant for on-chain gas cost, not hardware.
```

### 1.3 Three Approaches to ZK Development

```
APPROACH COMPARISON
===================

                    DSL-Based           Library-Based        zkVM-Based
                    ─────────           ─────────────        ──────────
Tools:              Circom, Cairo,      arkworks, halo2,     RISC Zero, SP1,
                    Noir, Leo           bellman, gnark       Valida, zkWASM

Input:              Custom DSL          Rust/Go code         Standard Rust/C
                    (domain-specific    (circuits defined    (normal programs,
                    circuit language)   as library calls)    no circuit logic)

Abstraction:        High                Low-Medium           Very High
                    (constraints are    (you see the         (you write normal
                    implicit in the     constraint system    code; the VM
                    language syntax)    directly)            handles everything)

Control:            Medium              Maximum              Minimum
                    (compiler decides   (you control every   (VM decides all
                    constraint layout)  gate and wire)       constraints)

Performance:        Good for standard   Best possible        Worst per-op
                    circuits            (hand-optimized)     (enormous traces)

Learning curve:     Low-Medium          High                 Very Low
                    (learn the DSL)     (learn ZK math +     (just write Rust)
                                        Rust generics)

HW engineer use:    Prototyping,        Benchmarking,        Understanding
                    learning ZK         custom proof         why hardware
                    concepts            systems, reference   acceleration is
                                        implementations      critical

Constraint system:  Fixed per tool      You choose           Fixed by VM
                    (Circom -> R1CS,    (R1CS, PLONKish,     (usually STARK/AIR)
                    Noir -> ACIR)       custom)
```

### 1.4 The Compilation Pipeline for Each Approach

```
DSL-BASED (Circom example):
  circuit.circom
       |
       v
  circom compiler (Rust binary)
       |
       +---> circuit.r1cs        (binary constraint system)
       +---> circuit.wasm        (witness generator, runs in browser/node)
       +---> circuit.cpp         (witness generator, for rapidsnark)
       +---> circuit.sym         (symbol table for debugging)
       |
       v
  snarkjs / rapidsnark
       |
       +---> Powers of Tau       (universal setup ceremony)
       +---> Phase 2 zkey        (circuit-specific trusted setup)
       |
       v
  proof.json + public.json       (the ZK proof + public signals)


LIBRARY-BASED (Halo 2 example):
  my_circuit.rs
       |
       v
  Rust compiler (circuit is defined via halo2 API calls)
       |
       v
  halo2_proofs::plonk::create_proof()
       |
       +---> PLONKish constraint system (built at compile/run time)
       +---> KZG/IPA polynomial commitments
       +---> Proof bytes
       |
       v
  halo2_proofs::plonk::verify_proof()


zkVM-BASED (RISC Zero example):
  my_program.rs (standard Rust, no ZK knowledge needed)
       |
       v
  RISC-V cross-compiler (target: riscv32im)
       |
       v
  RISC Zero zkVM execution
       |
       +---> Execute program in emulated RISC-V CPU
       +---> Record execution trace (every CPU state transition)
       +---> Execution trace has millions/billions of rows
       |
       v
  STARK prover (over BabyBear field)
       |
       +---> AIR constraints for RISC-V instruction set
       +---> NTT for polynomial operations
       +---> FRI commitment (Merkle hash-heavy)
       |
       v
  STARK proof (optionally wrapped in Groth16 SNARK for on-chain)
```

---

## 2. Circom + snarkjs — The Entry-Level Stack

Circom is the most widely-deployed ZK circuit language. It was created by the iden3 team and has been used in production for Tornado Cash, Semaphore, zkIdentity, Dark Forest, and dozens of other projects. For a hardware engineer, Circom is the fastest path to understanding how high-level circuit descriptions become R1CS constraints and then Groth16 proofs.

### 2.1 The Circom Language

Circom is a domain-specific language (DSL) for defining arithmetic circuits that compile to R1CS. It is not a general-purpose programming language — it has no heap, no dynamic memory, no recursion. Everything reduces to field arithmetic over a prime field (typically BN128 or BLS12-381).

#### Core Concepts

```
CIRCOM LANGUAGE PRIMITIVES
==========================

1. SIGNALS — the wires of the circuit (field elements)
   - input signal:   signal input a;     // provided by prover
   - output signal:  signal output c;    // public output
   - intermediate:   signal mid;         // internal wire

   Signals are IMMUTABLE after assignment. They correspond to
   columns in the R1CS witness vector.

2. VARIABLES — compile-time or runtime helpers (NOT in R1CS)
   - var i = 0;                          // loop counter, etc.
   - Variables can be mutable, used for computation
   - They do NOT generate constraints

3. CONSTRAINTS — the actual R1CS rows
   - ===   constraint only:        a * b === c;
   - <==   assign + constrain:     c <== a * b;
   - <--   assign only (UNSAFE):   c <-- a / b;  // no constraint!

   CRITICAL: <-- does NOT generate a constraint.
   It computes a value for witness generation but adds NO
   constraint to R1CS. You MUST add a separate === constraint.
   This is the #1 source of Circom security bugs.

4. TEMPLATES — parameterized circuit blueprints
   template Multiplier(n) {
       // like a class/function that defines a subcircuit
   }

5. COMPONENTS — instantiated templates
   component mult = Multiplier(4);
   mult.a <== x;
   mult.b <== y;

6. PRAGMA — version declaration
   pragma circom 2.0.0;  // required at top of file

7. MAIN COMPONENT — the circuit entry point
   component main {public [b]} = MyCircuit();
   // declares which inputs are public vs private
```

#### Constraint Operators in Detail

```
THE THREE OPERATORS — UNDERSTANDING THE DIFFERENCE
===================================================

1.  c <== a * b;
    Equivalent to:
      c <-- a * b;    // compute witness value
      c === a * b;    // add R1CS constraint
    This is the SAFE default. Use this whenever possible.

2.  c === a * b;
    Constraint only. Does not assign a value to c.
    Used when c was already assigned via <-- or another <==.

3.  c <-- a * b;
    Assignment only. Computes the value for the witness but
    adds NO constraint. The prover could put ANY value in c.

    DANGER: This is necessary for non-quadratic operations:
      c <-- a / b;     // division is not a native R1CS op
      c * b === a;     // must manually add the constraint

    R1CS only supports constraints of the form:
      (linear_combination) * (linear_combination) = (linear_combination)
    So division, comparison, bit decomposition all require <--
    followed by manual constraints.
```

### 2.2 Circom Code Example: Proving Knowledge of a Multiplication

```circom
// File: multiplier.circom
pragma circom 2.0.0;

// A simple template: prove that a * b = c
template Multiplier2() {
    // Declare signals
    signal input a;      // private input
    signal input b;      // private input
    signal output c;     // public output

    // This single line does two things:
    // 1. Computes c = a * b (witness generation)
    // 2. Adds R1CS constraint: a * b === c
    c <== a * b;
}

// Instantiate the main component
// {public [c]} means c is a public output
component main = Multiplier2();
```

This generates exactly 1 R1CS constraint (one row in the A, B, C matrices):
```
Constraint #1:
  A = [0, 1, 0, 0]    (coefficient 1 at signal 'a')
  B = [0, 0, 1, 0]    (coefficient 1 at signal 'b')
  C = [0, 0, 0, 1]    (coefficient 1 at signal 'c')

Witness vector s = [1, a, b, c]
Check: (A.s) * (B.s) = (C.s)  =>  a * b = c   ✓
```

### 2.3 More Complex Example: Proving Knowledge of a Hash Preimage

```circom
// File: hash_preimage.circom
pragma circom 2.0.0;

include "circomlib/circuits/poseidon.circom";

// Prove: "I know a secret value 'preimage' whose Poseidon hash
//         equals the public 'hash' value."
template HashPreimage() {
    signal input preimage;     // private: the secret
    signal input hash;         // public: the known hash

    // Instantiate the Poseidon hash with 1 input
    component poseidon = Poseidon(1);
    poseidon.inputs[0] <== preimage;

    // Constrain: hash of preimage must equal the public hash
    hash === poseidon.out;
}

component main {public [hash]} = HashPreimage();
```

This circuit uses `circomlib`'s Poseidon implementation, which internally expands to ~300 R1CS constraints (Poseidon is designed to be SNARK-friendly, requiring far fewer constraints than SHA-256 which needs ~25,000+ constraints).

### 2.4 Compilation and snarkjs Workflow

```bash
# STEP 0: Install tools
npm install -g circom snarkjs
# Or install circom from source (Rust):
# git clone https://github.com/iden3/circom.git
# cd circom && cargo build --release

# STEP 1: Compile the circuit
circom multiplier.circom --r1cs --wasm --sym -o build/
#
# Output files:
#   build/multiplier.r1cs     — binary R1CS constraint system
#   build/multiplier_js/      — directory with WASM witness generator
#     multiplier.wasm          — WebAssembly witness calculator
#     generate_witness.js      — Node.js wrapper
#     witness_calculator.js    — WASM interface
#   build/multiplier.sym      — symbol file (signal names <-> indices)

# STEP 2: View circuit info
snarkjs r1cs info build/multiplier.r1cs
# Output: # of constraints, # of private inputs, # of public outputs

snarkjs r1cs print build/multiplier.r1cs build/multiplier.sym
# Output: human-readable constraint equations

# STEP 3: Powers of Tau ceremony (Phase 1 — universal, reusable)
# The power (12 here) means max 2^12 = 4096 constraints
snarkjs powersoftau new bn128 12 pot12_0000.ptau -v
snarkjs powersoftau contribute pot12_0000.ptau pot12_0001.ptau \
    --name="First contribution" -v
# (In production, multiple parties contribute for security)
snarkjs powersoftau prepare phase2 pot12_0001.ptau pot12_final.ptau -v

# STEP 4: Circuit-specific setup (Phase 2 — Groth16 only)
snarkjs groth16 setup build/multiplier.r1cs pot12_final.ptau \
    multiplier_0000.zkey
snarkjs zkey contribute multiplier_0000.zkey multiplier_0001.zkey \
    --name="Contributor 1" -v
snarkjs zkey export verificationkey multiplier_0001.zkey \
    verification_key.json

# STEP 5: Create input file
echo '{"a": "3", "b": "11"}' > input.json

# STEP 6: Generate witness
cd build/multiplier_js
node generate_witness.js multiplier.wasm ../../input.json witness.wtns
cd ../..

# STEP 7: Generate proof
snarkjs groth16 prove multiplier_0001.zkey \
    build/multiplier_js/witness.wtns proof.json public.json

# STEP 8: Verify proof
snarkjs groth16 verify verification_key.json public.json proof.json
# Output: [INFO]  snarkJS: OK!

# STEP 9 (Optional): Generate Solidity verifier
snarkjs zkey export solidityverifier multiplier_0001.zkey verifier.sol
```

### 2.5 circomlib — Standard Library

circomlib is the standard library of pre-built, audited circuit templates:

```
CIRCOMLIB MODULES
=================

Category          Template                  Constraints    Use Case
─────────────────────────────────────────────────────────────────────
Hashing           Poseidon(nInputs)         ~300/input     ZK-friendly hash
                  MiMCSponge(nInputs,       ~300/input     Alternative hash
                    nOutputs, nRounds)
                  Pedersen(n)               ~1500          Curve-based hash

Signatures        EdDSAVerifier()           ~6000          Signature check
                  EdDSAMiMCSpongeVerifier() ~6000          Sig + MiMC hash

Bitwise           Num2Bits(n)               n              Field -> bits
                  Bits2Num(n)               n              Bits -> field
                  Num2BitsNeg(n)            ~n             Negative handling

Comparators       LessThan(n)              ~2n            a < b (n-bit)
                  GreaterThan(n)            ~2n            a > b
                  IsZero()                  2              a == 0
                  IsEqual()                 3              a == b
                  ForceEqualIfEnabled()     1              Conditional eq

Arithmetic        BinSum(n, ops)           varies          Binary addition

Multiplexers      Mux1(), Mux2(), Mux4()  varies          Selection

Merkle Trees      MerkleTreeChecker(levels) ~300*levels   Membership proof
                  (custom, using Poseidon)

EC Operations     BabyAdd()                6              Edwards curve add
                  BabyDbl()                5              Edwards curve dbl
                  EscalarMulFix(n)         ~6n            Fixed-base mul
                  EscalarMulAny(n)         ~6n            Variable-base mul
```

### 2.6 Limitations of Circom

```
CIRCOM LIMITATIONS — CRITICAL FOR HARDWARE ENGINEERS
=====================================================

1. R1CS ONLY
   - No custom gates (every constraint is a * b = c)
   - No lookup tables (expensive bit decomposition)
   - No direct support for PLONKish or AIR
   - Cannot use UltraPlonk features (range checks via lookups)

2. NO NATIVE RECURSION
   - Cannot verify a Groth16 proof inside a Circom circuit efficiently
   - BN128 pairing verification in R1CS requires ~1M constraints
   - Recursive proof composition must use external tools

3. PERFORMANCE
   - snarkjs (JavaScript) is slow: ~10-60 seconds for medium circuits
   - rapidsnark (C++) is 4-10x faster but still CPU-only
   - ICICLE-snark (GPU) is the fastest but requires NVIDIA GPU
   - No native GPU acceleration in the standard toolchain

4. DEVELOPER EXPERIENCE
   - No type system beyond field elements
   - <-- operator is a security footgun
   - Debugging is difficult (symbolic execution, no print statements)
   - Error messages from the compiler can be cryptic

5. FIELD RESTRICTIONS
   - Hardcoded to bn128 or bls12-381 (large ~254-bit primes)
   - Cannot use small fields (BabyBear, Goldilocks, Mersenne31)
   - This means Circom circuits cannot leverage the performance
     gains of small-field STARKs
```

### 2.7 rapidsnark — The Fast C++ Alternative

rapidsnark is a drop-in replacement for snarkjs's proof generation step, written in C++ with Intel/ARM assembly optimizations:

```bash
# Use rapidsnark instead of snarkjs for proof generation
# (witness generation still uses the WASM from circom)

# Build rapidsnark from source
git clone https://github.com/iden3/rapidsnark.git
cd rapidsnark
git submodule update --init --recursive
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Generate proof with rapidsnark (same inputs as snarkjs)
./prover multiplier_0001.zkey witness.wtns proof.json public.json

# Performance comparison (typical, medium circuit ~50K constraints):
#   snarkjs:    ~15 seconds
#   rapidsnark: ~2 seconds
#   ICICLE:     ~0.3 seconds (GPU)
```

### 2.8 When to Use Circom

```
USE CIRCOM WHEN:
  ✓ Learning ZK circuit development for the first time
  ✓ Prototyping a new ZK application idea
  ✓ Building Groth16-based production circuits (smallest proofs)
  ✓ Following established tutorials and documentation
  ✓ Building for ecosystems with existing Circom infrastructure
  ✓ Your circuit complexity is moderate (< 1M constraints)
  ✓ You need browser-based proving (snarkjs + WASM)

DO NOT USE CIRCOM WHEN:
  ✗ You need custom gates or lookup tables (use Halo 2 or Noir)
  ✗ You need STARK-based proofs (use Cairo)
  ✗ You need recursive proof composition (use Halo 2 or Nova)
  ✗ You need to prove general computation (use a zkVM)
  ✗ You need maximum prover performance (use arkworks or GPU provers)
  ✗ Circuit complexity exceeds ~10M constraints efficiently
```

---

## 3. Halo 2 — Production PLONKish Framework

Halo 2 is the most powerful and flexible circuit development framework in production. Developed by the Zcash team for the Orchard shielded protocol, it implements PLONKish arithmetization with custom gates and lookup tables. The PSE (Privacy and Scaling Explorations) fork adds KZG commitment support, making it the backbone of Scroll and Taiko zkEVMs.

### 3.1 Architecture Overview

```
HALO 2 ARCHITECTURE
====================

                    +─────────────────────+
                    |   Your Circuit       |
                    |   (impl Circuit)     |
                    +─────────────────────+
                              |
                    +---------+---------+
                    |                   |
                    v                   v
            configure()          synthesize()
            (define gates,       (assign values
             columns, lookups)    to regions)
                    |                   |
                    v                   v
            ConstraintSystem     Layouter
            (gate definitions,   (floor planner,
             column types,        region allocation)
             lookup args)
                    |                   |
                    +--------+----------+
                             |
                             v
                    PLONKish Constraint System
                    (columns × rows matrix)
                             |
                    +--------+----------+
                    |                   |
                    v                   v
              IPA Backend        KZG Backend
              (Zcash original)   (PSE fork)
              (no trusted setup) (trusted setup,
                                  smaller proofs)
                    |                   |
                    v                   v
              Proof bytes        Proof bytes
              (recursive-        (EVM-verifiable)
               friendly)
```

### 3.2 PLONKish Arithmetization — The Column Model

Unlike R1CS (which uses matrices and a witness vector), PLONKish uses a **table** with typed columns:

```
PLONKISH TABLE STRUCTURE
========================

  Row  | Advice_0 | Advice_1 | Advice_2 | Fixed_0 | Selector | Instance
  ─────+──────────+──────────+──────────+─────────+──────────+──────────
   0   |   a_0    |   b_0    |   c_0    |  const  |    s_0   |  pub_0
   1   |   a_1    |   b_1    |   c_1    |  const  |    s_1   |  pub_1
   2   |   a_2    |   b_2    |   c_2    |  const  |    s_2   |
   3   |   a_3    |   b_3    |   c_3    |  const  |    s_3   |
  ...  |   ...    |   ...    |   ...    |  ...    |   ...    |   ...
  2^k  |          |          |          |         |          |

Column types:
  ADVICE    — private witness values (prover fills these)
  FIXED     — constants known at circuit definition time
  INSTANCE  — public inputs (verifier knows these)
  SELECTOR  — binary flags that activate/deactivate gates per row

Custom gate example (multiplication gate):
  s_mul(row) * (advice_0(row) * advice_1(row) - advice_2(row)) = 0

  When s_mul = 1: enforces a * b = c
  When s_mul = 0: constraint is trivially satisfied

Key advantage over R1CS:
  - Multiple operations per row (not limited to one multiplication)
  - Custom gates can encode complex expressions
  - Lookup tables replace expensive bit-decomposition
  - Relative references: gates can reference adjacent rows
    e.g., advice_0(row) + advice_0(row + 1) = advice_0(row + 2)
    This enables efficient sequential computations (Fibonacci, etc.)
```

### 3.3 The Circuit API

Halo 2 circuits in Rust implement the `Circuit` trait:

```rust
use halo2_proofs::{
    circuit::{AssignedCell, Layouter, SimpleFloorPlanner, Value},
    plonk::{
        Advice, Circuit, Column, ConstraintSystem, Error,
        Fixed, Instance, Selector,
    },
    poly::Rotation,
    dev::MockProver,
    pasta::Fp,  // Pallas curve field (for IPA backend)
};

// Step 1: Define the circuit configuration (columns + gates)
#[derive(Debug, Clone)]
struct MulConfig {
    advice: [Column<Advice>; 2],    // two advice columns
    instance: Column<Instance>,      // public input column
    s_mul: Selector,                 // multiplication gate selector
}

// Step 2: Define the "chip" — implements the circuit logic
struct MulChip {
    config: MulConfig,
}

impl MulChip {
    fn construct(config: MulConfig) -> Self {
        MulChip { config }
    }

    fn configure(
        meta: &mut ConstraintSystem<Fp>,
    ) -> MulConfig {
        // Create columns
        let advice = [meta.advice_column(), meta.advice_column()];
        let instance = meta.instance_column();
        let s_mul = meta.selector();

        // Enable equality constraints (for copy constraints)
        meta.enable_equality(advice[0]);
        meta.enable_equality(advice[1]);
        meta.enable_equality(instance);

        // Define the custom gate: s_mul * (a * b - out) = 0
        meta.create_gate("mul", |meta| {
            let s = meta.query_selector(s_mul);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let out = meta.query_advice(advice[0], Rotation::next());

            // When s_mul is active: a * b must equal out
            vec![s * (a * b - out)]
        });

        MulConfig { advice, instance, s_mul }
    }

    fn assign_mul(
        &self,
        mut layouter: impl Layouter<Fp>,
        a: Value<Fp>,
        b: Value<Fp>,
    ) -> Result<AssignedCell<Fp, Fp>, Error> {
        layouter.assign_region(
            || "multiply",
            |mut region| {
                // Enable the multiplication selector on row 0
                self.config.s_mul.enable(&mut region, 0)?;

                // Assign a to advice[0], row 0
                region.assign_advice(
                    || "a", self.config.advice[0], 0, || a,
                )?;

                // Assign b to advice[1], row 0
                region.assign_advice(
                    || "b", self.config.advice[1], 0, || b,
                )?;

                // Assign a*b to advice[0], row 1 (the output)
                let out = a * b;  // Value<Fp> multiplication
                region.assign_advice(
                    || "a*b", self.config.advice[0], 1, || out,
                )
            },
        )
    }
}

// Step 3: Define the top-level circuit
#[derive(Default)]
struct MulCircuit {
    a: Value<Fp>,
    b: Value<Fp>,
}

impl Circuit<Fp> for MulCircuit {
    type Config = MulConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> MulConfig {
        MulChip::configure(meta)
    }

    fn synthesize(
        &self,
        config: MulConfig,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let chip = MulChip::construct(config);

        // Perform the multiplication and get the output cell
        let out = chip.assign_mul(
            layouter.namespace(|| "mul"),
            self.a,
            self.b,
        )?;

        // Expose the output as a public instance
        layouter.constrain_instance(
            out.cell(), chip.config.instance, 0,
        )?;

        Ok(())
    }
}

// Step 4: Test with MockProver
fn main() {
    let a = Fp::from(3);
    let b = Fp::from(7);
    let out = a * b;  // = 21

    let circuit = MulCircuit {
        a: Value::known(a),
        b: Value::known(b),
    };

    // k = 4 means 2^4 = 16 rows available
    let prover = MockProver::run(4, &circuit, vec![vec![out]])
        .unwrap();

    // Verify all constraints pass
    prover.assert_satisfied();
}
```

### 3.4 Lookup Tables — The PLONKish Superpower

Lookup tables are what make PLONKish dramatically more efficient than R1CS for many operations:

```
LOOKUP TABLE EXAMPLE: RANGE CHECK
==================================

Problem: Prove that a value x is in [0, 255] (fits in 8 bits).

R1CS approach (Circom):
  - Decompose x into 8 bits: b0, b1, ..., b7
  - Constrain each bit: bi * (bi - 1) = 0  (8 constraints)
  - Reconstruct: x = b0 + 2*b1 + 4*b2 + ... + 128*b7 (1 constraint)
  - Total: 9 constraints, 8 auxiliary signals

PLONKish with lookup (Halo 2):
  - Pre-define a table column with values [0, 1, 2, ..., 255]
  - Add one lookup argument: x ∈ table
  - Total: 1 lookup (amortized cost ~1-2 constraints)
  - No auxiliary signals needed!

Savings per range check: ~4x fewer constraints, ~8x fewer signals.
For a circuit with thousands of range checks (common in zkEVMs),
this translates to MASSIVE constraint reduction.

Other lookup uses:
  - Bitwise operations (XOR, AND tables)
  - Byte-level operations
  - Hash function S-boxes
  - Elliptic curve point validation
```

### 3.5 IPA vs KZG — Two Commitment Backends

```
HALO 2 COMMITMENT SCHEME COMPARISON
====================================

                        IPA (Inner Product      KZG (Kate-Zaverucha-
                        Argument)               Goldberg)
                        ──────────────          ────────────────────
Repository:             zcash/halo2             privacy-scaling-
                                                explorations/halo2

Trusted setup:          NONE                    Required
                        (transparent)           (universal ceremony)

Proof size:             Larger                  Smaller
                        (~10-20 KB)             (~1-5 KB)

Verification:           O(n) (but amortized     O(1) with pairings
                        via accumulation)

Recursion:              Native (accumulation    Possible but harder
                        scheme)

EVM verification:       Expensive               Cheap (~300K gas)

Used by:                Zcash Orchard           Scroll zkEVM
                                                Taiko zkEVM
                                                PSE projects

Hardware profile:       Curve operations,       MSM-heavy,
                        multi-scalar mul        pairing operations

For HW engineers:       Less MSM-heavy;         MSM is THE bottleneck;
                        more recursive-          this is what GPUs/
                        friendly                 FPGAs accelerate
```

### 3.6 Cost Model for Hardware Engineers

```
HALO 2 COST MODEL
==================

The "cost" of a Halo 2 circuit is determined by:

  Total cells = num_rows × num_columns
  Proving time ∝ total_cells × log(total_cells)  [for NTT]
                + num_columns × total_rows        [for MSM/commit]

Key parameters:
  k (circuit size):   2^k rows available. Larger k = slower proving.
  num_advice_cols:    More columns = wider circuit = more MSM work.
  num_fixed_cols:     Constants. Committed once during keygen.
  gate_degree:        Higher degree = fewer rows but larger polys.
  num_lookups:        Each lookup adds ~2 columns of overhead.

Rule of thumb for proving time:
  k=14 (16K rows):  ~0.5-2 seconds
  k=18 (256K rows): ~5-20 seconds
  k=22 (4M rows):   ~60-300 seconds

MockProver is essential for development:
  - Checks every constraint without generating a real proof
  - Reports which cells violate which constraints
  - Runs in milliseconds even for large circuits
  - Always test with MockProver before real proving
```

### 3.7 Who Uses Halo 2

```
HALO 2 IN PRODUCTION
=====================

Zcash Orchard:     IPA-based, recursive proof composition for
                   shielded transactions. The original use case.

Scroll zkEVM:      KZG-based (PSE fork), full EVM-equivalent
                   zero-knowledge rollup. One of the largest
                   Halo 2 deployments.

Taiko:             Type 1 zkEVM using Halo 2 + KZG.

PSE (Privacy &     Multiple projects: zkEVM, Semaphore v4,
Scaling            Bandada, zkID. The team maintains the
Explorations):     KZG fork of halo2.

Axiom:             halo2-lib and halo2-base libraries for
                   simplified circuit development on top of Halo 2.
                   halo2-ecc for elliptic curve operations.
```

### 3.8 Limitations

```
HALO 2 LIMITATIONS
===================

1. STEEP LEARNING CURVE
   - The API is verbose: 150+ lines for a simple multiplication
   - Must understand PLONKish arithmetization deeply
   - Chip/Region/Layouter abstraction is non-trivial
   - Rust generics + trait system adds complexity

2. RUST-ONLY
   - No JavaScript, Python, or other language bindings
   - Full Rust toolchain required
   - Compile times can be long (5-15 minutes for large circuits)

3. DEVELOPMENT SPEED
   - Complex circuits take weeks-months to implement
   - Compare: equivalent Noir circuit might take days
   - Debugging constraint failures requires understanding
     the column layout

4. API INSTABILITY
   - Multiple forks (Zcash, PSE, Axiom) with diverging APIs
   - No single "standard" version
   - Migration between forks requires significant refactoring
```

---

## 4. Cairo + STARK Provers — The STARK Ecosystem

Cairo (CPU Algebraic Intermediate Representation) is the programming language for STARK-based proofs. Unlike Circom and Halo 2 (which target SNARK proof systems), Cairo targets STARKs, which means transparent setup (no trusted ceremony), post-quantum security assumptions, and a fundamentally different computational profile that is hash-heavy and NTT-heavy rather than MSM-heavy.

### 4.1 Cairo Language Evolution

```
CAIRO VERSION COMPARISON
========================

                    Cairo 0                 Cairo 1 / Sierra
                    ─────────               ──────────────────
Syntax:             Python-like             Rust-like
                    func main():            fn main() {
                      [ap] = 5                let x: felt252 = 5;
                      ap += 1                }

Type system:        Minimal (felt only)     Rich (felt252, u8, u16,
                                            u32, u64, u128, bool,
                                            Array, Dict, etc.)

Safety:             Low (manual AP/FP       High (ownership, borrow
                    management)             checker, no dangling refs)

Compilation:        Cairo 0 -> CASM        Cairo 1 -> Sierra -> CASM
                    (direct)                (two-stage, safe IR)

Memory model:       Write-once felt         Write-once felt252
                    (immutable after        (same fundamental model,
                     first write)           but higher-level abstractions)

In production:      StarkEx, early          Starknet (current),
                    Starknet                all new development

Status:             Legacy (maintenance     Active development
                    only)
```

### 4.2 Cairo's Execution Model

```
CAIRO VM ARCHITECTURE
=====================

  Cairo Source (.cairo)
       |
       v
  Sierra (Safe Intermediate Representation)
       |
       v
  CASM (Cairo Assembly)
       |
       v
  CairoVM Execution
       |
       +---> REGISTERS: Only 3 registers
       |       pc  (program counter)
       |       ap  (allocation pointer — next free memory cell)
       |       fp  (frame pointer — current function's base)
       |
       +---> MEMORY: Write-once model
       |       Address space: felt252 -> felt252 mapping
       |       Once cell[addr] = v, it is IMMUTABLE
       |       No overwrites, no deletes
       |       This is key for STARK-friendliness:
       |         Memory consistency = simple algebraic constraint
       |
       +---> EXECUTION TRACE:
       |       Every step records (pc, ap, fp, memory_accesses)
       |       Trace is a matrix: rows = steps, cols = state
       |       This trace IS the witness for the STARK proof
       |
       +---> BUILT-IN FUNCTIONS:
               range_check    — prove value in [0, 2^128)
               pedersen       — Pedersen hash
               poseidon       — Poseidon hash
               ecdsa          — ECDSA signature verification
               bitwise        — bitwise operations
               ec_op          — elliptic curve operations

               Built-ins are NOT software functions.
               They are dedicated AIR columns with specialized
               constraints, like hardware co-processors.
```

### 4.3 From Cairo Program to STARK Proof

```
CAIRO -> AIR -> STARK PIPELINE
===============================

1. EXECUTION
   Cairo program runs in the CairoVM.
   Every instruction produces a row in the execution trace.

   Trace table (simplified):
   Step | pc  | ap  | fp  | op0 | op1 | res | dst
   ─────+─────+─────+─────+─────+─────+─────+─────
     0  | 0   | 100 | 100 | 5   | 3   | 15  | 15
     1  | 2   | 101 | 100 | 15  | 7   | 22  | 22
     2  | 4   | 102 | 100 | 22  | 1   | 23  | 23
    ...

2. AIR CONSTRAINTS
   The trace must satisfy transition constraints:
   For each row i:
     - Instruction decoding: flags are binary, mutually exclusive
     - Operation constraint: res = op0 * op1 (or op0 + op1, etc.)
     - Memory consistency: sorted memory access permutation argument
     - Built-in constraints: range check, hash, etc.

   These are polynomial equations over the trace:
     P(trace[i], trace[i+1]) = 0 for all i

3. LOW-DEGREE EXTENSION (LDE)
   Trace columns are interpolated as polynomials.
   Evaluated on a larger domain (blowup factor, typically 4-16x).

   THIS IS WHERE NTT HAPPENS:
     - Forward NTT: trace values -> polynomial coefficients
     - Coset evaluation: coefficients -> extended domain values
     - This is the single most expensive operation in STARK proving

4. CONSTRAINT COMPOSITION
   Transition constraint polynomials are combined into
   a single composition polynomial using random challenges.

5. FRI COMMITMENT
   The composition polynomial is committed using FRI
   (Fast Reed-Solomon Interactive Oracle Proof):
     - Build Merkle trees over polynomial evaluations
     - Hash operations dominate (Poseidon or Keccak)
     - Multiple rounds of folding (each halves the degree)

   THIS IS WHERE HASHING HAPPENS:
     - Merkle tree construction: O(n) hash operations
     - FRI query responses: O(log n) paths per query
     - Total hashing: ~40% of STARK prover time

6. STARK PROOF OUTPUT
   Contains: FRI commitments, query responses, Merkle paths
   Size: ~50-200 KB (much larger than Groth16's 128 bytes)
```

### 4.4 Stone Prover — StarkWare's Original

Stone is StarkWare's C++ STARK prover that has been in production since 2020:

```
STONE PROVER
=============

Language:       C++
Field:          Large prime field (251-bit Stark field)
Hash:           Keccak-256 for Merkle trees
Performance:    ~100-1000 ms for typical Starknet transactions
Status:         Replaced by Stwo on Starknet mainnet (late 2025)
Open source:    Yes (github.com/starkware-libs/stone-prover)

Computational profile:
  - NTT/IFFT: ~40% of prover time (large-field, 251-bit)
  - Hashing:  ~35% of prover time (Keccak Merkle trees)
  - Memory:   ~25% (trace storage, polynomial evaluation)

Hardware implication:
  Large-field (251-bit) arithmetic is EXPENSIVE on standard
  hardware. 251-bit multiply requires multiple 64-bit multiplies.
  This is a primary motivation for moving to small fields.
```

### 4.5 Stwo Prover — Next-Generation Circle STARKs

Stwo (pronounced "stew") is StarkWare's next-generation prover, now live on Starknet mainnet. It represents a paradigm shift in STARK proving by using small-field arithmetic.

```
STWO PROVER — ARCHITECTURE
============================

Language:       Rust
Field:          Mersenne31 (M31): p = 2^31 - 1
                This is the EIGHTH Mersenne prime.

Why M31 matters for hardware:
  ┌──────────────────────────────────────────────────────────┐
  │  M31 arithmetic fits in 32-bit integers!                 │
  │                                                          │
  │  Multiplication: a * b mod (2^31 - 1)                    │
  │    = (a * b) mod (2^31 - 1)                              │
  │    = ((a*b) >> 31) + ((a*b) & 0x7FFFFFFF)                │
  │    Simple bit shift + add + conditional subtract!         │
  │                                                          │
  │  Compare with BN254 field (254 bits):                    │
  │    Requires 4x4 = 16 64-bit multiplies + reduction       │
  │    That is ~50x more expensive per field operation        │
  │                                                          │
  │  On modern CPUs/GPUs with 32-bit integer ALUs:           │
  │    M31 multiply = 1 cycle                                │
  │    BN254 multiply = 50+ cycles                           │
  │                                                          │
  │  For FPGAs:                                              │
  │    M31 uses a single 32-bit DSP slice                    │
  │    BN254 uses 16+ DSP slices in a pipeline               │
  └──────────────────────────────────────────────────────────┘

Circle STARKs innovation:
  Traditional STARKs require the field to have a large
  multiplicative subgroup of order 2^k (for NTT).
  M31 has multiplicative order p-1 = 2^31 - 2, which is
  NOT a power of 2.

  Solution: Circle STARKs use the CIRCLE GROUP over M31.
  The set of points (x, y) where x^2 + y^2 = 1 over M31
  forms a group of order 2^31 (a perfect power of 2!).
  This enables efficient NTT-like operations.

  This was a joint breakthrough by StarkWare (David Levit,
  Shahar Papini) and Polygon (Ulrich Haboeck).

Performance:
  - 100x faster than Stone
  - Proves a Poseidon2 hash chain in ~0.5 seconds on M3 Pro laptop
  - 28x faster than RISC Zero precompile on Keccak
  - 39x faster than SP1 precompile on Keccak
  - SIMD-optimized for AVX-512 and ARM NEON
  - GPU support via ICICLE-Stwo (Ingonyama)

Architecture:
  - Modular AIR fragments: each operator has its own AIR
  - Fragments compose into a single STARK proof
  - LogUp-based permutation arguments for cross-table lookups
  - Commitment: Mixed-degree Merkle (hash-based, no elliptic curves)
```

### 4.6 Cairo + Starknet Ecosystem

```
STARKNET ARCHITECTURE
=====================

                    ┌──────────────────────┐
                    │  User Transaction     │
                    │  (Cairo smart         │
                    │   contract call)      │
                    └──────────┬───────────┘
                               │
                    ┌──────────v───────────┐
                    │  Sequencer            │
                    │  (orders & executes   │
                    │   transactions)       │
                    └──────────┬───────────┘
                               │
                    ┌──────────v───────────┐
                    │  Starknet OS          │
                    │  (Cairo program that  │
                    │   processes all txns) │
                    └──────────┬───────────┘
                               │
                    ┌──────────v───────────┐
                    │  Stwo Prover (SHARP)  │
                    │  (generates STARK     │
                    │   proof of execution) │
                    └──────────┬───────────┘
                               │
                    ┌──────────v───────────┐
                    │  STARK -> SNARK       │
                    │  Wrapper (optional)   │
                    │  (compress for L1)    │
                    └──────────┬───────────┘
                               │
                    ┌──────────v───────────┐
                    │  Ethereum L1          │
                    │  Verification         │
                    └──────────────────────┘
```

### 4.7 Hardware Implications of the STARK Ecosystem

```
STARK HARDWARE ACCELERATION TARGETS
====================================

1. NTT / CIRCLE-NTT
   - Dominant operation in LDE (low-degree extension)
   - M31 field: 32-bit arithmetic, perfect for SIMD/GPU
   - Circle NTT has different butterfly structure than standard NTT
   - FPGA: streaming NTT pipelines, on-chip butterfly networks
   - GPU: massive parallelism in butterfly stages
   - ASIC: dedicated NTT cores with on-chip memory

2. HASH FUNCTIONS (Poseidon2 over M31)
   - Merkle tree construction for FRI commitment
   - Poseidon2 over M31: ~50% fewer rounds than Poseidon over BN254
   - FPGA: pipelined Poseidon2 round functions
   - GPU: parallel hash of many leaves
   - ASIC: dedicated Poseidon2 core

3. FIELD ARITHMETIC (M31, BabyBear, Goldilocks)
   - Foundation of all operations
   - M31: single 32-bit multiply + shift
   - BabyBear (p = 15 * 2^27 + 1): 32-bit, used by RISC Zero
   - Goldilocks (p = 2^64 - 2^32 + 1): 64-bit, used by Plonky2
   - Extension fields: M31 -> CM31 (complex) -> QM31 (degree 4)
     Operations over QM31 = 4 M31 operations (Karatsuba)

4. PERMUTATION ARGUMENTS (LogUp)
   - Cross-table consistency checks
   - Involves batch field inversions
   - Montgomery batch inverse algorithm
```

---

## 5. Noir — The Simplicity-First DSL

Noir is a Rust-inspired, backend-agnostic ZK DSL developed by Aztec Labs. It represents the latest generation of ZK languages, designed for developer productivity without sacrificing prover performance. Noir occupies the sweet spot between Circom's simplicity and Halo 2's power.

### 5.1 Language Overview

```
NOIR LANGUAGE FEATURES
======================

Syntax:           Rust-like (fn, let, struct, impl, match, if/else)
Type system:      Static typing with generics
                  Field (native field element)
                  u8, u16, u32, u64 (unsigned integers with range checks)
                  bool, [T; N] (arrays), (T1, T2) (tuples)
                  str<N> (fixed-length strings)
                  Struct, Enum support

Key properties:
  - Constraints are IMPLICIT (unlike Circom's explicit operators)
  - assert() generates constraints automatically
  - if/else works naturally (Circom cannot do conditional constraints)
  - Standard library includes cryptographic primitives
  - Backend-agnostic: compiles to ACIR, not to a specific proof system
```

### 5.2 Noir Code Example

```rust
// File: src/main.nr

// Prove: "I know two private numbers that multiply to a public product"
fn main(x: Field, y: Field, product: pub Field) {
    // This assert generates constraints automatically.
    // No need for <== or === operators.
    assert(x * y == product);
}

// More complex example: hash preimage with standard library
use std::hash::poseidon;

fn main(preimage: Field, hash: pub Field) {
    // Noir's standard library includes Poseidon
    let computed_hash = poseidon::bn254::hash_1([preimage]);
    assert(computed_hash == hash);
}

// Example with structs, loops, and conditionals:
struct VoteProof {
    voter_id: Field,
    vote: u8,         // 0 or 1 (automatically range-checked!)
    nullifier: Field,
}

fn main(proof: VoteProof, merkle_root: pub Field, nullifier_hash: pub Field) {
    // Range check is automatic: vote is u8, so 0 <= vote <= 255
    // Additional constraint: vote must be 0 or 1
    assert((proof.vote == 0) | (proof.vote == 1));

    // Compute nullifier hash
    let computed = poseidon::bn254::hash_1([proof.nullifier]);
    assert(computed == nullifier_hash);

    // Merkle membership proof would go here...
}
```

### 5.3 ACIR — The Backend Bridge

```
NOIR COMPILATION PIPELINE
=========================

  Noir source (.nr)
       |
       v
  Noir Compiler (nargo)
       |
       v
  ACIR (Abstract Circuit Intermediate Representation)
       |
       |  ACIR is a backend-agnostic representation of the circuit.
       |  It contains:
       |    - Opcode list (arithmetic, memory, Brillig VM)
       |    - Witness map (variable <-> index)
       |    - Public input/output specification
       |
       +──────────+──────────+──────────+
       |          |          |          |
       v          v          v          v
  Barretenberg  Halo2     Arkworks   Future
  (UltraPlonk  backend   (Marlin/   backends
   + lookups)             Groth16)

  ACIR opcodes include:
    - Arithmetic gates (polynomial constraints)
    - Range constraints (via lookup tables in BB)
    - SHA256, Blake2s, Pedersen, Poseidon (black-box functions)
    - ECDSA verification
    - Schnorr verification
    - Keccak256
    - Recursive proof verification

  Key insight: ACIR black-box functions are implemented differently
  by each backend. Barretenberg uses lookup tables for range checks
  (efficient), while an R1CS backend would use bit decomposition
  (expensive).
```

### 5.4 Nargo Toolchain

```bash
# Install Noir (nargo)
curl -L https://raw.githubusercontent.com/noir-lang/noirup/main/install | bash
noirup

# Create a new project
nargo new my_circuit
cd my_circuit

# Project structure:
# my_circuit/
#   Nargo.toml          -- project configuration
#   src/
#     main.nr           -- circuit source code
#   Prover.toml         -- prover inputs (private + public)
#   Verifier.toml       -- verifier inputs (public only)

# Edit src/main.nr with your circuit

# Edit Prover.toml with inputs:
# x = "3"
# y = "7"
# product = "21"

# Compile to ACIR
nargo compile
# Output: target/my_circuit.json (ACIR + ABI)

# Execute (compute witness, check constraints)
nargo execute
# This runs the circuit with inputs from Prover.toml
# and checks all constraints without generating a proof

# Generate proof (using Barretenberg backend)
nargo prove
# Output: proofs/my_circuit.proof

# Verify proof
nargo verify
# Output: Proof verified successfully

# Run tests (Noir supports test functions)
nargo test
```

### 5.5 Noir vs Circom vs Halo 2

```
FEATURE COMPARISON
==================

                    Circom          Noir            Halo 2
                    ──────          ────            ──────
Language feel:      Custom DSL      Rust-like       Rust eDSL

Lines for a         ~20             ~5              ~150
simple multiply:

Constraint          Explicit        Implicit        Explicit
management:         (<==, ===)      (assert)        (create_gate)

Type safety:        None (felt      Strong (u8,     Rust type
                    only)           bool, struct)   system

If/else:            Cannot          Natural if/     Selector-
                    constrain       else with       based
                    conditionally   constraints     conditional

Proof system:       Groth16 only    UltraPlonk      PLONKish
                    (R1CS)          (via BB),       (IPA or KZG)
                                    backend-agnostic

Lookup tables:      No              Yes (via BB)    Yes (native)

Recursion:          No (native)     Yes (built-in)  Yes (IPA
                                                    accumulation)

Testing:            Manual          nargo test      MockProver

Ecosystem:          Large           Growing         Large
                    (circomlib)     (std library)   (Zcash, Scroll)

Best for:           Learning,       Production      Maximum
                    Groth16,        apps, fast      optimization,
                    prototyping     development     zkEVM-class
```

---

## 6. arkworks — The Rust Cryptographic Library

arkworks is not a circuit DSL or a proof system — it is a modular Rust library ecosystem for building ZK proof systems and cryptographic primitives from scratch. For hardware engineers, arkworks is the reference implementation: when you need to understand exactly what algorithm your FPGA/GPU needs to accelerate, arkworks source code is where you look.

### 6.1 Modular Architecture

```
ARKWORKS CRATE ECOSYSTEM
=========================

┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                         │
│  ark-circom (import Circom circuits)                        │
│  ark-crypto-primitives (Merkle trees, PRFs, commitments)    │
└─────────────────────────────────┬───────────────────────────┘
                                  │
┌─────────────────────────────────v───────────────────────────┐
│                    PROOF SYSTEM LAYER                        │
│  ark-groth16     (Groth16 SNARK)                            │
│  ark-marlin      (universal SNARK)                          │
│  ark-spartan     (no trusted setup)                         │
│  ark-gemini      (elastic SNARK)                            │
└─────────────────────────────────┬───────────────────────────┘
                                  │
┌─────────────────────────────────v───────────────────────────┐
│                    CONSTRAINT LAYER                          │
│  ark-relations   (R1CS trait definitions)                   │
│  ark-r1cs-std    (R1CS gadgets: field ops, curve ops,       │
│                   boolean, comparisons, hashing)            │
│  ark-snark       (generic SNARK trait)                      │
└─────────────────────────────────┬───────────────────────────┘
                                  │
┌─────────────────────────────────v───────────────────────────┐
│                    ALGEBRA LAYER (Foundation)                │
│  ark-ff          (finite field arithmetic)                  │
│    - Field, PrimeField, FftField traits                     │
│    - Montgomery representation                              │
│    - Fp64, Fp256, Fp384 implementations                     │
│    - FFT/NTT implementations                                │
│                                                             │
│  ark-ec          (elliptic curve arithmetic)                │
│    - CurveGroup, AffineRepr traits                          │
│    - Short Weierstrass, Twisted Edwards, Montgomery curves  │
│    - Pairing-friendly: BN254, BLS12-381, BW6-761, MNT4/6   │
│    - MSM (multi-scalar multiplication) implementations      │
│                                                             │
│  ark-poly        (polynomial arithmetic)                    │
│    - DensePolynomial, SparsePolynomial                      │
│    - Radix-2 FFT/NTT                                        │
│    - Polynomial commitment traits                           │
│                                                             │
│  ark-serialize   (efficient binary serialization)           │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Key API Examples

#### Finite Field Operations (ark-ff)

```rust
use ark_ff::{Field, PrimeField, FftField};
use ark_bn254::Fr;  // BN254 scalar field

fn field_operations() {
    // Create field elements
    let a = Fr::from(42u64);
    let b = Fr::from(7u64);

    // Arithmetic
    let c = a + b;           // addition
    let d = a - b;           // subtraction
    let e = a * b;           // multiplication (modular)
    let f = a.square();      // a^2 (optimized)
    let g = a.double();      // 2*a (optimized)
    let h = a.inverse().unwrap();  // modular inverse

    // Verify: h * a = 1
    assert_eq!(h * a, Fr::one());

    // Exponentiation
    let i = a.pow(&[5u64]);  // a^5

    // Conversion
    let bytes = a.into_bigint().to_bytes_le();
    let from_bytes = Fr::from_le_bytes_mod_order(&bytes);

    // FFT-related
    // Fr implements FftField, so it has roots of unity
    let omega = Fr::get_root_of_unity(1024).unwrap(); // 1024-th root
    assert_eq!(omega.pow(&[1024u64]), Fr::one());
}
```

#### Elliptic Curve Operations (ark-ec)

```rust
use ark_ec::{CurveGroup, AffineRepr, VariableBaseMSM};
use ark_bn254::{G1Projective as G1, G1Affine, Fr};
use ark_std::rand::thread_rng;
use ark_std::UniformRand;

fn curve_operations() {
    let rng = &mut thread_rng();

    // Random group elements
    let p = G1::rand(rng);
    let q = G1::rand(rng);

    // Group operations (projective coordinates — faster)
    let r = p + q;           // point addition
    let s = p.double();      // point doubling
    let scalar = Fr::rand(rng);
    let t = p * scalar;      // scalar multiplication

    // Convert to affine (for serialization/verification)
    let p_affine: G1Affine = p.into_affine();

    // MSM — THE key operation for hardware acceleration
    let n = 1024;
    let bases: Vec<G1Affine> = (0..n)
        .map(|_| G1Affine::rand(rng))
        .collect();
    let scalars: Vec<Fr> = (0..n)
        .map(|_| Fr::rand(rng))
        .collect();

    // This is what FPGAs/GPUs accelerate:
    let msm_result = G1::msm(&bases, &scalars).unwrap();
    // Internally uses Pippenger's algorithm
    // Time complexity: O(n / log n) group operations
}
```

#### Building a Simple Constraint System (ark-relations)

```rust
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystemRef, SynthesisError,
};
use ark_r1cs_std::{
    fields::fp::FpVar,
    alloc::AllocVar,
    eq::EqGadget,
};
use ark_bn254::Fr;

// Define a circuit: prove knowledge of x such that x^3 + x + 5 = y
struct CubeCircuit {
    x: Option<Fr>,  // private witness
    y: Option<Fr>,  // public input
}

impl ConstraintSynthesizer<Fr> for CubeCircuit {
    fn generate_constraints(
        self,
        cs: ConstraintSystemRef<Fr>,
    ) -> Result<(), SynthesisError> {
        // Allocate private input x
        let x = FpVar::new_witness(cs.clone(), || {
            self.x.ok_or(SynthesisError::AssignmentMissing)
        })?;

        // Allocate public input y
        let y = FpVar::new_input(cs.clone(), || {
            self.y.ok_or(SynthesisError::AssignmentMissing)
        })?;

        // Compute x^2
        let x_sq = &x * &x;

        // Compute x^3
        let x_cu = &x_sq * &x;

        // Compute x^3 + x + 5
        let five = FpVar::constant(Fr::from(5u64));
        let result = x_cu + &x + five;

        // Constrain result == y
        result.enforce_equal(&y)?;

        Ok(())
    }
}
```

### 6.3 How Hardware Engineers Use arkworks

```
ARKWORKS FOR HARDWARE ENGINEERS
================================

1. REFERENCE IMPLEMENTATIONS
   When designing an MSM accelerator, look at:
     ark-ec/src/scalar_mul/variable_base/mod.rs
   This is Pippenger's algorithm — the standard MSM algorithm.
   Your hardware must produce identical results.

2. BENCHMARKING
   use ark_std::test_rng;
   use std::time::Instant;

   let start = Instant::now();
   let result = G1::msm(&bases, &scalars).unwrap();
   let cpu_time = start.elapsed();
   // Compare this with your FPGA/GPU implementation

3. ALGORITHM EXPLORATION
   arkworks lets you swap curves, fields, and proof systems:
     - Change from BN254 to BLS12-381: swap one line
     - Change from Groth16 to Marlin: swap the prover call
     - Test with different MSM sizes, field sizes, NTT lengths

4. TEST VECTOR GENERATION
   Generate known-good inputs/outputs for your hardware:
     - Random field elements with known products
     - NTT input/output pairs
     - MSM input/output pairs
     - Serialized points for FPGA consumption
```

---

## 7. zkVMs — General-Purpose ZK Virtual Machines

zkVMs represent the highest abstraction level in ZK development: write standard programs in Rust or C, and the VM automatically generates zero-knowledge proofs of correct execution. This simplicity comes at a cost — enormous execution traces that make hardware acceleration not just beneficial but **essential**.

### 7.1 RISC Zero

```
RISC ZERO ARCHITECTURE
=======================

  Standard Rust code
       |
       v
  RISC-V cross-compiler (riscv32im-risc0-zkvm-elf)
       |
       v
  ┌─────────────────────────────────────┐
  │  RISC Zero zkVM                      │
  │                                      │
  │  ┌──────────┐    ┌───────────────┐  │
  │  │  Guest    │    │  Host         │  │
  │  │  Program  │<-->│  Application  │  │
  │  │  (runs    │    │  (provides    │  │
  │  │   inside  │    │   inputs,     │  │
  │  │   zkVM)   │    │   receives    │  │
  │  │           │    │   proof)      │  │
  │  └──────────┘    └───────────────┘  │
  │       |                              │
  │       v                              │
  │  Execution trace                     │
  │  (every RISC-V instruction           │
  │   becomes multiple trace rows)       │
  │       |                              │
  │       v                              │
  │  Segment proofs (STARKs)             │
  │  Field: BabyBear (p = 15*2^27 + 1)  │
  │       |                              │
  │       v                              │
  │  Recursive aggregation               │
  │  (combine segment proofs)            │
  │       |                              │
  │       v                              │
  │  Succinct STARK proof                │
  │  (optionally -> Groth16 wrapper)     │
  └─────────────────────────────────────┘

Key: BabyBear (p = 2013265921 = 15 * 2^27 + 1)
  - 31-bit prime, fits in 32-bit integer
  - Large multiplicative subgroup of order 2^27
    (perfect for NTT, unlike M31 which needs Circle STARKs)
  - Single 32-bit multiply for field multiplication
  - Similar hardware efficiency to M31
```

#### RISC Zero Code Example

```rust
// ====== Guest program (runs inside zkVM) ======
// guest/src/main.rs
#![no_main]
risc0_zkvm::guest::entry!(main);

use risc0_zkvm::guest::env;

fn main() {
    // Read private input from host
    let secret: u64 = env::read();

    // Perform computation (this is what gets proved)
    let result = secret * secret + 1;

    // Commit public output (included in the proof)
    env::commit(&result);
}

// ====== Host program (runs the zkVM) ======
// host/src/main.rs
use risc0_zkvm::{default_prover, ExecutorEnv};
use my_guest::GUEST_ELF;  // compiled guest binary

fn main() {
    // Set up the execution environment with inputs
    let env = ExecutorEnv::builder()
        .write(&42u64)  // private input: secret = 42
        .unwrap()
        .build()
        .unwrap();

    // Run the prover
    let prover = default_prover();
    let receipt = prover.prove(env, GUEST_ELF).unwrap();

    // The receipt contains the proof + public outputs
    let result: u64 = receipt.journal.decode().unwrap();
    println!("Proven result: {}", result);  // 42*42 + 1 = 1765

    // Verify the proof
    receipt.verify(GUEST_IMAGE_ID).unwrap();
}
```

### 7.2 SP1 (Succinct Processor 1)

```
SP1 ARCHITECTURE
================

  Standard Rust code
       |
       v
  RISC-V cross-compiler
       |
       v
  ┌─────────────────────────────────────┐
  │  SP1 zkVM                            │
  │                                      │
  │  Key differentiator: PRECOMPILES     │
  │                                      │
  │  Core RISC-V execution               │
  │       +                              │
  │  Specialized precompile STARKs:      │
  │    - SHA-256 (dedicated AIR)         │
  │    - Keccak-256 (dedicated AIR)      │
  │    - BN254 add/mul (dedicated AIR)   │
  │    - BLS12-381 operations            │
  │    - Ed25519 verification            │
  │    - Secp256k1 verification          │
  │                                      │
  │  Proving stack: Plonky3 (STARK)      │
  │  Field: BabyBear                     │
  │                                      │
  │  SP1 Turbo: GPU-accelerated prover   │
  │  SP1 Hypercube: next-gen, real-time  │
  │    Ethereum block proving            │
  └─────────────────────────────────────┘

Performance benchmarks (as of late 2025):
  SP1 Hypercube:
    - Ethereum block in ~10.3 seconds (200 RTX 4090 cluster)
    - Single RTX 4090: ~1 min 55 sec per block
    - 93% of blocks proved in real-time (< 12 seconds)

  SP1 Turbo (earlier):
    - Single RTX 4090: ~3.5-4 minutes per block
    - Multi-GPU: < 40 seconds
```

### 7.3 Valida

```
VALIDA — CUSTOM ISA FOR PROVING
================================

Unlike RISC Zero and SP1 (which emulate RISC-V), Valida uses a
CUSTOM instruction set architecture designed from scratch for
efficient STARK proving.

Key design decisions:

  1. NO GENERAL-PURPOSE REGISTERS
     - Stack-based architecture (like JVM, WASM)
     - All operands addressed relative to Frame Pointer (FP)
     - Eliminates register allocation complexity
     - Eliminates register spill/reload (major trace overhead)

  2. WORD SIZE = 32 bits
     - Matches BabyBear/M31 field elements
     - Every memory word is one field element
     - No byte-packing/unpacking overhead

  3. MINIMAL INSTRUCTION SET
     - Only instructions that are cheap to constrain
     - Complex operations decomposed into simpler ones
     - Each instruction has a small, fixed-size AIR

  4. PROVING STACK
     - Uses Plonky3 (same as SP1)
     - BabyBear field
     - Modular AIR per instruction type

Compiler support:
  - Rust (alpha), C, WASM, LLVM IR -> Valida machine code
  - The compiler toolchain handles the ISA translation

Performance advantage:
  - Benchmarked faster than RISC Zero, SP1, and Jolt
  - On SHA-256 and Fibonacci: better CPU efficiency AND wall time
  - Fewer trace rows per computation step
```

### 7.4 zkWASM and zkEVM Approaches

```
GENERAL-PURPOSE ZK EXECUTION COMPARISON
========================================

Approach    ISA         Field        Pros                 Cons
─────────── ─────────── ──────────── ──────────────────── ────────────────
RISC Zero   RISC-V      BabyBear     Standard ISA,        Large traces
                                      mature ecosystem     for complex ops

SP1         RISC-V      BabyBear     Precompiles for      Still RISC-V
                                      crypto ops,          overhead for
                                      GPU acceleration     non-precompile

Valida      Custom      BabyBear     Minimal trace        New, smaller
                                      overhead, optimal    ecosystem
                                      for proving

zkWASM      WebAssembly varies       Web-native,          Complex opcodes,
                                      cross-platform       worse proving
                                                          efficiency

zkEVM       EVM         varies       Direct Ethereum      Most complex,
(Type 1)    opcodes                   compatibility        highest overhead
                                                          per instruction

zkEVM       Modified    varies       Better performance   Less compatible,
(Type 2-4)  EVM                       than Type 1          app changes needed

The performance hierarchy (proving efficiency):
  Custom ISA (Valida) > RISC-V (RISC Zero, SP1)
    > WASM (zkWASM) > EVM (zkEVM)

The compatibility hierarchy (opposite direction):
  EVM (zkEVM) > WASM > RISC-V > Custom ISA
```

### 7.5 Why zkVMs Make Hardware Acceleration Critical

```
zkVM TRACE SIZES — THE HARDWARE IMPERATIVE
============================================

A simple Fibonacci(100) computation:
  - Standard CPU: ~200 instructions, < 1 microsecond
  - RISC Zero trace: ~10,000 rows × 200+ columns
  - Total cells: ~2,000,000 field elements
  - Proof time (CPU): ~5-10 seconds

A SHA-256 hash:
  - Standard CPU: ~1000 cycles
  - RISC Zero trace: ~100,000 rows (without precompile)
  - SP1 with precompile: ~5,000 rows (20x reduction)
  - Proof time varies 10-100x based on trace size

An Ethereum block (~150 transactions):
  - Standard CPU: ~50ms to execute
  - zkEVM trace: billions of cells
  - Proof time (CPU): hours
  - Proof time (GPU cluster): seconds to minutes
  - This is why hardware acceleration is not optional

The scaling problem:
  trace_size ∝ number_of_instructions
  proving_time ∝ trace_size × log(trace_size)
  memory ∝ trace_size × blowup_factor

  For a 1-billion-row trace (realistic for a block):
    NTT alone: ~10 billion field multiplications
    Hashing: ~1 billion hash operations
    Memory: ~64 GB at 64 bytes per extended cell

  WITHOUT hardware acceleration: hours
  WITH GPU acceleration: minutes
  WITH custom FPGA/ASIC: seconds

  This is the fundamental business case for ZK hardware.
```

---

## 8. Choosing the Right Tool — Decision Framework

### 8.1 Decision Matrix

```
USE CASE DECISION MATRIX
=========================

"I want to..."                      Tool             Proof System
──────────────────────────────────  ───────────────  ────────────────
Learn ZK circuits for the           Circom +         Groth16
first time                          snarkjs

Build a simple identity/            Circom or        Groth16 or
membership proof                    Noir             UltraPlonk

Build a privacy mixer or            Circom           Groth16
credential system (like TC)         (with circomlib)

Build a zkEVM or complex            Halo 2           PLONKish + KZG
rollup circuit                      (PSE fork)

Build for Starknet ecosystem        Cairo 1          STARK (Stwo)

Write a general-purpose ZK          RISC Zero        STARK + Groth16
application in normal Rust          or SP1           wrapper

Build a custom proof system         arkworks         Any (you build it)

Prototype quickly, ship fast        Noir             UltraPlonk (BB)

Benchmark MSM/NTT for               arkworks         Groth16 or PLONK
hardware acceleration design

Understand what zkVMs               RISC Zero        STARK
need to accelerate                  source code

Maximize prover performance         Halo 2 +        PLONKish +
for a specific circuit              hand-optimization KZG

Deploy cheapest on-chain            Circom           Groth16
verification                        (Groth16)        (~200K gas)
```

### 8.2 For Hardware Engineers Specifically

```
HARDWARE ENGINEER'S TOOL GUIDE
===============================

PHASE 1: UNDERSTANDING (weeks 1-4)
  Start with Circom + snarkjs:
    - Build 3-5 simple circuits
    - Understand R1CS constraints
    - See how signals map to the witness vector
    - Understand what Groth16 prove/verify actually computes

PHASE 2: PROFILING (weeks 5-8)
  Move to arkworks:
    - Write MSM benchmarks with different sizes (2^10 to 2^22)
    - Write NTT benchmarks with different lengths
    - Profile with perf/vtune to see CPU bottlenecks
    - Generate test vectors for your hardware designs
    - Understand Pippenger's algorithm at the code level

PHASE 3: PRODUCTION SYSTEMS (weeks 9-12)
  Study Halo 2 (for SNARK acceleration):
    - Understand PLONKish costs: rows, columns, degree
    - Profile Scroll's zkEVM circuits
    - Identify MSM/NTT sizes in real production circuits

  Study RISC Zero / SP1 (for STARK acceleration):
    - Understand trace generation and AIR constraints
    - Profile NTT over BabyBear/M31 fields
    - Profile hash-based commitment (Poseidon2 Merkle trees)
    - Understand the recursive proof pipeline

PHASE 4: SPECIALIZATION (ongoing)
  Choose your acceleration target:
    - MSM accelerator -> study arkworks MSM, Groth16 prover
    - NTT accelerator -> study arkworks NTT, STARK provers
    - Hash accelerator -> study Poseidon2, FRI commitment
    - Full prover accelerator -> study ICICLE, cuZK, PipeZK
```

### 8.3 The "Prove in STARK, Verify in SNARK" Pattern

```
THE STARK -> SNARK WRAPPER PATTERN
====================================

Problem:
  STARK proofs are large (~50-200 KB) and expensive to verify on-chain.
  SNARK proofs (Groth16) are tiny (~128 bytes) and cheap (~200K gas).
  But STARKs have no trusted setup and are post-quantum.

Solution:
  1. Generate a STARK proof of the main computation (fast, no setup)
  2. Write a SNARK circuit that VERIFIES the STARK proof
  3. Generate a SNARK proof of the STARK verification
  4. Post the small SNARK proof on-chain

  ┌───────────────────────┐
  │  Main Computation     │
  │  (millions of steps)  │
  └───────────┬───────────┘
              │ STARK prove (fast, transparent)
              v
  ┌───────────────────────┐
  │  STARK Proof           │
  │  (~100 KB)            │
  └───────────┬───────────┘
              │ Express STARK verification as a circuit
              v
  ┌───────────────────────┐
  │  SNARK Circuit         │
  │  (verifies STARK)     │
  └───────────┬───────────┘
              │ SNARK prove (Groth16, requires trusted setup)
              v
  ┌───────────────────────┐
  │  Groth16 Proof         │
  │  (~128 bytes)          │
  │  Cheap on-chain verify │
  └───────────────────────┘

Who uses this pattern:
  - Polygon zkEVM: STARK proof -> recursive compression -> Groth16
  - Starknet: STARK proof -> (optionally) SNARK wrapper for L1
  - RISC Zero: STARK segments -> recursive combine -> Groth16 wrapper
  - SP1: STARK proof -> SNARK wrapper via Groth16

Tool support:
  - RISC Zero: built-in Groth16 wrapper (bonsai)
  - SP1: built-in SNARK wrapping
  - StarkWare: custom STARK->SNARK pipeline in SHARP
  - circom-pairing: Groth16 verifier circuit in Circom (~1M constraints)

Hardware implication:
  The SNARK wrapping step is MSM-heavy (Groth16).
  The STARK step is NTT+hash-heavy.
  A full proving pipeline accelerator needs BOTH capabilities.
```

---

## 9. Projects

### Project 1: Circom Fundamentals — Private Voting Circuit (Beginner)

```
OBJECTIVE: Build a zero-knowledge private voting system where:
  - Voters prove they are in an authorized voter list (Merkle tree)
  - Votes are cast without revealing voter identity
  - Double-voting is prevented via nullifiers

SKILLS: Circom syntax, circomlib usage, snarkjs workflow, R1CS

STEPS:
  1. Create a Merkle tree membership circuit using circomlib's Poseidon
     - Template: MerkleTreeChecker(levels) with 8 levels (256 voters)
     - Input: leaf (voter commitment), path elements, path indices
     - Output: computed root (constrained to equal public root)

  2. Add nullifier computation
     - nullifier = Poseidon(secret, proposal_id)
     - This prevents double-voting: same secret always gives same nullifier

  3. Full circuit:
     template Vote(levels) {
         signal input secret;          // private
         signal input pathElements[levels];  // private (Merkle path)
         signal input pathIndices[levels];   // private
         signal input root;            // public (Merkle root)
         signal input nullifierHash;   // public (for double-vote check)
         signal input proposalId;      // public

         // Compute voter leaf
         component leafHasher = Poseidon(1);
         leafHasher.inputs[0] <== secret;

         // Verify Merkle membership
         component tree = MerkleTreeChecker(levels);
         tree.leaf <== leafHasher.out;
         for (var i = 0; i < levels; i++) {
             tree.pathElements[i] <== pathElements[i];
             tree.pathIndices[i] <== pathIndices[i];
         }
         tree.root === root;

         // Compute and verify nullifier
         component nullifier = Poseidon(2);
         nullifier.inputs[0] <== secret;
         nullifier.inputs[1] <== proposalId;
         nullifierHash === nullifier.out;
     }
     component main {public [root, nullifierHash, proposalId]}
         = Vote(8);

  4. Compile, run trusted setup, generate proof, verify

  5. MEASURE: Time the proof generation with both snarkjs and
     rapidsnark. Record the number of R1CS constraints.

DELIVERABLE: Working circuit + proof + constraint count analysis
```

### Project 2: Halo 2 Fibonacci Circuit with Custom Gates (Intermediate)

```
OBJECTIVE: Build a Fibonacci sequence circuit in Halo 2 that
  demonstrates custom gates, relative rotation references, and
  the MockProver testing workflow.

SKILLS: Halo 2 API, PLONKish arithmetization, Rust, custom gates

CIRCUIT DESIGN:
  Three advice columns: a, b, c
  Custom gate: s_fib * (a(cur) + b(cur) - c(cur)) = 0
  This enforces: c = a + b at every activated row.

  Row layout:
    Row 0: a=1,  b=1,  c=2   (fib(1), fib(2), fib(3))
    Row 1: a=1,  b=2,  c=3   (copy: a(1)=b(0), b(1)=c(0))
    Row 2: a=2,  b=3,  c=5
    Row 3: a=3,  b=5,  c=8
    ...
    Row n: a=fib(n), b=fib(n+1), c=fib(n+2)

  Copy constraints link adjacent rows:
    a(row+1) = b(row)
    b(row+1) = c(row)

STEPS:
  1. Define FibConfig with 3 advice columns, 1 selector, 1 instance
  2. Implement configure() with the addition gate
  3. Implement synthesize() that fills the Fibonacci table
  4. Test with MockProver
  5. Generate and verify a real proof (IPA or KZG backend)
  6. EXPERIMENT: Vary the number of Fibonacci steps (10, 100, 1000)
     and measure proving time. Plot the relationship.

DELIVERABLE: Working Halo 2 circuit + proving time measurements
```

### Project 3: Noir Hash Chain with Recursive Verification (Intermediate)

```
OBJECTIVE: Build a circuit in Noir that proves knowledge of a chain
  of hash preimages, demonstrating Noir's ergonomic syntax, standard
  library, and recursive proof capability.

SKILLS: Noir syntax, nargo workflow, Barretenberg backend

CIRCUIT:
  // src/main.nr
  use std::hash::poseidon;

  fn main(
      chain: [Field; 10],        // private: 10 preimages
      final_hash: pub Field,     // public: end of chain
  ) {
      let mut current = chain[0];
      for i in 1..10 {
          // Each element hashes to the next
          let h = poseidon::bn254::hash_1([current]);
          assert(h == chain[i]);
          current = chain[i];
      }
      // Final hash must match public input
      let final_computed = poseidon::bn254::hash_1([current]);
      assert(final_computed == final_hash);
  }

STEPS:
  1. Install Noir and create the project with nargo
  2. Implement the hash chain circuit
  3. Compute valid inputs (start from a random value, hash forward)
  4. Fill in Prover.toml with the chain values
  5. Compile, prove, and verify
  6. COMPARE: Implement the same circuit in Circom
     - Count constraints in both
     - Compare code complexity (lines of code)
     - Compare proving time

DELIVERABLE: Noir + Circom implementations + comparison analysis
```

### Project 4: Cairo STARK Program with Stwo Benchmarking (Advanced)

```
OBJECTIVE: Write a Cairo program, generate its execution trace,
  and understand STARK proving. Profile the key operations
  (NTT, hashing) that hardware engineers need to accelerate.

SKILLS: Cairo 1 language, STARK concepts, trace analysis

PROGRAM (Cairo 1):
  // Compute and verify a chain of Poseidon hashes
  use core::poseidon::poseidon_hash_span;

  fn main() {
      let mut values: Array<felt252> = array![1, 2, 3, 4, 5];
      let mut i: u32 = 0;

      loop {
          if i >= 100 {
              break;
          }
          // Hash the current array
          let hash = poseidon_hash_span(values.span());

          // Append hash and continue
          values.append(hash);
          i += 1;
      };

      // The final hash is the public output
      let final_hash = *values.at(values.len() - 1);
      // Print or assert the result
  }

STEPS:
  1. Install Cairo toolchain (scarb)
  2. Write and run the program
  3. Examine the execution trace structure:
     - How many steps does the VM take?
     - How many memory cells are used?
     - How many built-in invocations (Poseidon)?
  4. If possible, run with Stone prover and measure:
     - Total proving time
     - NTT time (via profiling or prover logs)
     - Hash time (Merkle tree construction)
  5. Compare the trace statistics with equivalent Circom circuit

DELIVERABLE: Cairo program + trace analysis + timing breakdown
```

### Project 5: MSM/NTT Performance Profiling Across Frameworks (Advanced)

```
OBJECTIVE: Create a benchmarking suite that measures the core
  operations (MSM, NTT, field arithmetic) across multiple
  libraries, generating data for hardware acceleration design
  decisions.

SKILLS: arkworks API, Rust benchmarking, performance analysis

BENCHMARKS TO IMPLEMENT:

  1. MSM Benchmark (arkworks):
     - Sizes: 2^10, 2^12, 2^14, 2^16, 2^18, 2^20, 2^22
     - Curves: BN254 (G1), BLS12-381 (G1)
     - Measure: time, throughput (points/second)

  2. NTT Benchmark (arkworks):
     - Sizes: 2^10 through 2^24
     - Fields: BN254::Fr, BLS12-381::Fr
     - Measure: time, throughput (elements/second)
     - Compare forward NTT vs inverse NTT

  3. Field Arithmetic Benchmark:
     - Operations: multiply, add, inverse, square
     - Fields: BN254::Fr (254-bit), BabyBear (31-bit), if available
     - Batch sizes: 10K, 100K, 1M
     - Measure: operations per second

  4. Hash Benchmark:
     - Poseidon hash over BN254 field
     - Measure: hashes per second
     - Compare with SHA-256 (as baseline)

  5. End-to-End Prover Benchmark:
     - Simple circuit (e.g., 2^16 constraint multiply chain)
     - Groth16 (via arkworks): total time, MSM%, NTT%, other%
     - PLONK (via halo2): total time, commit%, NTT%, other%

RUST SKELETON:

  use ark_bn254::{G1Projective as G1, G1Affine, Fr};
  use ark_ec::VariableBaseMSM;
  use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
  use ark_std::{rand::thread_rng, UniformRand};
  use std::time::Instant;

  fn bench_msm(size: usize) {
      let rng = &mut thread_rng();
      let bases: Vec<G1Affine> = (0..size)
          .map(|_| G1Affine::rand(rng)).collect();
      let scalars: Vec<Fr> = (0..size)
          .map(|_| Fr::rand(rng)).collect();

      let start = Instant::now();
      let _result = G1::msm(&bases, &scalars).unwrap();
      let elapsed = start.elapsed();

      println!("MSM size 2^{}: {:?} ({:.0} points/sec)",
          (size as f64).log2() as u32,
          elapsed,
          size as f64 / elapsed.as_secs_f64());
  }

  fn bench_ntt(size: usize) {
      let rng = &mut thread_rng();
      let domain = Radix2EvaluationDomain::<Fr>::new(size).unwrap();
      let mut coeffs: Vec<Fr> = (0..size)
          .map(|_| Fr::rand(rng)).collect();

      let start = Instant::now();
      domain.fft_in_place(&mut coeffs);
      let elapsed = start.elapsed();

      println!("NTT size 2^{}: {:?} ({:.0} elements/sec)",
          (size as f64).log2() as u32,
          elapsed,
          size as f64 / elapsed.as_secs_f64());
  }

OUTPUT FORMAT:
  Generate a CSV with columns:
    operation, size, field/curve, time_ms, throughput

  This data directly informs your hardware design:
    - MSM throughput target = 10x CPU throughput
    - NTT throughput target = 100x CPU throughput
    - These define your FPGA/ASIC performance requirements

DELIVERABLE: Benchmarking suite + CSV results + analysis document
  comparing CPU baseline with published GPU/FPGA results
```

---

## 10. Resources

### 10.1 Official Documentation

```
TOOL              URL
────────────────  ──────────────────────────────────────────────
Circom            https://docs.circom.io/
snarkjs           https://github.com/iden3/snarkjs
circomlib         https://github.com/iden3/circomlib
Halo 2 Book       https://zcash.github.io/halo2/
Halo 2 (PSE)      https://github.com/privacy-scaling-explorations/halo2
Cairo Book         https://book.cairo-lang.org/
Stwo Prover        https://github.com/starkware-libs/stwo
Noir Docs          https://noir-lang.org/docs/
Barretenberg       https://barretenberg.aztec.network/docs/
arkworks           https://arkworks.rs/
RISC Zero          https://dev.risczero.com/
SP1                https://docs.succinct.xyz/
Valida             https://www.lita.foundation/
```

### 10.2 Tutorials and Courses

```
RESOURCE                                    FOCUS
──────────────────────────────────────────  ─────────────────────
RareSkills Circom Tutorial                  Circom from scratch
  rareskills.io/post/circom-tutorial

0xPARC ZK Learning Group                    Circom + applied ZK
  learn.0xparc.org/

ZK Security Halo2 Course                    Halo2 deep dive
  github.com/zksecurity/halo2-course

Halo2 Book Simple Example                   Halo2 first circuit
  zcash.github.io/halo2/user/simple-example.html

Starklings (Cairo exercises)                Cairo 1 exercises
  github.com/shramee/starklings-cairo1

Noir Quick Start                            Noir first circuit
  noir-lang.org/docs/getting_started/quick_start

Cyfrin Updraft Noir Course                  Noir programming
  updraft.cyfrin.io/courses/noir-programming-and-zk-circuits

RISC Zero Getting Started                   zkVM first project
  dev.risczero.com/api/getting-started

arkworks Tutorial                           arkworks R1CS
  github.com/marcozecchini/arkworks-tutorial
```

### 10.3 Example Repositories

```
REPOSITORY                                  WHAT IT DEMONSTRATES
──────────────────────────────────────────  ─────────────────────
tornadocash/tornado-core                    Production Circom:
  github.com/tornadocash/tornado-core       Merkle + Poseidon

semaphore-protocol/semaphore                Production Circom:
  github.com/semaphore-protocol/semaphore   identity + signals

zcash/halo2 (examples/)                     Halo2 simple example,
  github.com/zcash/halo2                    Fibonacci, etc.

scroll-tech/zkevm-circuits                  Production Halo2:
  github.com/scroll-tech/zkevm-circuits     full EVM in ZK

starkware-libs/stwo-cairo                   Cairo + Stwo prover
  github.com/starkware-libs/stwo-cairo

noir-lang/noir (examples/)                  Noir example circuits
  github.com/noir-lang/noir

risc0/risc0 (examples/)                     RISC Zero guest/host
  github.com/risc0/risc0                    examples

succinctlabs/sp1 (examples/)                SP1 example programs
  github.com/succinctlabs/sp1

ingonyama-zk/icicle                         GPU-accelerated ZK
  github.com/ingonyama-zk/icicle            (MSM, NTT, hashing)
```

### 10.4 Hardware Acceleration References

```
RESOURCE                                    FOCUS
──────────────────────────────────────────  ─────────────────────
PipeZK (ISCA 2021)                          Pipelined ZK accelerator
  people.iiis.tsinghua.edu.cn/~gaomy/       architecture
  pubs/pipezk.isca21.pdf

cuZK (GPU acceleration)                     GPU NTT + MSM for ZK
  eprint.iacr.org/2022/1321

Jane Street — ZK on FPGAs                   MSM+NTT on FPGAs
  blog.janestreet.com/                      with Hardcaml
  zero-knowledge-fpgas-hardcaml/

ICICLE (Ingonyama)                          GPU acceleration library
  github.com/ingonyama-zk/icicle            for MSM, NTT, Poseidon

SZKP (NSF)                                  Scalable ZK accelerator
  Architectural study of ZK workloads

ICICLE-Snark                                Fastest Groth16 prover
  ingonyama.com/post/icicle-snark-          (GPU-accelerated)
  the-fastest-groth16-implementation

ICICLE-Stwo                                 GPU-accelerated Stwo
  medium.com/@ingonyama/                    STARK prover
  introducing-icicle-stwo
```

### 10.5 Community Resources

```
Discord / Chat:
  - Circom: iden3 Discord
  - Halo 2: Zcash Discord (#halo2)
  - Cairo: Starknet Discord
  - Noir: Aztec Discord (#noir)
  - RISC Zero: RISC Zero Discord
  - SP1: Succinct Discord
  - ZK General: ZKResearch Telegram, ZKHack Discord

Forums:
  - Ethereum Research (ethresear.ch) — ZK rollup discussions
  - ZKProof.org — Standards body for ZK
  - Starknet Forum (community.starknet.io)

Events:
  - ZKHack (periodic hackathons + puzzles)
  - ZK Summit (annual conference)
  - Starknet events (ETH Denver, Starknet sessions)
  - Progcrypto (applied ZK)
```

---

## Summary: The Complete Map

```
THE ZK TOOL LANDSCAPE — COMPLETE MAP
======================================

                        ABSTRACTION LEVEL
          Low                                    High
          (more control)                         (less control)
          |                                       |
          v                                       v

    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ arkworks │  │ Halo 2   │  │ Circom   │  │ zkVMs    │
    │          │  │          │  │ Noir     │  │ RISC Zero│
    │ Library  │  │ Framework│  │ Cairo    │  │ SP1      │
    │ (Rust)   │  │ (Rust)   │  │          │  │ Valida   │
    │          │  │          │  │ DSLs     │  │          │
    │ Custom   │  │ PLONKish │  │ R1CS/    │  │ General  │
    │ proof    │  │ circuits │  │ AIR/     │  │ purpose  │
    │ systems  │  │          │  │ ACIR     │  │ ZK       │
    └──────────┘  └──────────┘  └──────────┘  └──────────┘
         |              |              |              |
         v              v              v              v
    ┌──────────────────────────────────────────────────────┐
    │              PROVER BACKENDS                          │
    │  Groth16 │ PLONK │ STARK │ Marlin │ HyperPlonk      │
    └──────────────────────────────────────────────────────┘
         |              |              |
         v              v              v
    ┌──────────────────────────────────────────────────────┐
    │         CORE OPERATIONS (Hardware Targets)            │
    │                                                      │
    │    MSM          NTT/FFT         Hash                 │
    │    (elliptic    (polynomial     (Merkle tree         │
    │     curve       domain          commitment)          │
    │     commit)     conversion)                          │
    │                                                      │
    │    Field Arithmetic (modular multiply/add/inverse)   │
    └──────────────────────────────────────────────────────┘
         |              |              |
         v              v              v
    ┌──────────────────────────────────────────────────────┐
    │         HARDWARE ACCELERATION                         │
    │                                                      │
    │    GPU           FPGA           ASIC                 │
    │    (ICICLE,      (Hardcaml,     (future              │
    │     cuZK)        PipeZK)        dedicated)           │
    └──────────────────────────────────────────────────────┘

For the hardware cryptographer:
  - Phase 5 (this guide) taught you WHAT the tools produce
  - Phase 6 will teach you HOW to accelerate the output
  - The bridge: every circuit you write above becomes
    MSM + NTT + Hash operations below. Your hardware
    accelerates the bottom. Understanding the top tells
    you the SIZE and SHAPE of the workloads.
```

---

**Next:** Phase 6 — ZK Hardware Acceleration. You now understand what ZK tools produce and why the computational bottlenecks exist. Phase 6 will teach you how to design FPGA, GPU, and ASIC architectures that accelerate MSM, NTT, and hash operations by 10-1000x over CPU baselines, making real-time ZK proving possible.
