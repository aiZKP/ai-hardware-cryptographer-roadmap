# Polynomial Arithmetic, NTT, and Polynomial Commitment Schemes for Zero-Knowledge Cryptography

---

## 1. Polynomial Evaluation and Interpolation over Finite Fields

### 1.1 Lagrange Interpolation

**Problem:** Given n+1 points {(x_0, y_0), (x_1, y_1), ..., (x_n, y_n)} where all x_i are distinct elements of a finite field F_p, find the unique polynomial P(x) of degree <= n passing through all points.

**Exact Formula (Classical Lagrange):**

```
P(x) = sum_{i=0}^{n} y_i * L_i(x)
```

where the Lagrange basis polynomials are:

```
L_i(x) = product_{j=0, j!=i}^{n} (x - x_j) / (x_i - x_j)
```

Expanded:

```
L_i(x) = (x - x_0)(x - x_1)...(x - x_{i-1})(x - x_{i+1})...(x - x_n)
          ---------------------------------------------------------------
          (x_i - x_0)(x_i - x_1)...(x_i - x_{i-1})(x_i - x_{i+1})...(x_i - x_n)
```

Each L_i(x) has the property: L_i(x_j) = delta_{ij} (1 if i=j, 0 otherwise).

**Computational Cost:**

| Operation | Cost |
|---|---|
| Computing all basis polynomials from scratch | O(n^2) field multiplications |
| Evaluating P(x) at a single new point (naive) | O(n^2) |
| Precomputing barycentric weights w_k | O(n^2) one-time cost |
| Evaluating P(x) at a new point (barycentric form) | O(n) per evaluation |

**Barycentric Form (more efficient for repeated evaluation):**

First, precompute the barycentric weights:

```
w_k = 1 / product_{j=0, j!=k}^{n} (x_k - x_j)
```

Then the interpolating polynomial at any point x is:

```
P(x) = [ sum_{k=0}^{n} (w_k / (x - x_k)) * y_k ] / [ sum_{k=0}^{n} (w_k / (x - x_k)) ]
```

This requires O(n^2) to precompute the weights, but only O(n) to evaluate at each new point.

**Significance for ZK:** Lagrange interpolation is used to encode witness values into polynomials. In PLONK, the witness polynomial is interpolated from gate evaluations over a multiplicative subgroup H = {1, omega, omega^2, ..., omega^{n-1}}. In Groth16, the QAP (Quadratic Arithmetic Program) is built using Lagrange basis polynomials over roots of unity.

---

### 1.2 Horner's Method for Polynomial Evaluation

**Problem:** Evaluate P(x) = c_n * x^n + c_{n-1} * x^{n-1} + ... + c_1 * x + c_0 at a point z in F_p.

**Horner's Nested Form:**

```
P(z) = c_0 + z * (c_1 + z * (c_2 + z * (... + z * (c_{n-1} + z * c_n)...)))
```

**Algorithm:**

```
result = c_n
for i from n-1 down to 0:
    result = result * z + c_i
return result
```

**Computational Cost:**

| Operation | Count |
|---|---|
| Field multiplications | exactly n |
| Field additions | exactly n |
| Total | 2n field operations |

**Complexity:** O(n), which is **provably optimal**. Alexander Ostrowski (1954) proved that the number of additions is minimal, and Victor Pan (1966) proved the number of multiplications is minimal.

**In ZK context:** Used whenever a prover needs to evaluate a committed polynomial at a specific challenge point. For example, in PLONK, the prover evaluates several polynomials at the challenge point zeta using Horner's method.

---

### 1.3 Polynomial Identity Testing and the Schwartz-Zippel Lemma

**Schwartz-Zippel Lemma (Formal Statement):**

Let f(x_1, x_2, ..., x_n) be a non-zero polynomial of total degree d over a finite field F_p. Let S be a finite subset of F_p. If r_1, r_2, ..., r_n are chosen independently and uniformly at random from S, then:

```
Pr[f(r_1, r_2, ..., r_n) = 0] <= d / |S|
```

**For the univariate case** (most common in ZK):

If f(x) and g(x) are distinct polynomials of degree at most d over F_p, and z is chosen uniformly at random from F_p, then:

```
Pr[f(z) = g(z)] <= d / p
```

**Concrete security example:** If p ~ 2^254 (e.g., BN254 or BLS12-381 scalar field) and polynomials are of degree d = 2^20 (about 1 million), then:

```
Pr[false positive] <= 2^20 / 2^254 = 1/2^234 ~ 1/10^70
```

This is astronomically small -- far smaller than any meaningful attack threshold.

**Why this is the foundation of ZK:**

Nearly all ZK proof systems rely on the Schwartz-Zippel Lemma to achieve succinctness. The key insight: instead of checking that two polynomials are identical (which requires comparing all coefficients -- O(d) work), the verifier picks a random point z and checks that f(z) = g(z). If they are not identical, this check catches the fraud with overwhelming probability.

This transforms verification from polynomial-time to constant-time (one evaluation and comparison), which is the core reason ZK proofs can be succinct.

**Application in PLONK/Groth16:** The prover must show that certain polynomial identities hold over the entire evaluation domain H. Instead of checking at every point in H, the verifier picks a random challenge z and checks the identity at that single point. By Schwartz-Zippel, if the identity fails at any point of H, it will fail at z with probability >= 1 - d/p.

---

### 1.4 Vanishing Polynomials and Their Role

**Definition:** For a multiplicative subgroup H = {1, omega, omega^2, ..., omega^{n-1}} where omega is a primitive n-th root of unity in F_p, the vanishing polynomial is:

```
Z_H(x) = product_{i=0}^{n-1} (x - omega^i) = x^n - 1
```

The equality Z_H(x) = x^n - 1 holds because the omega^i are exactly all the n-th roots of unity, and x^n - 1 = product_{i=0}^{n-1} (x - omega^i) by the factoring of cyclotomic polynomials.

**Key Property:** Z_H(x) = 0 for all x in H, and Z_H(x) != 0 for all x not in H.

**Evaluation efficiency:** Computing Z_H(z) for a specific z requires:

```
Z_H(z) = z^n - 1
```

This takes only O(log n) multiplications (via repeated squaring), NOT O(n). This is a crucial efficiency gain.

**Role in PLONK:**

In PLONK, the prover must demonstrate that constraint polynomials are zero on all of H. This is equivalent to showing that Z_H(x) divides the constraint polynomial. Concretely, if C(x) encodes the constraint, the prover must produce a quotient polynomial T(x) such that:

```
C(x) = T(x) * Z_H(x)
```

The verifier checks at random z: C(z) = T(z) * Z_H(z), where Z_H(z) = z^n - 1 is trivially computable.

**Role in Groth16:**

In Groth16's QAP, the target polynomial is t(x) = Z_H(x). The prover must show that:

```
A(x) * B(x) - C(x) = H(x) * t(x)
```

where A, B, C encode the R1CS constraint system and H is the quotient polynomial. Computing f(x)/t(x) is done efficiently using a coset FFT.

**Coset trick:** Popular implementations compute polynomial division by t(x) by evaluating f(x) on a coset k*H = {k, k*omega, k*omega^2, ...} where k is not in H (so Z_H is non-zero on the coset). The values f(k*omega^i) / Z_H(k*omega^i) give the evaluations of the quotient, which can then be interpolated back.

---

## 2. NTT (Number Theoretic Transform) -- The FFT over Finite Fields

### 2.1 Definition

The Number Theoretic Transform is the Discrete Fourier Transform performed over a finite field F_p instead of the complex numbers. It replaces complex roots of unity with finite-field roots of unity.

**Forward NTT:** Given a polynomial f(x) = sum_{i=0}^{n-1} a_i * x^i represented by its coefficient vector (a_0, a_1, ..., a_{n-1}), the NTT produces its evaluation at all n-th roots of unity:

```
f_k = sum_{i=0}^{n-1} a_i * omega^{ik}   (mod p),     for k = 0, 1, ..., n-1
```

where omega is a primitive n-th root of unity in F_p (i.e., omega^n = 1 mod p and omega^j != 1 for 0 < j < n).

In matrix form: the NTT is multiplication by the Vandermonde matrix:

```
| f_0   |   | 1    1       1        ...  1             |   | a_0   |
| f_1   |   | 1    omega   omega^2   ...  omega^{n-1}   |   | a_1   |
| f_2   | = | 1    omega^2 omega^4   ...  omega^{2(n-1)} | * | a_2   |
| ...   |   | ...                                        |   | ...   |
| f_{n-1}|   | 1    omega^{n-1} ...  omega^{(n-1)^2}    |   | a_{n-1}|
```

**What NTT computes:** It evaluates the polynomial at the n points {1, omega, omega^2, ..., omega^{n-1}}. This converts from coefficient representation to evaluation representation.

---

### 2.2 Connection to Polynomial Multiplication

**The key insight:** Polynomial multiplication in coefficient representation takes O(n^2), but in evaluation representation it is just pointwise multiplication -- O(n).

**Multiply-via-NTT pipeline:**

```
Given: f(x), g(x) of degree < n/2 each (product has degree < n)

Step 1: F = NTT(f)       -- O(n log n)   -- evaluate f at roots of unity
Step 2: G = NTT(g)       -- O(n log n)   -- evaluate g at roots of unity
Step 3: H = F . G        -- O(n)         -- pointwise multiply: H_k = F_k * G_k
Step 4: h = INTT(H)      -- O(n log n)   -- interpolate back to coefficients

Total: O(n log n) instead of O(n^2)
```

This is the single most important algorithmic primitive in ZK proof systems. Without NTT, polynomial operations in PLONK/STARK provers would be prohibitively expensive.

---

### 2.3 Radix-2 Cooley-Tukey Butterfly

**Decimation-in-Time (DIT) decomposition:** For n = 2m, split the polynomial into even and odd indexed coefficients:

```
f(x) = f_even(x^2) + x * f_odd(x^2)
```

where:

```
f_even(y) = a_0 + a_2*y + a_4*y^2 + ... + a_{n-2}*y^{m-1}
f_odd(y)  = a_1 + a_3*y + a_5*y^2 + ... + a_{n-1}*y^{m-1}
```

Each is a polynomial of degree m-1 = n/2 - 1.

**Butterfly operation:** For k = 0, 1, ..., m-1:

```
X_k       = E_k + omega^k * O_k
X_{k+m}   = E_k - omega^k * O_k
```

where:
- E_k = NTT of f_even, evaluated at k-th point
- O_k = NTT of f_odd, evaluated at k-th point
- omega^k = the twiddle factor

**Why this works:** The n-th roots of unity have the property that omega^{k+m} = -omega^k (the "half-rotation" property), so the same E_k and O_k values are reused with just a sign change.

**Single butterfly diagram:**

```
    E_k ----+----> E_k + W * O_k = X_k
             \  /
              \/
              /\
             /  \
    O_k --[*W]---> E_k - W * O_k = X_{k+m}
```

where W = omega^k is the twiddle factor for that stage.

**Cost per butterfly:** 1 field multiplication (by twiddle factor) + 2 field additions/subtractions.

---

### 2.4 Twiddle Factors

Twiddle factors are powers of the primitive root of unity used in each butterfly stage.

**Definition:** For an n-point NTT with primitive n-th root of unity omega, the twiddle factor at stage s and position k is:

```
W(s, k) = omega^{k * n / 2^{s+1}}
```

At the first stage (combining pairs), the twiddle factors are omega^{0*n/2} = {1, -1} (trivial).
At the second stage (combining groups of 4), the twiddle factors are omega^{k*n/4} for k=0,1.
At the final stage (full size), the twiddle factors are omega^k for k = 0, ..., n/2 - 1.

**Precomputation:** All n/2 distinct twiddle factors can be precomputed and stored in a table. Total storage: n/2 field elements.

**Properties exploited:**
- omega^0 = 1 (skip multiplication for trivial twiddle)
- omega^{n/2} = -1 (subtraction instead of multiplication)
- omega^{n/4} is the "imaginary unit" analog (useful for radix-4)

---

### 2.5 Inverse NTT (INTT)

**Formula:**

```
a_i = (1/n) * sum_{k=0}^{n-1} f_k * omega^{-ik}   (mod p)
```

The INTT is the NTT with two modifications:
1. Replace omega with omega^{-1} (the multiplicative inverse of omega mod p)
2. Multiply all results by n^{-1} (the multiplicative inverse of n mod p)

**Implementation detail:** The factor of 1/n can be:
- Applied to all outputs at the end (most common)
- Applied to all inputs at the beginning
- Folded into the twiddle factors of the last stage

**Key requirement:** n must have a multiplicative inverse in F_p, which requires gcd(n, p) = 1. Since p is prime and n < p (in practice, n ~ 2^20 while p ~ 2^254), this is always satisfied.

---

### 2.6 Computational Cost

**Exact operation count for radix-2 NTT of size n = 2^k:**

```
Number of stages:          log_2(n) = k
Butterflies per stage:     n/2
Total butterflies:         (n/2) * log_2(n)
```

Each butterfly requires:
- 1 field multiplication (by twiddle factor)
- 2 field additions (one add, one subtract)

**Total for one n-point NTT:**

| Operation | Count |
|---|---|
| Field multiplications | (n/2) * log_2(n) |
| Field additions | n * log_2(n) |

**Concrete example for n = 2^20 (1,048,576 points) over BN254 scalar field:**

```
Multiplications: (2^20 / 2) * 20 = 2^19 * 20 = 524,288 * 20 = 10,485,760 ~ 10 million
Additions:       2^20 * 20 = 1,048,576 * 20 = 20,971,520 ~ 21 million
```

Each field multiplication over BN254 involves 254-bit modular multiplication (typically implemented via Montgomery multiplication -- about 4-8 machine multiplications on a 64-bit CPU using schoolbook or Karatsuba methods for the 256-bit limbs).

**Note:** Some of these multiplications are "trivial" (twiddle factor = 1 or -1) and can be optimized away. In practice, roughly 25% of twiddle factors at each stage are trivial.

---

### 2.7 Memory Access Patterns and Cache Unfriendliness

**The core problem:** The butterfly pattern at stage s accesses elements with stride 2^s (or 2^{k-s} depending on formulation). In early stages, the stride is small (adjacent elements -- cache-friendly). In later stages, the stride grows to n/2, causing cache misses on every access.

**Bit-reversal permutation:** The input to a decimation-in-time (DIT) NTT must be in bit-reversed order, or equivalently, the output appears in bit-reversed order. This permutation itself is cache-unfriendly for large n, as it scatters data across the entire array.

**Concrete impact:**
- For n = 2^20 with 32-byte field elements, the array is 32 MB
- At later stages, stride = 2^19 * 32 bytes = 16 MB between paired elements
- A typical L2 cache is 256 KB to 1 MB -- every access is a cache miss
- GPU implementations suffer from uncoalesced memory access: strided access wastes memory bandwidth because each memory transaction fetches a full cache line but only one element is used

**Optimization approaches:**
1. **Four-step / six-step NTT:** Decompose the NTT as a matrix transpose plus smaller NTTs. The transpose can be done cache-obliviously, and the sub-NTTs fit in cache.
2. **On-the-fly twiddle generation:** Instead of reading twiddle factors from memory (doubling memory bandwidth), generate them on the fly (reported 4.2x speedup on GPUs).
3. **Thread block coalescing (GPU):** Multiple thread blocks perform adjacent-point NTTs so GPU memory accesses coalesce.

---

### 2.8 Field Choice Constraints: Why p-1 Must Have a Large Power-of-2 Factor

**Requirement:** For NTT of size n = 2^k, we need a primitive n-th root of unity in F_p. This exists if and only if n divides p-1. Therefore, for radix-2 NTT of size 2^k, we need 2^k | (p-1).

**The 2-adic valuation:** The 2-adic valuation v_2(p-1) determines the maximum NTT size: n_max = 2^{v_2(p-1)}.

**Concrete curve parameters:**

| Curve | Scalar field r | Bits | v_2(r-1) | Max NTT size |
|---|---|---|---|---|
| BN254 | 21888242871839275222246405745257275088548364400416034343698204186575808495617 | 254 | 28 | 2^28 ~ 268M |
| BLS12-381 | 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001 | 255 | 32 | 2^32 ~ 4.3B |

**BLS12-381 was explicitly designed** with v_2(r-1) = 32 to support large NTTs for zkSNARK schemes. The hex representation of r-1 ends in `...ffffffff00000000`, clearly showing the factor of 2^32.

**BN254** supports NTTs up to 2^28 ~ 268 million points, which is sufficient for most current ZK circuits.

**What if n > n_max?** You cannot do a standard radix-2 NTT. Options include:
- Bluestein's algorithm (works for arbitrary n, uses a convolution of size 2n)
- Mixed-radix with other prime factors of p-1
- Zero-padding to the next power of 2

---

### 2.9 NTT as Percentage of Prover Time

The computational breakdown varies significantly by proof system:

| System | NTT/FFT | MSM | Other |
|---|---|---|---|
| **Groth16** (e.g., Rapidsnark) | ~30% | ~60-70% | ~5-10% |
| **PLONK** (pairing-based) | ~10% | ~85% | ~5% |
| **STARKs** (FRI-based) | ~60-80% | 0% (no EC) | 20-40% (hashing) |

**Key observations:**
- In pairing-based SNARKs (Groth16, PLONK), MSM dominates because the commitment step involves multi-scalar multiplication over elliptic curves
- In STARKs, there are no elliptic curves -- NTT/FFT and hashing are the dominant costs
- On CPU, approximately 98% of Rapidsnark's processing time is dedicated to NTT + MSM combined
- The often-cited "70-80% NTT" figure applies specifically to STARK provers where the polynomial evaluation over large domains (via NTT) and the FRI folding (which involves smaller NTTs at each round) together dominate

---

### 2.10 Mixed-Radix NTT: Radix-4, Radix-8

**Motivation:** Higher radix reduces the number of NTT stages and thus the multiplicative depth (number of sequential multiplications).

**Radix-4 butterfly:** Processes 4 elements at once. For inputs (a, b, c, d) and twiddle factors:

```
A = a + b + c + d
B = (a - c) + i*(b - d)     (where i = omega^{n/4})
C = a + b - c - d           (note: only additions/subtractions for some terms)
D = (a - c) - i*(b - d)
```

**Comparison:**

| Metric | Radix-2 | Radix-4 | Radix-8 |
|---|---|---|---|
| Stages for n-point NTT | log_2(n) | log_4(n) = log_2(n)/2 | log_8(n) = log_2(n)/3 |
| Mults per butterfly | 1 | 3 | ~7 |
| Total multiplications | (n/2)*log_2(n) | (3n/4)*log_4(n) = (3n/8)*log_2(n) | ~similar savings |
| Multiplicative depth | log_2(n) | log_2(n)/2 | log_2(n)/3 |

**Radix-4 advantage:** Reduces total multiplications by ~25% compared to radix-2 (ratio: 3/8 vs 1/2 per log_2(n) factor). Some implementations report ~20% reduction in modular multiplications and ~33% reduction in modular additions.

**Radix-8 advantage:** Further reduction in multiplicative depth (important for hardware pipelines and FHE where multiplicative depth is a constraint).

**Hybrid approaches:** Use radix-4 or radix-8 for larger stages (better arithmetic efficiency) and radix-2 for the final stages (simpler control logic). This is common in FPGA and ASIC implementations.

---

## 3. KZG (Kate-Zaverucha-Goldberg) Polynomial Commitment Scheme

### 3.1 Trusted Setup: The Structured Reference String (SRS)

**Setup procedure:** A trusted party (or distributed MPC ceremony) picks a random secret tau in F_p (the "toxic waste") and computes:

**SRS elements in G1:**

```
[1]_1, [tau]_1, [tau^2]_1, ..., [tau^d]_1
```

where [x]_1 denotes x * G_1 (scalar multiplication of the G1 generator by x).

In standard notation: g_1, tau*g_1, tau^2*g_1, ..., tau^d*g_1.

**SRS elements in G2** (only a few needed):

```
[1]_2, [tau]_2
```

i.e., g_2 and tau*g_2.

**SRS Size:** d+1 elements in G1 + 2 elements in G2.

**Concrete sizes for BLS12-381:**

| Component | Count | Bytes each | Total |
|---|---|---|---|
| G1 points (compressed) | d+1 | 48 bytes | 48*(d+1) |
| G2 points (compressed) | 2 | 96 bytes | 192 |

For degree d = 2^20 - 1 (about 1 million):

```
G1: 2^20 * 48 bytes = 1,048,576 * 48 = 50,331,648 bytes ~ 48 MB
G2: 192 bytes (negligible)
Total: ~48 MB
```

For degree d = 2^28 - 1 (about 268 million, large circuits):

```
G1: 2^28 * 48 bytes ~ 12.8 GB
```

After setup, tau must be irrecoverably destroyed -- this is the "toxic waste."

---

### 3.2 Commitment

**Formula:** To commit to polynomial f(x) = sum_{i=0}^{d} c_i * x^i:

```
C = f(tau) * G_1 = sum_{i=0}^{d} c_i * [tau^i]_1
```

This is a **multi-scalar multiplication (MSM)**: computing a linear combination of d+1 known G1 points with the polynomial coefficients as scalars.

**Observation:** The prover never learns tau. It computes the commitment using the SRS points [tau^i]_1 from the setup. The commitment C is a single G1 element (48 bytes compressed on BLS12-381).

**Prover cost:** One MSM of size d+1. Using Pippenger's algorithm, this costs approximately O(d / log d) group additions, which is sublinear in naive cost.

---

### 3.3 Opening Proof

**Goal:** Prove that f(z) = v for a public evaluation point z and claimed value v.

**Quotient polynomial:** If f(z) = v, then (x - z) divides f(x) - v, so we can compute:

```
q(x) = (f(x) - v) / (x - z)
```

This is a polynomial of degree d-1 (one less than f).

**Opening proof:**

```
pi = q(tau) * G_1 = sum_{i=0}^{d-1} q_i * [tau^i]_1
```

Again, this is an MSM, now of size d.

**Prover cost:** Computing q(x) requires O(d) field operations (synthetic division), plus one MSM of size d for the proof.

---

### 3.4 Verification Equation

**The verifier checks:**

```
e(C - v*G_1, G_2) = e(pi, [tau]_2 - z*G_2)
```

Equivalently written:

```
e(C - [v]_1, [1]_2) = e(pi, [tau - z]_2)
```

**Why this works (derivation):**

```
C = [f(tau)]_1
C - [v]_1 = [f(tau) - v]_1
pi = [q(tau)]_1

We need to verify: f(tau) - v = q(tau) * (tau - z)

LHS of pairing check: e([f(tau)-v]_1, [1]_2) = e(G_1, G_2)^{f(tau)-v}
RHS of pairing check: e([q(tau)]_1, [tau-z]_2) = e(G_1, G_2)^{q(tau)*(tau-z)}

These are equal iff f(tau) - v = q(tau)*(tau - z), which holds because
f(x) - v = q(x)*(x - z) is a polynomial identity (true for all x),
so in particular at x = tau.
```

**Verifier cost:** 2 group multiplications (compute C - v*G_1 and [tau]_2 - z*G_2) + 2 pairings. This is **constant** regardless of polynomial degree.

---

### 3.5 Batch Opening

**Scenario 1: One polynomial, multiple points.**

Given f(x) and evaluation points {z_1, ..., z_m} with claimed values {v_1, ..., v_m}:

Define the accumulator polynomial: A(x) = product_{i=1}^{m} (x - z_i)

Define the remainder polynomial R(x) as the unique polynomial of degree < m such that R(z_i) = v_i for all i (computed via Lagrange interpolation).

Compute the quotient: q(x) = (f(x) - R(x)) / A(x)

Proof: pi = [q(tau)]_1

Verification: e(C - [R(tau)]_1, G_2) = e(pi, [A(tau)]_2)

The verifier needs [A(tau)]_2 which can be computed from the SRS G2 elements (or precomputed for common A).

**Scenario 2: Multiple polynomials, same point.**

Given f_1, ..., f_t with commitments C_1, ..., C_t and evaluations f_i(z) = v_i:

The verifier sends random gamma in F_p. Define:

```
h(x) = sum_{i=1}^{t} gamma^{i-1} * (f_i(x) - v_i) / (x - z)
```

The proof is pi = [h(tau)]_1, and verification batches everything into a single pairing check.

**Scenario 3: Multiple polynomials, multiple points (BDFG20 protocol).**

This is the most general case, used in PLONK. The key idea: group polynomials by their evaluation points, create linearized combinations using verifier randomness, and reduce to a single pairing check.

---

### 3.6 Properties

| Property | Status | Notes |
|---|---|---|
| **Binding** | Computationally binding under d-SDH assumption | Breaking binding requires solving discrete log in G1 |
| **Hiding** | Not perfectly hiding (deterministic) | To add hiding: commit to f(x) + r*Z_H(x) with random r |
| **Homomorphic** | Yes -- additively homomorphic | C_f + C_g = commitment to f+g; alpha*C_f = commitment to alpha*f |
| **Succinct** | Proof size O(1), verifier time O(1) | Independent of polynomial degree |

**Homomorphic property detail:** Given commitments C_f = [f(tau)]_1 and C_g = [g(tau)]_1:

```
C_f + C_g = [f(tau)]_1 + [g(tau)]_1 = [(f+g)(tau)]_1 = C_{f+g}
```

This is used extensively in PLONK's linearization trick.

---

### 3.7 Cost Summary

| Operation | Prover | Verifier |
|---|---|---|
| Commit | 1 MSM of size d+1 | -- |
| Open (single point) | O(d) division + 1 MSM of size d | 2 EC mults + 2 pairings |
| Open (k points) | O(d) + 1 MSM | 2 EC mults + 2 pairings |
| Batch open (t polys, 1 point) | t MSMs (amortized) | 2 EC mults + 2 pairings |

**Concrete timing (approximate, BLS12-381 on modern CPU):**
- Pairing: ~1-2 ms each
- MSM of 2^20 points: ~500 ms - 1 s (Pippenger)
- Verifier total: ~2-4 ms (constant!)
- Prover total: dominated by MSM

---

### 3.8 Trusted Setup Ceremonies

**Zcash Powers of Tau (2016-2018):**
- One of the first large-scale ceremonies for Sprout and Sapling
- Phase 1: universal powers of tau (reusable across circuits)
- Phase 2: circuit-specific setup
- 6 participants in the original 2016 ceremony; expanded in later rounds
- Security: 1-of-N trust assumption (if even one participant destroyed their randomness honestly, the setup is secure)

**Ethereum KZG Ceremony (2023):**
- Purpose: generate SRS for EIP-4844 (proto-danksharding)
- Ran for 208 days (January 13 - August 8, 2023)
- 141,416 contributions -- the largest trusted setup ceremony ever
- Generated 2^12, 2^13, 2^14, 2^15 powers in G1 and 65 powers in G2 over BLS12-381
- 1-of-N trust: as long as one of the 141,416 contributors honestly destroyed their randomness, the SRS is secure

### 3.9 The Toxic Waste Problem

If any party learns the secret tau, they can:

1. **Forge proofs:** Create a valid-looking proof pi for any false statement f(z) = v' (where v' != f(z)) by computing the "quotient" directly using knowledge of tau.

2. **Break binding:** Construct two different polynomials that map to the same commitment.

**Mitigation:** Multi-party computation (MPC) ceremonies ensure tau = tau_1 + tau_2 + ... + tau_N (or multiplicative combination), where each participant contributes randomness. If even one participant destroys their share, tau is unrecoverable.

---

## 4. IPA (Inner Product Argument) -- Bulletproofs-Style Commitment

### 4.1 Overview and Comparison with KZG

| Property | KZG | IPA |
|---|---|---|
| Trusted setup | Required (SRS with toxic waste) | None (transparent) |
| Cryptographic assumption | Pairing + d-SDH | Discrete log only |
| Proof size | O(1) -- 1 group element | O(log n) -- log_2(n) group elements |
| Verifier time | O(1) -- 2 pairings | O(n) -- n group operations |
| Pairing-friendly curves needed | Yes | No (works on any prime-order group) |
| Post-quantum | No (pairings broken by quantum) | No (discrete log broken by quantum) |

---

### 4.2 Pedersen Vector Commitment

**Setup:** Choose a group G of prime order p with:
- A vector of generators **G** = (G_0, G_1, ..., G_{n-1}) -- n independent random group elements
- An additional generator H (for blinding)

No trusted setup: the generators can be derived from a hash function (nothing-up-my-sleeve).

**Commitment to coefficient vector a = (a_0, ..., a_{n-1}):**

```
C = <a, G> + r*H = sum_{i=0}^{n-1} a_i * G_i + r * H
```

where r is a random blinding factor.

**Properties:**
- Perfectly hiding (due to r*H)
- Computationally binding (under discrete log assumption)
- Commitment size: 1 group element (same as KZG!)

---

### 4.3 Polynomial Commitment via IPA

**Key observation:** Polynomial evaluation is an inner product. For polynomial p(x) = sum a_i * x^i, evaluating at z gives:

```
p(z) = <a, b>   where  a = (a_0, a_1, ..., a_{n-1})  and  b = (1, z, z^2, ..., z^{n-1})
```

So proving p(z) = v reduces to proving that the inner product of the committed vector a with the public vector b equals v.

---

### 4.4 The Recursive Halving Protocol

**Goal:** Prove <a, b> = v where a is committed in C and b is publicly known.

**Protocol (log_2(n) rounds):**

In each round j (for j = 1, ..., k = log_2(n)):

**Split:** Divide a, b, **G** into left and right halves:
- a = (a_lo, a_hi), each of length n/2^j
- b = (b_lo, b_hi)
- **G** = (G_lo, G_hi)

**Prover computes cross-terms:**

```
L_j = <a_lo, G_hi> + r_L * H + <a_lo, b_hi> * U
R_j = <a_hi, G_lo> + r_R * H + <a_hi, b_lo> * U
```

where U is a generator associated with the inner product claim.

**Prover sends** L_j and R_j to verifier (2 group elements per round).

**Verifier sends** random challenge u_j.

**Both sides fold:**

```
a' = u_j * a_lo + u_j^{-1} * a_hi          (vector of half length)
b' = u_j^{-1} * b_lo + u_j * b_hi
G' = u_j^{-1} * G_lo + u_j * G_hi
```

**After k rounds:** Vectors are reduced to single elements a', b', G'. The verifier checks:

```
a' * G' + r' * H + a'*b' * U  =  C' + sum_{j=1}^{k} (u_j^2 * L_j + u_j^{-2} * R_j)
```

---

### 4.5 Proof Size and Verification Cost

**Proof size:** 2 * log_2(n) group elements (one L and one R per round) + 2 field elements (final a', blinding).

For n = 2^20: proof = 2 * 20 = 40 group elements + 2 scalars. On a 256-bit curve with 32-byte compressed points: ~40 * 32 + 2 * 32 = 1,344 bytes ~ 1.3 KB.

Compare: KZG proof = 1 group element = 48 bytes (BLS12-381).

**Verification cost:**

The verifier must compute G' = u_1^{-1}*G_{lo,1} + u_1*G_{hi,1} at each round, which ultimately requires reconstructing an MSM over all n original generators:

```
G_final = sum_{i=0}^{n-1} s_i * G_i
```

where s_i depends on all challenges u_1, ..., u_k. This is an MSM of size n, costing **O(n)** group operations.

**This O(n) verifier cost is the main drawback of IPA vs. KZG.**

---

### 4.6 Used By

| System | Notes |
|---|---|
| **Bulletproofs** (Monero) | Range proofs, no trusted setup |
| **Halo 2** (Zcash) | PLONK + IPA, recursive proof composition without trusted setup |
| **Kimchi** (Mina/o1-labs) | Extended PLONK with IPA |

**Halo 2's key innovation:** By using IPA instead of KZG, Halo 2 eliminates the trusted setup requirement while achieving recursive proof composition. The O(n) verifier cost is handled via accumulation: instead of fully verifying each IPA proof, proofs are accumulated, and only one final O(n) verification is performed.

---

### 4.7 Hardware Implications

MSM is the bottleneck for both KZG and IPA:
- **KZG prover:** MSM of size n for commitment
- **IPA prover:** MSM of size n for commitment (same)
- **IPA verifier:** MSM of size n for reconstructing the generator (this is what makes IPA verification expensive)

Pippenger's algorithm for MSM of n points costs approximately n / log_2(n) group additions, which is the best known.

---

## 5. FRI (Fast Reed-Solomon Interactive Oracle Proofs of Proximity)

### 5.1 What FRI Proves

FRI proves that a function f: D -> F_p (given as evaluations on a domain D) is "close" to a polynomial of degree < d. More precisely, it proves that the Reed-Solomon codeword is delta-close to some low-degree polynomial, where delta relates to the code rate rho.

**Terminology:**
- Domain D: a subgroup of F_p* of size |D| = n
- Degree bound: d (the polynomial has degree < d)
- Code rate: rho = d/n (ratio of degree to domain size, typically rho = 1/2 or 1/4)
- "Close to low-degree" = within the unique decoding radius of the Reed-Solomon code

---

### 5.2 The Folding/Splitting Process

**Step 1: Split.** Decompose f(x) into even and odd parts:

```
f(x) = f_E(x^2) + x * f_O(x^2)
```

where:
- f_E(y) = (f(x) + f(-x)) / 2 -- the even-coefficient polynomial
- f_O(y) = (f(x) - f(-x)) / (2x) -- the odd-coefficient polynomial

If deg(f) < d, then deg(f_E) < d/2 and deg(f_O) < d/2.

**Step 2: Fold.** The verifier sends a random challenge alpha. The prover computes:

```
f*(y) = f_E(y) + alpha * f_O(y)
```

This is a random linear combination of f_E and f_O, and has degree < d/2.

**Step 3: Domain reduction.** The new polynomial f* is evaluated on the squared domain D* = {x^2 : x in D}. Since D is a multiplicative subgroup, D* has size |D|/2 (because x and -x map to the same x^2).

**Explicit folding formula from evaluations:**

```
f*(omega^{2i}) = (1/2) * [(1 + alpha * omega^{-i}) * f(omega^i) + (1 - alpha * omega^{-i}) * f(-omega^i)]
```

This can be computed entirely from the evaluations of f on D, without ever knowing the coefficients.

---

### 5.3 Number of Rounds

Starting with degree d and domain size n:

```
Round 1: degree d/2, domain size n/2
Round 2: degree d/4, domain size n/4
...
Round k: degree d/2^k, domain size n/2^k
```

After log_2(d) rounds, the polynomial is reduced to a constant (degree 0). The prover sends this constant directly.

**Total rounds:** ceil(log_2(d))

In practice, the prover may stop a few rounds early and send a polynomial of small degree (e.g., degree 16) directly, to save on communication.

---

### 5.4 Merkle Tree Commitments

At each FRI round, the prover commits to the evaluations of the current polynomial on its domain using a **Merkle tree**:

1. Compute all evaluations: {f*(omega^{2i})} for i = 0, ..., |D*|/2 - 1
2. Hash each evaluation to get leaves of a Merkle tree
3. Send the Merkle root to the verifier

**Query phase:** The verifier picks random indices and asks the prover to reveal:
- The evaluation at that index
- The Merkle authentication path (sibling hashes from leaf to root)

The verifier checks:
1. The Merkle path is consistent with the committed root
2. The folding relation is satisfied: f*(y) is consistent with f(x) and f(-x) via the folding formula

---

### 5.5 Proof Size

**FRI proof consists of:**
- log_2(d) Merkle roots (one per round): log_2(d) hashes
- For each query (s queries total):
  - log_2(d) authentication paths, each of depth log_2(n/2^i) for round i
  - Evaluation values at each round

**Total proof size:**

```
O(s * log_2(d) * log_2(n)) = O(lambda * log^2(d) / log(1/rho))
```

where s ~ lambda / log(1/rho) queries are needed for lambda bits of security, and each query has a Merkle path of depth ~log(n).

Since n ~ d/rho, and log(n) ~ log(d), the total is approximately:

```
O(lambda * log^2(d))    hash values
```

**Concrete example (128-bit security, d = 2^20, rho = 1/4):**
- Rounds: ~20
- Queries: s ~ 128 / log_2(4) = 64
- Per query: ~20 authentication paths of average depth ~10-20
- Total: roughly 64 * 20 * 15 ~ 19,200 hash values
- At 32 bytes/hash: ~600 KB

In practice, optimized implementations achieve ~50-200 KB for these parameters.

Compare: KZG proof = 48 bytes, IPA proof ~ 1.3 KB.

---

### 5.6 No Trusted Setup, No Pairings, No Elliptic Curves

FRI is a "transparent" scheme requiring only:
- A finite field F_p with roots of unity
- A collision-resistant hash function

**No elliptic curve operations at all.** The prover does field arithmetic (NTT for polynomial evaluation) and hashing (for Merkle trees). The verifier does field arithmetic (checking folding relations) and hashing (verifying Merkle paths).

**This is why FRI/STARKs are plausibly post-quantum secure:** they rely only on hash function collision resistance, not on discrete log or pairing assumptions.

---

### 5.7 Connection to STARKs

**FRI is THE polynomial commitment scheme for STARKs.**

STARK = Scalable Transparent ARgument of Knowledge

The STARK prover:
1. Encodes the computation trace as polynomials
2. Commits to these polynomials using Merkle trees (FRI-style)
3. Proves constraint satisfaction using the STARK IOP
4. Uses FRI to prove low-degree of quotient polynomials

FRI provides both the commitment mechanism and the low-degree test that underpins STARK soundness.

---

### 5.8 ZK-Friendly Hash Functions

Since FRI/STARK provers must hash extensively (for Merkle trees), and sometimes ZK circuits must verify STARK proofs (recursive composition), the choice of hash function matters enormously.

**Traditional vs. algebraic hashes:**

| Hash | R1CS constraints (SNARK) | AET (STARK) | Notes |
|---|---|---|---|
| SHA-256 | ~25,000 | ~140,000 | Bit-manipulation heavy, terrible in ZK circuits |
| MiMC | 628 | 1,944 | Simple x^3 or x^7 round function |
| Poseidon | 276 | 425 | HADES strategy, partial S-box layers |
| Rescue | 264 | 198 | Alternating forward/inverse S-boxes |
| Griffin | ~130-200 | ~100-200 | Horst construction, fewer inverse computations |
| Anemoi | ~130-200 | ~100-200 | Newer, claims 2-3x over Rescue |

**Why algebraic hashes are cheaper:** They use simple power maps (x -> x^alpha) over the field, which translate directly to a small number of field multiplications in the arithmetic circuit. SHA-256's bitwise operations (AND, XOR, rotations) require decomposing field elements into bits, which costs hundreds of constraints per operation.

**SHA-256 is 50-100x more expensive** than Poseidon/Rescue when computed inside a STARK.

---

## 6. Comparison Table of Polynomial Commitment Schemes

| Property | **KZG** | **IPA (Bulletproofs)** | **FRI (STARKs)** |
|---|---|---|---|
| **Trusted setup** | Yes (SRS with toxic waste) | No (transparent) | No (transparent) |
| **Cryptographic assumption** | Pairing + d-SDH | Discrete log | Collision-resistant hash |
| **Proof size** | O(1) -- 48 bytes (BLS12-381) | O(log n) -- ~1-2 KB | O(log^2 n) -- ~50-200 KB |
| **Prover time** | O(n log n) for NTT + O(n/log n) for MSM | O(n log n) for NTT + O(n) for MSM | O(n log n) for NTT + O(n log n) for hashing |
| **Verifier time** | O(1) -- 2 pairings (~2 ms) | O(n) -- n group ops | O(log^2 n * polylog) -- ~10-50 ms |
| **Post-quantum secure** | No | No | Plausibly yes |
| **Homomorphic** | Yes (additively) | Yes (additively) | No |
| **Curves required** | Pairing-friendly (BLS12-381, BN254) | Any prime-order group | None |
| **Hardware bottleneck** | MSM (prover), pairings (verifier) | MSM (both prover and verifier) | NTT + hashing |
| **Aggregation** | Efficient batching (constant proof) | Accumulation (Halo 2) | Aggregation via recursion |
| **Used in** | PLONK, Groth16, EIP-4844 | Bulletproofs, Halo 2, Kimchi | STARKs, RISC Zero, StarkWare |
| **Concrete proof (d=2^20)** | ~48 bytes | ~1.3 KB | ~50-200 KB |
| **Concrete verifier time** | ~2-4 ms (constant) | ~100-500 ms (depends on n) | ~10-50 ms |

**Summary of tradeoffs:**

- **KZG:** Smallest proofs, fastest verification, but requires trusted setup and is not post-quantum.
- **IPA:** No trusted setup, no pairings, small-ish proofs, but O(n) verifier is expensive for large n.
- **FRI:** No trusted setup, no elliptic curves, plausibly post-quantum, but largest proofs.

---

## 7. Best Resources for Learning

### Foundational Papers

1. **KZG original paper:** A. Kate, G. Zaverucha, I. Goldberg. "Polynomial Commitments" (2010). IACR ePrint 2010/079.
   - The primary academic reference.

2. **Bulletproofs / IPA:** B. Bunz, J. Bootle, D. Boneh, A. Poelstra, P. Wuille, G. Maxwell. "Bulletproofs: Short Proofs for Confidential Transactions and More" (2018). IEEE S&P.

3. **FRI original paper:** E. Ben-Sasson, I. Bentov, Y. Horesh, M. Riabzev. "Fast Reed-Solomon Interactive Oracle Proofs of Proximity" (2018). ICALP.

4. **PLONK:** A. Gabizon, Z. Williamson, O. Ciobotaru. "PLONK: Permutations over Lagrange-bases for Oecumenical Noninteractive arguments of Knowledge" (2019). IACR ePrint 2019/953.

5. **Groth16:** J. Groth. "On the Size of Pairing-based Non-interactive Arguments" (2016). EUROCRYPT.

6. **BDFG20 (batch KZG):** D. Boneh, J. Drake, B. Fisch, A. Gabizon. "Efficient polynomial commitment schemes for multiple points and polynomials" (2020). IACR ePrint 2020/081.

### Online Courses

7. **Berkeley ZKP MOOC** -- https://rdi.berkeley.edu/zkp-course/
   - Lecture 6: Polynomial Commitments (KZG, IPA)
   - Lecture 8: FRI-based Commitments and Fiat-Shamir
   - High quality, mathematically rigorous, freely available.

### Blog Posts and Tutorials

8. **Dankrad Feist: KZG Polynomial Commitments** -- https://dankradfeist.de/ethereum/2020/06/16/kate-polynomial-commitments.html
   - Excellent intuitive explanation with correct formulas.

9. **Dankrad Feist: Inner Product Arguments** -- https://dankradfeist.de/ethereum/2021/07/27/inner-product-arguments.html
   - Clear walkthrough of the IPA recursive halving.

10. **Alin Tomescu: KZG Commitments** -- https://alinush.github.io/kzg
    - Very detailed, includes batch opening, complexities, and implementation notes.

11. **Alin Tomescu: FRI** -- https://alinush.github.io/fri
    - Detailed FRI reference with formal definitions.

12. **Alan Szepieniec: Anatomy of a STARK** -- https://aszepieniec.github.io/stark-anatomy/
    - Part 3 covers FRI in depth, Part 6 covers optimization.
    - One of the best practical STARK tutorials.

13. **Vitalik Buterin: STARKs** -- https://vitalik.eth.limo/general/2017/11/09/starks_part_1.html (Parts 1-3)
    - Accessible introduction, good for intuition.

14. **Vitalik Buterin: Halo and more** -- https://vitalik.eth.limo/general/2021/11/05/halo.html
    - Covers IPA, accumulation, and recursive composition.

15. **RareSkills: Schwartz-Zippel Lemma** -- https://rareskills.io/post/schwartz-zippel-lemma
    - Concrete examples of why polynomial identity testing matters for ZK.

16. **LambdaClass: IPA and Polynomial Commitment** -- https://blog.lambdaclass.com/ipa-and-a-polynomial-commitment-scheme/
    - Step-by-step walkthrough with code.

17. **LambdaClass: How to code FRI from scratch** -- https://blog.lambdaclass.com/how-to-code-fri-from-scratch/
    - Hands-on implementation guide.

18. **Remco Bloemen: NTT** -- https://2-umb.com/23/ntt/
    - Deep dive into NTT algorithms, mixed-radix, and optimization.

19. **Cryptography Caffe: NTT Gentle Introduction** -- https://cryptographycaffe.sandboxaq.com/posts/ntt-02/
    - Part II covers Cooley-Tukey in detail.

20. **Ben Edgington: BLS12-381 For The Rest Of Us** -- https://hackmd.io/@benjaminion/bls12-381
    - Essential reference for understanding the curve parameters.

### Books

21. **Justin Thaler: "Proofs, Arguments, and Zero-Knowledge"** (2022)
    - The most comprehensive textbook. Covers polynomial commitments (KZG, IPA, FRI) with full proofs.
    - Freely available: https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.html

22. **David Pointcheval, et al.: "A Graduate Course in Applied Cryptography"** by Boneh & Shoup
    - Chapter on polynomial commitments provides rigorous foundations.

### Interactive / Code-Based Resources

23. **The halo2 Book** -- https://zcash.github.io/halo2/
    - Official documentation for Halo 2, covers IPA-based polynomial commitment in practice.

24. **ZKDocs** -- https://www.zkdocs.com/
    - Clear technical specifications for KZG, IPA, and related protocols.

25. **ICICLE (Ingonyama)** -- https://dev.ingonyama.com/
    - GPU-accelerated NTT and MSM implementations, good for understanding hardware optimization.

26. **Zellic: ZK-Friendly Hash Functions** -- https://www.zellic.io/blog/zk-friendly-hash-functions/
    - Detailed comparison of Poseidon, Rescue, Griffin with constraint counts.

### Recent Research (2024-2025)

27. **STIR: Reed-Solomon Proximity Testing with Fewer Queries** (2024) -- IACR ePrint 2024/390
    - Improves FRI proof size by 1.29x to 2.25x.

28. **Greyhound: Fast Polynomial Commitments from Lattices** (Crypto 2024)
    - First concretely efficient post-quantum polynomial commitment from standard lattice assumptions.

29. **FRIttata** (2025) -- IACR ePrint 2025/1285
    - FRI-based PCS for distributed proof generation, transparent and plausibly post-quantum.

---

*Document compiled March 2026. All formulas verified against primary sources. Field-specific parameters (BN254, BLS12-381) verified against curve specifications.*
