# C++ and SIMD (Phase 1 §4 — Sub-Track 1)

**Parent:** [C++ and Parallel Computing](../Guide.md)

> *Modern C++ gives you high-level syntax with low-level performance. Master the language features first — they are the building blocks of every parallel framework that follows.*

**Prerequisites:** Basic C programming (from Phase 1 §3 OS work). Familiarity with pointers, arrays, functions.

---

## Why Modern C++ First

Every parallel computing technology in this curriculum — SIMD intrinsics, OpenMP, oneTBB, CUDA, HIP, SYCL — is built on C++. But not the C++ of the 1990s. Modern C++ (C++11 through C++17) introduced features specifically designed for performance, safety, and parallel programming:

- **Lambdas** → used in oneTBB, SYCL, modern CUDA, parallel STL
- **Move semantics** → zero-copy data transfer, critical for GPU memory
- **Templates** → generic CUDA kernels, CUTLASS, CuTe
- **Parallel STL** → built-in SIMD + threading with one line of code
- **`constexpr`** → compile-time computation for HPC optimization

Learn these features now. You'll use every single one in Sub-Tracks 2–5.

---

## Section 1: Modern C++17 for Parallel Computing

### 1.1 Type Deduction (`auto`, `decltype`)

**`auto` — let the compiler figure out the type:**
```cpp
auto x = 10;          // int
auto y = 3.14;        // double
auto z = vec.begin(); // std::vector<int>::iterator — much cleaner
```

`auto` reduces verbosity and prevents type mismatch bugs. Use it for local variables, especially with complex types (iterators, template results).

**`decltype` — get the type of an expression:**
```cpp
int a = 5;
decltype(a) b = 10;   // b is int (same type as a)

// Useful in templates:
template<typename T, typename U>
auto multiply(T a, U b) -> decltype(a * b) {
    return a * b;
}
```

**Why it matters for parallel computing:** Generic code (templates, parallel algorithms) needs type deduction constantly. You'll see `auto` and `decltype` in every CUDA template library (CUTLASS, CuTe).

---

### 1.2 Smart Pointers (Memory Safety via RAII)

**The problem with raw pointers:**
```cpp
int* p = new int(5);
// ... 200 lines of code ...
// Did you remember to delete p? Memory leak.
// Did you delete it twice? Undefined behavior.
```

**The solution — RAII (Resource Acquisition Is Initialization):**
```cpp
#include <memory>

// Exclusive ownership — automatically freed when scope ends
auto p = std::make_unique<int>(5);
// No delete needed. Ever.

// Shared ownership — freed when last reference dies
auto q = std::make_shared<int>(10);
auto r = q;  // Reference count = 2
// Freed when both q and r go out of scope
```

**Three smart pointer types:**

| Type | Ownership | Overhead | Use when |
|------|-----------|----------|----------|
| `std::unique_ptr` | Exclusive (one owner) | Zero (same as raw pointer) | Default choice. Single owner. |
| `std::shared_ptr` | Shared (reference counted) | Atomic ref count (control block overhead, impl-dependent) | Multiple owners needed |
| `std::weak_ptr` | Non-owning observer | Minimal | Break circular references |

**Rule:** Never use `new`/`delete` in modern C++. Use `make_unique` and `make_shared`.

**Why it matters for parallel computing:**
- GPU memory (`cudaMalloc`/`cudaFree`) follows the same RAII pattern — wrap in a smart pointer or custom RAII class
- Thread-safe reference counting (`shared_ptr`) is essential for multi-threaded data sharing
- Memory leaks in HPC = your 80 GB GPU runs out of memory mid-training

---

### 1.3 Range-Based Loops

```cpp
std::vector<float> data = {1.0, 2.0, 3.0, 4.0};

// Modern (clean, less error-prone)
for (auto& x : data) {
    x *= 2.0f;
}

// Old style (verbose, index bugs)
for (int i = 0; i < data.size(); i++) {
    data[i] *= 2.0f;
}
```

**Why it matters:** Range-based loops work naturally with STL algorithms and parallel STL (`std::execution::par`). They're also easier for the compiler to auto-vectorize.

---

### 1.4 Structured Bindings (C++17)

Unpack tuples, pairs, and structs directly:

```cpp
// Pair unpacking
std::pair<int, float> p = {1, 2.5f};
auto [id, value] = p;   // id = 1, value = 2.5f

// Map iteration (clean)
std::map<std::string, int> scores = {{"alice", 95}, {"bob", 87}};
for (auto& [name, score] : scores) {
    std::cout << name << ": " << score << "\n";
}

// Function returning multiple values
auto [min_val, max_val] = std::minmax_element(data.begin(), data.end());
```

**Why it matters:** Cleaner data handling in performance-critical code. Used in profiling results, benchmark data, multi-return-value functions.

---

### 1.5 Lambda Functions (Critical for Parallel Computing)

Lambdas are **the single most important C++ feature for parallel programming**. Every parallel framework uses them.

**Basic lambda:**
```cpp
auto add = [](int a, int b) {
    return a + b;
};

int result = add(3, 4);  // 7
```

**Lambda with capture (access surrounding variables):**
```cpp
int multiplier = 10;

// [=] capture by value (copy)
auto scale_copy = [=](int x) { return x * multiplier; };

// [&] capture by reference (no copy, but thread-unsafe)
auto scale_ref = [&](int x) { return x * multiplier; };

// [multiplier] capture specific variable by value
auto scale_specific = [multiplier](int x) { return x * multiplier; };
```

**Where lambdas are used in this curriculum:**

| Framework | Lambda usage |
|-----------|-------------|
| **Parallel STL** | `std::sort(std::execution::par, v.begin(), v.end(), [](auto a, auto b) { return a > b; });` |
| **oneTBB** | `tbb::parallel_for(0, N, [&](int i) { C[i] = A[i] + B[i]; });` |
| **OpenMP** (C++17) | Used in task-based parallelism |
| **CUDA** (modern) | Device lambdas in `__device__` context |
| **SYCL** | `q.parallel_for(range, [=](id<1> i) { C[i] = A[i] + B[i]; });` |

**Key message:** If you don't understand lambdas, you can't write parallel code in modern C++.

---

### 1.6 `std::optional`, `std::variant`, `std::any` (C++17)

**`std::optional` — a value that may or may not exist:**
```cpp
#include <optional>

std::optional<int> find_index(const std::vector<int>& v, int target) {
    for (int i = 0; i < v.size(); i++) {
        if (v[i] == target) return i;
    }
    return std::nullopt;  // "not found" — no magic numbers like -1
}

auto result = find_index(data, 42);
if (result) {
    std::cout << "Found at index " << *result << "\n";
}
```

**`std::variant` — type-safe union:**
```cpp
#include <variant>

std::variant<int, float, std::string> v;
v = 42;         // holds int
v = 3.14f;      // now holds float
v = "hello";    // now holds string

// Visit pattern (useful for dispatch)
std::visit([](auto&& val) { std::cout << val << "\n"; }, v);
```

**Why it matters:** Safer API design. Avoids null pointer bugs and unsafe unions. Better error handling in production HPC code.

---

### 1.7 Move Semantics (Performance Core)

**The problem — expensive copies:**
```cpp
std::vector<float> create_large_buffer() {
    std::vector<float> buf(10000000);  // 40 MB
    // ... fill buffer ...
    return buf;  // Does this copy 40 MB? NO — move semantics!
}
```

**Copy vs move:**
```cpp
std::vector<int> a = {1, 2, 3, 4, 5};

// Copy: duplicates all data (expensive)
std::vector<int> b = a;           // b is a copy, a unchanged

// Move: transfers ownership (cheap — just pointer swap)
std::vector<int> c = std::move(a); // c owns the data, a is now empty
```

**How move works internally:**
```
Before move:
  a → [heap: 1, 2, 3, 4, 5]    (owns the buffer)

After std::move(a) → c:
  a → nullptr                    (empty, moved-from)
  c → [heap: 1, 2, 3, 4, 5]    (now owns the buffer)
```

No data was copied. Just three pointer/size assignments. O(1) instead of O(n).

**When to use `std::move`:**
- Returning large objects from functions (compiler often does this automatically — NRVO)
- Transferring ownership of buffers to another thread
- Moving data into containers: `vec.push_back(std::move(large_object));`

**Why it matters for parallel computing:**
- GPU memory transfers are expensive. Move semantics = zero-copy mindset.
- Thread pools move work items between threads without copying.
- CUDA's `cudaMemcpyAsync` + pinned memory is the GPU equivalent of move semantics.

---

### 1.8 Multithreading Basics (`std::thread`, `std::mutex`)

**Launching a thread:**
```cpp
#include <thread>

void compute(int id) {
    std::cout << "Thread " << id << " running\n";
}

int main() {
    std::thread t1(compute, 1);
    std::thread t2(compute, 2);

    t1.join();  // Wait for t1 to finish
    t2.join();  // Wait for t2 to finish
}
```

**Protecting shared data with mutex:**
```cpp
#include <mutex>

std::mutex mtx;
int shared_counter = 0;

void increment(int n) {
    for (int i = 0; i < n; i++) {
        std::lock_guard<std::mutex> lock(mtx);  // RAII lock
        shared_counter++;
    }  // lock released here automatically
}
```

**Better: `std::scoped_lock` (C++17) for multiple mutexes:**
```cpp
std::mutex m1, m2;
{
    std::scoped_lock lock(m1, m2);  // Locks both, prevents deadlock
    // ... critical section ...
}
```

**`std::atomic` for simple shared variables:**
```cpp
#include <atomic>

std::atomic<int> counter{0};

void increment(int n) {
    for (int i = 0; i < n; i++) {
        counter++;  // Thread-safe, no mutex needed
    }
}
```

**Why it matters:** `std::thread` and `std::mutex` are the primitives that OpenMP and oneTBB abstract over. Understanding them helps you debug race conditions and deadlocks in any parallel framework.

---

### 1.9 Parallel STL (C++17)

The most powerful single feature for parallelism — add one argument to any STL algorithm and it runs in parallel:

```cpp
#include <algorithm>
#include <execution>
#include <vector>

std::vector<float> data(10000000);

// Sequential (single core)
std::sort(std::execution::seq, data.begin(), data.end());

// Parallel (multi-threaded across all cores)
std::sort(std::execution::par, data.begin(), data.end());

// Parallel + vectorized (SIMD + threads)
std::sort(std::execution::par_unseq, data.begin(), data.end());
```

**Execution policies:**

| Policy | What it does | When to use |
|--------|-------------|-------------|
| `seq` | Sequential (default) | Small data, debugging |
| `par` | Multi-threaded | Large data, independent elements |
| `par_unseq` | Multi-threaded + SIMD vectorization | Maximum throughput, no dependencies |
| `unseq` (C++20) | SIMD only (single thread) | Vectorizable loop, single core |

**Works with many STL algorithms:**
```cpp
// Parallel transform (apply function to every element)
std::transform(std::execution::par, a.begin(), a.end(), b.begin(), c.begin(),
    [](float x, float y) { return x + y; });

// Parallel reduce (sum)
float total = std::reduce(std::execution::par, data.begin(), data.end());

// Parallel for_each
std::for_each(std::execution::par_unseq, data.begin(), data.end(),
    [](float& x) { x = std::sqrt(x); });
```

**Why this is important:** `par_unseq` combines SIMD vectorization with multi-threading in one line. No intrinsics, no OpenMP pragmas, no oneTBB templates. It's the easiest path to parallel code — and it works with any STL-compatible container.

> **`par_unseq` requirements:** Your loop body must have **no data races** (independent elements) AND **no loop-carried dependencies**. If element `i` depends on element `i-1`, `par_unseq` produces undefined behavior — not just wrong results, but undefined behavior.

> **Warning — not always faster:** Parallel STL has overhead (thread pool spin-up, synchronization). For small arrays (< ~100K elements), `seq` is often faster. In HPC and production ML systems, teams frequently prefer **OpenMP** or **oneTBB** for more predictable, tunable, and debuggable parallelism. Benchmark before committing to `par` or `par_unseq`.

**Compiler support:** GCC 9+ (with `-ltbb`), MSVC 2017+, Intel DPC++. Clang support is improving.

---

### 1.10 Templates & Generic Programming

**Function templates:**
```cpp
template<typename T>
T add(T a, T b) {
    return a + b;
}

add(3, 4);        // T = int
add(3.14, 2.71);  // T = double
```

**Class templates:**
```cpp
template<typename T, int N>
struct Vector {
    T data[N];
    T& operator[](int i) { return data[i]; }
};

Vector<float, 4> v;  // 4-element float vector
```

**Why templates matter for parallel computing:**
- CUDA kernels are often templated on data type and tile size: `matmul<float, 32><<<grid, block>>>(...)`
- CUTLASS is entirely template-based — configurable GEMM dimensions, precisions, and tiling
- CuTe (CUDA Template Engine) uses templates for layout and copy abstractions
- Zero-cost abstraction: templates are resolved at compile time, no runtime overhead

---

### 1.11 `constexpr` (Compile-Time Computation)

**Move computation from runtime to compile time:**
```cpp
constexpr int square(int x) {
    return x * x;
}

constexpr int tile_size = square(16);  // Computed at compile time = 256

// Use in array declarations, template parameters, etc.
float buffer[tile_size];  // float buffer[256] — no runtime cost
```

**`constexpr` + `if` (C++17):**
```cpp
template<typename T>
void process(T value) {
    if constexpr (std::is_integral_v<T>) {
        // Integer-specific code — compiled away for floats
    } else {
        // Float-specific code — compiled away for integers
    }
}
```

**Why it matters for HPC:**
- Tile sizes, block dimensions, and buffer sizes in GPU kernels are often compile-time constants
- `constexpr` enables the compiler to optimize aggressively (unroll loops, eliminate branches)
- Used extensively in CUTLASS for configurable kernel parameters

---

### 1.12 How C++ Features Map to Parallel Frameworks

| C++ Feature | Used in | How |
|-------------|---------|-----|
| **Lambdas** | oneTBB, SYCL, Parallel STL, modern CUDA | Kernel functions, task bodies, parallel_for callbacks |
| **Move semantics** | GPU memory, thread pools | Zero-copy buffer transfer, work item passing |
| **Templates** | CUDA, CUTLASS, CuTe, CK | Configurable kernels, generic algorithms |
| **`constexpr`** | CUDA, HLS, HPC libraries | Compile-time tile sizes, buffer dimensions |
| **`auto`** | Everywhere | Complex return types, template deduction |
| **Smart pointers** | Resource management | RAII wrappers for GPU memory, file handles |
| **Parallel STL** | CPU parallelism | One-line SIMD + threading |
| **`std::thread`/`mutex`** | Manual threading | Foundation for OpenMP, oneTBB internals |
| **`std::atomic`** | Lock-free algorithms | Counters, flags in multi-threaded code |
| **Structured bindings** | Data processing | Clean iteration over results, tuples |

---

## Section 2: SIMD on CPUs

> *The gateway to GPU thinking — same operation, multiple data.*

Now that you have the C++ foundation, apply it to the first level of parallelism: SIMD vector instructions.

### What SIMD Is

CPU vector instructions that process 4, 8, 16, or 32 data elements in one instruction:

| Instruction Set | Width | Data per instruction | Platform |
|----------------|-------|---------------------|----------|
| SSE (1999) | 128-bit | 4x float32 | x86 |
| AVX (2011) | 256-bit | 8x float32 | x86 |
| AVX-512 (2017) | 512-bit | 16x float32 | x86 (server) |
| ARM NEON | 128-bit | 4x float32 | ARM (Jetson, mobile) |

### Parallel Mental Models

SIMD is the first step in a hierarchy of data-parallel abstractions. Every level below uses the same fundamental idea:

| Concept | CPU SIMD | GPU (CUDA) | Hardware |
|---------|----------|------------|----------|
| **Data parallelism** | AVX lanes (8× float) | Warp (32 threads) | Vector unit |
| **Task parallelism** | `std::thread` / OpenMP | Thread blocks | Core clusters |
| **Memory hierarchy** | L1/L2 cache | Shared / global memory | SRAM / DRAM |
| **Latency hiding** | ILP + SIMD | Warp scheduling | Pipeline design |

Learn SIMD well and the GPU mental model becomes obvious — not a new paradigm, but SIMD at 32× width with explicit memory tiers.

---

### Auto-Vectorization (Compiler Does It)

The easiest path — write a simple loop, let the compiler vectorize:

```cpp
// This loop is auto-vectorizable
void add_vectors(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
```

```bash
# Check if compiler vectorized your loop
g++ -O2 -march=native -fopt-info-vec-optimized my_code.cpp
# Output: "loop vectorized using 32 byte vectors" = success
```

**Rules for auto-vectorization:**
- No loop-carried dependencies (`a[i] = a[i-1] + 1` cannot vectorize)
- Simple data types (float, int — not complex objects)
- Aligned memory helps (`alignas(32) float data[1024];`)
- Use `-O2` or `-O3` with `-march=native`

### Manual Intrinsics (Full Control)

When auto-vectorization fails or you need specific instructions:

```cpp
#include <immintrin.h>

void add_vectors_avx(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i += 8) {
        __m256 va = _mm256_load_ps(&a[i]);   // Load 8 floats from a
        __m256 vb = _mm256_load_ps(&b[i]);   // Load 8 floats from b
        __m256 vc = _mm256_add_ps(va, vb);   // Add 8 pairs at once
        _mm256_store_ps(&c[i], vc);           // Store 8 results to c
    }
}
```

**Key intrinsics patterns:**

| Operation | Intrinsic | What it does |
|-----------|-----------|-------------|
| Load aligned | `_mm256_load_ps(ptr)` | Load 8 **32-byte-aligned** floats — **crashes if misaligned** |
| Load unaligned | `_mm256_loadu_ps(ptr)` | Load 8 floats (any alignment) — safer default |
| Store | `_mm256_store_ps(ptr, v)` | Store 8 floats (aligned) |
| Stream store | `_mm256_stream_ps(ptr, v)` | Write bypassing cache — use for large write-once buffers |
| Add | `_mm256_add_ps(a, b)` | 8 parallel additions |
| Multiply | `_mm256_mul_ps(a, b)` | 8 parallel multiplications |
| **FMA** | `_mm256_fmadd_ps(a, b, c)` | 8 parallel `a*b + c` — **one instruction, not two; also more precise** |
| Broadcast | `_mm256_set1_ps(x)` | Fill all 8 lanes with same value |
| Horizontal add | `_mm256_hadd_ps(a, b)` | Add adjacent pairs within each 128-bit half |
| Compare | `_mm256_cmp_ps(a, b, _CMP_GT_OS)` | Per-lane compare → all-0s or all-1s mask |
| Blend | `_mm256_blendv_ps(a, b, mask)` | Select per-lane: `mask[i] ? b[i] : a[i]` |
| Permute (cross-lane) | `_mm256_permutevar8x32_ps(v, idx)` | Full cross-lane reorder by index vector |
| Gather | `_mm256_i32gather_ps(base, idx, 4)` | Load from non-contiguous addresses |

> **`_mm256_load_ps` alignment:** If `ptr` is not 32-byte aligned, this generates a segfault (`#GP` fault) at runtime — not a compile error. When in doubt, use `_mm256_loadu_ps` (the `u` = unaligned). On modern CPUs the performance difference is negligible.

> **FMA is the core of deep learning:** `_mm256_fmadd_ps(a, b, c)` computes `a*b + c` in a single fused instruction. GEMM (matrix multiplication), convolutions, and transformer attention inner loops are entirely composed of FMA. This is why FLOPS counts are usually reported as "FLOPS of FMA." On AVX2, 8-wide FMA at 2 ops/cycle = **16 GFLOPS/GHz per core**.

**Aligned memory:**
```cpp
// Aligned allocation (required for _mm256_load_ps)
alignas(32) float data[1024];

// Or dynamic allocation
float* data = (float*)aligned_alloc(32, 1024 * sizeof(float));
```

---

### Top 10 Most-Used Intrinsics in Real Codebases

Analysis of SIMD usage across GitHub repositories (production ML frameworks, databases, parsers) shows the same ~10 intrinsics appear in 80%+ of vectorized code. These are the ones worth memorizing first.

| Rank | Intrinsic | ISA | What it does | Why it's everywhere |
|------|-----------|-----|--------------|---------------------|
| 1 | `_mm256_add_ps(a, b)` | AVX | Add 8 float32 | Core of every FP loop |
| 2 | `_mm256_loadu_ps(ptr)` | AVX | Load 8 floats, unaligned | Default load — no alignment constraint |
| 3 | `_mm256_storeu_ps(ptr, v)` | AVX | Store 8 floats, unaligned | Default store |
| 4 | `_mm256_fmadd_ps(a, b, c)` | FMA+AVX2 | `a*b + c` in one instruction | GEMM, convolution, attention |
| 5 | `_mm_add_ps(a, b)` | SSE | Add 4 float32 (128-bit) | SSE compat code, horizontal ops |
| 6 | `_mm_set1_epi32(x)` | SSE2 | Broadcast int to all 4 lanes | Loading integer constants |
| 7 | `_mm256_cmp_ps(a, b, pred)` | AVX | Compare → bitmask | Conditional selection, NaN handling |
| 8 | `_mm_loadu_si128(ptr)` | SSE2 | Load 128-bit integer vector | String/byte processing |
| 9 | `_mm_shuffle_epi32(v, imm)` | SSE2 | Reorder 32-bit int lanes | Data rearrangement, AoS→SoA |
| 10 | `_mm_popcnt_u32(x)` | SSE4.2 | Count set bits in 32-bit int | Database filters, Hamming distance |

**Patterns behind the rankings:**

- **`_ps` (packed single) dominates** — float32 is the workhorse of ML inference and scientific computing. Learn `_ps` variants first.
- **Unaligned loads/stores are preferred** (`_loadu_`, `_storeu_`) — on Haswell and newer, the penalty for cache-line-crossing loads is ≤1 cycle. The code simplicity is worth it.
- **SSE2 128-bit still common** — many codebases use SSE2 for compatibility or for 128-bit sub-operations (horizontal reduction, remainder loops, byte processing).
- **`_mm_shuffle_epi32` is the workhorse for integer rearrangement** — appears in virtually every AoS→SoA conversion, hash function, and string parser.
- **`_mm_popcnt_u32/u64` is an outlier** — not strictly a SIMD register instruction (operates on scalar registers), but uses the SSE4.2 ISA feature bit and dominates database and bit-manipulation code (Hamming distance, counting set bits in filters).

```cpp
// The most common inner loop pattern in real ML code (ranks 2, 4, 3):
for (int i = 0; i < n; i += 8) {
    __m256 a = _mm256_loadu_ps(&A[i]);     // rank 2
    __m256 b = _mm256_loadu_ps(&B[i]);     // rank 2
    acc      = _mm256_fmadd_ps(a, b, acc); // rank 4
}
_mm256_storeu_ps(out, acc);                // rank 3
```

> **Source:** Usage statistics from [simd.info intrinsic statistics](https://simd.info/blog/intrinsic_statistics_repositories_insights_and_patterns/) and analysis of GitHub repositories including production ML frameworks, databases, and codec libraries.

---

### Cross-Lane Limitations (The AVX2 Gotcha)

Most AVX2 "256-bit" instructions actually execute as **two independent 128-bit halves**. This is the single most surprising architectural quirk in AVX2 and causes subtle bugs when you expect full cross-lane operations.

```cpp
// shuffle_ps looks like it works on 256-bit, but it only shuffles within each 128-bit half:
__m256 v = {7, 6, 5, 4, 3, 2, 1, 0};  // lanes 0-7
__m256 s = _mm256_shuffle_ps(v, v, 0b00011011);
// Result: {4,5,6,7, 0,1,2,3} — reversed within each half, NOT across the full register

// To truly cross lanes, you need explicit cross-lane moves:
// Swap the two 128-bit halves:
__m256 swapped = _mm256_permute2f128_ps(v, v, 0x01);

// Full cross-lane element reorder (AVX2) — the most flexible option:
__m256i idx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);  // reverse order
__m256 reversed = _mm256_permutevar8x32_ps(v, idx);       // true 8-element permute
```

| Operation | Cross-lane? | Notes |
|-----------|-------------|-------|
| `_mm256_shuffle_ps` | No | Within each 128-bit half only |
| `_mm256_hadd_ps` | No | Adjacent pairs within each half |
| `_mm256_permute2f128_ps` | Yes | Swaps/copies 128-bit halves |
| `_mm256_permutevar8x32_ps` | Yes | Full 8-element permute (AVX2 only) |

> **Rule of thumb:** If an intrinsic says "256-bit" but was available in AVX (not AVX2), suspect it operates on two 128-bit halves. Verify with [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/) or [uops.info](https://uops.info/).

---

### Inspecting SIMD Registers (Debugging)

You can't `printf` a `__m256`. The standard pattern is a union, which lets you read individual lanes:

```cpp
// Union for safe lane inspection (from hands-on-simd-programming)
union float8 {
    __m256  v;
    float   a[8];
    float8() : v(_mm256_setzero_ps()) {}
    float8(__m256 x) : v(x) {}
};

// Usage
float8 result(_mm256_fmadd_ps(va, vb, vc));
printf("lane 0: %f, lane 3: %f\n", result.a[0], result.a[3]);

// Or with structured binding in C++17:
auto [l0,l1,l2,l3,l4,l5,l6,l7] = result.a;
```

**Horizontal sum — reducing a vector to a scalar:**

This pattern appears everywhere: dot products, attention scores, reductions.

```cpp
// Sum all 8 floats in a __m256 to a scalar
float hsum(__m256 v) {
    // Step 1: add pairs within each 128-bit half → [a+e, b+f, c+g, d+h | a+e, b+f, c+g, d+h]
    __m128 lo  = _mm256_castps256_ps128(v);          // lower 128 bits
    __m128 hi  = _mm256_extractf128_ps(v, 1);        // upper 128 bits
    __m128 sum = _mm_add_ps(lo, hi);                 // add the two halves
    // Step 2: horizontal adds until scalar
    sum = _mm_hadd_ps(sum, sum);                     // [a+b+e+f, c+d+g+h, ...]
    sum = _mm_hadd_ps(sum, sum);                     // [total, total, ...]
    return _mm_cvtss_f32(sum);                       // extract lane 0
}
```

**Stream stores for large write-once buffers:**

```cpp
// Non-temporal store: bypasses CPU cache entirely
// Use when writing large output buffers you won't read back soon
// (e.g., preprocessing output, frame buffer writes)
for (int i = 0; i < n; i += 8) {
    __m256 result = /* compute */;
    _mm256_stream_ps(&out[i], result);   // skip cache, write direct to memory
}
_mm_sfence();  // memory fence — ensure all stream writes are visible

// When to use: write-once buffers > L3 cache size
// When NOT: if you'll read the data back soon (defeats the purpose)
```

**Gather — loading from non-contiguous addresses:**

```cpp
// Gather: load from base_ptr + indices[i] * scale
float table[1024] = { /* lookup table */ };
__m256i indices = _mm256_set_epi32(7, 3, 15, 0, 42, 8, 1, 100);  // arbitrary indices
__m256 gathered = _mm256_i32gather_ps(table, indices, 4);          // scale=4 (sizeof float)

// Note: gather throughput ≈ scalar loads — no bandwidth benefit for the gather itself.
// The value: the *computation* after gather runs vectorized (8 FMAs instead of 8 scalar ops).
// Good for: embedding lookups, sparse vector ops, non-sequential access patterns.
```

---

### Cache Lines and Memory Alignment

The CPU fetches memory in **64-byte cache lines** — not individual bytes. This is the physical reason SIMD alignment matters and the root cause of most false-sharing bugs in multithreaded code.

```
Cache line = 64 bytes = 16 floats = 8 doubles = 2 AVX registers

[float 0][float 1]...[float 15]   ← one cache line fetch
         └─ AVX load ──┘└─ AVX load ──┘  ← if misaligned, spans two lines
```

**Why it matters:**

| Scenario | Cache line impact |
|----------|------------------|
| `_mm256_load_ps` aligned | One cache line touched per load |
| `_mm256_loadu_ps` misaligned | May touch two cache lines — up to 2× memory traffic |
| False sharing (threads) | Two threads write different variables in same 64-byte line → cache ping-pong |
| `alignas(64)` struct | Guarantees struct starts at cache line boundary |

```cpp
// False sharing — avoid this
struct Workers {
    int counter_a;  // same 64-byte line as counter_b
    int counter_b;
};

// Fix: pad to cache line boundary
struct alignas(64) WorkerA { int counter; };
struct alignas(64) WorkerB { int counter; };
```

**Rule:** Align SIMD buffers to 32 bytes (`alignas(32)`) for AVX. Align per-thread data to 64 bytes (`alignas(64)`) to prevent false sharing.

---

### Data Layout: AoS vs SoA

How you arrange data in memory determines whether SIMD can vectorize it. This is one of the most impactful micro-architecture decisions in HPC.

**Array of Structs (AoS) — natural but SIMD-unfriendly:**
```cpp
struct Particle { float x, y, z, w; };
Particle particles[N];  // Memory: xyzw xyzw xyzw xyzw ...

// SIMD load picks up interleaved x,y,z,w — not what we want
// Must gather/scatter or use expensive shuffles
```

**Struct of Arrays (SoA) — SIMD-friendly:**
```cpp
struct Particles {
    float x[N], y[N], z[N], w[N];
};
Particles p;  // Memory: xxxx... yyyy... zzzz... wwww...

// Now AVX loads 8 consecutive x values in one instruction
for (int i = 0; i < N; i += 8) {
    __m256 vx = _mm256_load_ps(&p.x[i]);  // 8 x-coords, perfectly packed
    // process...
}
```

| Layout | Memory pattern | SIMD efficiency | Typical use |
|--------|---------------|-----------------|-------------|
| AoS | `xyzw xyzw xyzw` | Low (shuffles needed) | Game objects, general code |
| SoA | `xxxx yyyy zzzz` | High (contiguous lanes) | Physics simulations, ML kernels |
| AoSoA | `[xxxx yyyy][xxxx yyyy]` | High + cache-friendly | Intel oneDNN, CUTLASS |

> **Deep learning connection:** cuDNN uses NCHW (channels × height × width) and NHWC layouts. Switching between them is exactly an AoS↔SoA transformation. Tensor Core efficiency depends on the right layout.

---

### `std::simd` (C++26, Experimental)

The future of portable SIMD — no intrinsics, no platform-specific code:

```cpp
#include <experimental/simd>
namespace stdx = std::experimental;

void add_vectors(float* a, float* b, float* c, int n) {
    using V = stdx::native_simd<float>;  // Auto-selects best width
    for (int i = 0; i < n; i += V::size()) {
        V va(&a[i], stdx::element_aligned);
        V vb(&b[i], stdx::element_aligned);
        V vc = va + vb;
        vc.copy_to(&c[i], stdx::element_aligned);
    }
}
```

Available in GCC 11+ with `-std=c++23 -lstdc++exp`. Not yet production-ready but worth tracking.

### Measurement First

> *"Measure, don't guess. The bottleneck is never where you think it is."*

Before writing a single intrinsic, profile to confirm what kind of bottleneck you actually have. Optimizing the wrong thing wastes time.

```bash
# perf stat — hardware counter overview
perf stat -e instructions,cycles,cache-misses,fp_arith_inst_retired.256b_packed_single \
    ./my_program

# Key ratios to check:
#   instructions/cycles < 2.0  → probably stalled on memory
#   cache-misses high           → data layout problem (try SoA)
#   256b_packed_single low      → auto-vectorization failed, add -fopt-info-vec

# Confirm vectorization happened
g++ -O2 -march=native -fopt-info-vec-optimized my_code.cpp
# "loop vectorized using 32 byte vectors" = success

# VTune (Intel) — visual hotspots and vectorization analysis
vtune -collect hotspots ./my_program
```

### Roofline Model

The **Roofline model** tells you whether your kernel is compute-bound or memory-bound — and therefore whether SIMD helps.

```
Peak FLOPS  = freq × cores × SIMD_width × FMA_factor
Peak BW     = DRAM bandwidth (GB/s)
AI          = FLOPs / bytes_accessed  (arithmetic intensity)

If  AI > Peak FLOPS / Peak BW  → compute-bound  (SIMD / FMA helps)
If  AI < Peak FLOPS / Peak BW  → memory-bound   (better layout / caching helps)
```

**Example for a modern desktop CPU:**
```
Peak AVX2 FP32 = 3.6 GHz × 8 lanes × 2 (FMA) = ~58 GFLOPS/core
Peak DRAM BW   = ~50 GB/s
Ridge point AI = 58 / 50 ≈ 1.2 FLOP/byte

Vector add:  AI = 4 bytes out / 8 bytes in = 0.5 FLOP/byte → memory-bound
Dot product: AI = 2N FLOP / 2N×4 bytes    = 0.25 FLOP/byte → memory-bound
Matrix mult: AI = O(N³) FLOP / O(N²) bytes → compute-bound for large N ✓
```

**Implication:** SIMD gives the full 8× speedup only on compute-bound kernels. For memory-bound kernels (like vector add), the bottleneck is DRAM bandwidth — SIMD won't help much beyond reducing loop overhead.

### SIMD → GPU: Mental Model Bridge

The jump from SIMD to CUDA looks large but the underlying idea is the same. The key difference is *who* picks the element index:

```cpp
// ── SIMD mindset: one thread, one instruction covers N elements ──
for (int i = 0; i < n; i += 8) {
    __m256 va = _mm256_load_ps(&a[i]);
    __m256 vb = _mm256_load_ps(&b[i]);
    _mm256_store_ps(&c[i], _mm256_add_ps(va, vb));
}
// You write the stride (i += 8). The hardware runs 8 lanes silently.

// ── GPU / SIMT mindset: one thread per element, hardware spawns thousands ──
__global__ void add(float* a, float* b, float* c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) c[i] = a[i] + b[i];
}
// You write one element. The hardware runs 32 (warp) or thousands in parallel.
```

| Aspect | AVX (CPU SIMD) | CUDA warp (GPU SIMT) |
|--------|---------------|----------------------|
| Parallel width | 8× float32 | 32 threads |
| Register file | `__m256` (256-bit) | 32× 32-bit registers per thread |
| Memory | L1/L2 cache | Shared memory / global memory |
| Divergence cost | Branch kills vectorization | Warp divergence serializes threads |
| Programmer model | Explicit lanes (intrinsics) | Implicit (thread index) |

---

### Connection to the Stack

- **L1 (Application):** cuDNN and CUTLASS use vectorized memory loads internally
- **L2 (Compiler):** MLIR's `vector` dialect (Phase 4C) targets exactly this abstraction level
- **L5 (Architecture):** When you design an accelerator's vector unit, you're designing custom SIMD hardware
- **Sub-Track 3 (CUDA):** GPU "SIMT" is SIMD scaled to thousands of threads — same mental model

---

## Performance Traps and Common Mistakes

These are the issues that consume weeks of debugging in real HPC and ML systems. Learn them here.

| Trap | What happens | Fix |
|------|-------------|-----|
| **`shared_ptr` in hot loops** | Atomic ref-count inc/dec every iteration → cache line contention | Use raw pointer or `unique_ptr` inside hot paths; bump `shared_ptr` count once outside the loop |
| **False sharing** | Two threads update different variables in the same 64-byte cache line → invisible serialization | Pad thread-local data to `alignas(64)` |
| **Unaligned `_mm256_load_ps`** | Segfault (`#GP`) at runtime, not compile time | Use `_mm256_loadu_ps` unless you've verified 32-byte alignment with `alignas(32)` |
| **Branching inside vectorized loops** | Compiler cannot vectorize; scalar fallback runs | Move conditionals outside the loop; use SIMD blend instructions (`_mm256_blendv_ps`) |
| **Memory-bound SIMD** | SIMD adds complexity, negligible speedup | Profile first (Roofline); restructure data layout (SoA) before adding intrinsics |
| **`par_unseq` with dependencies** | Silent data corruption or undefined behavior | Only use when elements are **completely independent** — no reads from neighboring indices |
| **Ignoring NUMA** | Multi-socket systems: thread allocates memory on socket 0, socket 1 thread reads it → 2× bandwidth | Allocate memory on the NUMA node that will use it (`numactl`, `mbind`) |

---

## Projects

1. **Modern C++ warmup** — Write a matrix class using `std::vector`, move semantics, templates, and operator overloading. Verify that returning a large matrix from a function uses move (not copy) by checking with `-fno-elide-constructors`.
2. **Lambda benchmark** — Implement the same computation three ways: raw loop, `std::for_each` with lambda, and `std::transform` with lambda. Benchmark all three with `-O2`. Verify the compiler generates identical assembly (`-S`).
3. **Parallel STL sort** — Sort 10M floats with `std::execution::seq`, `par`, and `par_unseq`. Measure speedup. Plot time vs array size.
4. **Auto-vectorization** — Write a dot product loop. Compile with `-O2 -march=native -fopt-info-vec`. Check if the compiler vectorized it. If not, restructure the loop until it does.
5. **AVX intrinsics dot product** — Implement dot product using `_mm256_fmadd_ps`. Benchmark against scalar and auto-vectorized versions. Measure GFLOPS.
6. **Aligned vs unaligned** — Benchmark `_mm256_load_ps` (aligned) vs `_mm256_loadu_ps` (unaligned) for a large array. Measure the performance difference.
7. **AoS → SoA benchmark** — Implement vector dot product for an array of `Vec3` structs (AoS) and compare to the SoA layout version. Observe the speedup gap (expect 10–40× from layout alone).
8. **Horizontal sum** — Implement a dot product that uses `_mm256_fmadd_ps` to accumulate 8 at a time, then uses the horizontal sum idiom to reduce to a scalar. Verify against scalar result.
9. **Tiny MHA block** — Implement a batched multi-head attention forward pass for seq=8, head_dim=16, heads=4 using raw AVX2 intrinsics. Pre-transpose weight matrices. Benchmark against scalar version. Observe that the speedup is ~2–3×, not 8× — explain why (memory-bound).

---

## Resources

### Learning

| Resource | What it covers |
|----------|---------------|
| *A Tour of C++* (Stroustrup) | Modern C++ overview, concise |
| [cppreference.com](https://en.cppreference.com/) | Definitive C++ reference |
| *C++ Concurrency in Action* (Williams) | Threads, atomics, memory model |
| [SIMD for C++ Developers (const.me)](http://const.me/articles/simd/simd.pdf) | Best practitioner tutorial: loads, arithmetic, shuffles, masking, cross-lane gotchas. 23 pages. Read this first. |
| [hands-on-simd-programming](https://github.com/yuninxia/hands-on-simd-programming) | Progressive labs from basic intrinsics → image processing → full MHA block → quantized GPT decoder. Run `./runme.sh` to see benchmarks. |
| [awesome-simd](https://github.com/awesome-simd/awesome-simd) | Curated index of real-world SIMD libraries, tools, blogs, and references |
| [Agner Fog Optimization Guides](https://www.agner.org/optimize/) | Definitive reference on CPU micro-architecture, instruction latencies, calling conventions |

### Reference and Tooling

| Resource | What it covers |
|----------|---------------|
| [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/) | Searchable reference for every x86 SIMD intrinsic |
| [uops.info](https://uops.info/) | Instruction latency, throughput, and port usage for every CPU micro-architecture |
| [Compiler Explorer (godbolt.org)](https://godbolt.org/) | See assembly output for any C++ code — verify vectorization happened |
| [SIMD-Visualiser](https://github.com/piotte13/SIMD-Visualiser) | Browser-based visual diagram of what each intrinsic does to register contents |
| [Felix Cloutier x86 reference](https://www.felixcloutier.com/x86/) | HTML x86/x64 instruction reference |

### Portable SIMD Libraries (skip raw intrinsics)

| Library | Language | What it is |
|---------|----------|------------|
| [Google Highway](https://github.com/google/highway) | C++ | Best portable SIMD: length-agnostic, runtime dispatch, SSE/AVX/AVX-512/NEON/SVE |
| [xsimd](https://github.com/QuantStack/xsimd) | C++ | Header-only wrappers for SSE/AVX/AVX-512/NEON |
| [SIMDe](https://github.com/simd-everywhere/simde) | C/C++ | Emulates x86 intrinsics on ARM and other targets |
| [Intel ISPC](https://ispc.github.io/) | ISPC | C-like language that compiles to optimal SIMD for SSE/AVX/AVX-512/NEON/PS5/Xbox |

### Real-World SIMD Libraries to Study

| Library | What it does |
|---------|-------------|
| [simdjson](https://github.com/lemire/simdjson) | JSON parsing at >2 GB/s using SIMD — canonical example of real-world vectorized parsing |
| [SimSIMD](https://github.com/ashvardanian/SimSIMD) | SIMD-accelerated similarity measures (cosine, L2) for embedding vectors |
| [ncnn](https://github.com/Tencent/ncnn) | On-device neural network inference with hand-tuned SIMD for ARM and x86 |
| [StringZilla](https://github.com/ashvardanian/StringZilla) | SIMD substring search, fuzzy matching, sorting |

### Blogs (practitioners writing production SIMD code)

| Blog | Author |
|------|--------|
| [lemire.me/blog](https://lemire.me/blog/) | Daniel Lemire — simdjson, SIMDComp, practical vectorization |
| [branchfree.org](https://branchfree.org/) | Geoff Langdale — branchless and SIMD techniques |
| [0x80.pl/notesen](http://0x80.pl/notesen.html) | Wojciech Muła — low-level SIMD algorithms |

---

## Next

→ [**Sub-Track 2 — OpenMP and oneTBB**](../OpenMP%20and%20OneTBB/Guide.md) — scale from SIMD to multi-core CPU parallelism.
