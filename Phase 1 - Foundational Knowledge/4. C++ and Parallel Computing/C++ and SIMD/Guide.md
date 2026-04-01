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
| `std::shared_ptr` | Shared (reference counted) | Atomic ref count (~16 bytes) | Multiple owners needed |
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
| Load | `_mm256_load_ps(ptr)` | Load 8 aligned floats |
| Load unaligned | `_mm256_loadu_ps(ptr)` | Load 8 floats (any alignment) |
| Store | `_mm256_store_ps(ptr, v)` | Store 8 floats |
| Add | `_mm256_add_ps(a, b)` | 8 parallel additions |
| Multiply | `_mm256_mul_ps(a, b)` | 8 parallel multiplications |
| FMA | `_mm256_fmadd_ps(a, b, c)` | 8 parallel `a*b + c` (one instruction!) |
| Broadcast | `_mm256_set1_ps(x)` | Fill all 8 lanes with same value |

**Aligned memory:**
```cpp
// Aligned allocation (required for _mm256_load_ps)
alignas(32) float data[1024];

// Or dynamic allocation
float* data = (float*)aligned_alloc(32, 1024 * sizeof(float));
```

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

### Profiling SIMD Code

```bash
# perf (Linux) — see which instructions are hot
perf stat -e instructions,cycles,cache-misses ./my_program

# Check vectorization with perf
perf record -e fp_arith_inst_retired.256b_packed_single ./my_program
perf report

# VTune (Intel) — visual analysis of vectorization
vtune -collect hotspots ./my_program
```

**Key metric:** If your code is compute-bound, SIMD should give 4-8x speedup (SSE→AVX). If it doesn't, you're memory-bound — focus on data layout and cache.

### Connection to the Stack

- **L1 (Application):** cuDNN and CUTLASS use vectorized memory loads internally
- **L2 (Compiler):** MLIR's `vector` dialect (Phase 4C) targets exactly this abstraction level
- **L5 (Architecture):** When you design an accelerator's vector unit, you're designing custom SIMD hardware
- **Sub-Track 3 (CUDA):** GPU "SIMT" is SIMD scaled to thousands of threads — same mental model

---

## Projects

1. **Modern C++ warmup** — Write a matrix class using `std::vector`, move semantics, templates, and operator overloading. Verify that returning a large matrix from a function uses move (not copy) by checking with `-fno-elide-constructors`.
2. **Lambda benchmark** — Implement the same computation three ways: raw loop, `std::for_each` with lambda, and `std::transform` with lambda. Benchmark all three with `-O2`. Verify the compiler generates identical assembly (`-S`).
3. **Parallel STL sort** — Sort 10M floats with `std::execution::seq`, `par`, and `par_unseq`. Measure speedup. Plot time vs array size.
4. **Auto-vectorization** — Write a dot product loop. Compile with `-O2 -march=native -fopt-info-vec`. Check if the compiler vectorized it. If not, restructure the loop until it does.
5. **AVX intrinsics dot product** — Implement dot product using `_mm256_fmadd_ps`. Benchmark against scalar and auto-vectorized versions. Measure GFLOPS.
6. **Aligned vs unaligned** — Benchmark `_mm256_load_ps` (aligned) vs `_mm256_loadu_ps` (unaligned) for a large array. Measure the performance difference.

---

## Resources

| Resource | What it covers |
|----------|---------------|
| *A Tour of C++* (Stroustrup) | Modern C++ overview, concise |
| [cppreference.com](https://en.cppreference.com/) | Definitive C++ reference |
| [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/) | All SIMD intrinsics with search |
| [Compiler Explorer (godbolt.org)](https://godbolt.org/) | See assembly output for any C++ code |
| *C++ Concurrency in Action* (Williams) | Threads, atomics, memory model |

---

## Next

→ [**Sub-Track 2 — OpenMP and oneTBB**](../OpenMP%20and%20OneTBB/Guide.md) — scale from SIMD to multi-core CPU parallelism.
