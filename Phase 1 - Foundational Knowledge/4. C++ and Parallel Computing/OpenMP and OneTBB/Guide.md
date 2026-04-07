# OpenMP and oneTBB

Part of [Phase 1 section 4 — C++ and Parallel Computing](../Guide.md).

**Goal:** Shared-memory **CPU parallelism** with **OpenMP** (directive-based) and **oneTBB** (task-based algorithms and flow graphs) so structured parallel patterns feel familiar before CUDA.

---

## 1. Baseline: `std::thread` and Synchronization

Before frameworks, understand what they abstract over:

```cpp
#include <thread>
#include <mutex>
#include <atomic>

// Raw thread
std::thread t([]{ /* work */ });
t.join();

// Mutex for shared state
std::mutex mtx;
std::lock_guard<std::mutex> lock(mtx);  // RAII, released on scope exit

// Atomic for simple counters (no mutex needed)
std::atomic<int> counter{0};
counter.fetch_add(1, std::memory_order_relaxed);
```

**Key concepts:**
- **Data races** — two threads access the same memory, at least one writes, no synchronization → undefined behavior. Not a crash, just undefined.
- **Lock granularity** — coarse locks are safe but serialize; fine locks are fast but deadlock-prone.
- **Atomics** — cheaper than mutexes for single-variable shared state. Same concept as CUDA's `atomicAdd`.
- **Profiling first** — find hotspots with `perf stat` or VTune before parallelizing. The bottleneck is rarely the obvious loop.

OpenMP and oneTBB both build on these primitives. Understanding `std::thread` helps debug race conditions in any framework.

---

## 2. OpenMP

### 2.0 Mental Model: Fork-Join

OpenMP uses the **fork-join model**. The program starts with one thread (the *master*). When it hits a parallel region, it forks into a team of threads. When the region ends, all threads join back into one.

```
main thread ────────────────────────────────────────────────►
                    │                               │
            ┌───────┴──────────────────────────┐   │
            │    #pragma omp parallel           │   │
            │                                   │   │
thread 0 ───┤── work ── work ── work ────────── ┤───┤
thread 1 ───┤── work ── work ── work ────────── ┤───┤
thread 2 ───┤── work ── work ── work ────────── ┤───┤
thread 3 ───┤── work ── work ── work ────────── ┤───┤
            │                                   │   │
            └─────────────── implicit barrier ──┘   │
                                                     │
main thread continues ───────────────────────────────►
```

**The compiler does this for you:** `#pragma omp parallel for` is essentially a loop split + thread pool dispatch + barrier, all generated automatically.

**Compile:** `g++ -fopenmp -O2 my_code.cpp`

---

### 2.1 Your First Parallel Loop

```cpp
#include <omp.h>
#include <vector>

int N = 1'000'000;
std::vector<float> a(N), b(N), c(N);

// Sequential: thread 0 does all N iterations
for (int i = 0; i < N; i++)
    c[i] = a[i] + b[i];

// Parallel: 8 threads each do ~125,000 iterations
#pragma omp parallel for
for (int i = 0; i < N; i++)
    c[i] = a[i] + b[i];
```

That's it. One line added. The compiler splits `[0, N)` into chunks, assigns each chunk to a thread, and inserts a barrier at the end.

**What each thread sees:**

```
Thread 0:  i = 0 … 124,999
Thread 1:  i = 125,000 … 249,999
Thread 2:  i = 250,000 … 374,999
...
Thread 7:  i = 875,000 … 999,999
```

This is safe here because each `i` writes to a different `c[i]`. No two threads touch the same memory.

---

### 2.2 The Classic Race Condition

**Wrong — data race:**

```cpp
int sum = 0;

#pragma omp parallel for
for (int i = 0; i < N; i++)
    sum += a[i];   // ← RACE: multiple threads read-modify-write sum simultaneously

// sum is wrong. Possibly different on every run.
```

**Why it breaks:**

```
Thread 0 reads sum = 5
Thread 1 reads sum = 5     ← same value, before thread 0 wrote back
Thread 0 writes sum = 5 + a[0] = 7
Thread 1 writes sum = 5 + a[1] = 6   ← overwrites thread 0's result!
```

One update is lost. This is a classic **read-modify-write race**.

**Fix: `reduction` clause**

```cpp
int sum = 0;

#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < N; i++)
    sum += a[i];   // each thread accumulates its own private sum
                   // all private sums are added together at the end
```

Each thread gets its own private copy of `sum` (initialized to 0). After the loop, OpenMP adds all private copies into the original `sum`. No race.

---

### 2.3 Data Sharing Clauses

Every variable referenced inside a parallel region is either **shared** (one copy, all threads see it) or **private** (each thread has its own copy). OpenMP's default: variables declared outside the region are shared.

**No** — `private` and `firstprivate` are mutually exclusive for the same variable. A variable can only appear in one clause. They are alternatives that differ only in initialization:

```cpp
int x = 10;

// Option A — private(x): each thread gets its own x, value is UNINITIALIZED
// Use when: you assign x before reading it inside the loop anyway
#pragma omp parallel for private(x)
for (int i = 0; i < N; i++) {
    x = compute(i);   // x is assigned first → uninitialized value never read → safe
    a[i] *= x;
}

// Option B — firstprivate(x): each thread gets its own x, initialized to 10
// Use when: you read x before assigning it (need the original value)
#pragma omp parallel for firstprivate(x)
for (int i = 0; i < N; i++) {
    a[i] = x + i;    // x is read first → needs the initial value 10 → must use firstprivate
    x = compute(i);  // then overwritten — fine, it's private
}
```

**Rule:** use `private` when you always write before read. Use `firstprivate` when you need the original value inside the loop.

#### When `private` is the right choice

Use `private` when the variable is **purely a scratch/temp** — it gets completely overwritten at the start of every iteration, so the uninitialized value never matters:

```cpp
char buf[64];        // scratch buffer — we always snprintf before using it
float tmp;           // intermediate result — always assigned before read
int  row, col;       // 2D indices derived from i — always computed fresh

// All three are scratch: always written first → private is correct
#pragma omp parallel for private(buf, tmp, row, col)
for (int i = 0; i < N; i++) {
    row = i / COLS;                       // written first
    col = i % COLS;                       // written first
    tmp = heavy_compute(matrix[row][col]); // written first
    snprintf(buf, sizeof(buf), "(%d,%d)=%.2f", row, col, tmp);  // written first
    store_result(i, buf, tmp);
}
```

Think of it this way: **if you would declare the variable *inside* the loop body, it belongs in `private`**. `private` is just moving a loop-local variable outside (maybe because you need it in the pragma or it's a fixed-size buffer).

```cpp
// These are equivalent:

// Version A — variable declared inside loop (naturally private, no clause needed)
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    float tmp = a[i] * b[i];   // each iteration's own tmp
    result[i] = tmp + c[i];
}

// Version B — variable declared outside, made private with clause
float tmp;
#pragma omp parallel for private(tmp)
for (int i = 0; i < N; i++) {
    tmp = a[i] * b[i];         // same effect
    result[i] = tmp + c[i];
}
```

> **Prefer Version A** when possible — declare scratch variables inside the loop body. Use `private` only when you *must* declare the variable outside (e.g., fixed-size arrays, C99 VLAs, or compatibility constraints).

---

#### `lastprivate` — Getting the Value from the Last Iteration

`lastprivate` is private during the loop, but after the loop ends, the value from the **sequentially last iteration** (highest `i`) is copied back to the original variable.

**The problem it solves:** after a parallel loop, you sometimes need to know what a variable held in the final iteration — as if the loop ran serially.

```cpp
// Imagine you're sweeping x from 0.0 to 1.0 and computing sin(x) at each step.
// After the loop, you want sin(x) at the last step — i.e., sin(x_max).

const int N  = 1000;
const double dx = 1.0 / (N - 1);

double x      = 0.0;   // current x value
double sin_x  = 0.0;   // sin at current x

#pragma omp parallel for lastprivate(x, sin_x)
for (int i = 0; i < N; i++) {
    x     = i * dx;          // each iteration computes its own x
    sin_x = std::sin(x);     // and its own sin_x
    output[i] = sin_x;
}

// After the loop:
// x     = (N-1) * dx  = 1.0        ← value from i = N-1 (last iteration)
// sin_x = sin(1.0)    ≈ 0.8415     ← value from i = N-1 (last iteration)
printf("At x=%.4f, sin(x)=%.4f\n", x, sin_x);
```

**What each thread sees vs what you get back:**

```
Thread 0 processes i = 0..249:
  last x in its range = 249 * dx = 0.249
  last sin_x          = sin(0.249)

Thread 1 processes i = 250..499:
  last x in its range = 499 * dx = 0.499
  last sin_x          = sin(0.499)

Thread 2 processes i = 500..749:
  last x  = 0.749,  sin_x = sin(0.749)

Thread 3 processes i = 750..999:   ← sequentially last range
  last x  = 999 * dx = 1.0        ← THIS gets copied back
  sin_x   = sin(1.0)              ← THIS gets copied back

After join: x = 1.0, sin_x = sin(1.0) — as if the loop ran serially
```

> **`lastprivate` copies from the thread that handled the highest `i`**, not the thread that finished last in wall-clock time. The result is deterministic regardless of scheduling.

**Common real use cases:**

| Pattern | Why `lastprivate` |
|---------|------------------|
| Running total / recurrence after a sweep | Need final accumulated value |
| Finding the last element matching a condition | Carry the match out of the loop |
| Mesh traversal — record position of last node processed | Node index/coords after full sweep |
| Numerical integration — endpoint value needed post-loop | Value of integrand at upper bound |

**`lastprivate` + `firstprivate` on the same variable** — this *is* legal (they are not the same clause):

```cpp
double x = 0.5;   // starting x

// firstprivate: each thread starts with x = 0.5
// lastprivate:  after loop, x = value from last iteration
#pragma omp parallel for firstprivate(x) lastprivate(x)
for (int i = 0; i < N; i++) {
    x = x * 0.99 + i * dx;   // reads x (needs firstprivate) and updates it
    output[i] = x;
}
// x here = value of x after i = N-1, with each thread having started at 0.5
```

---

#### Full Clause Reference

| Clause | Initialized? | Written back after loop? | Use when |
|--------|:-----------:|:------------------------:|---------|
| `shared(var)` | — (one copy) | — (always live) | Read-only inside loop, or safely written with sync |
| `private(var)` | **No** | No | Scratch — always written before read |
| `firstprivate(var)` | **Yes** (from original) | No | Needs original value inside loop |
| `lastprivate(var)` | No | **Yes** (from last `i`) | Need loop's final value after it ends |
| `firstprivate` + `lastprivate` | **Yes** | **Yes** | Needs original value AND final value |
| `reduction(op:var)` | Yes (identity) | Yes (combined) | Accumulate across all iterations |

> **Rule of thumb:** If all threads only read → `shared`. Scratch variable → `private`. Needs original value → `firstprivate`. Need value after loop ends → `lastprivate`. Accumulate → `reduction`.

---

### 2.4 Schedules — How Work Is Divided

```cpp
// Static (default): divide evenly before the loop starts
// Thread 0 gets [0, N/P), thread 1 gets [N/P, 2N/P), ...
// Best when: each iteration costs the same amount of time
#pragma omp parallel for schedule(static)
for (int i = 0; i < N; i++) { /* uniform work */ }

// Static with chunk: interleave chunks of k iterations
// Thread 0 gets 0-7, 32-39, 64-71, ...   (chunk=8, 4 threads)
// Helps with cache locality in some patterns
#pragma omp parallel for schedule(static, 8)
for (int i = 0; i < N; i++) { /* */ }

// Dynamic: each thread takes the next k iterations when it becomes free
// Overhead: ~synchronization per chunk fetch
// Best when: iterations have unpredictable/varying cost
#pragma omp parallel for schedule(dynamic, 64)
for (int i = 0; i < N; i++) { /* variable-cost work */ }

// Guided: starts with large chunks, shrinks over time
// Reduces scheduling overhead while handling tail imbalance
// Best when: later iterations are lighter than earlier ones
#pragma omp parallel for schedule(guided)
for (int i = 0; i < N; i++) { /* */ }

// Runtime: schedule determined by OMP_SCHEDULE env variable
// OMP_SCHEDULE="dynamic,32" ./my_program
#pragma omp parallel for schedule(runtime)
for (int i = 0; i < N; i++) { /* */ }
```

**When to pick which:**

```
All iterations take the same time?     → static (lowest overhead)
Iterations have wildly different cost? → dynamic
Don't know, want adaptive?             → guided
Tuning at runtime without recompile?   → runtime
```

---

### 2.5 Reductions

Supported operators out of the box:

```cpp
int sum = 0, product = 1, max_val = INT_MIN;

#pragma omp parallel for reduction(+:sum) reduction(*:product) reduction(max:max_val)
for (int i = 0; i < N; i++) {
    sum      += a[i];
    product  *= b[i];
    max_val   = std::max(max_val, a[i]);
}
```

Built-in operators: `+`, `*`, `-`, `&`, `|`, `^`, `&&`, `||`, `min`, `max`.

**Custom reduction (C++ only, OpenMP 4.0+):**

```cpp
struct Vec3 { float x, y, z; };

#pragma omp declare reduction(vec_add : Vec3 : \
    omp_out.x += omp_in.x; \
    omp_out.y += omp_in.y; \
    omp_out.z += omp_in.z) \
    initializer(omp_priv = Vec3{0,0,0})

Vec3 total{0,0,0};

#pragma omp parallel for reduction(vec_add:total)
for (int i = 0; i < N; i++)
    total += forces[i];
```

---

### 2.6 Collapse — Parallelizing Nested Loops

```cpp
// Without collapse: only the outer loop is parallelized
// If outer loop has fewer iterations than threads → wasted threads
#pragma omp parallel for
for (int i = 0; i < 4; i++)
    for (int j = 0; j < 1000; j++)
        A[i][j] = B[i][j] * C[i][j];

// With collapse(2): outer * inner = 4000 iterations are parallelized
// Each thread gets a portion of the flattened 4000-iteration space
#pragma omp parallel for collapse(2)
for (int i = 0; i < 4; i++)
    for (int j = 0; j < 1000; j++)
        A[i][j] = B[i][j] * C[i][j];
```

> Use `collapse` when the outer loop count is small relative to the thread count.

---

### 2.7 Critical Sections and Atomics

When `reduction` is not enough (complex shared state):

```cpp
// critical: only one thread at a time
// Correct, but slow (acts like a mutex around the block)
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    #pragma omp critical
    {
        global_map[key(i)] += value(i);
    }
}

// atomic: faster for single memory operations (uses hardware atomics)
int counter = 0;
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    #pragma omp atomic
    counter++;           // ← hardware atomic, no mutex overhead
}

// atomic with operation
#pragma omp atomic update
total += a[i];

#pragma omp atomic read
int val = shared_var;

#pragma omp atomic write
shared_var = computed;
```

**Critical vs Atomic:**
- `#pragma omp critical` — mutex, any code block, higher overhead
- `#pragma omp atomic` — hardware instruction, single read/write/update only, much faster

---

### 2.8 Barrier and Nowait

```cpp
#pragma omp parallel
{
    // Threads do independent work
    do_phase_one(omp_get_thread_num());

    // Implicit barrier at end of 'parallel for' — all threads wait here
    // by default before continuing

    // 'nowait' removes the barrier — threads proceed immediately when done
    #pragma omp for nowait
    for (int i = 0; i < N; i++)
        a[i] = compute(i);

    // threads that finish early do NOT wait — they proceed here
    // only safe if subsequent code doesn't depend on all threads finishing

    // Explicit barrier — synchronize all threads at a specific point
    #pragma omp barrier
    // all threads are here now
}
```

---

### 2.9 Tasks — Irregular Parallelism

Tasks are for work that doesn't fit a regular loop: **recursive algorithms, tree traversal, linked lists**.

```cpp
// Recursive parallel tree sum using tasks
int sum_tree(Node* node) {
    if (!node) return 0;

    int left_sum, right_sum;

    #pragma omp parallel   // create thread team
    #pragma omp single     // only ONE thread creates tasks (others wait for tasks)
    {
        #pragma omp task shared(left_sum)
        left_sum = sum_tree(node->left);

        #pragma omp task shared(right_sum)
        right_sum = sum_tree(node->right);

        #pragma omp taskwait  // wait for both tasks before using results
    }

    return node->value + left_sum + right_sum;
}
```

**Fibonacci (classic task example):**

```cpp
int fib(int n) {
    if (n < 2) return n;

    int x, y;
    #pragma omp task shared(x) firstprivate(n)
    x = fib(n - 1);

    #pragma omp task shared(y) firstprivate(n)
    y = fib(n - 2);

    #pragma omp taskwait
    return x + y;
}

// Call from inside a parallel region:
int result;
#pragma omp parallel
#pragma omp single
result = fib(30);
```

> **`single` is critical:** without it, every thread would create tasks, creating an explosion. `single` ensures only one thread spawns the top-level tasks; the rest of the team executes them.

---

### 2.10 Sections — Different Code on Different Threads

```cpp
#pragma omp parallel sections
{
    #pragma omp section
    {
        printf("Thread %d: loading data\n", omp_get_thread_num());
        load_data();
    }

    #pragma omp section
    {
        printf("Thread %d: initializing config\n", omp_get_thread_num());
        init_config();
    }

    #pragma omp section
    {
        printf("Thread %d: warming up cache\n", omp_get_thread_num());
        warmup();
    }
}
// All three run in parallel, all done here
```

Use `sections` for a fixed, small number of different concurrent operations. For dynamic work, use `tasks`.

---

### 2.11 SIMD Pragma

```cpp
// Ask the compiler to vectorize (SIMD) one loop on a single thread
#pragma omp simd
for (int i = 0; i < N; i++)
    c[i] = a[i] * b[i] + d[i];

// Parallelize across threads AND vectorize within each thread's chunk
#pragma omp parallel for simd
for (int i = 0; i < N; i++)
    c[i] = a[i] * b[i];

// simd with reduction (e.g. sum with SIMD accumulation)
float sum = 0.0f;
#pragma omp simd reduction(+:sum)
for (int i = 0; i < N; i++)
    sum += a[i];
```

The `simd` pragma is a hint. The compiler still decides if vectorization is safe. Add `-fopt-info-vec` to see what was vectorized.

---

### 2.12 Thread Info and Environment

```cpp
// Query runtime info
int n_threads = omp_get_num_threads();   // inside parallel region
int thread_id = omp_get_thread_num();    // 0 to n_threads-1
int max_threads = omp_get_max_threads(); // outside parallel region
int n_procs = omp_get_num_procs();       // hardware thread count

// Set thread count
omp_set_num_threads(8);

// Timing
double t0 = omp_get_wtime();
// ... work ...
double elapsed = omp_get_wtime() - t0;
```

**Environment variables (set before running):**

```bash
OMP_NUM_THREADS=8             # number of threads
OMP_SCHEDULE="dynamic,64"     # schedule for 'runtime' clauses
OMP_PROC_BIND=close           # bind threads to nearby hardware (NUMA)
OMP_PLACES=cores              # binding granularity: threads, cores, sockets
GOMP_SPINCOUNT=100000         # how long to spin before sleeping (GNU)
```

---

### 2.13 Nested Parallelism

```cpp
omp_set_nested(1);  // enable nested parallel regions

void outer_task() {
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < 4; i++) {
        // Inner parallel region: each of 4 threads spawns 2 more
        #pragma omp parallel for num_threads(2)
        for (int j = 0; j < N; j++)
            compute(i, j);
    }
}
```

> Nested parallelism often over-subscribes cores. Usually better to `collapse(2)` instead.

---

### 2.14 Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Shared write without sync | Wrong results, non-deterministic | `reduction`, `atomic`, or `critical` |
| Capturing `i` by reference in tasks | Task reads wrong `i` | `firstprivate(i)` on the task |
| `single` missing in task-creation code | Task explosion | Add `#pragma omp single` around task creation |
| Calling `omp_get_num_threads()` outside parallel | Returns 1 always | Call inside the parallel region |
| Missing `taskwait` before using task results | Use before ready | `#pragma omp taskwait` |
| `nowait` on dependent loops | Uses wrong data | Remove `nowait` or add explicit `barrier` |
| Long critical sections | Serializes threads | Narrow the critical section, use `atomic` for simple ops |

---

### 2.15 OpenMP vs oneTBB

| | OpenMP | oneTBB |
|--|--------|--------|
| API style | Compiler directives (`#pragma`) | C++ templates and lambdas |
| Learning curve | Lower — one pragma per feature | Higher — need to know template types |
| Granularity | Loop-level | Task-level |
| Load balancing | Static/dynamic/guided schedules | Work-stealing (automatic, adaptive) |
| Flow graphs | No | Yes (`flow_graph`) |
| Per-thread storage | `threadprivate` / `private` clause | `enumerable_thread_specific` |
| Nested parallelism | Manual | Built-in |
| SIMD hints | `#pragma omp simd` | Must use manual intrinsics |
| Best for | Existing loops, Fortran interop, scientific HPC | New C++ code, complex graphs, irregular tasks |

**Resources:** [OpenMP specifications](https://www.openmp.org/specifications/) · [OpenMP API Reference Card (PDF)](https://www.openmp.org/wp-content/uploads/OpenMP-4.5-1115-CPP-web.pdf)

---

## 3. oneTBB (oneAPI Threading Building Blocks)

### 3.0 Mental Model: Work-Stealing Scheduler

oneTBB is a **task-based** parallel programming library. Instead of managing threads directly, you express *what* can run in parallel — the runtime distributes work using a **work-stealing** scheduler.

**How work-stealing works:**

```
Thread 0's deque:  [task A] [task B] [task C]  ← pushes/pops from right (LIFO, cache-friendly)
Thread 1's deque:  [] ← empty, idle
Thread 2's deque:  [task X]
Thread 3's deque:  [] ← empty, idle

Step 1: Thread 0 pops task C from its own deque (right end, hot in cache)
Step 2: Thread 1 is idle → STEALS task A from Thread 0's LEFT end
Step 3: Thread 3 is idle → STEALS task X from Thread 2
```

**Why steal from the left?**
- Owner pops from the **right** (LIFO) — processes its own tasks in stack order, cache-friendly
- Stealer pops from the **left** (FIFO) — takes the biggest, oldest tasks, maximizing stolen work size

**Result:** Threads never sit idle as long as any thread has work. No need to manually tune thread counts for imbalanced workloads.

**Install:**

```bash
# Ubuntu/Debian
sudo apt install libtbb-dev

# CMake integration
find_package(TBB REQUIRED)
target_link_libraries(my_target TBB::tbb)
```

**Header:**

```cpp
#include "oneapi/tbb.h"
using namespace oneapi::tbb;
```

---

### 3.1 `parallel_for` — Parallel Loop

The fundamental building block. Splits a range into chunks and executes each chunk on an available thread.

**Simple 1D integer form:**

```cpp
int N = 1'000'000;
float a[N], b[N], c[N];

// Compact form — runtime picks chunk size automatically
parallel_for(0, N, [&](int i) {
    c[i] = a[i] + b[i];
});
```

**`blocked_range` form — gives you the subrange, loop inside:**

```cpp
// Better: lets you write inner loops with fewer lambda calls
parallel_for(blocked_range<int>(0, N),
    [&](const blocked_range<int>& r) {
        for (int i = r.begin(); i < r.end(); i++) {
            c[i] = a[i] + b[i];
        }
    }
);
```

`blocked_range<T>(begin, end)` is the half-open interval `[begin, end)`. The lambda receives a subrange `r` — call `.begin()` and `.end()` to iterate it.

**Why `blocked_range` over compact form?**
- Avoids per-element lambda call overhead
- Lets you initialize buffers once per chunk (instead of per element)
- Required for custom chunk-level logic (SIMD, temp allocation)

**2D range — matrix or image processing:**

```cpp
parallel_for(blocked_range2d<int>(0, rows, 0, cols),
    [&](const blocked_range2d<int>& r) {
        for (int i = r.rows().begin(); i < r.rows().end(); i++)
            for (int j = r.cols().begin(); j < r.cols().end(); j++)
                out[i][j] = process(in[i][j]);
    }
);
```

**3D range:**

```cpp
parallel_for(blocked_range3d<int>(0, D, 0, H, 0, W),
    [&](const blocked_range3d<int>& r) {
        for (int d = r.pages().begin(); d < r.pages().end(); d++)
            for (int h = r.rows().begin(); h < r.rows().end(); h++)
                for (int w = r.cols().begin(); w < r.cols().end(); w++)
                    vol[d][h][w] = f(d, h, w);
    }
);
```

#### Partitioners — Control Chunking

The third optional argument controls how work is divided:

```cpp
// auto_partitioner (default): runtime tunes chunk size automatically
parallel_for(blocked_range<int>(0, N), body);

// affinity_partitioner: reuses same data → same thread (cache-warm)
// Declare static so it persists between calls
static affinity_partitioner ap;
parallel_for(blocked_range<int>(0, N), body, ap);

// simple_partitioner: chunk = grainsize exactly, no adaptive splitting (work-stealing still active)
parallel_for(blocked_range<int>(0, N, 1000), body, simple_partitioner());

// static_partitioner: divide evenly upfront, no stealing
// Deterministic: same thread always gets same range
parallel_for(blocked_range<int>(0, N), body, static_partitioner());
```

| Partitioner | Chunk size | Adaptive splitting | Work-stealing | Use when |
|-------------|------------|--------------------|---------------|---------|
| `auto_partitioner` | Dynamic | Yes | Yes | Most cases — default |
| `affinity_partitioner` | Adaptive | Yes | Yes | Re-running same loop over same data (cache-warm) |
| `simple_partitioner` | Fixed ≈ grainsize | No | Yes | Chunk needs fixed-size temp buffer; work is uniform; N is huge |
| `static_partitioner` | Pre-divided, fixed mapping | No | No | NUMA/cache locality tuning; strict reproducible thread mapping |

**Grainsize tuning:** Each chunk should take ≥ ~100,000 clock cycles (~50 µs at 2 GHz). Too-small chunks = scheduling overhead dominates. Rule: if loop body takes 10 ns, grainsize of 10,000+ is reasonable.

#### Complete Example 1 — Squaring an Array

The clearest way to see the transition from serial to parallel:

```cpp
#include <oneapi/tbb.h>
#include <iostream>
#include <vector>

int main() {
    const size_t N = 1'000'000;
    std::vector<int> data(N, 3);    // all 3s
    std::vector<int> result(N);

    // Serial version:
    // for (size_t i = 0; i < N; ++i)
    //     result[i] = data[i] * data[i];

    // Parallel version — one line change:
    oneapi::tbb::parallel_for(size_t(0), N, [&](size_t i) {
        result[i] = data[i] * data[i];   // each thread handles a range of i
    });

    std::cout << "result[0]=" << result[0]
              << "  result[N-1]=" << result[N-1] << "\n";
    // Expected: 9  9
}
```

**What the runtime does:**

```
N = 1,000,000   threads = 8

Thread 0:  i = 0 … 124,999       → result[i] = data[i]²
Thread 1:  i = 125,000 … 249,999 → result[i] = data[i]²
...
Thread 7:  i = 875,000 … 999,999 → result[i] = data[i]²

All done → join back
```

No race condition: each thread writes to a different `result[i]`.

---

#### Complete Example 2 — 2D Image Box Blur

`blocked_range2d` splits a 2D pixel grid into rectangular tiles. Each tile stays hot in CPU cache because rows and columns are contiguous.

```cpp
#include <oneapi/tbb.h>
#include <iostream>
#include <vector>

int main() {
    const size_t W = 800, H = 600;

    // 2D image as flat vector, row-major: pixel(y,x) = image[y * W + x]
    std::vector<int> image (H * W, 100);   // all pixels = 100
    std::vector<int> blurred(H * W, 0);

    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range2d<size_t>(0, H, 0, W),
        [&](const oneapi::tbb::blocked_range2d<size_t>& tile) {
            for (size_t y = tile.rows().begin(); y < tile.rows().end(); ++y) {
                for (size_t x = tile.cols().begin(); x < tile.cols().end(); ++x) {
                    // 3×3 box blur — average the 9 surrounding pixels
                    int sum = 0;
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            size_t nx = (x + dx < W) ? x + dx : x;
                            size_t ny = (y + dy < H) ? y + dy : y;
                            sum += image[ny * W + nx];
                        }
                    }
                    blurred[y * W + x] = sum / 9;
                }
            }
        }
    );

    std::cout << "blurred pixel(300,400) = " << blurred[300 * W + 400] << "\n";
    // Expected: 100 (uniform image, blur changes nothing)
}
```

**Why `blocked_range2d` beats two nested `parallel_for` calls:**

```
blocked_range2d splits the grid into rectangular tiles:

  ┌───────┬───────┬───────┐
  │ T0    │ T1    │ T2    │   Each tile = contiguous memory block
  │       │       │       │   → good L1/L2 cache reuse within tile
  ├───────┼───────┼───────┤
  │ T3    │ T4    │ T5    │
  │       │       │       │
  └───────┴───────┴───────┘

Two nested parallel_for → each row is a separate task → too many tiny tasks
blocked_range2d    → each tile is one task → right granularity
```

The `tile.rows()` and `tile.cols()` accessors give you the subrange for this tile. Always use flat `vector<T>` with `[y * W + x]` indexing — `vector<vector<T>>` breaks spatial locality.

---

### 3.2 `parallel_reduce` — Parallel Reduction

For loops that accumulate a result: sum, min, max, dot product, histogram.

**Lambda form (most common):**

```cpp
// Sum of array
float total = parallel_reduce(
    blocked_range<int>(0, N),
    0.0f,                                        // identity value
    [&](const blocked_range<int>& r, float init) -> float {
        for (int i = r.begin(); i < r.end(); i++)
            init += a[i];
        return init;
    },
    [](float x, float y) -> float {             // combine partial results
        return x + y;
    }
);
```

**Arguments:**
1. Range to iterate
2. Identity value (initial value for each partial result)
3. Body: takes a subrange + running partial → returns new partial
4. Combine: merges two partial results into one

**Dot product:**

```cpp
float dot = parallel_reduce(
    blocked_range<int>(0, N), 0.0f,
    [&](const blocked_range<int>& r, float init) -> float {
        for (int i = r.begin(); i < r.end(); i++)
            init += a[i] * b[i];
        return init;
    },
    std::plus<float>{}
);
```

**Find minimum with index:**

```cpp
struct MinResult { float val; int idx; };

auto result = parallel_reduce(
    blocked_range<int>(0, N),
    MinResult{FLT_MAX, -1},
    [&](const blocked_range<int>& r, MinResult curr) {
        for (int i = r.begin(); i < r.end(); i++)
            if (a[i] < curr.val) curr = {a[i], i};
        return curr;
    },
    [](MinResult a, MinResult b) {
        return a.val < b.val ? a : b;
    }
);
```

> **Common mistake:** Do not reset the accumulator to zero inside the body — `init` carries prior partial results. Resetting it discards prior work.

**Deterministic reduce** — same result every run regardless of scheduling:

```cpp
float result = parallel_deterministic_reduce(
    blocked_range<int>(0, N, 1000),  // explicit grainsize required
    0.0f,
    [&](const blocked_range<int>& r, float init) {
        return std::accumulate(&a[r.begin()], &a[r.end()], init);
    },
    std::plus<float>{}
);
// Floating-point result identical on every run
```

#### Complete Example 3 — Monte Carlo π Estimation

Monte Carlo π estimation is a classic parallel workload: throw random points at a unit square, count how many land inside the quarter-circle, then estimate π from the ratio.

```
  1 ┤        ╭──────╮
    │      ╭─╯      │
    │    ╭─╯  ●●●   │   ● = inside circle  (x²+y² ≤ 1)
    │  ╭─╯ ●●●●●●●  │   ○ = outside circle
    │ ─╯ ●●●●●●●●●  │
    │  ○○○●●●●●●●●  │
    │  ○○○○●●●●●●●  │
    │  ○○○○○●●●●●   │
  0 ┼──────────────── 1
       π ≈ 4 × (hits / total)
```

**`parallel_reduce` lambda form — complete example:**

```cpp
#include <oneapi/tbb.h>
#include <iostream>
#include <random>

int main() {
    const size_t N = 10'000'000;

    // parallel_reduce<Range, Value>(range, identity, body, combine)
    //
    //  body(range, running_value) → new_value   (reduce a chunk)
    //  combine(a, b)              → merged      (merge two chunk results)
    long long hits = oneapi::tbb::parallel_reduce(
        oneapi::tbb::blocked_range<size_t>(0, N),

        0LL,          // identity — each thread starts its partial count at 0

        // ── body: called once per subrange, on the thread that owns it ──
        [](const oneapi::tbb::blocked_range<size_t>& r, long long init) {
            // Construct a separate RNG per invocation.
            // A single shared mt19937 would need a mutex on every call,
            // serializing the whole loop. Per-thread RNGs = zero contention.
            std::mt19937 rng(std::random_device{}());
            std::uniform_real_distribution<double> dist(0.0, 1.0);

            for (size_t i = r.begin(); i < r.end(); ++i) {
                double x = dist(rng), y = dist(rng);
                if (x*x + y*y <= 1.0) ++init;   // accumulate into running total
            }
            return init;   // hand partial count back to TBB
        },

        // ── combine: called to merge two partial counts into one ──
        std::plus<long long>{}
    );

    double pi = 4.0 * hits / N;
    std::cout << "Estimated π = " << pi << "\n";
    // Typical output: Estimated π = 3.14159...
}
```

**How TBB splits and joins the work:**

```
           parallel_reduce(blocked_range(0, 10M), 0LL, body, combine)
                              [0 … 10,000,000)
                             /                \
                            /                  \
                [0 … 5,000,000)          [5,000,000 … 10M)
               /               \         /               \
       [0…2.5M)           [2.5M…5M)  [5M…7.5M)     [7.5M…10M)
       hits=1,963k        hits=1,964k hits=1,963k   hits=1,964k
              \               /         \               /
           combine(a,b) = a+b           combine(a,b) = a+b
               hits=3,927k                  hits=3,927k
                          \                 /
                           combine(a,b) = a+b
                               hits=7,854k
                           π ≈ 4×7,854k/10M ≈ 3.1416
```

Each leaf calls `body(subrange, 0LL)` — the identity `0LL` means every thread starts counting from zero. After all leaves finish, TBB walks back up the tree calling `combine` to sum the partial counts.

---

### 3.3 `parallel_scan` — Prefix Sum

A prefix sum (scan) gives every output element the running total up to that point:

```
input:  [ 1,  2,  3,  4,  5 ]
output: [ 1,  3,  6, 10, 15 ]
         ↑   ↑   ↑   ↑   ↑
         1  1+2 1+2+3 ...
```

Naively this is sequential — each output depends on the previous one. `parallel_scan` breaks it into two passes so chunks run in parallel:

```
Pass 1 — "pre-scan" (is_final = false):
  Chunk A [0..2]: partial sum = 1+2+3 = 6       (don't write output yet)
  Chunk B [3..4]: partial sum = 4+5   = 9       (don't write output yet)

Pass 2 — "final scan" (is_final = true):
  Chunk A [0..2]: carry-in = 0  → write 1, 3, 6
  Chunk B [3..4]: carry-in = 6  → write 10, 15

Combine: TBB calls combine(6, 9) = 15 to pass carry-in to Chunk B
```

The body lambda runs **twice per chunk** — once without writing (building partial sums), once with the correct carry-in (filling output). The `is_final` flag tells the body which pass it is.

```cpp
#include "oneapi/tbb/parallel_scan.h"

std::vector<float> in(N), out(N);

tbb::parallel_scan(
    tbb::blocked_range<int>(0, N),
    0.0f,                   // identity value (carry-in for first chunk)
    [&](const tbb::blocked_range<int>& r, float running, bool is_final) {
        for (int i = r.begin(); i < r.end(); i++) {
            running += in[i];
            if (is_final)
                out[i] = running;   // only write on second pass
        }
        return running;             // hand partial sum to TBB
    },
    [](float a, float b) { return a + b; }   // how to merge two partial sums
);
// out[i] = in[0] + in[1] + ... + in[i]
```

Use cases: cumulative sums, exclusive scan for stream compaction, computing CDF from a histogram.

---

### 3.4 `parallel_sort`

```cpp
#include "oneapi/tbb/parallel_sort.h"

std::vector<int> data(N);
// ...fill data...

// In-place sort (like std::sort but parallel)
parallel_sort(data.begin(), data.end());

// Custom comparator
parallel_sort(data.begin(), data.end(), std::greater<int>{});

// Sort struct by field
parallel_sort(records.begin(), records.end(),
    [](const Record& a, const Record& b) {
        return a.score > b.score;
    }
);
```

Uses a parallel quicksort variant with work-stealing. Typically 4–6× faster than `std::sort` on 8+ cores for large N.

> **Note:** `parallel_sort` is **not stable** (equal elements may reorder). Use `std::stable_sort` for stable ordering (no TBB parallel version exists for stable sort).

---

### 3.5 `parallel_for_each` — Unknown Iteration Space

For containers without random-access iterators (linked lists, sets) or when the iteration space grows during execution:

```cpp
std::list<WorkItem> items = get_work();

parallel_for_each(items.begin(), items.end(),
    [](WorkItem& item) {
        process(item);
    }
);
```

**Dynamic work addition with feeder** — BFS / tree traversal:

```cpp
parallel_for_each(roots.begin(), roots.end(),
    [](Node* node, feeder<Node*>& f) {
        process(node);
        for (Node* child : node->children)
            f.add(child);             // adds work dynamically to the pool
    }
);
// Runs until all nodes processed, including dynamically added ones
```

---

### 3.6 `parallel_pipeline` — Assembly Line

A pipeline is a sequence of stages where each item flows through all stages in order. Unlike a plain `parallel_for` (all items do the same thing), a pipeline lets **different items be at different stages at the same time** — exactly like a factory assembly line.

```
Without pipeline (serial):          With pipeline (3 items in-flight):
  Item 1: read → process → write
  Item 2: read → process → write    read₁ ──► process₁ ──► write₁
  Item 3: read → process → write    read₂ ──► process₂ ──► write₂
  (one item at a time)              read₃ ──► process₃ ──► write₃
                                    (all happening simultaneously)
```

The key constraint: a **serial** stage runs one item at a time (order preserved). A **parallel** stage runs multiple items simultaneously (order not preserved). You choose per-stage.

```
Stage 1 (serial_in_order)   Stage 2 (parallel)     Stage 3 (serial_in_order)
  File reading                 CPU transform           Writing results
  One file at a time           4 items at once         Must write in order
  ← can't parallelize          ← CPU-bound here →      ← must be ordered →
```

```cpp
#include "oneapi/tbb/pipeline.h"

const int max_tokens = 16;   // max items in-flight simultaneously

parallel_pipeline(max_tokens,
    // Stage 1: serial input (reads one item at a time)
    make_filter<void, InputData*>(
        filter_mode::serial_in_order,
        [&](flow_control& fc) -> InputData* {
            InputData* data = read_next();
            if (!data) { fc.stop(); return nullptr; }
            return data;
        }
    ) &
    // Stage 2: parallel processing (multiple items at once)
    make_filter<InputData*, OutputData*>(
        filter_mode::parallel,
        [](InputData* in) -> OutputData* {
            return transform(in);
        }
    ) &
    // Stage 3: serial output (writes in original order)
    make_filter<OutputData*, void>(
        filter_mode::serial_in_order,
        [&](OutputData* out) {
            write_result(out);
            delete out;
        }
    )
);
```

**Filter modes:**

| Mode | Ordering | Concurrency | Use when |
|------|---------|-------------|---------|
| `serial_in_order` | Preserved | 1 at a time | I/O, ordered output |
| `serial_out_of_order` | Not preserved | 1 at a time | Single-threaded transform |
| `parallel` | Not preserved | Multiple | CPU-bound transform, independent items |

**`max_tokens`** limits memory: the pipeline never has more than this many items in-flight simultaneously. If processing is fast but output is slow, tokens pile up — limit them to bound memory.

> **Throughput law:** Throughput = `max_tokens / slowest_serial_stage_latency`. A slow serial stage caps your throughput regardless of parallel stage speed.

---

### 3.7 `parallel_invoke` and `task_group` — Explicit Tasks

For a fixed number of independent tasks (fork-join):

```cpp
#include "oneapi/tbb/parallel_invoke.h"

// Run two functions in parallel, wait for both
parallel_invoke(
    []{ sort_left_half(); },
    []{ sort_right_half(); }
);
// Both are done here

// Up to N functions
parallel_invoke(
    []{ task_a(); },
    []{ task_b(); },
    []{ task_c(); },
    []{ task_d(); }
);
```

For a dynamic number of tasks:

```cpp
#include "oneapi/tbb/task_group.h"

task_group tg;

for (auto& item : work_units) {
    tg.run([&item]{ process(item); });
}

tg.wait();  // blocks until all tasks complete

// task_group::run() is thread-safe — tasks can add more tasks
task_group tg2;
tg2.run([&]{
    tg2.run([&]{ subtask_a(); });  // recursive is fine
    tg2.run([&]{ subtask_b(); });
});
tg2.wait();
```

---

### 3.8 `enumerable_thread_specific` — Per-Thread Storage

The problem with shared accumulators: every thread writes to the same memory location, causing data races or expensive lock contention.

```
threads → single hist[]         threads → own hist[]
  T0: hist[42]++  ─┐              T0: local_hist.local()[42]++   (no contention)
  T1: hist[42]++  ─┼─ race!       T1: local_hist.local()[42]++   (no contention)
  T2: hist[42]++  ─┘              T2: local_hist.local()[42]++   (no contention)
                                                    ↓
                                  merge all copies at the end
```

Each thread gets its own isolated copy. They never touch each other during the parallel phase. At the end you iterate over all copies and combine them.

**Problem without it:**

```cpp
// WRONG: all threads write to the same histogram → data race
std::vector<int> hist(256, 0);
tbb::parallel_for(0, N, [&](int i) {
    hist[image[i]]++;   // race condition!
});
```

**Fix with `enumerable_thread_specific`:**

```cpp
#include "oneapi/tbb/enumerable_thread_specific.h"

// Each thread gets its own private histogram
enumerable_thread_specific<std::vector<int>> local_hist(
    []{ return std::vector<int>(256, 0); }  // factory: how to create each copy
);

parallel_for(0, N, [&](int i) {
    local_hist.local()[image[i]]++;   // .local() returns THIS thread's copy
});

// Combine all per-thread histograms into one
std::vector<int> global_hist(256, 0);
for (auto& h : local_hist) {          // iterate over all thread-local copies
    for (int k = 0; k < 256; k++)
        global_hist[k] += h[k];
}
```

**Mechanics:**
- `.local()` returns a reference to this thread's copy. Creates it on first call.
- Iterating over `local_hist` gives one copy per thread that called `.local()`.
- Zero lock contention during the parallel phase.

**Another example — thread-local sum:**

```cpp
enumerable_thread_specific<float> thread_sum(0.0f);

parallel_for(blocked_range<int>(0, N), [&](const blocked_range<int>& r) {
    float& my_sum = thread_sum.local();
    for (int i = r.begin(); i < r.end(); i++)
        my_sum += a[i];
});

float total = 0.0f;
for (float s : thread_sum) total += s;
```

**Construction options:**

```cpp
// Default-constructed value
enumerable_thread_specific<int> ets1;          // each copy = int{}

// Value-initialized
enumerable_thread_specific<int> ets2(42);      // each copy = 42

// Factory function (for non-copyable or expensive init)
enumerable_thread_specific<std::vector<int>> ets3(
    []{ return std::vector<int>(1024, 0); }
);
```

---

### 3.9 `combinable<T>` — Simpler Per-Thread Accumulation

A simplified version of `enumerable_thread_specific` specifically for accumulating a single value:

```cpp
#include "oneapi/tbb/combinable.h"

combinable<float> partial_sum;

parallel_for(0, N, [&](int i) {
    partial_sum.local() += a[i];   // thread-local accumulate
});

// Combine all locals with a binary op
float total = partial_sum.combine([](float a, float b) {
    return a + b;
});
```

**vs `enumerable_thread_specific`:**
- `combinable<T>` — simpler API, designed specifically for reduction patterns
- `enumerable_thread_specific<T>` — more flexible, supports iteration, factory init, non-trivial types

Use `combinable` when you just want a thread-local accumulator. Use `enumerable_thread_specific` when you need to inspect each thread's value separately after the parallel phase.

---

### 3.10 Flow Graph — Data Flow and Dependence Graphs

For expressing complex parallel patterns as a graph of nodes and edges. The runtime automatically runs nodes when their inputs are ready — no manual synchronization.

```cpp
#include "oneapi/tbb/flow_graph.h"
using namespace oneapi::tbb::flow;

graph g;

// function_node<In, Out>: receives In, produces Out
// Second arg = max concurrency (1=serial, unlimited=max)
function_node<int, int> square(g, unlimited, [](int x) { return x * x; });
function_node<int, int> cube  (g, unlimited, [](int x) { return x * x * x; });

// join_node: wait for one input per port, emit as tuple
join_node<std::tuple<int,int>> join(g);

function_node<std::tuple<int,int>, void> printer(g, 1,
    [](const std::tuple<int,int>& t) {
        printf("square=%d  cube=%d\n", std::get<0>(t), std::get<1>(t));
    }
);

// Wire the graph
make_edge(square, input_port<0>(join));
make_edge(cube,   input_port<1>(join));
make_edge(join,   printer);

// Send inputs — both nodes run in parallel
square.try_put(5);
cube.try_put(5);

g.wait_for_all();  // always wait before graph goes out of scope
```

#### All Key Node Types

| Node | Description |
|------|-------------|
| `function_node<In, Out>` | Transforms `In` → `Out`, configurable concurrency |
| `source_node<Out>` *(deprecated)*<br>`input_node<Out>` | Generates tokens from a function; starts the graph |
| `broadcast_node<T>` | Fans out: sends one input to ALL successors |
| `join_node<tuple<...>>` | Fans in: waits for one from each port, emits tuple |
| `split_node<tuple<...>>` | Splits tuple: sends element N to port N |
| `buffer_node<T>` | Buffers messages until a successor requests them |
| `queue_node<T>` | FIFO buffer; passes messages to any available successor |
| `priority_queue_node<T>` | Priority-ordered buffer |
| `sequencer_node<T>` | Reorders out-of-order items back to sequence number order |
| `limiter_node<T>` | Limits in-flight messages (backpressure) |
| `overwrite_node<T>` | Stores last value; new successors get it immediately |
| `write_once_node<T>` | Stores first value; broadcasts to all registered receivers |
| `indexer_node<T...>` | Tagged union input: accepts any type from set, tags messages |

#### Fan-Out with `broadcast_node`

```cpp
broadcast_node<int> broadcaster(g);
function_node<int, void> worker_a(g, unlimited, [](int x){ do_a(x); });
function_node<int, void> worker_b(g, unlimited, [](int x){ do_b(x); });

make_edge(broadcaster, worker_a);
make_edge(broadcaster, worker_b);  // same input goes to both

broadcaster.try_put(42);  // both worker_a and worker_b receive 42
g.wait_for_all();
```

#### Reordering with `sequencer_node`

```cpp
// parallel_for may complete items out of order — use sequencer to restore order
sequencer_node<Frame> reorder(g, [](const Frame& f) {
    return f.sequence_number;   // sequencer uses this to order outputs
});

function_node<Frame, Frame> processor(g, unlimited, [](Frame f) {
    f.data = process(f.data);   // runs in parallel, out of order
    return f;
});

function_node<Frame, void> writer(g, serial, [](const Frame& f) {
    write_in_order(f);          // receives frames in sequence order
});

make_edge(processor, reorder);
make_edge(reorder, writer);
```

#### Backpressure with `limiter_node`

```cpp
// Prevent fast producer from overwhelming slow consumer
limiter_node<int> limiter(g, 8);   // max 8 messages in flight

make_edge(producer, limiter);
make_edge(limiter, slow_consumer);
make_edge(slow_consumer, limiter.decrement); // signal when done → unlocks limiter
```

#### Conditional Routing with `indexer_node`

```cpp
using IndexerType = indexer_node<int, float>;   // accepts int OR float

IndexerType idx(g);
function_node<IndexerType::output_type, void> router(g, unlimited,
    [](const IndexerType::output_type& msg) {
        if (msg.tag() == 0)       // int arrived
            handle_int(cast_to<int>(msg));
        else                       // float arrived
            handle_float(cast_to<float>(msg));
    }
);

make_edge(idx, router);
input_port<0>(idx).try_put(42);     // send int
input_port<1>(idx).try_put(3.14f);  // send float
```

> **Always call `g.wait_for_all()`** before the graph object is destroyed. Destroying a live graph is undefined behavior.

---

### 3.11 Concurrent Containers

**The problem:** standard containers (`std::vector`, `std::map`, `std::queue`) have no internal locking. If two threads write at the same time, you get data corruption, crashes, or silently wrong results.

```
std::vector<int> v;                // NOT thread-safe
parallel_for(0, N, [&](int i) {
    v.push_back(i);                // ← data race: undefined behaviour
});
```

The naive fix — wrapping everything in a `std::mutex` — works but serialises every access. TBB's concurrent containers solve this with fine-grained internal locking or lock-free algorithms:

```
std::vector + one global mutex     tbb::concurrent_vector
  Thread 0 writes → LOCK            Thread 0 writes ──┐
  Thread 1 waits...                  Thread 1 writes ──┤ all at once
  Thread 2 waits...                  Thread 2 writes ──┘
  (serialised)                       (concurrent, safe)
```

> Only use concurrent containers when multiple threads genuinely need to share the same container. If each thread works on its own data, a plain `std::vector` per thread (or `enumerable_thread_specific`) is faster.

#### `concurrent_vector`

Like `std::vector` but safe for concurrent `push_back`. Iterators and references stay valid after growth — unlike `std::vector` which may reallocate.

```cpp
#include "oneapi/tbb/concurrent_vector.h"

tbb::concurrent_vector<int> results;

tbb::parallel_for(0, N, [&](int i) {
    if (passes_filter(i))
        results.push_back(compute(i));   // safe from any thread
});
// results holds all passing items; order is non-deterministic
```

When to use: collecting results from a parallel filter where you don't know how many items will pass.

#### `concurrent_queue` / `concurrent_bounded_queue`

Classic producer-consumer queue. `try_pop` is non-blocking (returns false if empty); `pop` on `concurrent_bounded_queue` blocks until an item arrives.

```cpp
#include "oneapi/tbb/concurrent_queue.h"

tbb::concurrent_queue<WorkItem> q;

// Producer threads — can all push simultaneously
q.push(item_a);
q.push(item_b);

// Consumer threads — non-blocking
WorkItem item;
if (q.try_pop(item))
    process(item);   // got one

// Bounded variant — adds backpressure (push blocks when full)
tbb::concurrent_bounded_queue<WorkItem> bounded(100);
bounded.push(item);   // blocks if 100 items already queued
bounded.pop(item);    // blocks if queue is empty
```

When to use: streaming pipelines where producers and consumers run at different rates.

#### `concurrent_hash_map`

High-concurrency key-value store. Uses per-bucket locking — only the bucket being written is locked, so unrelated keys never contend.

```cpp
#include "oneapi/tbb/concurrent_hash_map.h"

tbb::concurrent_hash_map<std::string, int> freq;

tbb::parallel_for_each(words.begin(), words.end(), [&](const std::string& w) {
    tbb::concurrent_hash_map<std::string, int>::accessor acc;  // write lock
    freq.insert(acc, w);   // locks only the bucket for key w
    acc->second++;         // safe: this thread owns the bucket
});                        // accessor destructor releases the lock

// Read-only (shared lock — many readers can hold this simultaneously)
tbb::concurrent_hash_map<std::string, int>::const_accessor cacc;
if (freq.find(cacc, "hello"))
    printf("hello: %d\n", cacc->second);
```

`accessor` = write lock (exclusive). `const_accessor` = read lock (shared). Multiple threads can hold `const_accessor` on the same key simultaneously.

#### `concurrent_unordered_map`

Simpler API — drop-in for `std::unordered_map`, no accessor needed. Individual insertions and lookups are thread-safe, but iteration while modifying is not.

```cpp
#include "oneapi/tbb/concurrent_unordered_map.h"

tbb::concurrent_unordered_map<int, int> m;

tbb::parallel_for(0, N, [&](int i) {
    m.insert({i, i * i});    // safe
    // m[i] = i * i;         // also safe for insert; avoid for update
});
```

Use `concurrent_hash_map` when you need read-modify-write atomicity (e.g., incrementing a counter). Use `concurrent_unordered_map` when you only insert-or-lookup with no partial updates.

#### `concurrent_priority_queue`

```cpp
#include "oneapi/tbb/concurrent_priority_queue.h"

tbb::concurrent_priority_queue<int> pq;
pq.push(10); pq.push(5); pq.push(20);

int top;
pq.try_pop(top);   // top = 20 (max-heap by default)
```

#### Container comparison

| Container | Analogy | Key operation | Use for |
|-----------|---------|---------------|---------|
| `concurrent_vector` | Shared notepad | `push_back` | Collecting parallel results |
| `concurrent_queue` | Shared inbox | `push` / `try_pop` | Producer-consumer pipelines |
| `concurrent_bounded_queue` | Inbox with size limit | blocking `push`/`pop` | Backpressure control |
| `concurrent_hash_map` | Shared dictionary (with door locks per entry) | `accessor` read-modify-write | Word counts, shared caches |
| `concurrent_unordered_map` | Shared dictionary (simpler) | `insert` / `find` | Insert-once lookup tables |
| `concurrent_priority_queue` | Shared task queue sorted by priority | `push` / `try_pop` | Priority scheduling |

---

### 3.12 Scalable Memory Allocator

Standard `malloc`/`free` have a single global lock — a bottleneck when many threads allocate simultaneously. TBB's allocator uses per-thread memory pools:

```cpp
#include "oneapi/tbb/scalable_allocator.h"

// Use tbb_allocator with STL containers
std::vector<float, tbb::tbb_allocator<float>> vec(N);

// Use scalable_malloc directly (drop-in for malloc)
float* buf = (float*)scalable_malloc(N * sizeof(float));
scalable_free(buf);

// Replace all allocations globally (link with -ltbbmalloc_proxy)
// Or: set LD_PRELOAD=libtbbmalloc_proxy.so before running
```

Most impactful when many threads frequently allocate/free small objects (task objects, intermediate buffers in a pipeline).

---

### 3.13 `task_arena` — Control Thread Pool

By default, TBB creates one global thread pool that uses all hardware threads. Every `parallel_for`, `parallel_reduce`, etc. draws from that pool. `task_arena` lets you create a **separate, bounded execution zone** with its own concurrency limit and optionally pinned to specific NUMA nodes.

```
Default (global arena):          With task_arena:
  All 16 hardware threads          arena(4) → max 4 threads
  shared by everything             Work inside it can't steal
  ← any parallel_for steals        from the global pool
     from any other
```

**`task_arena` does NOT create new OS threads.** TBB's worker threads can participate in multiple arenas. What the arena controls is how many of them are *allowed in* at once.

#### Basic usage — limit parallelism

```cpp
#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/parallel_for.h"

tbb::task_arena arena(2);   // at most 2 threads

arena.execute([&] {
    tbb::parallel_for(0, 1000, [](int i) {
        // runs with at most 2 threads regardless of core count
        do_work(i);
    });
});
```

Useful when your program shares the CPU with a GPU or another process and you want to leave headroom.

#### Two independent subsystems

```cpp
tbb::task_arena physics(4);   // physics simulation: 4 threads
tbb::task_arena render(4);    // rendering: 4 threads

// Launch both concurrently — they draw from separate thread budgets
// No work-stealing between them
std::thread t1([&]{ physics.execute([&]{ parallel_for(0, N, sim_body);    }); });
std::thread t2([&]{ render.execute( [&]{ parallel_for(0, M, render_body); }); });
t1.join(); t2.join();
```

Without arenas, TBB's global pool would intermix work from both subsystems — threads doing physics could be stolen away to help with rendering mid-frame. Arenas prevent that.

#### NUMA-aware pinning (multi-socket servers)

```cpp
tbb::task_arena numa_arena(
    tbb::task_arena::constraints{}
        .set_numa_id(1)            // bind to NUMA socket 1
        .set_max_concurrency(8)    // use up to 8 threads from that socket
);

numa_arena.execute([&]{
    parallel_for(0, N, body);     // threads stay on socket 1's cores
});
// Data allocated on socket 1 memory is accessed locally → lower latency
```

#### When to use `task_arena`

| Scenario | Why `task_arena` helps |
|----------|----------------------|
| CPU + GPU running simultaneously | Limit CPU threads so GPU isn't starved for PCIe bandwidth |
| UI application with background processing | Prevent background `parallel_for` from consuming all cores |
| Multiple independent subsystems | Isolate them so one can't steal from the other |
| NUMA system (multi-socket server) | Pin arenas to sockets to keep data access local |
| Debugging: make parallel code deterministic | `task_arena(1)` forces single-thread execution |

---

### 3.14 `global_control` — Runtime Configuration

`task_arena` limits threads *locally* (inside one block). `global_control` sets a **program-wide policy** that applies to every TBB algorithm everywhere — including third-party libraries that use TBB internally.

```
task_arena(4).execute(...)      global_control(max_allowed_parallelism, 4)
  Only this block uses 4         Every parallel_for, parallel_reduce,
  Everything else: unlimited     pipeline, etc. in the entire process: ≤ 4
```

Uses RAII — settings revert automatically when the object goes out of scope:

```cpp
#include "oneapi/tbb/global_control.h"

{
    tbb::global_control ctrl(
        tbb::global_control::max_allowed_parallelism, 4
    );
    // ── all TBB work inside this scope uses at most 4 threads ──
    tbb::parallel_for(0, N, body_a);   // ≤ 4 threads
    some_library_call();               // if it uses TBB: also ≤ 4 threads
}
// ── ctrl destroyed → limit lifted, TBB uses all cores again ──
tbb::parallel_for(0, N, body_b);       // full thread count again
```

**Set thread stack size** — useful when tasks recurse deeply or allocate large local arrays:

```cpp
tbb::global_control stack_ctrl(
    tbb::global_control::thread_stack_size,
    8 * 1024 * 1024   // 8 MB per worker thread (default is typically 1–4 MB)
);
```

Each thread has a fixed stack. Deep recursion or large stack-allocated buffers silently overflow the default. Increase it before the first TBB task is created.

**`task_arena` vs `global_control`:**

| | `task_arena` | `global_control` |
|--|--|--|
| Scope | One `execute()` block | Entire process |
| Thread limit | Per isolated region | All TBB algorithms |
| Work-stealing isolation | Yes | No |
| Typical use | Independent subsystems, NUMA pinning | Embedding TBB in a GUI/server, global throttling |

---

### 3.15 Exception Handling and Cancellation

In normal single-threaded C++, an exception unwinds the call stack to the nearest `catch`. In a parallel loop, each iteration runs in a different thread with its own stack — there is no shared call stack to unwind.

TBB bridges this: when a worker thread throws, TBB catches it internally, cancels remaining *pending* iterations, then re-throws the exception on the calling thread once the loop completes.

```
Thread 0: iteration  0 → OK
Thread 1: iteration  5 → throws std::runtime_error("bad input")
                         ↑ TBB catches it
Thread 2: iteration 10 → OK (already running → finishes normally)
Pending iterations 11–N → CANCELLED (never start)
                         ↓ TBB re-throws on calling thread
Main thread: catch block runs
```

```cpp
try {
    tbb::parallel_for(0, N, [](int i) {
        if (bad_condition(i))
            throw std::runtime_error("bad input");
    });
} catch (const std::exception& e) {
    // Caught here on the calling thread
    // Already-running iterations completed; pending ones were skipped
    std::cerr << e.what() << "\n";
}
```

> Already-running iterations always finish. Only *pending* (not-yet-started) tasks are cancelled. You cannot abort a task mid-execution.

**Explicit cancellation without exceptions** — use `task_group_context` when you want to cancel from another thread or condition (e.g., a timeout, a "stop" button):

```cpp
tbb::task_group_context ctx;

// Start the parallel loop in one thread
tbb::parallel_for(
    tbb::blocked_range<int>(0, N),
    body,
    tbb::auto_partitioner(),
    ctx          // ← attach context
);

// From another thread (e.g., a watchdog or UI thread):
ctx.cancel_group_execution();
// → pending tasks stop being scheduled
// → running tasks finish naturally
// → parallel_for returns without throwing
```

**Cancellation flow:**

```
cancel_group_execution() called
          │
          ▼
Pending tasks   → skip (never execute)
Running tasks   → run to completion (cannot be interrupted)
parallel_for()  → returns normally (no exception thrown)
```

| Mechanism | Triggers how | Running tasks | Pending tasks | Throws? |
|-----------|-------------|---------------|---------------|---------|
| Worker throws exception | Exception in body | Finish | Cancelled | Yes, on caller |
| `ctx.cancel_group_execution()` | External call | Finish | Cancelled | No |

---

### 3.16 Work Isolation

TBB's work-stealing scheduler is aggressive: when a thread finishes a task and is waiting on an inner `parallel_for`, it doesn't idle — it looks for *any* available task in the pool, including tasks from the outer loop. This is great for throughput but dangerous when inner and outer tasks share data.

```
WITHOUT isolate:
  Outer task i=0 launches inner parallel_for (10 tasks)
  Outer task i=0's thread waits → steals outer task i=3
  Now i=0's thread is running i=3 while i=0's inner tasks run on other threads
  If inner tasks read/write data[i=0...] and outer task i=3 does too → race

WITH isolate:
  Outer task i=0 launches inner parallel_for inside isolate()
  Inner tasks are visible ONLY to threads already inside the isolate block
  Outer threads cannot steal them → no interleaving → safe
```

**When you need isolation — three questions:**

1. Does the inner loop write to data that outer tasks also touch?
2. Is that data not protected by a lock?
3. Are outer and inner tasks running at the same time?

If all three: use `isolate`.

```cpp
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/this_task_arena.h>

std::vector<int> data(100, 0);

tbb::parallel_for(0, 10, [&](int i) {
    // outer task owns data[i*10 ... i*10+9]
    // inner tasks write into exactly that range
    // → must not be stolen by another outer task's thread

    tbb::this_task_arena::isolate([&] {
        tbb::parallel_for(0, 10, [&](int j) {
            data[i*10 + j] += 1;   // safe: isolated from other outer tasks
        });
    });
});
```

**Without `isolate`:** Thread running outer `i=2` could steal inner tasks belonging to outer `i=7`, then write into `data[20..29]` while `i=2`'s thread is somewhere else writing into `data[20..29]` — data race.

**With `isolate`:** Inner tasks of `i=2` stay inside the isolate block. Only the thread that entered the block (and any helpers *it* spawns internally) can run those inner tasks.

**When NOT to isolate:**

If the inner loop works on independent data (no sharing with outer tasks), isolation only hurts performance — it prevents free threads from helping with the inner work. Leave it out.

| Situation | Use `isolate`? | Reason |
|-----------|---------------|--------|
| Inner tasks write to shared outer data | Yes | Prevent data race |
| Inner tasks read-only from shared data | Usually no | Reads are safe to steal |
| Inner tasks work on fully independent data | No | Let threads steal freely — faster |
| Need deterministic per-outer-task execution order | Yes | Isolate forces strict containment |

---

### 3.17 Design Patterns Summary

**Reduce:**

```cpp
float total = parallel_reduce(range, 0.0f, body, std::plus<float>{});
```

**Divide and Conquer (recursive):**

```cpp
void parallel_mergesort(int* a, int n) {
    if (n < THRESHOLD) { std::sort(a, a + n); return; }
    parallel_invoke(
        [&]{ parallel_mergesort(a, n/2); },
        [&]{ parallel_mergesort(a + n/2, n - n/2); }
    );
    std::inplace_merge(a, a + n/2, a + n);
}
```

**Map (elementwise):**

```cpp
parallel_for(blocked_range<int>(0, N), [&](const blocked_range<int>& r) {
    for (int i = r.begin(); i < r.end(); i++)
        out[i] = f(in[i]);
});
```

**Histogram (per-thread local + merge):**

```cpp
enumerable_thread_specific<std::vector<int>> local_hist(
    []{ return std::vector<int>(256, 0); });

parallel_for(0, N, [&](int i) {
    local_hist.local()[data[i]]++;
});

std::vector<int> hist(256, 0);
for (auto& h : local_hist)
    for (int k = 0; k < 256; k++) hist[k] += h[k];
```

**Prefix scan:**

```cpp
parallel_scan(range, 0.0f, scan_body, std::plus<float>{});
```

**Assembly line:**

```cpp
parallel_pipeline(max_tokens, input_filter & transform_filter & output_filter);
```

---

### 3.18 Connection to GPU Programming

oneTBB patterns map directly to GPU frameworks:

| oneTBB concept | GPU equivalent |
|---------------|---------------|
| `parallel_for` over `blocked_range` | CUDA kernel launch over thread grid |
| Work-stealing scheduler | Warp scheduler (hardware) |
| `affinity_partitioner` | L2 cache locality hints in CUDA |
| `enumerable_thread_specific` | Per-warp shared memory allocation |
| `combinable<T>` | `atomicAdd` into shared mem, then reduce |
| `concurrent_queue` | CUDA stream (async task queue) |
| `parallel_pipeline` | CUDA multi-stream pipeline |
| `flow_graph` nodes | CUDA graph nodes (`cudaGraph`) |
| `task_group` | CUDA dynamic parallelism |
| `task_arena` | CUDA stream priority + stream isolation |
| Scalable allocator | `cudaMallocAsync` (stream-ordered pool) |

Learning oneTBB's task decomposition makes CUDA's thread/block/grid hierarchy intuitive — they solve the same problem at different scales.

---

## Resources

| Resource | What it covers |
|----------|---------------|
| [oneTBB Developer Guide](https://uxlfoundation.github.io/oneTBB/main/tbb_userguide/) | Full tutorial: parallel_for, reduce, pipeline, flow graph, design patterns |
| [oneTBB API Reference](https://uxlfoundation.github.io/oneTBB/main/tbb_userguide/reference.html) | Complete API |
| [Intel oneTBB Get Started](https://www.intel.com/content/www/us/en/docs/onetbb/get-started-guide/2022-2/overview.html) | Installation, CMake, first steps |
| [OpenMP specifications](https://www.openmp.org/specifications/) | Official OpenMP standard |
| [OpenMP API Quick Reference Card](https://www.openmp.org/wp-content/uploads/OpenMP-4.5-1115-CPP-web.pdf) | All clauses at a glance |
| *Intel Threading Building Blocks* (Reinders) | Book: TBB patterns in depth |

---

## Next

→ [**CUDA and SIMT**](../CUDA%20and%20SIMT/Guide.md) — GPU parallelism: threads, warps, blocks, and memory hierarchy.
