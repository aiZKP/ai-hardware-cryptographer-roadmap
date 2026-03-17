# Lecture Note 05 (L10, L11): Lock-Free Programming (RCU, Atomics) & Deadlock / Priority Inversion

**Combines:** Lecture L10 (Lock-Free: RCU, Atomics, Memory Ordering) and Lecture L11 (Deadlock, Priority Inversion, PI Mutexes).

---

## How This Note Is Organized

1. **Part 1 — Lock-free:** Why avoid locks; C11 memory model; atomics and CAS; RCU; SPSC ring buffer; hazard pointers.
2. **Part 2 — Deadlock & priority inversion:** Coffman conditions; prevention (lock ordering, trylock); priority inversion and Mars Pathfinder; PI and priority ceiling; lockdep; watchdog.

---

# Part 1: Lock-Free Programming — RCU, Atomics & Memory Ordering

**Context:** Locks add cost even when uncontended: cache-line bouncing, context switches, priority inversion. On hot paths (camera pipeline, sensor fusion, model config), lock-free techniques use atomics and RCU to coordinate without mutual exclusion.

---

## Why Lock-Free?

- **Cache-line bouncing:** Lock variable ping-pongs between CPU caches; ~100–300 cycles on NUMA.
- **Context switches:** Blocked threads pay scheduler overhead (~1–10 µs).
- **Priority inversion:** Low-priority holder delays high-priority waiter (see Part 2).
- **Convoying:** Many threads queue behind one slow holder.

Lock-free uses **atomic** hardware instructions for coordination. "Lock-free" does not mean no coordination — it means coordination via atomics (and RCU) instead of mutexes; cost is more bounded and predictable.

---

## C11/C++11 Memory Model

CPUs and compilers reorder operations for performance. Without explicit ordering, a write in thread A may not be visible to thread B for an arbitrary time. The memory model defines ordering guarantees via `std::atomic<T>` / `_Atomic T`:

| Memory Order | Guarantee |
|--------------|-----------|
| `memory_order_relaxed` | Atomicity only; no ordering relative to other operations; use for counters. |
| `memory_order_acquire` | No load/store *after* this point may be reordered *before* it; pairs with release. |
| `memory_order_release` | No load/store *before* this point may be reordered *after* it; pairs with acquire. |
| `memory_order_acq_rel` | Both acquire + release; for atomic read-modify-write (e.g. CAS on success). |
| `memory_order_seq_cst` | Total global order across all threads; full fence; default for `std::atomic<>`. |

**Acquire-release pairing:** A **release** store to variable X "happens-before" a subsequent **acquire** load from X in another thread. All writes before the release are visible after the acquire.

```
  Thread A (producer)                    Thread B (consumer)
  data = 42;
  flag.store(1, memory_order_release);   int f = flag.load(memory_order_acquire);
  ───────────────────────────────────►  if (f == 1) assert(data == 42);  // guaranteed
```

> **Common pitfall:** Using `memory_order_relaxed` for a "data ready" flag only guarantees atomicity of the flag — it does *not* guarantee that the payload written before the flag store is visible to the reader. Always use **release** (producer) / **acquire** (consumer) for producer-consumer signaling.

---

## Hardware Memory Models & Kernel Barriers

Different CPUs have different default ordering. Kernel code often uses explicit barriers instead of C11 atomics:

| Architecture | Memory model | Barrier requirement |
|--------------|--------------|---------------------|
| x86-64 | TSO (Total Store Order) — strong | Few explicit barriers; LOCK prefix for atomics. |
| ARM64 | Weakly ordered | Explicit `DMB`/`DSB`; `LDAR`/`STLR` for acquire/release. |
| RISC-V | Weak (RVWMO) | `FENCE`; `LR`/`SC` for LL/SC atomics. |

**Kernel barrier macros:**
- `smp_mb()` — full memory barrier (load and store in both directions).
- `smp_rmb()` — read (load) barrier only.
- `smp_wmb()` — write (store) barrier only.

On ARM64 (e.g. Jetson Orin), every acquire/release in C11 translates to LDAR/STLR. Code that runs correctly on x86 without barriers may **silently fail** on ARM64. Always use C11 atomics or kernel barrier macros for shared variables.

---

## Atomic Operations (Kernel)

Map to single indivisible hardware instructions: `LOCK ADD`/`XADD`/`CMPXCHG` on x86, `LDADD`/`CAS` on ARMv8.1 LSE.

```c
atomic_t counter = ATOMIC_INIT(0);
atomic_inc(&counter);                   // LOCK XADD / LDADD
atomic_dec_and_test(&counter);          // decrement; return true if zero (refcounts)
int old = atomic_cmpxchg(&counter, 5, 10);  // if counter==5 set to 10; return old value
atomic_add_return(n, &counter);         // add n, return new value (rate limiting)
```

- `atomic_t` is 32-bit; `atomic64_t` for 64-bit. Use for: reference counts, flags, per-CPU statistics.

## Compare-and-Swap (CAS)

CAS atomically tests a value and replaces it only if it matches — the foundation of most lock-free structures.

```cpp
std::atomic<int> val{0};
int expected = 0;
bool ok = val.compare_exchange_strong(expected, 1,
    std::memory_order_acq_rel,   // success: acquire + release
    std::memory_order_acquire);  // failure: acquire only; expected gets current value
// On failure, 'expected' holds actual value — retry loop uses it
```

- **compare_exchange_weak:** may spuriously fail on LL/SC (ARM, RISC-V); preferred inside retry loops (avoids extra load). Retry loop must re-read dependent state and recompute desired value, not blindly retry with stale data.

**ABA problem:** Pointer value is A, then B, then A again (same address, new object). CAS sees "A" and succeeds, but the object at A was freed and reused — use-after-free. **Solutions:** (1) **Version tag** — pack a monotonic counter with the pointer in a 128-bit CAS (`CMPXCHG16B` on x86); (2) **Hazard pointers** — reader publishes pointer before use; reclaimer scans HP slots before freeing; (3) **RCU** — grace period ensures old object is not freed until readers are done.

---

## SPSC Ring Buffer (Lock-Free)

Single producer, single consumer: only **acquire**/ **release** on head and tail indices; no lock, no cache-line bouncing on payload.

```c
#define N 256   // power of 2 for modulo via bitmask
T buf[N];
atomic_size_t head = 0, tail = 0;

// Producer (single thread only)
buf[tail % N] = item;
atomic_store_explicit(&tail, tail + 1, memory_order_release);

// Consumer (single thread only)
size_t t = atomic_load_explicit(&tail, memory_order_acquire);
if (head != t) {
    item = buf[head % N];
    atomic_store_explicit(&head, head + 1, memory_order_release);
}
```

The **release** store (producer) and **acquire** load (consumer) on `tail` are the synchronization point: the consumer is guaranteed to see the payload in `buf[tail%N]` once it sees the updated tail. Throughput is limited only by memory bandwidth. Used in openpilot VisionIPC for zero-copy camera frames at 30 Hz. **Constraint:** exactly one producer and one consumer; with multiple producers or consumers, head/tail updates become races (MPMC needs more complex sync).

---

## RCU (Read-Copy-Update)

Linux kernel’s primary mechanism for read-mostly shared data. **O(1) read-side cost:** no locks, no atomics, no cache-line writes — only preempt disable/enable.

### Reader side

```c
rcu_read_lock();
ptr = rcu_dereference(gp);   // READ_ONCE + compiler barrier
/* use *ptr — guaranteed valid until rcu_read_unlock */
rcu_read_unlock();
```

### Writer side (copy → modify → publish → wait → free)

1. Allocate and populate a new version of the object.
2. Publish: `rcu_assign_pointer(gp, new_obj)` (smp_wmb + pointer store).
3. Wait for a **grace period**: all pre-existing readers have finished.
4. Free the old object (no reader can still hold it).

```c
new_obj = kmalloc(sizeof(*new_obj), GFP_KERNEL);
*new_obj = *old_obj;
new_obj->field = updated_value;
rcu_assign_pointer(gp, new_obj);
synchronize_rcu();   // blocks until grace period
kfree(old_obj);
```

**call_rcu(&old->rcu_head, free_fn):** asynchronous form; writer does not block; used in interrupt context or when writer must not block.

### Grace period

A **grace period** is the interval after `rcu_assign_pointer()` until every CPU has passed a **quiescent state** (context switch, entry to idle, or return to user space). The kernel tracks these events; when all CPUs have quiesced, no pre-existing reader can still hold the old pointer.

### RCU variants

| Variant | Read section can sleep? | Use case |
|---------|-------------------------|----------|
| Classic RCU | No | Routing tables, task_struct lookup, module parameters |
| SRCU (Sleepable RCU) | Yes | Notifier chains, subsystem registrations |
| PREEMPT_RCU | No (but preemptible) | CONFIG_PREEMPT kernels |

**rcu_nocbs=`<cpulist>`:** Offloads `call_rcu` callbacks to `rcuoc` kthreads on non-isolated CPUs. On an isolated inference core, normal RCU would run callbacks there at unpredictable times; `rcu_nocbs=` removes that latency source.

---

## kfifo (Kernel SPSC FIFO)

```c
DECLARE_KFIFO(my_fifo, int, 64);   // size power of 2
INIT_KFIFO(my_fifo);
kfifo_put(&my_fifo, value);        // producer
kfifo_get(&my_fifo, &value);       // consumer
kfifo_len(&my_fifo);
```

Used in kernel drivers for RX buffers between ISR (producer) and process context (consumer). Kernel’s standard lock-free SPSC FIFO.

## Hazard Pointers

Userspace alternative to RCU for lock-free reclamation:

- Reader **publishes** the pointer it is about to dereference into a per-thread hazard pointer slot.
- Before freeing, the reclaimer scans all hazard pointer slots for that pointer.
- If found → defer free; if not found → free is safe.

Used in Folly (Meta), Java `java.util.concurrent`; `std::hazard_pointer` in C++26.

---

## Part 1 Summary Table

| Technique | Reader cost | Writer cost | Safe in ISR? | Limitation |
|-----------|-------------|-------------|--------------|------------|
| Spinlock | Cache write + spin | Cache write | Yes | Wasted CPU; no sleep |
| CAS loop | Atomic RMW + retry | Atomic RMW + retry | Yes | ABA; retry cost |
| SPSC ring | Acquire load | Release store | Yes (producer or consumer) | Single producer AND consumer only |
| RCU | Preempt disable | Copy + grace period | No (synchronize_rcu blocks) | Read-mostly; writer pays |
| Hazard pointers | Publish + load | Scan HP slots | No | Reclamation overhead |
| kfifo | Acquire load | Release store | Yes | Kernel-only; SPSC only |

### Part 1 — Conceptual review

- **Why is relaxed wrong for producer-consumer signaling?** Relaxed guarantees only atomicity of the atomic variable; it does not guarantee that data written before the store is visible to the reader. Use release/acquire for “data ready” flags.
- **Why does RCU work without reader-side locks?** The kernel detects **quiescent states** (context switch, idle, return to user). A grace period ends only after every CPU has quiesced, so no pre-existing reader can still hold the old pointer.
- **When does CAS fail and what should the retry loop do?** CAS fails when another thread changed the value. On failure, the `expected` variable is updated with the current value; the loop should re-read dependent state, recompute the new value, and retry — not retry with stale computation.
- **RCU vs rwlock for model config?** With rwlock, writers block new readers. With RCU, readers are never blocked; the writer works on a copy and publishes atomically. For a 100 Hz inference loop reading config every cycle, RCU is the right choice.

---

# Part 2: Deadlock, Priority Inversion & PI Mutexes

**Context:** Deadlock = tasks waiting for each other forever. Priority inversion = high-priority task blocked by a low-priority one (via a shared resource). Both require design-time and runtime measures (ordering, PI, lockdep).

---

## Deadlock: Definition & Coffman Conditions

**Deadlock:** A set of processes are each waiting for a resource held by another in the set; no process can ever make progress.

```
  Process A holds L1, waiting for L2  ──►  Process B holds L2, waiting for L1
  Neither can proceed. Both wait forever.
```

**Coffman conditions** (all four must hold for deadlock to be possible; break any one to prevent it):

| Condition | Definition |
|-----------|------------|
| **Mutual exclusion** | At least one resource is non-sharable — only one process may hold it at a time |
| **Hold-and-wait** | A process holds at least one resource while waiting to acquire another |
| **No preemption** | Resources cannot be forcibly taken; only voluntary release |
| **Circular wait** | A cycle exists in the resource-allocation graph: P1→R1→P2→R2→P1 |

---

## Deadlock Prevention

Attack one of the four conditions at design time:

| Condition to break | Technique | Trade-off |
|--------------------|-----------|-----------|
| Hold-and-wait | Acquire all locks at once; release all if any acquisition fails | Less concurrency; retry logic |
| Circular wait | **Global lock ordering:** always acquire in a fixed, documented order | Discipline at all call sites; lockdep enforces |
| No preemption | `mutex_trylock()` with randomized exponential backoff | Retry overhead; livelock risk |
| Mutual exclusion | Lock-free data structures | Higher implementation complexity |

**Global lock ordering (step-by-step):**

1. Enumerate all mutexes/locks in the subsystem.
2. Assign each a numeric rank (e.g. Lock A = 1, Lock B = 2).
3. Mandate: always acquire in ascending rank order.
4. Document the ordering at the lock declaration site.
5. Use `lockdep_set_class` so lockdep can verify ordering automatically.
6. In code review, reject any patch that acquires a lower-ranked lock while holding a higher-ranked one.

> **Pitfall:** Ordering breaks when new locks are added without updating the global order — two developers can introduce an A→C→D and A→D cycle. Use lockdep in CI to catch cycles.

## Deadlock Detection & Recovery

- **Resource allocation graph:** Nodes = processes (P) and resources (R); edge P→R = P waits for R; edge R→P = R held by P. A cycle = deadlock.
- **Wait-for graph:** Simplified; edge P→Q means P waits for something held by Q; cycle = deadlock.

**Recovery options after detection:** (1) Abort one process and free its resources (choose victim by priority, runtime, resources held). (2) Preempt a resource from one process (requires rollback support). (3) Roll back to a safe checkpoint (requires checkpointing). **lockdep** in Linux detects *potential* deadlock at runtime when a new lock ordering is first seen — before an actual hang.

---

## Priority Inversion

**Mechanism (four steps):**

1. **L** (low priority) acquires mutex M.
2. **H** (high priority) wakes and tries to acquire M — blocks waiting for L.
3. **M** (medium priority) wakes and **preempts L** (M > L).
4. L never runs → cannot release M → H waits indefinitely despite being highest priority.

H is effectively running at L’s priority — **inverted**. Duration of inversion is **unbounded** without PI.

```
  Time ─────────────────────────────────────────────────────────►
  H (high):  [woken]──[BLOCKED on mutex held by L]────────────────►
  M (med):   ─────────────────────[RUNNING]────────────────────────►
  L (low):   [holds mutex]──[PREEMPTED by M]──────[never runs]
```

> **Insight:** Inversion can occur even when every task is correct. It’s an emergent effect of interactions; code review alone cannot catch it.

### Mars Pathfinder (1997)

Rover had periodic resets ~18 hours after landing. **Root cause:** unbounded priority inversion.

| Task | Priority | Role |
|------|----------|------|
| bc_dist | High | Data distribution bus; needed the mutex |
| bc_sched | Low | Bus scheduler; held the mutex |
| ASI_MET | Medium | Meteorological data; CPU-intensive |

**Sequence:** bc_sched (low) held mutex → bc_dist (high) blocked → ASI_MET (medium) preempted bc_sched and ran → bc_sched never ran → bc_dist missed its watchdog → VxWorks watchdog fired → full system reset. **Fix:** Enable **priority inheritance** on the shared mutex (feature existed in VxWorks but was off by default). Fix was uploaded to the rover via uplink.

**Lesson:** PI must be the default for any mutex shared between tasks of different priorities in safety-critical or inaccessible systems.

---

## Priority Inheritance (PI)

When H blocks on a mutex held by L, the kernel **temporarily boosts L** to H’s priority until L releases the mutex. **Transitive PI:** if L is also blocked on another mutex held by X, the boost propagates to X so the whole chain can run and release.

```
  BEFORE PI:                          AFTER PI:
  H (90): BLOCKED                     H (90): BLOCKED
  M (50): RUNNING                     M (50): BLOCKED (cannot preempt L)
  L (10): PREEMPTED                   L (10→90): RUNNING → releases mutex → H runs
```

- **Linux:** `rtmutex` implements PI; PREEMPT_RT converts most `spinlock_t` to `rtmutex` → PI system-wide (e.g. CAN, V4L2, GPU drivers) without driver changes.
- **Userspace:**

```c
pthread_mutexattr_t attr;
pthread_mutexattr_init(&attr);
pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT);
pthread_mutex_init(&mutex, &attr);
```

> **Pitfall:** PI is most effective with SCHED_FIFO/SCHED_RR. On SCHED_OTHER (CFS), “priority” is relative and the boost may not have the desired effect.

---

## Priority Ceiling Protocol

Each mutex has a **ceiling priority** = highest priority of any task that will ever acquire it. Any task that acquires it **immediately** runs at the ceiling for the duration — so a high-priority waiter never has to block (L is already at ceiling). Prevents inversion without runtime discovery; requires static analysis (all acquirers and their priorities known at design time).

| | PI | Priority ceiling |
|---|----|-------------------|
| When boost happens | When H blocks on L | On every acquisition |
| Overhead | Only when contention | Every acquisition |
| Suited to | Dynamic task sets, general RT | AUTOSAR, ARINC 653 (fixed task set) |

POSIX: `PTHREAD_PRIO_PROTECT`. **When mandatory:** AUTOSAR OS and ARINC 653 require formal bounded-inversion proofs; priority ceiling provides that statically.

---

## lockdep — Live Deadlock Detector

`CONFIG_PROVE_LOCKING=y`, `CONFIG_LOCK_STAT=y` (optional, per-lock contention stats).

- Assigns each lock a **lock class** (from static address in kernel binary).
- Records every acquisition chain: “lock A was held when lock B was acquired.”
- Reports **AB-BA cycles** the first time a new ordering is seen — before an actual hang.
- Reports **invalid context:** e.g. `mutex_lock()` from hardirq (sleeping lock in IRQ).

Example report:
```
WARNING: possible circular locking dependency detected
task/1234 is trying to acquire lock: (&lockB){...}, at: function_b+0x30
but task holds lock: (&lockA){...}, taken at: function_a+0x20
which lock already depends on the new lock.
```

- **lockdep_assert_held(&lock):** Documents and verifies that a lock is held at a given point.
- Enable in development/CI; disable in production (~10% overhead).

## Watchdog Timers

If a process does not kick the watchdog within the timeout, the system resets or enters a safe state — defense-in-depth against deadlocks that slip through. On Mars Pathfinder the watchdog **did** detect bc_dist missing its deadline; the real fix was enabling PI so the deadline was not missed. **Pitfall:** Treating watchdog reset as an acceptable “recovery” for priority inversion is wrong — e.g. on an ADAS controller it forces disengagement; the correct fix is to eliminate the inversion.

---

## Part 2 Summary Table

| Problem | Symptom | Detection | Solution |
|---------|---------|-----------|----------|
| Deadlock | Hang; tasks blocked forever | Resource graph; lockdep | Lock ordering; acquire-all; trylock |
| Priority inversion | High-priority task stalls | Latency; watchdog | PI mutex; priority ceiling |
| Livelock | CPU busy, no progress | Profiling | Backoff; arbitration |
| Starvation | Low-priority never runs | Wait monitoring | Aging; FIFO; boost |

### Part 2 — Conceptual review

- **Why can priority inversion occur with “correct” code?** It’s emergent: L holds the lock correctly, H waits correctly, M preempts correctly. The failure is in the system design (shared mutex across priority levels without PI).
- **Mars Pathfinder lesson:** PI existed in VxWorks but was off by default. Safety-critical systems must audit every mutex shared across priorities and enable PI.
- **PI vs priority ceiling?** PI boosts the holder only when a higher-priority waiter blocks (dynamic). Ceiling boosts the acquirer to ceiling on every acquisition (static). Ceiling has stronger predictability for fixed task sets (AUTOSAR, ARINC 653).
- **What does lockdep *not* report?** It does not report priority inversion — that needs latency analysis (e.g. cyclictest) or scheduling analysis.
- **Why is transitive PI important?** If H blocks on L and L blocks on X, X must also be boosted; otherwise L cannot run and cannot release the mutex for H.

---

## AI Hardware Connection

- **RCU:** Live model config and LoRA swaps without pausing inference; writer publishes new config, old freed after grace period. Readers (inference threads) never block.
- **SPSC ring + acquire/release:** Zero-copy camera path (e.g. openpilot VisionIPC between camerad and modeld); no mutex on the 30 Hz frame path; throughput limited by memory bandwidth.
- **rcu_nocbs=:** On Jetson/inference-dedicated cores, offloads RCU callbacks to helper CPUs so isolated RT cores don’t see callback jitter.
- **CAS-based queues:** Used in VisionIPC-style pipelines for multi-producer sensor aggregation; each sensor enqueues without a mutex; inference thread drains once per cycle.
- **Atomic refcounts (kref, std::atomic):** Manage DMA-BUF buffer lifetime across V4L2 and GPU consumers; buffer freed when last consumer decrements to zero.
- **PI mutex (rtmutex / PTHREAD_PRIO_INHERIT):** Required for openpilot controlsd/plannerd and cereal shared state with plannerd (medium) and loggers (low); Mars Pathfinder is a standard reference in ASIL-B safety reviews.
- **ROS2:** rclcpp real-time executor docs explicitly require PTHREAD_PRIO_INHERIT so control callbacks are not starved by data-logging threads.
- **lockdep:** Standard in camera, ISP, and DMA driver bring-up on Jetson/i.MX before deployment; AB-BA cycles must be eliminated.
- **Priority ceiling:** Used in AUTOSAR ECUs with fixed task sets; stronger for ASIL-D certified functions.
- **PREEMPT_RT:** System-wide spinlock→rtmutex gives PI automatically on CAN, V4L2, and GPU drivers without driver code changes.

---

*Lecture Note 05 — Combines L10 (Lock-Free: RCU, Atomics, Memory Ordering) and L11 (Deadlock, Priority Inversion, PI Mutexes).*
