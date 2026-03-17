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

CPUs and compilers reorder operations. Ordering is controlled by **memory order** on atomics:

| Order | Guarantee |
|-------|------------|
| `memory_order_relaxed` | Atomicity only; no ordering; use for counters. |
| `memory_order_acquire` | No later op reordered before this load; pairs with release. |
| `memory_order_release` | No earlier op reordered after this store; pairs with acquire. |
| `memory_order_seq_cst` | Total order across threads; default; full fence. |

**Acquire-release:** A **release** store to X "happens-before" a later **acquire** load from X on another thread. All writes before the release are visible after the acquire. Use **release**/ **acquire** for producer-consumer "data ready" flags — **relaxed** is wrong (no visibility guarantee for the payload).

---

## Hardware Memory Models & Kernel Barriers

| Architecture | Model | Barrier need |
|--------------|--------|---------------|
| x86-64 | TSO (strong) | Few; LOCK prefix for atomics. |
| ARM64 | Weak | Explicit DMB/DSB; LDAR/STLR for acquire/release. |
| RISC-V | Weak (RVWMO) | FENCE; LR/SC for atomics. |

Kernel: `smp_mb()`, `smp_rmb()`, `smp_wmb()`. On ARM64 (e.g. Jetson), code that "works" on x86 without barriers can fail; use C11 atomics or kernel barriers for shared state.

---

## Atomics & Compare-and-Swap (CAS)

- **Atomics:** `atomic_inc`, `atomic_dec_and_test`, `atomic_cmpxchg`, `atomic_add_return`. Map to single instructions (e.g. LOCK XADD, CAS). Use for refcounts, flags, per-CPU stats.
- **CAS:** Test value, replace only if it matches. Basis of most lock-free structures. On failure, `expected` is updated with current value — retry with that.

**ABA problem:** Pointer goes A → B → A; CAS sees same value but object was replaced (use-after-free risk). Mitigations: version tag (e.g. 128-bit CAS), hazard pointers, or RCU.

---

## SPSC Ring Buffer (Lock-Free)

Single producer, single consumer: use **release** store on tail (producer) and **acquire** load on tail (consumer). No lock; tail is the sync point. Throughput limited by memory bandwidth. Used in openpilot VisionIPC for zero-copy camera frames. **Relaxed** tail would not guarantee visibility of the written data.

---

## RCU (Read-Copy-Update)

**Readers:** `rcu_read_lock()`; `ptr = rcu_dereference(gp)`; use `*ptr`; `rcu_read_unlock()`. Cost: preempt disable/enable only; no cache-line writes.

**Writers:** (1) Allocate and fill new object. (2) Publish: `rcu_assign_pointer(gp, new_obj)`. (3) Wait for **grace period** (`synchronize_rcu()` or `call_rcu()`). (4) Free old object. Grace period = until every CPU has passed a quiescent state (context switch, idle, return to user). After that, no pre-existing reader can hold the old pointer.

**Variants:** Classic RCU (read section cannot sleep); SRCU (sleepable); PREEMPT_RCU (preemptible kernel). **rcu_nocbs=** offloads callbacks from specified CPUs (e.g. isolated RT/inference cores) to avoid jitter.

---

## kfifo, Hazard Pointers

- **kfifo:** Kernel SPSC FIFO; lock-free for single producer and single consumer; used between ISR and process context.
- **Hazard pointers:** Reader publishes the pointer it is using; reclaimer scans all HP slots before freeing; defer free if pointer is in use. Userspace alternative to RCU (e.g. Folly, C++26).

---

## Part 1 Summary Table

| Technique | Reader cost | Writer cost | Safe in ISR? | Limitation |
|-----------|-------------|-------------|--------------|------------|
| Spinlock | Cache write + spin | Cache write | Yes | Wasted CPU; no sleep |
| CAS loop | Atomic RMW + retry | Atomic RMW + retry | Yes | ABA; retry cost |
| SPSC ring | Acquire load | Release store | Yes | Single producer AND consumer |
| RCU | Preempt disable | Copy + grace period | No (sync_rcu blocks) | Read-mostly; writer pays |
| Hazard pointers | Publish + load | Scan HP slots | No | Reclamation overhead |

---

# Part 2: Deadlock, Priority Inversion & PI Mutexes

**Context:** Deadlock = tasks waiting for each other forever. Priority inversion = high-priority task blocked by a low-priority one (via a shared resource). Both require design-time and runtime measures (ordering, PI, lockdep).

---

## Deadlock: Definition & Coffman Conditions

**Deadlock:** Set of processes each waiting for a resource held by another in the set; no one can make progress.

**Coffman conditions** (all four must hold): (1) **Mutual exclusion** — resource not shareable. (2) **Hold-and-wait** — hold one resource while waiting for another. (3) **No preemption** — resources cannot be taken by force. (4) **Circular wait** — cycle in resource-allocation graph. Breaking any one prevents deadlock.

---

## Deadlock Prevention

| Break this | Technique | Trade-off |
|------------|-----------|-----------|
| Hold-and-wait | Acquire all locks at once; release all if any fails | Less concurrency; retry |
| Circular wait | **Global lock ordering:** always acquire in fixed order; lockdep enforces | Discipline at all call sites |
| No preemption | `mutex_trylock()` + backoff | Retry; livelock risk |
| Mutual exclusion | Lock-free structures | More complex |

**Global lock ordering:** Assign ranks to locks; always acquire in ascending order; document and use lockdep. New locks must be integrated into the ordering.

---

## Priority Inversion

**Mechanism:** (1) L (low) holds mutex. (2) H (high) blocks on that mutex. (3) M (medium) preempts L. (4) L never runs → cannot release mutex → H waits indefinitely. H is effectively running at L's priority — **inverted**.

**Mars Pathfinder (1997):** bc_sched (low) held mutex; bc_dist (high) blocked; ASI_MET (medium) ran; bc_sched never ran → bc_dist missed watchdog → reset. Fix: enable **priority inheritance** on the mutex (available in VxWorks but off by default).

---

## Priority Inheritance (PI)

When H blocks on a mutex held by L, the kernel **boosts L** to H's priority until L releases the mutex. **Transitive PI:** if L is also blocked on another lock held by X, boost propagates to X. Linux: `rtmutex`; PREEMPT_RT turns most `spinlock_t` into `rtmutex` → PI system-wide. Userspace: `pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT)`. PI is most effective with SCHED_FIFO/SCHED_RR; on SCHED_OTHER (CFS) it has limited effect.

---

## Priority Ceiling

Each mutex has a **ceiling priority** = highest priority of any task that will ever take it. Any task that acquires it runs at ceiling for the duration. Prevents inversion without runtime discovery. Used in ARINC 653, AUTOSAR. Stronger for fixed task sets; every acquisition pays the ceiling cost.

---

## lockdep & Watchdog

- **lockdep:** Tracks lock acquisition order; reports potential AB-BA cycles and invalid use (e.g. sleeping lock in IRQ). Reports at first observation of a bad ordering. Enable in dev (`CONFIG_PROVE_LOCKING=y`); disable in production (overhead).
- **Watchdog:** Last resort; reset or safe state if a task does not kick in time. Mars Pathfinder watchdog correctly detected the missed deadline; the real fix was PI, not relying on reset.

---

## Part 2 Summary Table

| Problem | Symptom | Detection | Solution |
|---------|---------|-----------|----------|
| Deadlock | Hang; tasks blocked forever | Resource graph; lockdep | Lock ordering; acquire-all; trylock |
| Priority inversion | High-priority task stalls | Latency; watchdog | PI mutex; priority ceiling |
| Livelock | CPU busy, no progress | Profiling | Backoff; arbitration |
| Starvation | Low-priority never runs | Wait monitoring | Aging; FIFO; boost |

---

## AI Hardware Connection

- **RCU:** Live model config / LoRA swaps without pausing inference; writer publishes new config, old freed after grace period.
- **SPSC ring + acquire/release:** Zero-copy camera path (e.g. VisionIPC); no mutex in 30 Hz frame path.
- **rcu_nocbs=** on isolated inference cores removes RCU callback jitter.
- **PI mutex (rtmutex / PTHREAD_PRIO_INHERIT):** Required for openpilot controlsd/plannerd and ROS2 RT executors; Mars Pathfinder is a standard reference in ASIL-B reviews.
- **lockdep:** Standard in driver bring-up (camera, ISP, DMA) before deployment.
- **Priority ceiling:** Used in AUTOSAR ECUs where task set is fixed; stronger for ASIL-D.

---

*Combines Lectures L10, L11 (Lock-Free: RCU, Atomics; Deadlock, Priority Inversion, PI Mutexes).*
