# Lecture 9: Synchronization, Deadlock, Semaphores

**Source:** [CS124 Lec09](https://users.cms.caltech.edu/~donnie/cs124/lectures/CS124Lec09.pdf)

---

## Software Mutual Exclusion: Peterson's Algorithm

**Two-thread** mutual exclusion using only shared memory (no hardware support):

```c
// Shared
int turn = 0;
int interested[2] = {0, 0};

void enter_critical(int me) {
  int other = 1 - me;
  interested[me] = 1;
  turn = me;
  while (interested[other] && turn == me)
    ;  // spin
}

void leave_critical(int me) {
  interested[me] = 0;
}
```

- **Does not scale** to many threads
- **Assumes** sequential consistency (no instruction reordering)
- **Historical** — shows software-only solution is possible

---

## Hardware Mutual Exclusion

### Disabling Interrupts

- **Single CPU:** Disable interrupts in critical section — no preemption
- **Does not work** for multi-CPU — other CPUs can still run
- **Used** for very short sections (e.g., updating run queue)

### Memory Barriers

- **Prevent reordering** of memory operations by compiler/CPU
- **Examples:** `mfence`, `lfence`, `sfence` (x86); `__sync_synchronize()` (GCC)
- **Needed** for lock-free algorithms, correct use of atomics

### Optimization Barriers

- **Compiler barrier:** `asm volatile("" ::: "memory")` — prevents compiler from moving loads/stores across it
- **CPU barrier:** Also prevents CPU reordering

---

## Spinlocks

- **Busy-wait** until lock is free
- **Used when** wait is expected to be short (e.g., in kernel, holding lock for microseconds)
- **Inefficient** if held long — wastes CPU
- **Implementation:** Atomic test-and-set, compare-and-swap (CAS)

```c
// Pseudocode
void spin_lock(lock_t *l) {
  while (atomic_cas(&l->val, 0, 1) != 0)
    ;  // spin
}
void spin_unlock(lock_t *l) {
  l->val = 0;
}
```

---

## Deadlocks

**Deadlock:** Set of processes blocked, each waiting for a resource held by another.

### Four Conditions (all required)

1. **Mutual exclusion** — resource cannot be shared
2. **Hold and wait** — process holds some resources, waits for more
3. **No preemption** — cannot forcibly take resource from holder
4. **Circular wait** — cycle in resource allocation graph

### Deadlock Prevention

- **Eliminate one condition:**
  - **No hold-and-wait:** Acquire all resources at once (conservative)
  - **Preemption:** Allow抢占 (complex for some resources)
  - **No circular wait:** Order resources, acquire in order (e.g., always lock A before B)

### Deadlock Avoidance

- **Banker's algorithm:** Only grant request if system stays in safe state
- **Requires** knowing future resource needs — often impractical

### Deadlock Detection and Recovery

- **Detect** cycles in resource graph
- **Recovery:** Abort a process, preempt resources
- **Used when** prevention/avoidance too costly

---

## Semaphores

**Semaphore:** Integer counter + wait queue. Operations: **P** (down, wait) and **V** (up, signal).

### Counting Semaphore

- **Value** = number of available resources
- **P:** Decrement; block if 0
- **V:** Increment; wake one waiter if any

### Binary Semaphore (Mutex)

- **Value** 0 or 1
- **Used for** mutual exclusion
- **P** before critical section, **V** after

### Mutex Locks

- **Mutex** = mutual exclusion lock
- **Lock** before CS, **unlock** after
- **Often** implemented with semaphore (binary) or futex (Linux)

---

## Thinking Like a Kernel Programmer

- **Identify** shared data and critical sections
- **Choose** lock granularity — fine-grained (more locks, less contention) vs coarse (simpler, more contention)
- **Avoid** deadlock — lock ordering, trylock, timeout
- **Consider** interrupt context — use spinlock, not mutex (mutex can sleep)

---

## Pintos Threads Project

- **Implement** thread switching, synchronization primitives
- **Alarm clock** — thread sleep
- **Priority scheduling**
- **Advanced:** MLFQS (multi-level feedback queue scheduler)

---

## Summary

| Primitive | Use |
|-----------|-----|
| Spinlock | Short critical sections, interrupt context |
| Mutex | Longer sections, can sleep |
| Semaphore | Resource counting, signaling |
| Peterson | Software-only (2 threads) |
