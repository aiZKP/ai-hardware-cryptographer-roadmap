# Lecture 8: Kernel Stacks, Interrupts, Reentrancy

**Source:** [CS124 Lec08](https://users.cms.caltech.edu/~donnie/cs124/lectures/CS124Lec08.pdf)

---

## Kernel Stacks

- **Each process** has a **kernel stack** — used when process is in kernel (syscall, fault, interrupt)
- **Separate from user stack** — kernel doesn't trust user stack; protects kernel from user
- **Kernel stack** holds: saved registers, local variables, call frames during syscall/interrupt handling

### Kernel Stack Initialization

- **Allocated** when process is created
- **Fixed size** (e.g., 8 KB) — overflow = kernel panic
- **Per-thread** in multithreaded processes (each thread has own kernel stack)

---

## Kernel Control Paths

- **Path of execution** through kernel code
- **Can be interrupted** — e.g., syscall handling interrupted by timer
- **Reentrant kernel:** Multiple kernel control paths can be active (e.g., one per CPU, or nested interrupts)

---

## Process Context vs Interrupt Context

| | Process context | Interrupt context |
|---|-----------------|-------------------|
| **Triggered by** | Syscall, fault from user | Hardware interrupt |
| **Associated process** | Yes (current) | No (or "current" may be undefined) |
| **Can sleep?** | Yes (e.g., wait for I/O) | No — cannot block |
| **Can access user memory?** | Yes (with care) | Risky — no user process |

**Rule:** In interrupt context, cannot call functions that might sleep (e.g., `kmalloc` with GFP_KERNEL, `mutex_lock`).

---

## IA32 Protected-Mode Interrupt Mechanics

- **IDT (Interrupt Descriptor Table):** Maps interrupt vector → handler
- **Vector 0–31:** Exceptions (faults, traps, aborts)
- **Vector 32+:** Hardware interrupts (IRQs)
- **On interrupt:** CPU saves state, switches to kernel, jumps to handler

---

## Overlapping Interrupt Handlers

- **Interrupts can nest** — handler for IRQ 1 can be interrupted by IRQ 2
- **Or:** Kernel can disable interrupts to prevent nesting
- **Trade-off:** Disabling = simpler, but higher latency for other interrupts

### Linux: Critical, Noncritical, Deferrable

- **Critical:** Must run immediately; interrupts disabled
- **Noncritical:** Run soon; may defer
- **Softirqs, tasklets:** Deferrable work — run after interrupt, before returning to user

---

## Preemptive vs Non-Preemptive Kernels

### Non-Preemptive

- **Process in kernel** runs until it voluntarily yields or blocks
- **No timer interrupt** to preempt kernel code
- **Simpler** — fewer race conditions in kernel
- **Worse latency** — long syscall delays other processes

### Preemptive

- **Kernel can be preempted** (except in critical sections)
- **Timer** can switch to another process even during syscall
- **Better responsiveness**
- **More complex** — must protect shared data (spinlocks, etc.)

### Dispatch Latency

- **Time** from when high-priority task becomes ready until it runs
- **Preemptive kernel** reduces dispatch latency

---

## Race Conditions (Heisenbugs)

- **Concurrent access** to shared data without synchronization
- **Result:** Non-deterministic bugs — "disappears when debugging"
- **Fix:** Critical sections, mutual exclusion

---

## Critical Sections and Mutual Exclusion

- **Critical section:** Code that accesses shared resource
- **Mutual exclusion:** Only one thread in critical section at a time
- **Requirements:** No two in CS simultaneously; progress (someone eventually enters); bounded waiting

---

## Summary

| Concept | Description |
|---------|-------------|
| Kernel stack | Per-process stack for kernel execution |
| Interrupt context | Cannot sleep; no user process |
| Reentrant kernel | Multiple kernel paths active |
| Preemptive kernel | Timer can preempt kernel code |
| Critical section | Must have mutual exclusion |
