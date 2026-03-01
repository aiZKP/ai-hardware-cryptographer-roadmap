# Lecture 3: Traps, Interrupts, Multitasking, OS Structure

**Source:** [CS124 Lec03](https://users.cms.caltech.edu/~donnie/cs124/lectures/CS124Lec03.pdf)

---

## Traps for System Calls

### How a System Call Works

1. **User program** executes a special instruction (e.g., `int 0x80` on x86, `syscall` on x86-64)
2. **CPU switches** to kernel mode, saves user context
3. **Trap handler** runs — identifies syscall number, dispatches to service routine
4. **Service routine** performs work (e.g., read from file)
5. **Return** — restore user context, switch back to user mode

### Exceptional Control Flow

| Type | Cause | Return? | Use |
|------|-------|---------|-----|
| **Trap** | Intentional (syscall) | Yes | System calls |
| **Interrupt** | External (timer, device) | Yes | I/O completion, timer tick |
| **Fault** | Recoverable error (page fault) | Yes | Demand paging |
| **Abort** | Unrecoverable (hardware error) | No | Double fault, triple fault |

### IA32 Double-Fault and Triple-Fault

- **Double fault:** CPU faults while handling a fault — e.g., page fault in page fault handler
- **Triple fault:** CPU faults while handling double fault — **CPU resets** (hardware reset)

---

## Cooperative vs Preemptive Multitasking

### Cooperative Multitasking

- **Process yields** voluntarily (e.g., `yield()`)
- **No timer interrupts** to force switch
- **Risk:** Buggy process can monopolize CPU
- **Examples:** Early Mac OS, Windows 3.x

### Preemptive Multitasking

- **Timer interrupt** periodically — OS can switch to another process
- **Fair sharing** even if one process never yields
- **Modern OSes** use this

### Hardware Timer Support

- **Programmable interval timer (PIT)** — e.g., 100 Hz tick
- **APIC (Advanced Programmable Interrupt Controller)** — per-CPU timers, IPIs

---

## IA32 APIC

- **Local APIC:** Per CPU — timer, IPI (inter-processor interrupt)
- **I/O APIC:** Routes external interrupts to CPUs
- **Used for:** Timer interrupts, IPIs for scheduling, TLB shootdown

---

## OS Structure

### Separation of Policy and Mechanism

- **Mechanism:** *How* to do something (e.g., context switch)
- **Policy:** *What* to do (e.g., which process to run next)
- **Benefit:** Change policy without changing mechanism

### Simple Structure (MS-DOS)

- **No layers** — applications could access hardware directly
- **No protection** — one bug could crash system
- **Resident in low memory**

### Monolithic Kernel

- **All OS code in one address space** — one big kernel
- **Efficient** — no IPC overhead for internal calls
- **Examples:** Linux, BSD, classic UNIX
- **Risk:** Bug in driver can crash entire kernel

### Layered Structure (THE OS, Dijkstra)

- **Layers:** Hardware → Layer 0 (scheduler) → Layer 1 (memory) → … → Layer 5 (user)
- **Each layer** uses only services of lower layers
- **Structured** but can have performance issues (e.g., layer N calling layer 0)

### Modular Kernel

- **Kernel modules** — loadable components (e.g., Linux loadable modules)
- **Core kernel** + optional modules (drivers, filesystems)
- **Flexible** — add functionality without recompiling

---

## Summary

| Concept | Description |
|---------|-------------|
| Trap | Intentional switch to kernel (syscall) |
| Fault | Recoverable error (e.g., page fault) |
| Preemptive | Timer forces process switch |
| Monolithic | All OS in one kernel |
| Modular | Loadable kernel modules |
