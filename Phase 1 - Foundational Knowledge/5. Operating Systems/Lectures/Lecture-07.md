# Lecture 7: The Thread Abstraction

**Source:** [CS124 Lec07](https://users.cms.caltech.edu/~donnie/cs124/lectures/CS124Lec07.pdf)

---

## Threads vs Processes

| | Process | Thread |
|---|---------|--------|
| **Address space** | Own | Shared with other threads in process |
| **Resources** | Own (files, etc.) | Shared |
| **Creation cost** | High (fork) | Lower |
| **Switch cost** | High (address space switch) | Lower (same address space) |

**Thread:** Lightweight unit of execution within a process. Shares address space with other threads.

---

## Threads and Performance

### Responsiveness

- **One thread blocks** (e.g., on I/O) — other threads can still run
- **GUI** stays responsive while I/O happens in background thread

### Scalability

- **Multiple cores** — threads can run in parallel
- **Single process** can use multiple CPUs

### Resource Sharing

- **Shared memory** — no IPC overhead for data sharing
- **Careful:** Need synchronization (locks, etc.)

### Economy

- **Cheaper** to create/switch threads than processes
- **Less memory** — shared address space

---

## Concurrency vs Parallelism

- **Concurrency:** Multiple tasks in progress; may interleave on one CPU
- **Parallelism:** Multiple tasks execute simultaneously on multiple CPUs
- **Concurrent** ≠ necessarily parallel (e.g., single-core time-sharing)

### Amdahl's Law

Speedup limited by sequential portion:

$$\text{Speedup} = \frac{1}{(1-P) + P/N}$$

- $P$ = fraction parallelizable
- $N$ = number of processors
- **Bottleneck:** Sequential part limits speedup

### Gustafson-Barsis' Law

With more processors, problem size can grow — different scaling model.

---

## Blocking vs Non-Blocking Operations

- **Blocking:** Thread waits until operation completes (e.g., `read()` on socket)
- **Non-blocking:** Returns immediately; may need to poll or use async I/O
- **Threads** allow blocking without freezing entire process — other threads run

---

## User-Space Threading vs Kernel Threads

### User-Space (Many-to-One)

- **Library** implements threads (e.g., GNU Pth, old Java green threads)
- **Kernel** sees one process
- **Pros:** Fast switch (no kernel call), portable
- **Cons:** One thread blocks on I/O → entire process blocks; cannot use multiple CPUs

### Kernel Threads (One-to-One)

- **Each thread** is a kernel schedulable entity
- **OS** schedules threads
- **Pros:** True parallelism, blocking I/O doesn't block process
- **Cons:** More overhead (kernel involvement)

### Many-to-Many (Hybrid)

- **M** user threads map to **N** kernel threads ($M \geq N$)
- **Scheduler activations** — kernel notifies user-level scheduler when thread blocks
- **Flexibility** — can have many user threads, fewer kernel threads

---

## Scheduler Activations, Upcalls

- **Upcall:** Kernel calls into user space (e.g., "thread T blocked")
- **User-level scheduler** can switch to another thread
- **Reduces** kernel involvement while allowing parallelism

---

## Summary

| Model | User Threads | Kernel Threads | Blocking I/O | Multi-CPU |
|-------|--------------|----------------|--------------|-----------|
| Many-to-one | Many | 1 | Blocks process | No |
| One-to-one | 1:1 | 1:1 | OK | Yes |
| Many-to-many | Many | N | OK (with upcalls) | Yes |
