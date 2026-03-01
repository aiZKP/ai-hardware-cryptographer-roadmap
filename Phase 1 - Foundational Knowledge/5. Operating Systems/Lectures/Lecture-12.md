# Lecture 12: Real-Time Scheduling, Linux Schedulers

**Source:** [CS124 Lec12](https://users.cms.caltech.edu/~donnie/cs124/lectures/CS124Lec12.pdf)

---

## Real-Time Scheduling

### Soft vs Hard Real-Time

| | Soft | Hard |
|---|------|------|
| **Deadline miss** | Degraded quality | System failure |
| **Example** | Video playback | Flight control |
| **Guarantee** | Best effort | Deterministic |

### Latency Metrics

- **Event latency:** Time from event (e.g., interrupt) to start of handling
- **Interrupt latency:** Time from interrupt to handler start
- **Dispatch latency:** Time from process becoming ready until it runs

---

## Periodic Processes

- **Period** $T$ — process must run every $T$ time units
- **Execution time** $C$ — needs $C$ time each period
- **Utilization** $U = C/T$ — fraction of CPU used

---

## Rate-Monotonic Scheduling (RMS)

- **Priority** = inverse of period — shorter period = higher priority
- **Preemptive** — higher priority preempts lower
- **Upper bound** on CPU utilization for $n$ processes:
  $$U \leq n(2^{1/n} - 1)$$
  - $n=1$: 100%
  - $n=2$: ~83%
  - $n \to \infty$: $\ln 2 \approx 69\%$
- **Sufficient** (not necessary) — if $U$ below bound, RMS succeeds

---

## Earliest Deadline First (EDF)

- **Dynamic** priority — always run process with earliest deadline
- **Preemptive**
- **Optimal** for single CPU — if any scheduler can meet deadlines, EDF can
- **Utilization** up to 100% (if feasible)
- **Overhead** — must recalculate on each event

---

## Admission Control

- **Before** accepting new real-time task, check if schedulable
- **RMS:** Check utilization bound
- **EDF:** Check $\sum C_i/T_i \leq 1$
- **Reject** if cannot guarantee

---

## Linux 2.4 Scheduler

- **O(n)** — scanned all runnable processes each tick
- **Single run queue**
- **Did not scale** to many CPUs/processes

---

## Linux 2.6 O(1) Scheduler

- **O(1)** — constant time to pick next process
- **140 priority levels** — two arrays (active, expired)
- **Active:** Processes to run; when empty, swap with expired
- **Per-CPU** run queues
- **Interactive** bonus — I/O-bound processes got priority boost

---

## Linux Completely Fair Scheduler (CFS)

- **Default** scheduler since 2.6.23
- **Goal:** Fair — each process gets proportional CPU time
- **Virtual runtime (vruntime):** Tracks how much CPU process has used (weighted by nice value)
- **Red-black tree** of runnable processes, keyed by vruntime
- **Pick** leftmost (smallest vruntime)
- **No** fixed time slice — variable based on load
- **Real-time** processes (FIFO, RR) have separate class, higher priority than CFS

---

## Summary

| Scheduler | Use Case |
|-----------|----------|
| Rate-Monotonic | Periodic, static priority |
| EDF | Periodic, dynamic priority, optimal |
| CFS | General-purpose, fairness |
| O(1) | Legacy Linux, fixed priorities |
