# Lecture 11: Process Scheduling

**Source:** [CS124 Lec11](https://users.cms.caltech.edu/~donnie/cs124/lectures/CS124Lec11.pdf)

---

## Scheduler and Dispatcher

- **Scheduler:** Chooses which ready process runs next (policy)
- **Dispatcher:** Performs context switch to that process (mechanism)
- **switch_threads()** — save current, restore next

---

## Scheduling Metrics

| Metric | Definition |
|--------|------------|
| **CPU utilization** | % of time CPU is busy |
| **Throughput** | # of processes completed per unit time |
| **Turnaround time** | Time from submission to completion |
| **Waiting time** | Time spent in ready queue |
| **Response time** | Time from submission to first response (interactive) |

---

## CPU Bursts and I/O Bursts

- **CPU burst:** Process uses CPU
- **I/O burst:** Process waits for I/O
- **Process** alternates between them
- **CPU-bound:** Long CPU bursts
- **I/O-bound:** Short CPU bursts, long I/O waits
- **Scheduler** often favors I/O-bound for responsiveness

---

## Preemptive vs Non-Preemptive

- **Non-preemptive (cooperative):** Process runs until it blocks or yields
- **Preemptive:** Scheduler can take CPU away (e.g., on timer interrupt)

---

## First-Come First-Served (FCFS)

- **Simplest:** Ready queue is FIFO
- **Non-preemptive** (typically)
- **Convoy effect:** One long process blocks many short ones
- **No starvation**

---

## Round-Robin (RR)

- **FCFS** with **time slice (quantum)**
- **Preemptive:** When quantum expires, process goes to back of queue
- **Fair** — each process gets equal CPU time
- **Quantum** trade-off: Large → lower overhead, worse response; Small → better response, more context switches

---

## Shortest-Job-First (SJF)

- **Pick** process with shortest next CPU burst
- **Optimal** for average waiting time (provably)
- **Problem:** Need to know burst length — often unknown
- **Can estimate** from past behavior (exponential average)

### Shortest-Remaining-Time-First (SRTF)

- **Preemptive** SJF — if new process has shorter burst, preempt current
- **Better** average waiting time than SJF
- **Starvation** possible for long jobs

---

## Priority Scheduling

- **Each process** has priority
- **Run** highest-priority ready process
- **Preemptive** or non-preemptive
- **Starvation:** Low-priority may never run
- **Aging:** Increase priority of waiting processes over time

### Priority Inversion

- **High-priority** waits for lock held by **low-priority**
- **Medium-priority** runs — blocks low-priority from releasing lock
- **High-priority** effectively has low priority

### Solutions for Priority Inversion

- **Priority inheritance:** Low-priority holder inherits high-priority waiter's priority
- **Priority ceiling:** Lock has ceiling priority; holder runs at that priority
- **Random boosting:** (Less common)

---

## Multilevel Queue

- **Multiple queues** — e.g., foreground (interactive), background (batch)
- **Each queue** has its own scheduling (e.g., RR for foreground, FCFS for background)
- **Scheduling between queues** — e.g., foreground has higher priority

---

## Multilevel Feedback Queue

- **Processes move** between queues based on behavior
- **New process** → highest-priority queue
- **Use full quantum** → demote to lower queue
- **Give up CPU (I/O)** → promote or stay
- **Adapts** to CPU-bound vs I/O-bound
- **Used in** many real systems (e.g., Unix, Windows)

---

## Summary

| Algorithm | Preemptive? | Pros | Cons |
|-----------|-------------|------|------|
| FCFS | No | Simple | Convoy effect |
| RR | Yes | Fair | Overhead if quantum small |
| SJF | Optional | Optimal avg wait | Needs burst estimate |
| Priority | Optional | Flexible | Starvation, inversion |
