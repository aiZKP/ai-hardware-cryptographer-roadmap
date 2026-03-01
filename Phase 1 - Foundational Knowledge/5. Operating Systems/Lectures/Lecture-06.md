# Lecture 6: The Process Abstraction

**Source:** [CS124 Lec06](https://users.cms.caltech.edu/~donnie/cs124/lectures/CS124Lec06.pdf)

---

## Process Abstraction

A **process** is an instance of a program in execution. It includes:

- **Code** (text) — program instructions
- **Data** — global/static variables
- **Heap** — dynamically allocated memory
- **Stack** — local variables, return addresses
- **Resources** — file descriptors, sockets, etc.

**Key idea:** The OS gives each process the illusion of having its own CPU and memory.

---

## Process States

| State | Description |
|-------|-------------|
| **New** | Being created |
| **Ready** | In memory, waiting for CPU |
| **Running** | Executing on CPU |
| **Blocked** (waiting) | Waiting for I/O or event |
| **Suspended** | Swapped out; not in ready queue |
| **Terminated** | Finished, resources being reclaimed |

### State Transitions

- **New → Ready:** Admitted (e.g., by long-term scheduler)
- **Ready → Running:** Dispatched (short-term scheduler chose it)
- **Running → Ready:** Preempted (time slice expired) or yielded
- **Running → Blocked:** Waiting for I/O or event
- **Blocked → Ready:** I/O complete, event occurred
- **Running → Terminated:** Exit

---

## Process Control Block (PCB)

Also: **Task Control Block (TCB)**, **task_struct** (Linux)

**Contains:**

- **Process ID (PID)**
- **State** (ready, running, blocked, …)
- **Program counter** — next instruction
- **CPU registers** — saved on context switch
- **Memory management** — page table pointer, limits
- **I/O status** — open files, pending I/O
- **Accounting** — CPU time used, etc.
- **Scheduling info** — priority, time slice remaining

---

## Process Context Switch

**When:** Preemption (timer), voluntary yield, or blocking

**Steps:**

1. **Save** current process's CPU state (registers, PC) to its PCB
2. **Switch** page table (if per-process)
3. **Restore** next process's state from its PCB
4. **Resume** execution of next process

**Cost:** Context switch has overhead (cache/TLB flushes, kernel entry/exit). Too-frequent switching hurts performance.

---

## Process Management

### Run Queue (Ready Queue)

- **List of processes** in Ready state
- **Scheduler** picks from this queue when CPU is free
- **May be** multiple queues (e.g., per priority level)

### Wait Queues

- **Per resource** (e.g., per disk, per lock)
- **Blocked processes** wait here until resource available
- **Wakeup** when I/O completes or lock released

---

## Process Scheduling Levels

### Long-Term (Job Scheduling)

- **Admission control** — which jobs to admit into system
- **Controls degree of multiprogramming**
- **Batch systems** — important; interactive systems — less so

### Medium-Term

- **Swapping** — move process between memory and disk
- **Controls** which processes are in memory (ready to run)

### Short-Term (CPU Scheduling)

- **Which ready process** runs next
- **Runs frequently** (every time slice, or when process blocks)
- **Most studied** — FCFS, RR, SJF, etc.

---

## Summary

| Concept | Description |
|---------|-------------|
| Process | Program in execution; has address space, resources |
| PCB | Kernel data structure for each process |
| Context switch | Save one process, restore another |
| Run queue | Ready processes waiting for CPU |
| Wait queue | Blocked processes waiting for event |
