# User Plan: Example Code for Lectures 1–9

This folder contain**x**s **one small runnable program** that demonstrates ideas from **Lectures 1 through 9**:

- what a process and thread are
- how the program asks the kernel for help using system calls
- how real-time scheduling and CPU pinning work
- why memory locking and pre-faulting help real-time code
- how mutexes, read-write locks, and condition variables coordinate shared data

## What the Example Does

The demo is a tiny **real-time-style worker program**:

- It runs as a normal user-space process and asks the kernel for scheduling, CPU affinity, and memory-locking help.
- It can switch to **SCHED_FIFO** so the worker behaves more like a real-time task.
- It can pin itself to specific CPUs so the work stays on the cores you choose.
- It pre-faults memory and avoids allocations in the hot loop so timing is more predictable.
- It uses a mutex, a read-write lock, and a condition variable to show common ways threads share state safely.

Concepts from **L3 (interrupts)** and **L5 (boot/modules)** are mentioned in comments and notes; full kernel-side examples would need a driver or kernel module.

## Lecture → Code Mapping


| Lecture | Topic                                                 | Where in the code                                                                                                            |
| ------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **L1**  | OS as resource manager, user vs kernel (Ring 3 / EL0) | The program runs in user space; syscalls are the way it asks the kernel for help.                                            |
| **L2**  | Process, threads, `task_struct` (PID, tgid, affinity) | `main()` starts the process and a worker thread; `getpid()`, `gettid()`, and affinity show which task is running.            |
| **L3**  | Interrupts, top/bottom half                           | The demo only simulates an event-driven wakeup; real interrupts live in the kernel or driver.                                |
| **L4**  | System calls, vDSO                                    | `sched_setscheduler`, `sched_setaffinity`, `mlockall`, `gettid`, and `clock_gettime` all cross into kernel help when needed. |
| **L5**  | Boot, modules, device tree                            | Not shown in this user-space demo; the README explains how to inspect these on a real system.                                |
| **L6**  | Scheduling (CFS, SCHED_FIFO, nice)                    | `sched_setscheduler(SCHED_FIFO)` shows how a task can be given real-time priority.                                           |
| **L7**  | Real-Time Linux (PREEMPT_RT, latency)                 | `mlockall()`, pre-faulting, and no allocations in the loop keep timing more predictable.                                     |
| **L8**  | CPU affinity & isolation                              | `sched_setaffinity()` keeps the worker on chosen CPUs.                                                                       |
| **L9**  | Synchronization (mutex, rwlock, completion-like)      | `pthread_mutex_t`, `pthread_rwlock_t`, and `pthread_cond_t` show how threads protect shared data and wait for events.        |


## Build and Run

**Requires:** Linux (or WSL on Windows) with pthreads and `sched.h`.

```bash
cd "Phase 1 - Foundational Knowledge/4. Operating Systems/user-plan"
make
./rt_demo [options]
```

If `make` is not available, build manually:

```bash
gcc -Wall -O2 -pthread -o rt_demo rt_demo.c
```

**Options (see `./rt_demo --help`):**

- `--cpu 2,3` — Pin the program to CPUs 2 and 3.
- `--rt` — Use SCHED_FIFO priority 80. This usually needs root or `cap_sys_nice`.
- `--lock-memory` — Call `mlockall()`. This usually needs root or `cap_ipc_lock`.
- `--no-rt` — Run without RT scheduling.

**Running with real-time and locking (needs root):**

```bash
sudo ./rt_demo --rt --lock-memory --cpu 1
```

**Observing syscalls (L4):**

```bash
strace -e sched_setscheduler,sched_setaffinity,mlockall,gettid,clock_gettime ./rt_demo --rt --cpu 1 2>&1 | head -50
```

## L5 (Boot / Modules / Device Tree)

There is no kernel module in this repo. To see L5 in action:

- **Boot:** Watch `dmesg` during boot; use `systemd-analyze` to measure boot time.
- **Modules:** Use `lsmod`, `modinfo`, and `modprobe` to load and unload a module (for example `loop`).
- **Device Tree:** On ARM, inspect `/sys/firmware/devicetree/base/` or boot logs for the machine model.

## Requirements

- Linux with pthreads and `sched.h` (POSIX real-time optional).
- For `--rt` and `--lock-memory`: run as root or have `cap_sys_nice` and `cap_ipc_lock`.
- For affinity: `sched_setaffinity` is supported on multi-core systems.

## Files

- `README.md` — This file (lecture mapping and usage).
- `rt_demo.c` — Main example (process, threads, syscalls, scheduling, affinity, sync).
- `Makefile` — Builds `rt_demo`.

