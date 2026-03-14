# Operating Systems — Detailed Study Guide

**Based on [Caltech CS124 Spring 2024](https://users.cms.caltech.edu/~donnie/cs124/lectures/) — Operating Systems**

This guide provides a structured, in-depth treatment of operating systems concepts aligned with a university-level OS course. It complements **Linux Fundamentals** (practical usage) and **Embedded Systems Basics** (RTOS concepts) by covering the theoretical foundations: processes, threads, scheduling, memory management, synchronization, and filesystems.

**Why OS matters for AI hardware engineers:** Every AI workload runs on an OS—Linux on Jetson, embedded Linux on ADAS, RTOS on microcontrollers. Understanding process scheduling, memory management, and I/O helps you optimize inference latency, debug real-time constraints, and design systems that meet deterministic deadlines.

---

## Course Overview

| Aspect | Details |
|--------|---------|
| **Source** | Caltech CS124 (Donnie Pinkston) |
| **Format** | 24 lectures, PDF slides available |
| **Prerequisites** | C programming, basic computer architecture |
| **Key Projects** | Pintos (Stanford OS teaching kernel) — Threads, User Programs, VM, File System |

---

## Lecture Index

| # | Topic | Key Concepts |
|:-:|-------|---------------|
| [1](Lectures/Lecture-01.md) | Introduction & OS History | Mainframes, batch, multiprogramming, time-sharing, virtualization, RTOS |
| [2](Lectures/Lecture-02.md) | OS Components & UNIX I/O | System calls, kernel/user mode, file descriptors, pipes, shells |
| [3](Lectures/Lecture-03.md) | Traps, Interrupts & OS Structure | Traps, faults, aborts, preemption, monolithic vs microkernel |
| [4](Lectures/Lecture-04.md) | Microkernels & Exokernels | Mach, L4, message-passing IPC, hybrid kernels |
| [5](Lectures/Lecture-05.md) | Bootstrap & Firmware | BIOS, MBR, UEFI, ACPI, chain loading |
| [6](Lectures/Lecture-06.md) | Process Abstraction | Process states, PCB, context switch, run/wait queues |
| [7](Lectures/Lecture-07.md) | Threads | User vs kernel threads, threading models, Amdahl's Law |
| [8](Lectures/Lecture-08.md) | Kernel Stacks & Interrupts | Reentrant kernels, interrupt context, critical sections |
| [9](Lectures/Lecture-09.md) | Synchronization & Deadlock | Peterson's algorithm, spinlocks, semaphores, deadlock conditions |
| [10](Lectures/Lecture-10.md) | Advanced Synchronization | RCU, read/write locks, lock granularity |
| [11](Lectures/Lecture-11.md) | Process Scheduling | FCFS, RR, SJF, priority, multilevel queues |
| [12](Lectures/Lecture-12.md) | Real-Time & Linux Schedulers | EDF, rate-monotonic, CFS, O(1) scheduler |
| [13](Lectures/Lecture-13.md) | System Calls | Trap mechanics, argument passing, pointer validation |
| [14](Lectures/Lecture-14.md) | UNIX Signals | Signal handlers, pending/blocked masks, sigreturn |
| [15](Lectures/Lecture-15.md) | Virtual Memory & MMU | Address translation, paging, TLB, demand paging |
| [16](Lectures/Lecture-16.md) | Page Tables & Copy-on-Write | PTE bits, COW, fork/vfork |
| [17](Lectures/Lecture-17.md) | Frame Tables & Page Replacement | FIFO, Optimal, Belady's Anomaly |
| [18](Lectures/Lecture-18.md) | Page Replacement Policies | LRU, Clock, NRU, working set |
| [19](Lectures/Lecture-19.md) | Pintos VM Design | Project 4 design discussion |
| [20](Lectures/Lecture-20.md) | Page Allocation & Thrashing | Global vs local, working set model, Linux/Windows paging |
| [21](Lectures/Lecture-21.md) | Filesystems | Directories, allocation (contiguous, linked, indexed), inodes |
| [22](Lectures/Lecture-22.md) | File Locking & SSDs | flock, lockf, FTL, write amplification, TRIM |
| [23](Lectures/Lecture-23.md) | Pintos File System Design | Project 5 design discussion |
| [24](Lectures/Lecture-24.md) | Journaling Filesystems | Log-structured updates, crash consistency |
| [25](Lectures/Lecture-25.md) | Capstone Project | Custom Linux images with Yocto |
| [26](Lectures/Lecture-26.md) | **eBPF Deep Dive** | eBPF VM & ISA, verifier, BPF maps, libbpf CO-RE, XDP networking, sched_ext, AI system observability |

---

## Recommended Study Path

1. **Lectures 1–5:** Foundations — history, components, kernel architectures, boot process
2. **Lectures 6–9:** Processes, threads, synchronization — core concurrency
3. **Lectures 10–12:** Scheduling — from basic algorithms to Linux CFS
4. **Lectures 13–14:** System calls and signals — user/kernel boundary
5. **Lectures 15–20:** Memory management — VM, paging, replacement
6. **Lectures 21–24:** Filesystems — allocation, locking, journaling
7. **Lecture 25:** Capstone — Yocto build integration
8. **Lecture 26:** eBPF — programmable kernel observability, XDP networking, custom schedulers

---

## Resources

- **Primary:** [CS124 Lecture Slides (PDF)](https://users.cms.caltech.edu/~donnie/cs124/lectures/)
- **Textbook:** *Operating System Concepts* (Silberschatz, Galvin, Gagne) — "Dinosaur Book"
- **Projects:** [Pintos](https://web.stanford.edu/class/cs140/projects/pintos/) — Stanford OS teaching kernel
- **Linux Kernel:** [Linux Insides](https://0xax.gitbooks.io/linux-insides/) — kernel internals
- **Real platform (AGNOS):** [agnos-kernel-sdm845](https://github.com/commaai/agnos-kernel-sdm845) and [agnos-builder](https://github.com/commaai/agnos-builder) — **forked and custom-modified Linux** for comma 3X/four, built for **openpilot on the road**. [**AGNOS Guide**](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/4.%20Autonomous%20Driving/agnos/Guide.md) maps **all 26 OS lectures** to both kernel/builder paths and to the **development changes** in the fork (boot, device tree, camera/CAN drivers, scheduler/RT tuning, etc.).

---

## AI Hardware Connection

| OS Concept | AI/Embedded Relevance |
|------------|------------------------|
| **Real-time scheduling** | Inference deadlines, sensor fusion pipelines |
| **Memory management** | Model weights in RAM, DMA for camera buffers |
| **File descriptors & I/O** | Camera streams, CAN bus, logging |
| **Process/thread model** | modeld, camerad, plannerd in openpilot |
| **Kernel vs user mode** | Driver vs userspace inference |
| **eBPF** | Production profiling of GPU driver latency, scheduler analysis for RT inference, XDP for sensor data filtering, custom schedulers (sched_ext) for inference-priority scheduling |
