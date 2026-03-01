# Lecture 1: Introduction, Course Overview, OS History

**Source:** [CS124 Lec01](https://users.cms.caltech.edu/~donnie/cs124/lectures/CS124Lec01.pdf)

---

## Operating Systems: General Principles

An **operating system** is software that:

1. **Manages hardware resources** — CPU, memory, storage, I/O devices
2. **Provides abstractions** — processes, files, sockets
3. **Mediates access** — protection, isolation, fairness

**Key principle:** The OS is the layer between applications and bare hardware. It enables multiple programs to share resources safely and efficiently.

---

## History: 1950s to Present

### Mainframes (1950s–1960s)

- **Batch processing:** Jobs submitted as decks of punch cards; operator ran them sequentially
- **No interactivity:** Long turnaround time (hours)
- **Single user at a time:** One job occupied the entire machine

### Multiprogramming (1960s)

- **Multiple jobs in memory:** While one job waited for I/O, another could use the CPU
- **Increased utilization:** CPU no longer idle during I/O waits
- **Job scheduling:** OS chose which job to run next

### Time-Sharing and Multitasking (1960s–1970s)

- **Interactive use:** Multiple users at terminals, each with the illusion of a dedicated machine
- **Time slices:** CPU shared in small quanta (e.g., 100 ms)
- **Rapid switching:** Users perceived concurrent execution

### Minicomputers (1970s)

- **Smaller, cheaper:** PDP-11, VAX — departments could own computers
- **UNIX:** Developed at Bell Labs on PDP-7/11; portable, multi-user, time-sharing

### Microcomputers (1980s)

- **Personal computers:** IBM PC, Apple II — one user, one machine
- **MS-DOS:** Simple OS, no memory protection, single-tasking
- **Mac OS, Windows:** Gradual evolution to multitasking and protection

### Multiprocessor Systems (1990s–present)

- **SMP (Symmetric Multiprocessing):** Multiple CPUs share memory
- **NUMA:** Non-uniform memory access — some memory closer to certain CPUs
- **Multi-core:** Multiple cores on one chip

---

## Virtualization and Emulation

### Virtualization

- **Hypervisor (VMM):** Software that runs multiple OS instances (VMs) on one physical machine
- **Type 1 (bare-metal):** Hypervisor runs directly on hardware (VMware ESXi, Xen, KVM)
- **Type 2 (hosted):** Hypervisor runs on a host OS (VMware Workstation, VirtualBox)

### Emulation

- **Emulator:** Software that mimics different hardware (e.g., QEMU emulating ARM on x86)
- **Used for:** Cross-platform development, legacy software, debugging

### Hypervisors

- **Hardware support:** Intel VT-x, AMD-V — CPU assists for efficient virtualization
- **Paravirtualization:** Guest OS modified to cooperate with hypervisor (e.g., Xen paravirt drivers)

---

## Embedded Operating Systems

- **Resource-constrained:** Limited CPU, RAM, storage
- **Often no MMU:** Simpler memory model (e.g., FreeRTOS, Zephyr on small MCUs)
- **Determinism:** Predictable timing for control loops

---

## Real-Time Operating Systems (RTOS)

### Soft Real-Time

- **Best-effort deadlines:** Occasional misses acceptable (e.g., video playback)
- **Linux with RT patches:** Can be used for soft real-time

### Hard Real-Time

- **Guaranteed deadlines:** Missing a deadline is a failure (e.g., flight control, brake-by-wire)
- **Deterministic scheduling:** Rate-monotonic, EDF
- **Examples:** VxWorks, QNX, FreeRTOS

---

## Summary

| Era | Paradigm | Example |
|-----|----------|---------|
| 1950s | Batch | IBM 701 |
| 1960s | Multiprogramming | IBM OS/360 |
| 1970s | Time-sharing | UNIX |
| 1980s | Personal computing | MS-DOS, Mac OS |
| 1990s+ | Multiprocessor, virtualization | Linux, Windows, VMware |
| Embedded | RTOS | FreeRTOS, VxWorks |
