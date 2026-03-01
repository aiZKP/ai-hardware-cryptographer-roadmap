# Lecture 4: Microkernels, Exokernels, Hybrid Kernels

**Source:** [CS124 Lec04](https://users.cms.caltech.edu/~donnie/cs124/lectures/CS124Lec04.pdf)

---

## Microkernels

### Idea

- **Minimal kernel** — only essential services: address spaces, threads, IPC
- **Everything else** (filesystems, drivers, network stack) runs in **user-space servers**
- **Communication** via **message passing**

### Liedtke's Minimality Principle

> A concept is allowed in the kernel only if moving it outside would prevent the system from functioning.

- **Kernel:** Address spaces, threads, IPC
- **Not in kernel:** File system, drivers, network — run as servers

### Benefits

- **Reliability:** Bug in filesystem server doesn't crash kernel
- **Flexibility:** Replace servers without kernel changes
- **Security:** Smaller trusted computing base (TCB)

### Drawbacks

- **IPC overhead** — every file read crosses kernel boundary twice (request + reply)
- **Performance** — microkernel systems historically slower than monolithic
- **Complexity** — more moving parts, harder to design

---

## Message-Passing IPC

- **Send message** to server (e.g., "read file X, offset 0, 4096 bytes")
- **Server** runs in user space, handles request
- **Reply message** returns data
- **Synchronous** (block until reply) or **asynchronous** (callback later)

---

## CMU Mach

- **Microkernel** developed at CMU (later used in parts of macOS)
- **Tasks** (address spaces) and **threads**
- **Ports** — message queues for IPC
- **External pager** — user-space memory manager for demand paging

---

## L4 Family of Microkernels

- **Minimal** — ~10K lines of code
- **Fast IPC** — optimized for low latency
- **Used in:** Secure systems, embedded, seL4 (formally verified)

---

## Hybrid Kernels

- **Compromise:** Some services in kernel (for performance), some in user space
- **Example:** Windows NT — graphics, some drivers in kernel; others in user space
- **macOS X (XNU):** Mach microkernel + BSD layer (monolithic) — "hybrid"

---

## Exokernels

### Idea

- **Expose hardware** to applications with minimal abstraction
- **Library OS** — each application links its own OS library (filesystem, network)
- **Kernel** only provides **secure multiplexing** of physical resources
- **Application** decides how to use disk blocks, memory, etc.

### Benefits

- **Performance** — no unnecessary abstraction
- **Flexibility** — app can use custom filesystem, network protocol
- **Used in:** Research (ExOS), some cloud/container work

---

## Summary

| Architecture | Kernel Size | IPC | Use Case |
|--------------|-------------|-----|----------|
| Monolithic | Large | Function calls | Linux, BSD |
| Microkernel | Small | Message passing | L4, seL4, QNX |
| Hybrid | Medium | Mix | Windows NT, macOS |
| Exokernel | Minimal | Resource grants | Research |
