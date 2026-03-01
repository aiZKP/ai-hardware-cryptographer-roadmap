# Lecture 2: OS Components, Processor Modes, UNIX File I/O

**Source:** [CS124 Lec02](https://users.cms.caltech.edu/~donnie/cs124/lectures/CS124Lec02.pdf)

---

## Operating System Components

### 1. Program Execution

- **Load and run** programs
- **Context switching** between processes
- **Scheduling** — which process runs next

### 2. Resource Allocation

- **CPU time** — scheduling
- **Memory** — allocation, virtual memory
- **I/O devices** — device drivers, queues

### 3. Filesystems

- **File abstraction** — named, persistent storage
- **Directories** — hierarchical organization
- **Permissions** — access control

### 4. I/O Services

- **Device drivers** — abstract hardware
- **Buffering, caching** — improve performance
- **Interrupt handling** — asynchronous I/O completion

### 5. Communication

- **IPC** — pipes, sockets, shared memory, message queues
- **Networking** — TCP/IP stack

### 6. Accounting

- **Resource usage** — CPU time, memory, I/O
- **Billing, quotas** — in multi-user systems

### 7. Error Detection

- **Hardware errors** — memory, disk, network
- **Software errors** — exceptions, traps
- **Recovery** — restart, checkpoint, log

### 8. Protection and Security

- **Isolation** — processes cannot access each other's memory
- **Privilege levels** — kernel vs user
- **Authentication, authorization**

---

## Processor Operating Modes

### Kernel Mode (Supervisor Mode, Ring 0)

- **Full hardware access** — all instructions, all memory, all devices
- **OS code runs here** — device drivers, scheduler, memory manager
- **Privileged instructions** — e.g., halt, change page tables

### User Mode (Ring 3 on x86)

- **Restricted** — cannot access hardware directly, cannot execute privileged instructions
- **Applications run here** — isolated from each other and from kernel
- **System calls** — only way to request kernel services

### Protection Rings (x86)

| Ring | Typical Use |
|------|-------------|
| 0 | Kernel |
| 1–2 | (Often unused) |
| 3 | User applications |

---

## Virtual Memory: Kernel Space vs User Space

- **User space:** Each process has its own virtual address space (e.g., 0 to 2^48 - 1 on 64-bit)
- **Kernel space:** Kernel mappings (often shared across processes, or per-process in some designs)
- **MMU:** Translates virtual addresses to physical; enforces protection (user cannot access kernel pages)

---

## UNIX File I/O

### File Descriptors

- **Integer handle** for an open file (or pipe, socket, device)
- **0 = stdin**, **1 = stdout**, **2 = stderr** (by convention)
- **open(), read(), write(), close()** — low-level API

### stdin, stdout, stderr

- **stdin (0):** Standard input — keyboard, or redirected from file
- **stdout (1):** Standard output — terminal, or redirected to file
- **stderr (2):** Standard error — separate from stdout for error messages

### I/O Redirection

```bash
cmd < input.txt      # stdin from file
cmd > output.txt     # stdout to file
cmd 2> errors.txt    # stderr to file
cmd >> output.txt    # append stdout
cmd 2>&1             # stderr to stdout
```

### Pipes

- **cmd1 | cmd2:** stdout of cmd1 becomes stdin of cmd2
- **Kernel buffer** between processes
- **Synchronization:** Reader blocks if pipe empty; writer blocks if pipe full

### File Descriptor Duplication

- **dup(), dup2():** Create a copy of an fd pointing to the same open file
- **Used for:** Redirecting stdout to a socket, saving/restoring stdin

### Command Shell Operation

1. **Parse** command line (words, redirections, pipes)
2. **Fork** — create child process
3. **Child:** Set up redirections (dup2), exec the program
4. **Parent:** Wait for child (or run in background)

---

## Summary

| Concept | Description |
|---------|-------------|
| Kernel mode | Full privilege; OS runs here |
| User mode | Restricted; apps run here |
| System call | Trap to kernel for service |
| File descriptor | Integer handle for open file/pipe/device |
| Pipe | Unidirectional byte stream between processes |
