# Lecture 14: UNIX Signals

**Source:** [CS124 Lec14](https://users.cms.caltech.edu/~donnie/cs124/lectures/CS124Lec14.pdf)

---

## UNIX Signals Overview

**Signal:** Asynchronous notification to a process. Like a "software interrupt" — can be sent by kernel or another process.

**Examples:** SIGINT (Ctrl+C), SIGTERM (kill), SIGSEGV (segfault), SIGCHLD (child exited)

---

## Signal Handlers

- **Default action:** Terminate, ignore, core dump, stop, continue
- **User handler:** Process can install custom function via `signal()` or `sigaction()`
- **Handler** runs when signal is delivered (not when generated)

---

## Advanced: siginfo_t and ucontext_t

- **sigaction()** can provide `siginfo_t` — signal number, sender PID, fault address (e.g., for SIGSEGV)
- **ucontext_t** — CPU context (registers) at time of signal
- **Used for** debugging, custom fault handling

---

## Pending and Blocked Signal Masks

- **Pending:** Signal generated but not yet delivered
- **Blocked:** Process has blocked this signal — it stays pending until unblocked
- **Mask** — per-signal block bit
- **Delivery** when: Process unblocks signal, or returns from handler

---

## Signal Generation vs Delivery

- **Generation:** Signal sent (e.g., kill(), or kernel on fault)
- **Delivery:** Signal actually handled (handler runs or default action)
- **Can be delayed** if signal is blocked

---

## Kernel Signal Data Structures

- **Pending signal queue** (or bitmask) — per process
- **sigaction array** — per signal, per process: handler, flags, mask
- **Kernel** checks pending on return to user (from syscall, interrupt)

---

## Signal Delivery: Kernel vs User Handler

- **Kernel-handled:** Some signals (e.g., SIGKILL) — kernel does action, no user handler
- **User handler:** Kernel sets up stack frame, jumps to handler; handler returns via `sigreturn()` syscall

---

## User-Mode Handler Invocation

1. **Kernel** saves user context on stack
2. **Kernel** sets up handler stack frame (return address = sigreturn trampoline)
3. **Kernel** jumps to user handler
4. **Handler** returns → sigreturn trampoline
5. **sigreturn()** syscall — kernel restores saved context, resumes

---

## Signal Handlers and Interrupted System Calls

- **Slow syscall** (e.g., read from terminal) can block
- **Signal** arrives → handler runs
- **Syscall** returns with EINTR (interrupted)
- **Application** can retry or handle
- **SA_RESTART** — kernel restarts syscall automatically (for some syscalls)

---

## Signal Handlers and longjmp

- **longjmp** from handler can bypass normal cleanup
- **Undefined behavior** if longjmp crosses signal handler boundary incorrectly
- **sigsetjmp/siglongjmp** — save/restore signal mask

---

## Summary

| Concept | Description |
|---------|-------------|
| Pending | Generated, not yet delivered |
| Blocked | Will not deliver until unblocked |
| Handler | User function, runs in user context |
| sigreturn | Restores context after handler |
