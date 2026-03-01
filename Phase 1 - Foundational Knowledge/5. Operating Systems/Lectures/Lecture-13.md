# Lecture 13: System Calls — Mechanics and Argument Passing

**Source:** [CS124 Lec13](https://users.cms.caltech.edu/~donnie/cs124/lectures/CS124Lec13.pdf)

---

## System Call Mechanics

### Interrupt vs Trap

- **Interrupt:** External (hardware) — timer, device
- **Trap:** Software — syscall instruction (e.g., `int 0x80`, `syscall`)
- **Both** switch to kernel mode, save user state, run handler

### System Call Flow

1. **User** loads syscall number and args into registers (or stack)
2. **Execute** trap instruction
3. **CPU** switches to kernel mode, jumps to trap handler
4. **Handler** reads syscall number, dispatches to service routine
5. **Service routine** validates args, performs work
6. **Return** — restore user state, return to user

### System Call Service Routines

- **Each syscall** has a routine (e.g., `sys_read`, `sys_write`)
- **Table** maps syscall number → routine
- **Routine** may block (e.g., wait for I/O) — process is switched out

---

## Verifying Pointers from User Mode

**Problem:** User passes pointer to kernel (e.g., buffer for `read()`). User could pass:

- **Invalid** address (e.g., 0x0, unmapped)
- **Kernel** address (to read kernel memory)
- **Pointer to kernel** (attack)

**Kernel must validate** before dereferencing.

### Linux: Exception Tables

- **Access** user pointer in special way — can fault
- **If fault** in kernel (page fault on user address) — not necessarily bug
- **Exception table** — list of faulting instructions that are "allowed" (e.g., copy_from_user)
- **Handler** fixes up — returns error to user instead of oops

### Kernel Oops

- **Unexpected** fault in kernel (e.g., null dereference)
- **Kernel** prints register dump, stack trace
- **Process** may be killed; kernel continues (or panics)

---

## Pintos System Call Argument Passing

- **User** puts args on stack (or in registers)
- **Kernel** must read from user stack safely
- **Pintos** — args on user stack: syscall number, then arg1, arg2, ...
- **Validation:** Check pointer is in user space, points to valid mapped memory
- **Fetch** each arg — may need to handle page faults

---

## Summary

| Step | Action |
|------|--------|
| 1 | User sets syscall #, args |
| 2 | Trap instruction |
| 3 | Kernel handler, dispatch |
| 4 | Validate pointers |
| 5 | Service routine |
| 6 | Return to user |
