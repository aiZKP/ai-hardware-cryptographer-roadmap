# Lecture 16: Page Table Entries, Demand Paging, Copy-on-Write

**Source:** [CS124 Lec16](https://users.cms.caltech.edu/~donnie/cs124/lectures/CS124Lec16.pdf)

---

## Page Table Entries (PTE)

### Valid Bit

- **1:** Page is mapped, translation valid
- **0:** Page not in memory — **page fault** on access
- **OS** uses invalid entries for: unmapped, swapped, copy-on-write

### Dirty Bit (Modified Bit)

- **1:** Page has been written (modified)
- **0:** Page unchanged since loaded
- **Used for** page replacement — dirty pages must be written to swap before eviction
- **Set by** hardware on write (if supported)

### Other Bits

- **Accessed (Reference) bit:** Set on read or write — for LRU-like replacement
- **Writable, executable:** Permission bits
- **User/supervisor:** Kernel vs user access

---

## Aliasing

- **Same physical frame** mapped into multiple virtual addresses (or processes)
- **Shared memory** — e.g., shared libs, IPC
- **Copy-on-write** — initially shared, private copy on write

---

## Demand Paging

- **Load pages** only when first accessed (page fault)
- **Saves** memory and startup time — don't load unused code/data
- **Pure demand paging:** No preloading — every page faulted in

---

## Copy-on-Write (COW)

- **fork()** — child needs copy of parent's address space
- **Naive:** Copy all pages — expensive
- **COW:** Share pages, mark read-only; on write, copy page, then allow write
- **First write** causes page fault — kernel copies, updates mapping, resumes

### fork() vs vfork()

- **fork():** Copy (or COW) address space; child is independent
- **vfork():** Child shares parent's address space; parent suspended until child exec or exit
- **vfork** — optimization when child will immediately exec
- **Dangerous** — child must not modify memory before exec

---

## Memory Area Descriptors (Supplemental Page Tables)

- **Kernel** may not have full page table for process (e.g., lazy allocation)
- **Supplemental structure** — describes valid regions (file-backed, anonymous, stack)
- **On page fault:** Look up in supplemental table; if valid, allocate/load page; else segfault

---

## Summary

| Concept | Description |
|---------|-------------|
| Valid bit | Page present or not |
| Dirty bit | Page modified, must write back |
| Demand paging | Load on first access |
| COW | Share until write, then copy |
| Supplemental table | Describes valid regions for fault handling |
