# Lecture 15: Virtual Memory, MMU, Paging

**Source:** [CS124 Lec15](https://users.cms.caltech.edu/~donnie/cs124/lectures/CS124Lec15.pdf)

---

## Program Layout and ABI

### Program Addresses

- **Absolute:** Fixed addresses (rare, embedded)
- **Relocatable:** Base + offset; loader sets base
- **Position-independent (PIC):** Code works at any address (used in shared libs)

### Application Binary Interface (ABI)

- **Defines** calling convention, data layout, syscall interface
- **Examples:** x86-64 System V ABI, ARM AAPCS

---

## Virtual vs Physical Addresses

- **Virtual address:** What program uses (e.g., 0x400000)
- **Physical address:** Actual RAM location
- **MMU** translates virtual → physical
- **Benefits:** Isolation, more address space than physical RAM, relocation

---

## Memory Management Unit (MMU)

- **Hardware** that translates virtual addresses
- **Uses** page tables (in memory) — OS sets them up
- **TLB** (Translation Lookaside Buffer) — cache for translations
- **On TLB miss:** MMU walks page table (or faults to OS)

---

## Swapping and Backing Store

- **Swap:** Move process (or pages) to disk when memory full
- **Backing store:** Disk space for swapped pages
- **Swap partition** or **swap file**

---

## MMU Strategies

### Relocation and Limit Registers

- **Base register:** Physical start of process
- **Limit register:** Size
- **Physical = base + virtual** (if virtual < limit)
- **Contiguous** allocation — external fragmentation

### Segmentation

- **Multiple segments** (code, data, stack) — each has base + limit
- **Virtual address** = (segment, offset)
- **External fragmentation** — variable-sized segments

### Paging

- **Physical memory** divided into **frames** (fixed size, e.g., 4 KB)
- **Virtual memory** divided into **pages** (same size)
- **Page table:** Maps virtual page # → frame # (or "not present")
- **No external fragmentation** — all units same size

---

## Page Tables

- **Entry per page** — can be huge (e.g., 2^20 entries for 4 GB with 4 KB pages)
- **Hierarchical paging** — multi-level table (e.g., 2–4 levels on x86-64)
- **Hashed page tables** — for sparse address spaces (e.g., 64-bit)
- **Clustered** — group entries for efficiency

### TLB Miss

- **TLB** = cache of recent translations
- **Miss** = translation not in TLB
- **Hardware** walks page table (or **software** on some architectures)
- **Cost:** Multiple memory accesses per miss

---

## Summary

| Concept | Description |
|---------|-------------|
| Virtual address | Program's view |
| Physical address | RAM location |
| Page | Fixed-size unit (e.g., 4 KB) |
| Frame | Physical page-sized block |
| TLB | Translation cache |
