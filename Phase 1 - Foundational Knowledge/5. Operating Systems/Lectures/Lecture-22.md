# Lecture 22: File Locking, SSDs, Write Amplification

**Source:** [CS124 Lec22](https://users.cms.caltech.edu/~donnie/cs124/lectures/CS124Lec22.pdf)

---

## Concurrent File Access

- **Multiple processes** can open same file
- **Without coordination** — races, corruption
- **File locking** — coordinate access

---

## File Locking

### Advisory vs Mandatory

- **Advisory:** Lock is a hint — processes must cooperate; if they ignore, no enforcement
- **Mandatory:** Kernel enforces — read/write blocks if lock held by another
- **Unix** typically advisory (flock, lockf)

### Lock Scope

- **Whole file** — lock entire file
- **Region (byte-range)** — lock specific bytes (e.g., bytes 1000–2000)
- **Region** allows concurrent access to different parts

### flock() and lockf()

- **flock()** — whole-file, advisory (BSD)
- **lockf()** — byte-range, advisory (POSIX)
- **fcntl()** F_SETLK, F_SETLKW — byte-range, POSIX
- **Locks** are per-process (or per fd) — not inherited across fork (or are, depending on call)

---

## File Deletion and Data Remanence

- **unlink()** — remove directory entry; file data freed when no fd open
- **Data remanence** — deleted data may remain on disk until overwritten
- **Secure delete** — overwrite before freeing (slow, not always effective on SSDs)

---

## Free-Space Management

### Bitmap

- **One bit per block** — 0 = free, 1 = used
- **Find free block** — scan bitmap (or use auxiliary structure)
- **Compact** — small space

### Linked List of Free Blocks

- **Each free block** points to next
- **Head** pointer in superblock
- **Allocation** — take from head
- **Fragmentation** of free list — can use grouping

---

## Solid State Drives (SSDs)

### Flash Translation Layer (FTL)

- **Flash** — erase in large blocks (e.g., 128 KB), write in pages (e.g., 4 KB)
- **Cannot** overwrite in place — must erase first
- **FTL** — maps logical blocks to physical; hides erase/write asymmetry
- **Wear leveling** — spread writes across blocks to avoid burning out

### Erase Blocks, Read/Write

- **Read** — page granularity, fast
- **Write** — page granularity, but must erase block first (slow)
- **Erase** — block granularity, slow

### Write Amplification

- **Logical write** of 4 KB may require: read block, modify, erase block, write block
- **Amplification** = physical writes / logical writes
- **Random writes** — worst case (many blocks partially updated)
- **Sequential** — better (fewer block updates)

### TRIM Command

- **OS** tells SSD which blocks are unused (e.g., after file delete)
- **SSD** can erase blocks in background, improve future write performance
- **Without TRIM** — SSD doesn't know blocks are free; write amplification worse

### Effects on Random Writes

- **Random writes** to SSD — high write amplification, slower
- **Design** for sequential when possible (e.g., log-structured filesystems)
- **Fragmentation** — less of an issue than HDD (no seek) but write amplification matters

---

## Summary

| Concept | Description |
|---------|-------------|
| Advisory lock | Cooperative; not enforced |
| Mandatory lock | Kernel enforces |
| FTL | Maps logical to physical; wear leveling |
| Write amplification | Physical writes > logical |
| TRIM | Inform SSD of freed blocks |
