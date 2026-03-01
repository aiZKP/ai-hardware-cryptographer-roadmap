# Lecture 17: Frame Tables, Page Replacement Basics

**Source:** [CS124 Lec17](https://users.cms.caltech.edu/~donnie/cs124/lectures/CS124Lec17.pdf)

---

## Kernel Frame Tables

- **Frame table:** Kernel data structure — one entry per physical frame
- **Tracks:** Which process/page uses frame, whether dirty, etc.
- **Used for** page replacement — pick frame to evict

### Pinned Pages

- **Pinned:** Cannot be evicted (e.g., I/O buffer, kernel code)
- **OS** marks frames as pinned during DMA, etc.
- **Unpin** when safe

---

## Multiprocessor Memory

### SMP (Symmetric Multiprocessing)

- **All CPUs** share same memory — uniform access time
- **Cache coherency** — hardware keeps caches consistent (MESI, etc.)

### NUMA (Non-Uniform Memory Access)

- **Memory** attached to specific CPUs/nodes
- **Local** access faster than **remote**
- **OS** tries to allocate on same node as process (NUMA-aware allocation)

---

## Anonymous Memory vs Memory-Mapped Files

- **Anonymous:** Not backed by file (heap, stack) — backed by swap
- **Memory-mapped file:** Pages backed by file — read from file on fault; write back on eviction (if dirty)

---

## Swap Partitions and Swap Files

- **Swap partition:** Dedicated disk partition for swap
- **Swap file:** File used as swap (e.g., `/swapfile`)
- **Multiple** swap areas — can prioritize

---

## Page Replacement Policies

**When:** Page fault and no free frame — must evict a page.

### Reference String

- **Sequence** of page accesses (e.g., 1, 2, 3, 1, 4, 2, ...)
- **Used to** evaluate replacement algorithms
- **Page fault** when referenced page not in memory

### Belady's Anomaly

- **FIFO** with more frames can have *more* page faults than with fewer
- **Counterintuitive** — more memory, worse performance
- **LRU** does not have this anomaly

---

## FIFO Page Replacement

- **Evict** oldest page (first in)
- **Simple** — queue of frames
- **Belady's anomaly** — can perform poorly
- **Ignores** access pattern

---

## Optimal (Clairvoyant) Page Replacement

- **Evict** page that will not be used for longest time
- **Requires** future knowledge — not implementable in practice
- **Useful** as lower bound for comparison
- **Offline** algorithm for evaluation

---

## Summary

| Policy | Evict | Implementable? |
|--------|-------|----------------|
| FIFO | Oldest | Yes |
| Optimal | Used farthest in future | No (clairvoyant) |
| LRU | Least recently used | Approximations |
