# Lecture 18: Page Replacement Policies — LRU, Clock, Working Set

**Source:** [CS124 Lec18](https://users.cms.caltech.edu/~donnie/cs124/lectures/CS124Lec18.pdf)

---

## Least Recently Used (LRU)

- **Evict** page not used for longest time
- **Good** — matches locality (recently used likely to be used again)
- **No Belady's anomaly**

### Implementation with Counter

- **Each page** has timestamp (or counter) of last access
- **On access:** Update timestamp
- **On evict:** Scan all, pick smallest timestamp
- **Cost:** O(n) per eviction; need to update on every access

### Implementation with Linked List

- **Doubly linked list** — most recently used at head
- **On access:** Move page to head
- **Evict:** Tail (LRU)
- **Update** is O(1) with proper pointers

---

## Not Frequently Used (NFU)

- **Counter** per page — incremented on each access
- **Evict** lowest counter
- **Problem:** Counters only increase — old pages never "age out"
- **Not** true LRU

---

## Aging Policy

- **Periodically** shift counters right (divide by 2)
- **Add** 1 to counter if page accessed in current period
- **Approximates** LRU — recent access weighs more
- **Evict** lowest counter

---

## Second-Chance (Clock) Policy

- **Frames** in circular list
- **Reference bit** per frame — set on access
- **Clock hand** sweeps: if ref=0, evict; if ref=1, set 0, advance
- **Gives** second chance to recently used pages
- **More efficient** than full LRU — single sweep

---

## Not Recently Used (NRU)

- **Two bits:** Referenced (R), Modified (M)
- **Four classes:** (R=0,M=0), (R=0,M=1), (R=1,M=0), (R=1,M=1)
- **Evict** from lowest non-empty class
- **Periodically** clear R bits
- **Approximation** — cheap, good enough

---

## Working Set

- **Working set** of process = set of pages used in last $\tau$ time (window)
- **Idea:** Process needs its working set in memory for good performance
- **Thrashing:** Working sets exceed physical memory — constant page faults

### WSClock Policy

- **Combine** working set idea with clock
- **Evict** page not in working set (not accessed in window)
- **If** page dirty, schedule write, give second chance

---

## Page Buffering (Free Page-Frame List)

- **Keep** pool of free frames
- **Evicted** pages go to free list (or swap)
- **Pageout** can be asynchronous — don't block faulting process
- **Pre-cleaning** — write dirty pages before needed

---

## Emulating Accessed/Dirty Bits in Software

- **Hardware** may not provide R/M bits (e.g., some architectures)
- **Software:** Mark pages read-only; fault on write → set dirty, make writable
- **Read** — similar trick with execute/read permission
- **Overhead** — extra faults

---

## Adaptive Policies

- **Adjust** based on system state
- **Example:** If fault rate high, increase working set window or reduce multiprogramming
- **Page-fault frequency** — target fault rate, adjust allocation

---

## Summary

| Policy | Approximates | Cost |
|--------|--------------|------|
| LRU | Optimal for locality | High (timestamp/list) |
| Clock | LRU | Low |
| NRU | LRU | Low |
| Working set | Process needs | Medium |
