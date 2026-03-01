# Lecture 20: Page Allocation, Thrashing, Linux/Windows Paging

**Source:** [CS124 Lec20](https://users.cms.caltech.edu/~donnie/cs124/lectures/CS124Lec20.pdf)

---

## Page Allocation Policies

### Degree of Multiprogramming

- **More processes** in memory → more competition for frames
- **Too many** → each process has too few frames → thrashing
- **Optimal** — balance utilization vs performance

---

## Thrashing

- **High page fault rate** — process spends most time waiting for pages
- **CPU utilization** drops — processes blocked on I/O (swap)
- **Working sets** exceed physical memory
- **Fix:** Reduce multiprogramming (swap out entire processes), or add memory

---

## Global vs Local Replacement

### Global Replacement

- **Evict** from any process
- **Can** take frame from process A to give to process B
- **Simple** — one pool of frames
- **Risk:** One process can "steal" frames from others

### Local Replacement

- **Each process** has fixed (or proportional) allocation
- **Evict** only from same process
- **Fair** — process's behavior affects only itself
- **May** underutilize — process with unused frames doesn't share

---

## Equal vs Proportional Allocation

- **Equal:** Each process gets same number of frames
- **Proportional:** Allocate proportional to process size (or working set)
- **Proportional** often fairer for mixed workloads

---

## Working Set Model

- **Estimate** working set size per process
- **Allocate** at least working set
- **If** total working sets > memory → reduce multiprogramming

---

## Prepaging

- **Load** pages before they are needed (e.g., adjacent pages, next in file)
- **Reduces** faults if prediction is good
- **Wastes** memory and I/O if wrong

---

## Page-Fault Frequency (PFF)

- **Target** fault rate (e.g., 10 faults/sec)
- **If** process faults too often → give more frames
- **If** too few faults → take frames away
- **Adaptive** allocation

---

## Linux Paging

- **Per-process** page tables
- **Global** frame allocation (with NUMA awareness)
- **Swap** — swap partition or file
- **Page cache** — unified cache for file data and anonymous (swap cache)
- **kswapd** — background daemon to free pages
- **OOM killer** — if cannot free enough, kill process

---

## Windows Paging and Trimming

- **Working set** per process — kernel trims (takes pages) when memory pressure
- **Trimmed** pages stay in process's virtual space but are unmapped
- **Fault** brings them back (from standby list or disk)
- **Standby list** — pages no longer in working set but not yet evicted (soft fault = fast)

---

## Summary

| Concept | Description |
|---------|-------------|
| Thrashing | Too many processes, too few frames |
| Global replacement | Evict from any process |
| Local replacement | Evict only from same process |
| Working set | Pages needed for good performance |
| PFF | Adjust allocation by fault rate |
