# Lecture 19: Pintos Virtual Memory — Design Discussion

**Source:** [CS124 Lec19](https://users.cms.caltech.edu/~donnie/cs124/lectures/CS124Lec19.pdf)

---

## Pintos Project 4: Virtual Memory

**Goal:** Extend Pintos to support demand paging, memory-mapped files, and copy-on-write.

---

## Design Considerations

### Supplemental Page Table

- **Kernel** does not pre-allocate all page table entries
- **Supplemental structure** (e.g., hash table) maps virtual page → metadata
- **Metadata:** Type (anon, file), file/offset for file-backed, swap slot if evicted
- **On page fault:** Look up supplemental table; load from file/swap or allocate zero page

### Frame Table

- **Global** or per-process
- **Tracks** which frame holds which (process, page)
- **Eviction** when no free frame — pick victim (e.g., clock), write back if dirty, update PTE

### Swap Table

- **Swap slot** — unit of swap space (e.g., one page per slot)
- **Bitmap** or similar to track free slots
- **On evict:** Write page to swap, record slot in supplemental table
- **On fault:** Read from swap, free slot

### Memory-Mapped Files

- **mmap()** — map file into address space
- **Supplemental** entry: file, offset, length
- **Fault:** Read from file (lazy)
- **Evict:** Write dirty pages back to file (not swap)
- **munmap()** — unmap, flush dirty pages

### Copy-on-Write fork()

- **fork()** — share pages with parent, mark read-only
- **Supplemental** — mark as COW
- **Fault on write:** Copy page, update mapping to writable, resume
- **Reference count** if shared by multiple children

---

## Synchronization

- **Frame table** — global lock or per-frame
- **Supplemental** — per-process or fine-grained
- **Swap** — lock for allocation/deallocation

---

## Summary

Key components: **supplemental page table**, **frame table**, **swap**, **mmap**, **COW fork**.
