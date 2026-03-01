# Lecture 23: Pintos File System — Design Discussion

**Source:** [CS124 Lec23](https://users.cms.caltech.edu/~donnie/cs124/lectures/CS124Lec23.pdf)

---

## Pintos Project 5: File System

**Goal:** Implement a proper filesystem on top of Pintos's simple block device. Support subdirectories, growth, and synchronization.

---

## Design Considerations

### Buffer Cache

- **Cache** recently used disk blocks in memory
- **Reduce** disk I/O — read/write to cache, flush to disk
- **Eviction** — LRU or clock when cache full
- **Write-through** vs **write-back** — write-back better performance, need to handle crash

### Extensible Files

- **File** can grow — allocate more blocks as needed
- **Inode** or similar — direct + indirect blocks
- **Free map** — track free blocks (bitmap or similar)

### Subdirectories

- **Directory** is a file with special format — (name, inode#) entries
- **Path parsing** — walk components, open each directory
- **`.`** and `..` — current and parent
- **mkdir, rmdir** — create/remove directory entries

### Synchronization

- **Multiple processes** can access filesystem
- **Per-file** or **per-inode** lock for metadata
- **Buffer cache** — lock per block or per cache entry
- **Avoid** deadlock — lock ordering (e.g., parent before child)

### Persistence

- **Sync** — flush dirty blocks to disk
- **On** fsync, sync, or periodic
- **Crash** — without journaling, may need fsck (not in basic Pintos)

---

## Summary

Key components: **buffer cache**, **extensible files** (block allocation), **subdirectories**, **synchronization**.
