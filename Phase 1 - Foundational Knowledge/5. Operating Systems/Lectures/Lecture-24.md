# Lecture 24: Journaling Filesystems

**Source:** [CS124 Lec24](https://users.cms.caltech.edu/~donnie/cs124/lectures/CS124Lec24.pdf)

---

## The Crash Consistency Problem

**Scenario:** Updating filesystem requires multiple disk writes (e.g., create file: allocate inode, update directory, update free block bitmap). **Crash** in the middle → inconsistent state (e.g., inode allocated but not in directory, or vice versa).

**Without protection:** fsck (filesystem check) on reboot — slow, may lose data.

---

## Journaling (Write-Ahead Logging)

**Idea:** Log intended changes *before* applying them to the real structures. On crash, replay log or discard incomplete transaction.

### Log Structure

- **Journal** — circular log on disk (or dedicated area)
- **Transaction** — set of block updates (metadata, maybe data)
- **Write** to journal first (sequential, fast)
- **Then** apply to real locations
- **Commit** — mark transaction complete in journal

### Recovery

- **Replay** committed transactions that weren't applied
- **Discard** incomplete transactions
- **Fast** — no full fsck

---

## Metadata vs Data Journaling

### Metadata Journaling (Ordered Mode in ext3)

- **Log** only metadata (inodes, directory blocks, bitmap)
- **Data** written to final location first, then metadata logged and applied
- **Crash** — metadata may be inconsistent; data blocks may be orphaned
- **Simpler** — less logged
- **ext3** default — ordered mode

### Full Data Journaling

- **Log** data and metadata
- **Crash** — full recovery
- **Overhead** — data written twice (log + final)
- **Slower** for data-heavy workloads

---

## Log Structure

- **Circular** — wrap around when full
- **Checkpoint** — mark applied transactions as free in log
- **Journal superblock** — current head, sequence numbers

---

## Other Approaches

### Copy-on-Write (e.g., Btrfs, ZFS)

- **Never overwrite** in place — write new copy, update pointer
- **Crash** — old consistent state still valid
- **No** journal; different trade-offs

### Log-Structured Filesystem (LFS)

- **All writes** go to log (append-only)
- **Good** for SSDs (sequential writes)
- **Cleaner** — reclaim space from old log

---

## Summary

| Approach | Crash Recovery | Overhead |
|----------|----------------|----------|
| No protection | fsck | None |
| Metadata journal | Fast, metadata consistent | Moderate |
| Full journal | Full consistency | High (data 2x) |
| Copy-on-Write | Consistent | Different (fragmentation) |
