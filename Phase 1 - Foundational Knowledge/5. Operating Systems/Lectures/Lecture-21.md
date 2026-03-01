# Lecture 21: Filesystems — Structure and Allocation

**Source:** [CS124 Lec21](https://users.cms.caltech.edu/~donnie/cs124/lectures/CS124Lec21.pdf)

---

## Persistent Storage: Files and Filesystem

- **File:** Named, persistent byte sequence
- **Filesystem:** Organizes files on disk (or other block device)
- **Block device:** Storage addressed in fixed-size blocks (e.g., 512 B, 4 KB)

---

## File Names and Extensions

- **Name** — human-readable identifier (e.g., `report.pdf`)
- **Extension** — convention (e.g., `.pdf`) — often not enforced by OS
- **Metadata** — size, timestamps, permissions — stored separately (e.g., inode)

---

## Directories (Folders)

- **Directory:** Special file containing list of (name, inode#) or (name, metadata)
- **Hierarchical** — directory can contain subdirectories
- **Root directory** `/` — top of tree

### Directory Structures

- **Single-level:** All files in one directory — doesn't scale
- **Two-level:** Per-user directory — still limited
- **General graph:** Directories can contain subdirs — tree or DAG (with links)

---

## Paths

- **Absolute:** From root (e.g., `/home/user/file.txt`)
- **Relative:** From current directory (e.g., `./file.txt`, `../other/file.txt`)
- **Current directory** — per-process (e.g., `cwd`)

---

## Hard Links and Symbolic Links

### Hard Link

- **Multiple directory entries** point to same inode
- **Same file** — same data, delete when link count = 0
- **Cannot** cross filesystems (inode# is per-fs)
- **Cannot** link to directory (usually — avoid cycles)

### Symbolic Link (Symlink)

- **Special file** containing path to target
- **Can** cross filesystems, point to directories
- **Dangling** if target deleted
- **Followed** by kernel on open

---

## File Access Patterns

- **Sequential:** Read/write in order (e.g., 0, 1, 2, ...)
- **Direct (random):** Seek to offset, then read/write
- **Indexed:** Key → record (database-style)

---

## File Layout — Allocation Methods

### Contiguous Allocation

- **File** occupies consecutive blocks on disk
- **Pros:** Simple, fast sequential access
- **Cons:** External fragmentation, need to know size at creation
- **Compaction** — move files to consolidate free space (expensive)

### Extents

- **Extent** = (start block, length)
- **File** can have multiple extents
- **Reduces** fragmentation vs pure contiguous; more flexible

### Linked Allocation

- **Each block** has pointer to next
- **No** random access (must traverse)
- **FAT** — File Allocation Table: array of "next block" per block; faster traversal

### Indexed Allocation

- **Index block** for each file — list of block numbers
- **Random access** — lookup in index
- **Large files** — need multi-level index (linked index blocks, or tree)

---

## Index Structures

- **Linked sequence:** Index blocks chained
- **Multilevel:** Tree of index blocks (e.g., 2-level, 3-level)
- **Hybrid:** Direct blocks + indirect blocks (like Unix inodes)

---

## Ext2 Inodes

- **Inode** — metadata + block pointers
- **Direct blocks:** First N (e.g., 12) block numbers
- **Single indirect:** Block containing more block numbers
- **Double indirect:** Block of single indirect blocks
- **Triple indirect:** One more level
- **Supports** very large files with limited inode size

---

## Summary

| Allocation | Pros | Cons |
|------------|------|------|
| Contiguous | Simple, fast sequential | Fragmentation |
| Linked | No fragmentation | No random access |
| Indexed | Random access | Overhead |
| Extents | Balance | More complex |
