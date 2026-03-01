# Lecture 10: Advanced Synchronization — RCU, Lock Granularity

**Source:** [CS124 Lec10](https://users.cms.caltech.edu/~donnie/cs124/lectures/CS124Lec10.pdf)

---

## Synchronization on Shared Data Structures

**Challenge:** Multiple threads read/write shared structures (e.g., linked list, hash table). Need correctness and performance.

---

## Lock-Based Approaches

### Lock Contention

- **Many threads** competing for same lock → serialization
- **Reduces** parallelism

### Lock Overhead

- **Acquire/release** has cost (atomic operations, cache coherency)
- **Too many locks** → overhead dominates

### Lock Granularity

- **Coarse:** One lock for entire structure — simple, but high contention
- **Fine-grained:** Lock per node or per bucket — less contention, more complex, risk of deadlock

### Mutexes vs Read/Write Locks

- **Mutex:** Exclusive — only one reader or writer
- **Read/write lock (rwlock):** Multiple readers OR one writer
- **Use when:** Reads dominate, writes rare — readers can proceed concurrently

### Crabbing (B-tree)

- **Lock parent** before descending to child
- **Lock child**, then **unlock parent** — "crabbing" down tree
- **Reduces** lock scope while traversing

---

## Read-Copy-Update (RCU)

**Idea:** Readers proceed without locks. Writers create new version; old version reclaimed after all readers are done.

### Publish/Subscribe

- **Writer** creates new node/structure
- **Publish:** Atomic pointer update (e.g., `rcu_assign_pointer`)
- **Readers** see either old or new — both valid

### Copy to Insert/Update/Delete

- **Insert:** Allocate new node, link it, update parent pointer
- **Update:** Copy node, modify copy, swap pointer
- **Delete:** Unlink, but don't free immediately — wait for grace period

### Replacement vs Reclamation

- **Replacement:** Atomic pointer swap — readers see new data
- **Reclamation:** Free old memory — must wait until no reader holds reference

### Read-Side Critical Sections

- **Reader** enters critical section (e.g., `rcu_read_lock()`)
- **Cannot block** in critical section
- **Grace period:** Time after which all pre-existing readers have exited

### Classic RCU

- **Grace period** = all CPUs have passed a quiescent state (e.g., context switch, idle)
- **Used in** Linux kernel for many structures (e.g., routing table, task list)

### Sleepable RCU (SRCU)

- **Allows** blocking in read-side critical section
- **Different** grace period rules
- **Used when** readers must sleep

### Preemptible RCU (PREEMPT_RCU)

- **Preemptible kernel** — reader can be preempted
- **Grace period** accounts for preemption

---

## Summary

| Technique | Best For |
|-----------|----------|
| Mutex | Simple exclusive access |
| Rwlock | Read-heavy, few writes |
| Fine-grained locks | Reduce contention, more complexity |
| RCU | Read-mostly, rare updates, no reader locks |
