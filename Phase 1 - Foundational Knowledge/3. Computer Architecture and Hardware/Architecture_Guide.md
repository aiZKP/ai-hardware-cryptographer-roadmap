# Computer Architecture: From ISA to Microarchitecture

A comprehensive exploration of how modern processors work — from Instruction Set Architecture (ISA) through microarchitecture, pipelining, caching, and multi-core systems. Designed for engineers building AI accelerators, embedded systems, and custom hardware.

---

## 1. Instruction Set Architecture (ISA) Fundamentals

### Core Concepts

**What is an ISA?**
* The **contract between software and hardware** — defines the instruction format, registers, memory access model, and addressing modes that a CPU implements.
* Hardware can change internally (microarchitecture), but if the ISA remains compatible, software runs unchanged.
* Examples: x86-64, ARM64 (AArch64), RISC-V, MIPS.

**ISA Classification:**

* **RISC (Reduced Instruction Set Computer):**
    * Load-Store architecture: Only `load`/`store` access memory; arithmetic on registers.
    * Simple, uniform instructions (typically 32 bits).
    * Fewer instructions, but easier to optimize and parallelize.
    * Examples: ARM, RISC-V, MIPS, PowerPC.
    * Advantage: Simpler hardware design, better for pipelining.

* **CISC (Complex Instruction Set Computer):**
    * Memory-oriented: Instructions can operate directly on memory.
    * Variable-length instructions (x86: 1-15 bytes).
    * Larger instruction set (100s of instructions).
    * Examples: x86, x86-64, VAX.
    * Advantage: Denser code (fewer instructions), reduces memory footprint. Disadvantage: Complex decoder.

**ISA Components:**

1. **Registers:** Storage locations (fast, on-chip).
   * General-purpose registers (GPRs): For data and addresses.
   * Special-purpose registers: Program counter (PC), stack pointer (SP), status register.
   * Floating-point registers: For FP arithmetic (separate in some ISAs, shared in others).
   * Example: ARM64 has 31 general-purpose 64-bit registers + PC/SP.

2. **Data Types:** Integer (8/16/32/64-bit), floating-point (FP32/FP64), vectors (SIMD).

3. **Instructions:** Arithmetic (ADD, SUB, MUL), logic (AND, OR, XOR), memory (LOAD, STORE), control flow (BRANCH, JUMP), special (NOP, SYSCALL).

4. **Addressing Modes:** How to specify operands.
   * Register: `ADD R1, R2, R3` (add R2 and R3, store in R1).
   * Immediate: `ADD R1, R2, 5` (add 5 to R2, store in R1).
   * Indirect: `LOAD R1, [R2]` (load from memory address in R2).
   * Indexed: `LOAD R1, [R2 + offset]` (base + offset).
   * Auto-increment: `LOAD R1, [R2++]` (load from R2, then increment R2).

5. **Memory Model:**
   * **Byte-addressable:** Each byte has a unique address (x86, ARM, RISC-V).
   * **Endianness:** Byte order (little-endian vs. big-endian). ARM64 and x86 are little-endian.

6. **Instruction Encoding:** How bits map to opcode, registers, immediates.
   * Fixed-length (RISC, e.g., 32 bits for ARM): Simpler decode, easier to parallelize.
   * Variable-length (CISC, e.g., x86): Denser code, complex decode.

**ISA Specification Examples:**

* **ARM64 (ARMv8 & later):** 32-bit fixed-length instructions, 31 GPRs, load-store architecture, unified floating-point ISA.
* **x86-64:** 1-15 byte variable-length instructions, 8/16 GPRs (depending on mode), memory-to-memory operations, separate FP/SIMD registers.
* **RISC-V:** Modular ISA with base RVI (integer) + optional extensions (F=float, D=double, M=multiply, etc.). 32-bit base instruction (16-bit compressed variant).

### Instruction Formats & Encoding

**ARM64 Example Instruction Formats:**

```
Register-to-Register (arithmetic):
31 30 29 28 27 26 25 24 [opcode: 8 bits] [operands: 16 bits]
ADD X0, X1, X2  →  Opcode determines operation; X0=destination, X1/X2=sources.

Load from Memory:
LDR X0, [X1]  →  Load from address in X1, store in X0.

Branch:
B label  →  Branch to address specified by label (26-bit offset).
```

**x86-64 Example:**
```
89 C8           →  MOV EAX, ECX  (copy ECX to EAX)
48 89 C0        →  MOV RAX, RAX  (64-bit version, REX prefix required)
```

### Resources

* **"Computer Organization and Design: ARM Edition" by Patterson & Hennessy:** ISA fundamentals with ARM64 focus.
* **"ARM Architecture Reference Manual (ARMv8 & ARMv9)":** Official ISA specification.
* **"RISC-V ISA Specification":** Official open-source ISA guide.
* **"x86-64 Application Binary Interface (ABI)":** System V ABI for x86-64.

---

## 2. CPU Design & Microarchitecture

### From ISA to Hardware

**Microarchitecture** = How the ISA is **implemented in silicon**.

The same ISA can have multiple microarchitectures with different performance/power/area characteristics:
* **Apple M4:** Efficient, high-performance ARM64 implementation (wide, OoO).
* **Cortex-A53:** Budget ARM64 in-order implementation.
* Both execute the same ARM64 instructions but with vastly different performance.

### Single-Cycle Datapath

**Concept:** Simplest CPU — execute one instruction per cycle.

```
┌─────────────────────────────────────────────────────────────┐
│ Instruction Fetch (IF)                                      │
│   PC += 4; read instruction from memory                     │
├─────────────────────────────────────────────────────────────┤
│ Execute (EX)            Register Fetch (RF) in parallel     │
│   Operates on registers, memory addresses                   │
├─────────────────────────────────────────────────────────────┤
│ Memory (MEM)                                                │
│   Load/Store operations                                     │
├─────────────────────────────────────────────────────────────┤
│ Write-Back (WB)                                             │
│   Store result in register                                  │
└─────────────────────────────────────────────────────────────┘
```

**Problems:**
* **Critical path:** Entire datapath (fetch + execute + memory + write-back) must complete in one clock cycle — limits clock frequency.
* **Resource contention:** If two instructions need the ALU simultaneously, one stalls.

---

## 3. Pipelining

### Concept

**Pipeline** = Break execution into stages; multiple instructions progress in parallel, each at a different stage.

```
Cycle:   1    2    3    4    5    6    7
Instr 1: IF  |RF |EX |MEM|WB
Instr 2:     IF |RF |EX |MEM|WB
Instr 3:         IF |RF |EX |MEM|WB
Instr 4:             IF |RF |EX |MEM|WB
```

**Advantages:**
* **Throughput:** ~1 instruction per cycle (instead of 5 cycles per instruction).
* **Frequency:** Each stage is shorter; clock can run faster.

**Disadvantages:**
* **Pipeline Hazards:** Data, control, and structural hazards cause stalls.
* **Latency:** Still takes 5 cycles to retire one instruction; pipelining improves throughput, not latency.

### Types of Hazards

**1. Data Hazards**

Two instructions depend on each other's data:

```
ADD R1, R2, R3     # WB in cycle 5
SUB R4, R1, R5     # RF in cycle 3 (R1 not yet written!)
```

*Solution:*
* **Forwarding (Bypass):** Route EX result directly to next instr's input.
* **Stalling:** Insert NOP to delay dependent instruction.
* **Out-of-order execution:** Execute independent instructions first.

**2. Control Hazards**

Branch instruction changes PC; next instruction unknown until branch resolves:

```
BEQ R1, Label     # Resolved in EX (cycle 3)
AND R5, R6, R7    # What instruction goes here? Fetched in cycle 2!
```

*Solutions:*
* **Branch Prediction:** Guess next instruction (covered in Section 7).
* **Delay Slots (MIPS, older ISAs):** Instruction after branch always executes.

**3. Structural Hazards**

Two instructions need same hardware resource:
* Single memory port, single ALU, etc.

*Solution:* Duplicate resources or serialize access.

### Pipeline Stalling & Forwarding

**Example: 5-Stage Pipeline (IF → RF → EX → MEM → WB)**

```
Data Dependency Problem:
Cycle 1: LDR X0, [X1]        # Load X0 from memory
Cycle 2:     ADD X2, X0, X3  # Use X0 (not yet in register!)

Without fix: Stall 2 cycles for X0 to be written back.
With forwarding: Route MEM result directly to ADD's input.
```

---

## 4. Superscalar Architecture

### Concept

**Superscalar** = Fetch & execute **multiple instructions per cycle**, not just one.

```
Cycle 1:
  IF: Fetch instr 1, 2, 3 (3-wide fetch)
  EX: Instr 4, 5 can execute in parallel (if independent)
  MEM: Instr 6, 7 can access memory in parallel
  WB: Instr 8, 9, 10 write results
```

**Width:** 2-wide (2 instructions/cycle), 4-wide, 6-wide (Apple M4 is ~6-wide).

**Requirements:**
1. **Multiple instruction fetch/decode units.**
2. **Multiple functional units** (ALUs, FP units, load/store units).
3. **Larger register file** (to avoid contention).
4. **Renaming hardware** to avoid false dependencies.

### Limitations

* **Instruction-level parallelism (ILP) limit:** Code may not have 4+ independent instructions per cycle.
* **Complexity:** Increases exponentially; beyond ~6-wide becomes impractical.
* **Power consumption:** More hardware = more power.

**Real-world widths:**
* Intel Core (x86): 4-6 wide.
* ARM Cortex-A72: 3-wide.
* Apple M4: ~6-wide.

---

## 5. Memory Hierarchy & Caching

### The Memory Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│ Registers (CPU)          │ ~1 cycle access, KB size   │
├─────────────────────────────────────────────────────────┤
│ L1 Cache (on-chip)       │ ~4 cycles, 32-64 KB        │
├─────────────────────────────────────────────────────────┤
│ L2 Cache (on-chip)       │ ~12 cycles, 256 KB - 1 MB  │
├─────────────────────────────────────────────────────────┤
│ L3 Cache (on-chip, shared)│ ~40 cycles, 8-32 MB       │
├─────────────────────────────────────────────────────────┤
│ RAM (off-chip)           │ ~100-200 cycles, GB size   │
├─────────────────────────────────────────────────────────┤
│ Disk/SSD (storage)       │ ms latency, TB size        │
└─────────────────────────────────────────────────────────┘
```

### Cache Organization

**Direct-Mapped Cache:**
```
Address:     [Tag (high bits)] [Index (middle bits)] [Offset (low bits)]
             ↓                 ↓
             Compare Tag       Selects cache line
Memory Address 0x12345678:
  Tag: 0x123, Index: 0x45, Offset: 0x678
  → Cache line at index 0x45 must have tag 0x123 to hit.
```

* **Pros:** Simple, fast.
* **Cons:** Poor temporal locality; repeated accesses to same line might evict on conflict.

**Set-Associative Cache:**
```
Address → [Index] selects SET (N-way set)
         Cache checks all N entries in set for tag match
         (N = 2, 4, 8 for 2-way, 4-way, 8-way associativity)
```

* **Pros:** Better conflict avoidance than direct-mapped.
* **Cons:** Slower (parallel tag lookup) and more power than direct.

**Fully-Associative Cache:**
* Every tag is checked in parallel.
* Ideal but expensive (rarely used except for TLB).

### Cache Policies

**Replacement Policies (what to evict when full):**
* **LRU (Least Recently Used):** Evict line not used longest. Good for temporal locality.
* **FIFO:** Evict oldest line. Simpler hardware.
* **Random:** Pseudo-random eviction. Low overhead.

**Write Policies:**
* **Write-Through:** Write to cache AND memory simultaneously. Safe but slow.
* **Write-Back:** Write to cache only; mark dirty. Flush to memory on eviction. Faster but complex.

### Cache Performance

**Cache Hit Rate:**
* L1: 90-95% typical (tight working set).
* L2: 99%+ typical.
* L3: varies (90-99% depending on application).

**Cache Misses:**
1. **Compulsory (Cold):** First access to line; must load from memory.
2. **Capacity:** Cache too small for working set.
3. **Conflict:** Poor placement (e.g., direct-mapped hash collision).

### Virtual Memory & TLB

**Virtual Memory:**
* Programs use **virtual addresses** (0x00001000 - 4 GB in 32-bit).
* OS maps virtual → physical addresses (CPU + MMU).
* Allows: protection, isolation, larger address space than physical RAM.

**Translation Lookaside Buffer (TLB):**
* Cache for virtual → physical mappings.
* Typical: 4-way, 256-512 entries.
* Large TLB = fewer misses; small TLB = lower cost.

**TLB Miss Penalty:** 50-500 cycles to walk page tables and load TLB.

### Resources

* **"Computer Architecture: A Quantitative Approach" by Hennessy & Patterson:** Cache hierarchy, policies, and design.
* **"What Every Programmer Should Know About Memory" by Ulrich Drepper:** Practical cache optimizations.
* **Cache Simulation Tools (DineroIV, Cacti):** Model cache behaviors.

---

## 6. Out-of-Order (OoO) Execution

### Motivation

**In-order execution bottleneck:**
```
ADD R1, R2, R3     # 5 cycles (MEM latency)
SUB R4, R5, R6     # Stalls waiting for R1; independent!
```

**OoO solution:** Execute SUB before waiting for ADD:
```
Fetch:   ADD, SUB (in order)
Rename:  Assign temporary registers (remove false dependencies)
Execute: SUB first (if hardware available)
Retire:  In original order
```

### Key Components

**1. Instruction Window (Re-Order Buffer / Reservation Station)**
* Decouples fetch/decode (in-order) from execution (out-of-order).
* Holds 32-256 instructions waiting to execute.
* Larger window = more parallelism potential, but more complexity.

**2. Register Renaming**
```
Original code:
  ADD R1, R2, R3
  SUB R1, R4, R5  (depends on ADD's R1)

Renamed (internal):
  ADD P5, P2, P3  (use physical reg P5)
  SUB P6, P4, P5  (P5 = result of ADD)

No false dependency: P6 and P5 are different registers.
```

**3. Reservation Station (RS)**
* Holds instructions waiting for operands.
* When operands ready → send to execution unit.
* Multiple RSs (ALU RS, FP RS, Load/Store RS) for parallelism.

**4. Common Data Bus (CDB)**
* Broadcasts execution results to all RSs.
* Implements forwarding at scale.

### Execution Model

```
Instruction sequence (in program order):
  1. ADD R1, R2, R3
  2. MUL R4, R1, R5   (depends on ADD)
  3. SUB R6, R7, R8   (independent)
  4. DIV R9, R4, R10  (depends on MUL)

Out-of-order timeline:
  Cycle 1: Fetch & rename all 4.
  Cycle 2: SUB executes (no dependencies).
  Cycle 3: ADD executes (data ready).
  Cycle 4: MUL executes (R1 now available via CDB).
  Cycle 5: DIV executes (R4 now available).

Retire (in-order): All complete by cycle 5.
```

### Commit/Retire

* Results are computed OoO but **committed to architectural state in-order**.
* Handles exceptions/mispredictions: If instr causes fault, later instructions must be discarded.
* **Precise exceptions:** Architectural state reflects all instructions up to faulting instruction.

### OoO Limitations

* **Complexity:** Renaming logic, speculation, many data structures.
* **Power consumption:** Reservation stations, CDB, large register file.
* **Width limits:** Beyond 6 instructions/cycle, dispatch becomes bottleneck.
* **Memory ordering:** Must track loads/stores to ensure correct order.

---

## 7. Branch Prediction & Speculation

### The Branch Problem (Revisited)

```
Cycle 1: Fetch BEQ R1, Label
  Don't know next PC yet; can't fetch instr 2!

Cycle 4: Branch resolves in EX stage.
  Lost 3 cycles of pipeline fill!

With 4-wide superscalar: Lost ~12 instruction slots.
```

### Prediction Mechanisms

**1. Two-Level Branch Predictor (Correlating)**

```
Pattern History Table (PHT):
  n-bit history of recent branches → lookup table of predictions.

Example: 2-bit history + 2-bit saturation counter
  History = "T, T" (last 2 branches taken)
  → Look up counter for (history="11")
  → Predict "taken" or "not taken" based on counter value.

State diagram (2-bit saturating):
  Strongly Not Taken (00) → (01) → (10) → (11) Strongly Taken
  Transitions based on actual outcome.
```

**2. Gshare (Global-History + XOR)**
```
Global history (last N branches) XORed with PC bits
  → Index into PHT.

Captures program + global dynamics better than local-only.
```

**3. Branch Target Buffer (BTB)**
```
Stores (branch_address → target_address) pairs.

Fast lookup: BEQ at 0x1000 target = 0x2000
  → Predict target immediately without computing.
```

**4. Return Stack Buffer (RSB)**
```
Stack of return addresses.

CALL pushes return address; RET pops from stack.

Perfectly predicts most function returns (nested ~16 levels typical).
```

### Speculation & Recovery

**Speculative Execution:**
```
Predict branch not taken; fetch instr A.
  A is speculatively executed.

If prediction wrong: Discard A, fetch correct instr B.
  (Uses OoO pipeline to discard; physical registers freed).
```

**Misprediction Cost:**
* **Pipeline flush:** 10-20+ cycles latency penalty for deep pipelines.
* **Example:** M4 misprediction = ~8-12 cycle penalty.

**Branch Prediction Rates:**
* **Conditional branches (~80% of all branches):** Modern predictors: 94-98% accuracy.
* **Indirect branches (~20%):** Harder; BTB-based guessing: 85-95%.

### Hardware Details

**Apple M4 Branch Prediction (Example):**
* ~8,000-entry BTB.
* 2-level predictor with global history.
* RSB for return prediction.
* Prediction latency: 1 cycle (integrated into fetch).

### Limits & Challenges

* **Misprediction power:** Wrong path wasted energy.
* **Unpredictable branches:** Hash functions, indirect calls → high misprediction rate.
* **Hardware cost:** BTB, pattern history tables consume area/power.

---

## 8. Multi-Core & Cache Coherence

### Multi-Core Architecture

```
┌──────────────┐  ┌──────────────┐
│ Core 0       │  │ Core 1       │
│ L1I / L1D    │  │ L1I / L1D    │
│ L2           │  │ L2           │
└──────────────┘  └──────────────┘
         ↓              ↓
    ┌─────────────────────────┐
    │ L3 Cache (Shared)       │
    └─────────────────────────┘
         ↓
    Memory Controller & Main RAM
```

**Multi-Core Benefits:**
* 2 cores → 2x work (for independent tasks).
* Shared L3 cache → reduced bandwidth to main memory.
* Effective TLP (Thread-Level Parallelism).

**Challenges:**
* **Cache Coherence:** If Core 0 writes to address X, Core 1 must see updated value (not stale cache copy).
* **Synchronization:** Atomic operations, locks, barriers.

### Cache Coherence Protocols

**MESI Protocol (MSI + Exclusive state):**
```
States per cache line:
  M (Modified): This core wrote it (dirty).
  E (Exclusive): Loaded; unmodified; no other core has it.
  S (Shared): Multiple cores have valid copy.
  I (Invalid): Stale/not loaded.

Transitions:
  Read miss: M/E → S (if other core has it); else E (if only local).
  Write: Any state → M (invalidate other cores' copies).
```

**MOESI Protocol (MESI + Owned state):**
* **O (Owned):** Modified by this core, but others can read (reduce writes).
* Used in AMD, some ARM systems.

**Example: Two cores, shared variable X**
```
Core 0: X = 5 (writes)
  → X's line: M in Core 0; I in Core 1.

Core 1: Read X
  → Coherence request; Core 0 sends data.
  → X's line: S in both cores.

Core 0: X = 10 (write again)
  → Invalidate Core 1's copy.
  → X's line: M in Core 0; I in Core 1.
```

**Coherence Cost:**
* **Snooping-based (bus):** All cores watch all writes (broadcast). Works for 2-8 cores.
* **Directory-based (point-to-point):** Central directory tracks line locations. For 16+ cores.

---

## 9. Real-World ISA & Architecture Case Studies

### ARM64 (ARMv8 & ARMv9)

**ISA Characteristics:**
* Load-store RISC architecture.
* 31 general-purpose 64-bit registers, CC register, PC.
* Fixed 32-bit instructions (also 16-bit Thumb-2 compressed).
* Unified floating-point (FP32/FP64 + SIMD in vector registers).

**Example Instructions:**
```
ADD X0, X1, X2        # X0 = X1 + X2
LDR X0, [X1]          # Load from address in X1
STR X0, [X1]          # Store to address in X1
B my_label            # Unconditional branch
BEQ my_label          # Branch if equal (CC flag)
MOVI V0.4S, #1        # SIMD: Set 4x 32-bit values to 1
```

**Microarchitecture Examples:**
* **Cortex-A53:** In-order, 2-wide, low-power (embedded).
* **Cortex-A72:** Out-of-order, 3-wide, mid-range power/perf.
* **Cortex-A77:** OoO, 4-wide, high-performance.
* **Apple M4:** OoO, 6-wide, highest ILP extraction (P-cores); 2-4 wide efficiency cores (E-cores).

**Advantages:**
* Clean ISA, load-store simplicity.
* Excellent for energy efficiency (mobile/embedded).

**Disadvantages:**
* Denser code than x86 requires slightly more memory.

### x86-64

**ISA Characteristics:**
* CISC architecture (evolved from x86 and x86-32).
* Variable-length instructions (1-15 bytes); complex decoder.
* 16 general-purpose 64-bit registers (some restricted).
* Separate floating-point (FP stack) + SIMD (XMM/YMM/ZMM) registers.

**Example Instructions:**
```
mov rax, [rbx]         # Load from memory (x86 mem-to-reg)
add rax, rbx           # Register-to-register
add rax, 5             # Immediate operand
mov [rax], rbx         # Store to memory (base + offset)
```

**Microarchitecture Examples:**
* **Intel Core Ultra (Lunar Lake, 2024):** P-cores (Lion Cove, 4-wide OoO), E-cores (Skymont).
* **AMD Ryzen 7000 (Zen 5, 2024):** 4-wide OoO, strong ILP extraction.

**Advantages:**
* Backward compatibility (40+ years of x86 code still runs).
* Memory-to-register instructions; denser code.
* Dominant ecosystem.

**Disadvantages:**
* Complex decoder (high power, area cost).
* Variable-length instructions complicate fetching.

### RISC-V

**ISA Characteristics:**
* Modular, open-source RISC ISA.
* Base 32-bit instructions; RV64 for 64-bit.
* Extensions: M (multiply), F (float), D (double), A (atomic), V (vector), etc.

**Example Instructions (RV64I base):**
```
add x1, x2, x3         # x1 = x2 + x3
ld x1, 8(x2)           # Load doubleword from [x2 + 8]
beq x1, x2, label      # Branch if equal
```

**Advantages:**
* Radical simplicity: ~50 core instructions.
* Modular extensions (pick what you need).
* Open-source; no licensing fees.

**Disadvantages:**
* Young ecosystem (software still maturing).
* No dominant commercial implementation yet (SiFive, Ventana, others emerging).

### Comparison

| Feature | ARM64 | x86-64 | RISC-V |
|---------|-------|--------|--------|
| **ISA Type** | RISC | CISC | RISC |
| **Instruction Length** | 32-bit fixed | 1-15 bytes | 32-bit (RV64) |
| **Primary Market** | Mobile, embedded, servers | Desktops, servers | Emerging, research |
| **Software Ecosystem** | Mature | Most mature | Growing |
| **Encoding Density** | Medium | High | Medium |
| **Decoder Complexity** | Simple | Complex | Very simple |
| **Energy Efficiency** | Excellent | Good | Excellent |

---

## 10. Advanced Topics

### Speculative Execution & Security

**Spectre/Meltdown Context:**
* CPUs fetch & execute speculatively before branch resolves.
* Attacker code runs speculatively; can leak cache state (timing side-channel).
* Modern mitigations: LFENCE, RSB stuffing, TLB isolation, CET (Control-Flow Enforcement).

### SIMD & Vector Execution

**Single Instruction, Multiple Data:**
```
Traditional: ADD V[4] array elements → 4 cycles
SIMD:        VADD (4x FP32) in parallel → 1 cycle
```

**Modern vector ISAs:**
* ARM NEON: 128-bit vectors (2x FP64 or 4x FP32).
* x86 AVX-512: 512-bit vectors (16x FP32).
* RISC-V V extension: Scalable to 1024+ bits.

### Hardware Security

* **Control-flow integrity:** CET (x86) / CoreSight (ARM).
* **Authenticated encryption:** AES-NI (x86), ARMv8.3 Crypto.
* **Trusted execution:** SGX (x86), TrustZone (ARM).

---

## Learning Resources

### Textbooks

1. **"Computer Architecture: A Quantitative Approach" by Hennessy & Patterson** (Essential)
   * Gold standard; covers pipelining, caching, superscalar, OoO in depth.
   * ~1000 pages but skip chapters as needed.

2. **"Computer Organization and Design" by Patterson & Hennessy** (Alternative)
   * ARM64 edition available; more accessible than above.

3. **"Modern Processor Design" by Shen & Lipasti**
   * Superscalar & OoO design in detail; excellent figures.

4. **"Structured Computer Organization" by Tanenbaum**
   * Layered approach; good for big-picture understanding.

### Online References

* **ARM Architecture Reference Manual (ARMv8/ARMv9):** Official ISA spec.
* **RISC-V ISA Specification:** Open-source; modular.
* **x86-64 System V ABI:** Calling conventions, linking.
* **Wikichip / Chips & Cheese:** Community-sourced microarchitecture analysis.
* **AnandTech Deep Dives:** CPU reviews with architectural analysis.

### Simulation & Tools

* **SimpleSimulator / SimpleScalar:** Cycle-accurate CPU simulation.
* **Gem5:** Full-system CPU simulator (complex but flexible).
* **DineroIV:** Cache simulation.
* **CacTi:** Cache area/power estimation.
* **LLVM / GCC:** Compiler toolchains; build code for ARM/x86/RISC-V.

---

## Projects & Labs

### Lab 1: Single-Cycle CPU in Verilog

Design a minimal RISC CPU that executes one instruction per cycle.
* Support: LDR, STR, ADD, SUB, BEQ.
* Components: Fetch unit, decoder, ALU, register file, memory.
* Simulate with Icarus Verilog or ModelSim.
* **Deliverable:** Verilog code + test benches; cycle trace showing instruction execution.

### Lab 2: 5-Stage Pipeline Implementation

Extend Lab 1 with pipelining:
* Break into IF→RF→EX→MEM→WB stages.
* Implement forwarding to resolve data hazards.
* Add NOPs for unresolvable hazards (or stalling logic).
* Measure: CPI (cycles per instruction), throughput.
* **Deliverable:** Pipeline simulator + benchmark showing ~1 CPI for independent instructions.

### Lab 3: Branch Prediction Simulator

Build a cycle-accurate predictor simulator:
* Implement 2-level correlating predictor.
* Add BTB (Branch Target Buffer).
* Simulate on real branch traces (SPEC benchmarks).
* Measure: Hit rate, misprediction penalty, speedup vs. stalling.
* **Deliverable:** Prediction accuracy report; analysis of branch patterns.

### Lab 4: Cache Performance Analysis

Measure cache behavior on real code:
* Use Cachegrind / PAPI (Performance API).
* Run benchmarks (matrix multiply, merge sort, hash table).
* Analyze: Hit rate, misses, working set size.
* Simulate different cache configs (size, associativity, line size).
* **Deliverable:** Cache miss charts; optimization suggestions.

### Lab 5: Multi-Core Coherence Simulator

Simulate cache coherence in 2-4 core system:
* Implement MESI protocol.
* Simulate memory access patterns from multiple threads.
* Measure: Coherence miss rate, cache-to-cache transfers.
* Compare MESI vs. snooping vs. directory protocol.
* **Deliverable:** Coherence miss analysis; protocol trade-offs.

### Lab 6: ISA Comparison Project

Compare x86-64 vs. ARM64 vs. RISC-V:
* Compile same C/C++ code for all three ISAs.
* Measure: Code size, cycle count, memory bandwidth.
* Analyze: Instruction selection, register usage, memory reference patterns.
* **Deliverable:** Comparative benchmark report; architecture lessons.

### Lab 7: CPU Microarchitecture Study

Reverse-engineer a real CPU (e.g., M4 Pro):
* Use tools: Geekbench, SpecCPU, custom microbenchmarks.
* Deduce: Cache hierarchy, line size, prefetching, branch predictor.
* Measure: IPC, branch misprediction rate, cache miss rate.
* **Deliverable:** Microarchitecture model + analysis; validation against spec.

### Capstone: Simple Out-of-Order CPU Simulator

Build an OoO CPU simulator (C++/Python):
* Instruction fetch, decode, rename, dispatch to reservation stations.
* Out-of-order execute; in-order commit.
* Support data hazard resolution, branch speculation, memory ordering.
* Simulate on benchmark suite.
* **Deliverable:** Functional simulator; IPC comparison vs. in-order.

---

## Next Steps

**After mastering this material:**
1. **Proceed to Phase 2 (Embedded Systems):** Real ARM/RISC-V processors, RTOSes, device drivers.
2. **Specialize in FPGA design (Phase 3):** Implement custom CPUs on Xilinx.
3. **Move to AI accelerators (roadmap Phase 4 Track B — Jetson/GPU, and optionally Track A — FPGA):** GPU architectures, tensor operations, memory optimization.

**Recommended progression:**
* Weeks 1-2: ISA fundamentals, simple CPU design (Lab 1).
* Weeks 3-4: Pipelining, hazards, forwarding (Lab 2).
* Weeks 5-6: Branch prediction, advanced execution (Lab 3).
* Weeks 7-8: Caching, memory hierarchy (Lab 4).
* Weeks 9-10: Multi-core, coherence (Lab 5).
* Weeks 11-12: ISA comparison, architecture analysis (Lab 6-7).
* ongoing: Capstone OoO simulator (parallel).
