# Computer Architecture Course - Complete Overview

This comprehensive computer architecture course is designed for engineers building AI accelerators, embedded systems, and custom hardware. It provides deep technical understanding of how modern CPUs work, from instruction set design through microarchitecture implementation.

---

## Course Structure

The course is divided into **three comprehensive guides** that work together as a complete curriculum:

### 1. **Architecture_Guide.md** — Core Theory & Principles
The foundational material covering:
- **Instruction Set Architecture (ISA)** fundamentals — design philosophy, encoding, instruction formats
- **CPU Design & Microarchitecture** — from single-cycle to pipelined execution
- **Pipelining** — stages, hazards (data, control, structural), forwarding, stalling
- **Superscalar Architecture** — executing multiple instructions per cycle, ILP extraction
- **Memory Hierarchy & Caching** — cache organization, policies, coherence (MESI protocol)
- **Branch Prediction & Speculation** — prediction mechanisms, misprediction recovery
- **Out-of-Order Execution** — instruction windows, reservation stations, renaming
- **Multi-Core & Cache Coherence** — scaling challenges, coherency protocols
- **Real-World ISA Case Studies** — ARM64, x86-64, RISC-V comparison
- **Advanced Topics** — speculative execution security, SIMD, hardware security

**Best for:** Understanding the "why" behind CPU design decisions, theoretical foundations.

### 2. **Labs_and_ProjectsGuide.md** — Hands-On Implementation
Detailed laboratory exercises to validate concepts:
- **Lab 1: Single-Cycle CPU in Verilog** — Build a minimal CPU; understand datapath
- **Lab 2: 5-Stage Pipeline with Forwarding** — Add pipelining; resolve hazards
- **Lab 3: Branch Prediction Simulator** — Implement & test predictors on real benchmarks
- **Lab 4: Cache Performance Analysis** — Measure & optimize cache behavior
- **Lab 5: Multi-Core Coherence Simulator** — Simulate MESI protocol
- **Lab 6: ISA Comparison Project** — Compile code for x86/ARM/RISC-V; analyze
- **Lab 7: CPU Microarchitecture Reverse Engineering** — Deduce real CPU design via benchmarking
- **Capstone: Out-of-Order CPU Simulator** — Full OoO simulation; IPC measurement

**Best for:** Building intuition through building (Verilog, C++, Python simulations).

### 3. **Advanced_Topics_and_CaseStudies.md** — Real-World Details
Deep product analysis and advanced topics:
- **ARM64 Deep Dive** — Registers, instruction encoding, calling convention, ARMv8 vs ARMv9
- **x86-64 Deep Dive** — Variable-length encoding, instruction set, System V ABI
- **RISC-V Deep Dive** — Modular ISA design, extensions, calling convention
- **Case Study 1: Apple M4** — Unified memory, P/E core hybrid, performance specs
- **Case Study 2: AMD Ryzen 9 9950X** — Zen 5 cores, cache hierarchy, multi-core scaling
- **Case Study 3: Qualcomm Snapdragon X** — ARM-based PC, NPU integration, Windows ARM ecosystem
- **Speculative Execution Security** — Spectre/Meltdown, mitigations (LFENCE, IBPB), side-channel attacks
- **Virtual Memory & TLB** — Translation lookaside buffers, TLB misses, large pages
- **Memory Bandwidth Optimization** — Spatial/temporal locality, prefetching, tiling
- **Advanced Tools & Profiling** — Roofline model, Amdahl's law, bottleneck analysis

**Best for:** Connecting theory to commercial products, security concerns, performance optimization.

---

## Recommended Learning Path

### For Complete Beginners (12-16 weeks)

**Weeks 1-2: Fundamentals**
- Read **Architecture_Guide** § 1-2 (ISA, CPU Design)
- Complete **Lab 1** (Single-cycle CPU)

**Weeks 3-4: Pipelining**
- Read **Architecture_Guide** § 3 (Pipelining)
- Complete **Lab 2** (5-stage pipeline with forwarding)

**Weeks 5-6: Branch Prediction**
- Read **Architecture_Guide** § 7 (Branch Prediction)
- Complete **Lab 3** (Branch predictor simulation)

**Weeks 7-8: Caching**
- Read **Architecture_Guide** § 5 (Memory Hierarchy)
- Complete **Lab 4** (Cache performance analysis)

**Weeks 9-10: Multi-core & Advanced Execution**
- Read **Architecture_Guide** § 6, 8 (OoO, multi-core)
- Complete **Lab 5** (Cache coherence simulator)

**Weeks 11-12: Real Hardware**
- Read **Advanced_Topics_and_CaseStudies** (ISA deep dives, case studies)
- Complete **Labs 6-7** (ISA comparison, reverse engineering)

**Weeks 13+: Capstone**
- Complete **Lab 8** (OoO CPU simulator) in parallel with above
- Begin Phase 2 (Embedded Systems)

### For Experienced Programmers (6-8 weeks)

**Week 1: Accelerated Fundamentals**
- Skim **Architecture_Guide** § 1-3 (ISA, pipeline concepts)
- Start **Lab 1** (single-cycle CPU) in parallel

**Weeks 2-3: Deep Dive**
- Read **Architecture_Guide** § 4-8 (branch prediction, caching, OoO, multi-core)
- Complete **Labs 2-4** (pipeline, prediction, cache)

**Weeks 4-5: Real Hardware**
- Read **Advanced_Topics_and_CaseStudies** (ISA dives, case studies, security)
- Complete **Labs 6-7** (ISA analysis, reverse engineering)

**Weeks 6+: Capstone**
- **Lab 8** (OoO simulator) + begin Phase 2

### For Hardware Design Background (4-6 weeks)

**Week 1: ISA & Design Philosophy**
- Read **Architecture_Guide** § 1, 9
- Read **Advanced_Topics_and_CaseStudies** (ISA deep dives)

**Weeks 2-4: Advanced Topics**
- Read **Architecture_Guide** § 4-8 (speculative exec, OoO, coherence)
- Skim through **Labs 2-4** for context
- Focus on **Lab 8** (OoO CPU simulator)

**Weeks 5+: Specialization**
- Use course as reference for Phase 2/3 (Embedded Systems, FPGA design)

---

## Key Concepts Checklist

By the end of this course, you should understand:

### ISA & Hardware Interface
- [ ] Difference between ISA and microarchitecture
- [ ] Why x86 is CISC, ARM is RISC, and what the trade-offs are
- [ ] How instructions are encoded (fixed vs. variable length)
- [ ] Register usage and calling conventions
- [ ] Memory addressing modes

### CPU Pipeline & Hazards
- [ ] How pipelining increases throughput but not latency
- [ ] Three types of hazards: data, control, structural
- [ ] How forwarding resolves most data hazards
- [ ] Cost of branch misprediction
- [ ] Speculative execution and why it's needed

### Performance Metrics
- [ ] IPC (Instructions Per Cycle) and what limits it
- [ ] How to use the Roofline model to identify compute vs. memory bottlenecks
- [ ] Amdahl's Law and parallelization limits
- [ ] Cache hit/miss rates and how to optimize for locality

### Advanced Execution
- [ ] How out-of-order execution masks memory stalls
- [ ] Register renaming and physical vs. architectural registers
- [ ] Instruction window size and its impact on ILP
- [ ] Difference between speculative and architectural state

### Memory Hierarchy
- [ ] Cache organization: direct-mapped, set-associative, fully-associative
- [ ] Cache replacement policies (LRU, random)
- [ ] MESI cache coherence protocol
- [ ] TLB and virtual memory
- [ ] DRAM latency vs. row buffer locality

### Multi-Core Scaling
- [ ] Cache coherence challenges and solutions
- [ ] NUMA latency and proper thread/memory binding
- [ ] Scaling limits (Amdahl's Law, synchronization overhead)
- [ ] Unified memory (Apple Silicon) vs. discrete memory (x86/ARM servers)

### Real Hardware
- [ ] ARM64 ISA and microarchitecture (Cortex-A, Apple Silicon)
- [ ] x86-64 System V ABI and performance characteristics
- [ ] RISC-V modular ISA design
- [ ] How to reverse-engineer CPU microarchitecture via benchmarking

---

## Tools & Resources

### Essential Tools

**Simulation & Design:**
- Verilog/VHDL: ModelSim, Icarus Verilog, Vivado (free for learning)
- CPU Simulators: Gem5 (full-system), SimpleScalar (simpler), Cachegrind (cache analysis)
- Python: NumPy, Matplotlib (for analysis)

**Profiling & Analysis:**
- `perf` (Linux CPU profiling, branch tracing)
- `valgrind --tool=cachegrind` (cache miss analysis)
- `taskset` (CPU affinity for reproducible results)
- PAPI (Performance API for low-level counters)

**Compilers & Cross-Compilation:**
- GCC/Clang with ARM/RISC-V support: `arm-linux-gnueabihf-gcc`, `riscv64-unknown-linux-gnu-gcc`
- LLVM/Clang for IR analysis
- Objdump for disassembly analysis

### Recommended Textbooks

1. **"Computer Architecture: A Quantitative Approach" (Hennessy & Patterson)** — Gold standard; covers everything at depth
2. **"Computer Organization and Design" (Patterson & Hennessy, ARM Edition)** — More accessible version
3. **"Modern Processor Design" (Shen & Lipasti)** — Superscalar & OoO in detail
4. **"Structured Computer Organization" (Tanenbaum)** — Layered approach, good for big picture

### Online References

- **ARM Architecture Reference Manual** (official ISA spec)
- **RISC-V ISA Specification** (open-source ISA)
- **x86-64 System V ABI** (calling conventions)
- **Wikichip / Wikichip Fuse** (community CPU database)
- **AnandTech** (CPU reviews with architectural analysis)
- **Chips & Cheese** (microarchitecture reverse-engineering)
- **ServeTheHome** (server hardware analysis)

---

## Success Criteria

You've mastered this course when you can:

1. **Explain the CPU pipeline** to someone with only digital logic background; describe pipelining tradeoffs
2. **Identify bottlenecks** in code: determine if memory-bound or compute-bound using Roofline model
3. **Write processor simulators** in high-level languages (Python/C++) that execute real code correctly
4. **Reverse-engineer microarchitecture** of real CPUs via benchmarking (cache sizes, branch predictor type, TLB behavior)
5. **Optimize algorithms** for cache locality: design tiled matrix multiply, understand prefetching
6. **Analyze real ISAs** (ARM64, x86-64, RISC-V): read specs, compile code, interpret assembly
7. **Design simple CPUs** in Verilog: single-cycle, pipelined, with forwarding
8. **Understand security implications** of speculative execution: explain Spectre, describe mitigations

---

## Next Steps After Course

Once you complete this course, you're ready for:

### Phase 2: Embedded Systems & Operating Systems
- Understand ARM Cortex-M real-time processors
- Learn embedded OS concepts (FreeRTOS, Zephyr)
- Implement device drivers and hardware abstraction layers
- Build IoT applications on real embedded hardware

### Phase 3: FPGA Design (Xilinx)
- Implement custom CPU cores on FPGAs
- Understand hardware design workflows (Verilog → synthesis → P&R)
- Design AI accelerators (systolic arrays, tensor engines)
- Learn about memory controllers and interconnects

### Phase 4: GPU & AI Accelerator Design
- GPU architecture (NVIDIA, AMD): warps, blocks, memory hierarchy
- Tensor operations and systolic arrays
- Compiler optimizations for accelerators
- ML model optimization on edge hardware

---

## Course Maintenance & Tips

### Study Tips

1. **Theory Before Labs:** Read the guide sections before starting corresponding labs. Theory provides context; labs validate understanding.

2. **Hands-On First for Labs:** Don't read lab solutions; work through the problem. Struggling builds intuition.

3. **Use Benchmarking Tools:** Valgrind, `perf` are powerful. Spend time understanding their output.

4. **Real Hardware > Simulation:** When possible, profile real code on real CPUs (your laptop counts!).

5. **Connect to AI Hardware:** Every concept (pipelining, caching, multi-core) applies directly to GPU design (Phase 4).

### Extending the Course

- **Add newer ISAs:** RISC-V V (vector), ARM SVE, custom architectures (Google Tensor, Apple Neural Engine)
- **Security focus:** Add Spectre/Meltdown labs, side-channel exploits, CET, TrustZone
- **AI accelerators:** Add systolic array design, dataflow optimization, memory hierarchies for AI
- **Formal verification:** Use formal tools to verify correct pipeline behavior

---

## Credits & Attribution

This course synthesizes:
- Classic computer architecture textbooks (Hennessy & Patterson, Tanenbaum)
- Modern processor analysis (WikiChip, Chips & Cheese, AnandTech)
- AI accelerator design principles (from Phase 4 scope)
- Real benchmark data from CES 2026 announcements
- Labs inspired by EECS courses (UC Berkeley, MIT, Stanford)

---

## Contact & Questions

For questions on course content:
- Refer to official ISA specifications (ARM, x86, RISC-V)
- Check Wikichip for microarchitecture details
- Run your own benchmarks to validate theory
- Examine source code of simulators (Gem5, SimpleScalar) for implementation details
