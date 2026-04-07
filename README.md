<div align="center" markdown="1">

# AI Hardware Engineer Roadmap

**From firmware & AI applications to ML compilers — and ultimately, custom silicon.**

![AI Hardware Engineer Roadmap](Assets/images/ai-hardware-engineer.png)

</div>

A custom AI chip is an **8-layer vertical stack**. This roadmap builds vertical literacy across all 8 layers — from AI applications and ML compilers down to RTL design and silicon fabrication.

---

## The 8-Layer AI Chip Stack

| Layer | What it does | Key technologies |
|:-----:|-------------|-----------------|
| **L1** | **AI Application & Framework** | PyTorch, ONNX, Agentic AI, MLOps, quantization |
| **L2** | **Compiler & Graph Optimization** | MLIR dialects, TVM, LLVM, custom NPU lowering |
| **L3** | **Runtime & Driver** | C++ runtime, Linux kernel driver, CUDA-like API |
| **L4** | **Firmware & OS** | FreeRTOS, bootloader, embedded Linux, RTOS |
| **L5** | **Hardware Architecture** | Systolic arrays, HBM controllers, NoC, power domains |
| **L6** | **RTL & Logic Design** | SystemVerilog, UVM verification, FPGA prototyping |
| **L7** | **Physical Implementation** | EDA tools, place & route, timing closure |
| **L8** | **Fabrication & Packaging** | Foundry process, CoWoS, post-silicon bring-up |

<small>**L1–L6:** Hands-on projects throughout the curriculum &nbsp;|&nbsp; **L7–L8:** Theory and guided labs (OpenROAD, TinyTapeout)</small>

---

## How Phases Map to Layers

```
               L1          L2          L3          L4          L5          L6          L7          L8
            Application  Compiler   Runtime &   Firmware    Hardware      RTL &      Physical     Fab &
            & Framework  & Graph     Driver      & OS      Architecture   Logic      Implement.  Packaging
            ───────────┬───────────┬───────────┬───────────┬───────────┬───────────┬───────────┬───────────
Phase 1        ░░░     │           │    ░░░    │    ░░░    │    ███    │    ███    │           │
Phase 2                │           │    ░░░    │    ███    │           │           │           │     ░
Phase 3        ███     │           │           │           │           │           │           │
Phase 4A               │           │    ███    │           │    ██░    │    ███    │           │
Phase 4B               │           │    ███    │    ██░    │    ░░░    │           │           │     ░
Phase 4C       ░░░     │    ███    │    ░░░    │           │           │           │           │
Phase 5        ░░░     │    ░░░    │    ░░░    │           │    ███    │    ██░    │    ██░    │    ░░░
            ───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────

███ = primary coverage    ██░ = strong supporting    ░░░ = background    (blank) = minimal
```

---

## 5-Phase Curriculum

### Phase 1: Digital Foundations (6–12 months)

<small>*The language of hardware — from gates and Verilog to CUDA kernels.*</small>

| Topic | Key Skills | Layer |
|-------|------------|:-----:|
| [**Digital Design and HDL**](Phase%201%20-%20Foundational%20Knowledge/1.%20Digital%20Design%20and%20Hardware%20Description%20Languages/Guide.md) | Number systems, logic, memory; Verilog, testbenches, synthesis | L6 |
| [**Computer Architecture**](Phase%201%20-%20Foundational%20Knowledge/2.%20Computer%20Architecture%20and%20Hardware/Guide.md) | ISA, pipelines, caches, OoO, coherence; modern CPUs/GPUs/memory | L5 |
| [**Operating Systems**](Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Guide.md) | Processes, threads, scheduling, memory management, drivers | L3/L4 |
| [**C++ and Parallel Computing**](Phase%201%20-%20Foundational%20Knowledge/4.%20C%2B%2B%20and%20Parallel%20Computing/Guide.md) | C++ & SIMD · OpenMP & oneTBB · CUDA & SIMT · ROCm & HIP · OpenCL & SYCL | L1/L3 |

### Phase 2: Embedded Systems (6–12 months)

<small>*The boards and buses that sit next to inference — MCUs, RTOS, and embedded Linux.*</small>

| Topic | Key Skills | Layer |
|-------|------------|:-----:|
| [**Embedded Software**](Phase%202%20-%20Embedded%20Systems/2.%20Embedded%20Software/Guide.md) | ARM Cortex-M, FreeRTOS, SPI/UART/I2C/CAN, power, OTA | L4 |
| [**Embedded Linux**](Phase%202%20-%20Embedded%20Systems/3.%20Embedded%20Linux/Guide.md) | Yocto, PetaLinux, kernel, rootfs | L3/L4 |

### Phase 3: Artificial Intelligence (6–12 months)

<small>*The workloads your hardware must run.* Core + two tracks. &nbsp;·&nbsp; *Hub:* [**Phase 3 — Artificial Intelligence**](Phase%203%20-%20Artificial%20Intelligence/Guide.md)</small>

**Core (mandatory):**

| # | Topic | Layer |
|---|-------|:-----:|
| 1 | [**Neural Networks**](Phase%203%20-%20Artificial%20Intelligence/1.%20Neural%20Networks/Guide.md) — MLPs, CNNs, training, backpropagation | L1 |
| 2 | [**Deep Learning Frameworks**](Phase%203%20-%20Artificial%20Intelligence/2.%20Deep%20Learning%20Frameworks/Guide.md) — [micrograd](Phase%203%20-%20Artificial%20Intelligence/2.%20Deep%20Learning%20Frameworks/micrograd/Guide.md) → [PyTorch](Phase%203%20-%20Artificial%20Intelligence/2.%20Deep%20Learning%20Frameworks/PyTorch/Guide.md) → [tinygrad](Phase%203%20-%20Artificial%20Intelligence/2.%20Deep%20Learning%20Frameworks/tinygrad/Guide.md) | L1/L2 |

**Track A — Hardware & Edge AI** (→ Phase 4):

| # | Topic | Layer |
|---|-------|:-----:|
| 3 | [**Computer Vision**](Phase%203%20-%20Artificial%20Intelligence/Track%20A%20-%20Hardware%20and%20Edge%20AI/3.%20Computer%20Vision/Guide.md) — detection, segmentation, 3D vision, OpenCV | L1 |
| 4 | [**Sensor Fusion**](Phase%203%20-%20Artificial%20Intelligence/Track%20A%20-%20Hardware%20and%20Edge%20AI/4.%20Sensor%20Fusion/Guide.md) — camera/LiDAR/IMU, Kalman, BEVFusion | L1 |
| 5 | [**Voice AI**](Phase%203%20-%20Artificial%20Intelligence/Track%20A%20-%20Hardware%20and%20Edge%20AI/5.%20Voice%20AI/Guide.md) — STT (Whisper), TTS (Piper), VAD, keyword spotting | L1 |
| 6 | [**Edge AI & Model Optimization**](Phase%203%20-%20Artificial%20Intelligence/Track%20A%20-%20Hardware%20and%20Edge%20AI/6.%20Edge%20AI%20and%20Model%20Optimization/Guide.md) — quantization, pruning, deployment | L1 |

**Track B — Agentic AI & ML Engineering** (→ Phase 5 HPC/GenAI):

| # | Topic | Layer |
|---|-------|:-----:|
| 3 | [**Agentic AI & GenAI**](Phase%203%20-%20Artificial%20Intelligence/Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/3.%20Agentic%20AI%20and%20GenAI/Guide.md) — LLM agents, RAG, tool use | L1 |
| 4 | [**ML Engineering & MLOps**](Phase%203%20-%20Artificial%20Intelligence/Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/4.%20ML%20Engineering%20and%20MLOps/Guide.md) — training pipelines, model serving | L1 |
| 5 | [**LLM Application Development**](Phase%203%20-%20Artificial%20Intelligence/Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/5.%20LLM%20Application%20Development/Guide.md) — fine-tuning, RAG architecture | L1 |

---

### Phase 4: Hardware Deployment & Compilation (6–12 months each)

<small>*Deploy AI on real silicon and learn how compilers bridge models to hardware.*</small>

#### Track A — Xilinx FPGA

| Topic | Key Skills | Layer |
|-------|------------|:-----:|
| [**Xilinx FPGA Development**](Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/1.%20Xilinx%20FPGA%20Development/Guide.md) | Vivado, IP, timing, ILA/VIO | L6 |
| [**Zynq UltraScale+ MPSoC**](Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/2.%20Zynq%20UltraScale%2B%20MPSoC/Guide.md) | PS/PL, Linux on Zynq | L5/L6 |
| [**Advanced FPGA Design**](Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/3.%20Advanced%20FPGA%20Design/Guide.md) | CDC, floorplanning, power, PR | L6 |
| [**HLS**](Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/4.%20High-Level%20Synthesis%20%28HLS%29/Guide.md) | C→RTL, dataflow, pipelining | L5/L6 |
| [**Runtime & Drivers**](Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/5.%20Runtime%20and%20Driver%20Development/Guide.md) | XRT, DMA, kernel drivers, Vitis AI/FINN | L3 |
| [**Projects**](Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/6.%20Projects/Wireless-Video-FPGA.md) | 1080p→4K wireless video (VCU, MIPI, TDMA) | L3–L6 |

#### Track B — NVIDIA Jetson

| Topic | Key Skills | Layer |
|-------|------------|:-----:|
| [**Jetson Platform**](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Guide.md) | Orin Nano, JetPack 6.2.2, L4T, CUDA | L3 |
| [**Carrier Board**](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/2.%20Custom%20Carrier%20Board%20Design%20and%20Bring-Up/Guide.md) | Schematic, PCB, thermal, bring-up | L5 |
| [**L4T Customization**](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/3.%20L4T%20Customization/Guide.md) | Rootfs, kernel/DT, OTA | L3/L4 |
| [**FSP Firmware**](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/4.%20FSP%20%28Firmware%20Support%20Package%29%20Customization/Guide.md) | FreeRTOS on SPE/AON | L4 |
| [**App Development**](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/5.%20Application%20Development/Guide.md) | ML/AI, ROS 2, multimedia | L1/L3 |
| [**Security & OTA**](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/6.%20Security%20and%20OTA/Guide.md) | Secure boot, OP-TEE, A/B OTA | L4 |
| [**Manufacturing**](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/7.%20Compliance%20and%20Manufacturing/Guide.md) | FCC/CE, DFM, production flash | L8 |
| [**Runtime & Drivers**](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/8.%20Runtime%20and%20Driver%20Development/Guide.md) | CUDA runtime, TensorRT, DLA, DeepStream | L3 |

#### Track C — ML Compiler

| Part | Key Skills | Layer |
|------|------------|:-----:|
| [**Compiler Fundamentals**](Phase%204%20-%20Track%20C%20-%20ML%20Compiler%20and%20Graph%20Optimization/Guide.md) | Graph IR, LLVM, MLIR, TVM/tinygrad, custom backends | L2 |
| [**DL Inference Optimization**](Phase%204%20-%20Track%20C%20-%20ML%20Compiler%20and%20Graph%20Optimization/DL%20Inference%20Optimization/Guide.md) | Triton, CUTLASS, Flash-Attention, TensorRT-LLM, quantization | L1/L2 |

---

### Phase 5: Specialization Tracks (ongoing)

| Track | Focus | Guide |
|-------|-------|-------|
| **A: GPU Infrastructure** | Nvidia GPU (NCCL, NVLink) · AMD GPU (ROCm, HIP, MI300X) | [Guide →](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/1.%20GPU%20Infrastructure/Guide.md) |
| **B: HPC** | [CUDA-X Libraries](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/2.%20High%20Performance%20Computing/CUDA-X%20Libraries/Guide.md): 40+ GPU-accelerated libraries | [Guide →](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/2.%20High%20Performance%20Computing/Guide.md) |
| **C: Edge AI** | Efficient nets, quantization, Holoscan, real-time pipelines | [Guide →](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/3.%20Edge%20AI/Guide.md) |
| **D: Robotics** | Nav2, MoveIt, planning | [Guide →](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/4.%20Robotics/Guide.md) |
| **E: Autonomous Vehicles** | openpilot, tinygrad, BEV, safety, [Lauterbach](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/5.%20Autonomous%20Vehicles/6.%20Lauterbach%20TRACE32%20Debug/Guide.md) | [Guide →](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/5.%20Autonomous%20Vehicles/Guide.md) |
| **F: AI Chip Design** | Systolic arrays, dataflow, tinygrad↔hardware, ASIC path | [Guide →](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/6.%20AI%20Chip%20Design/Guide.md) |

---

## Layer → Job Title Quick Reference

| Layer | Job titles | Primary phases |
|:-----:|-----------|---------------|
| **L1** | ML Inference Eng · Edge AI Eng · Agentic AI Eng · GenAI Eng · MLOps Eng | Phase 3, 4B+C |
| **L2** | AI Compiler Eng · Graph Optimization Eng · Kernel Eng | Phase 4C |
| **L3** | GPU Runtime Eng · Linux Kernel Eng · Embedded Linux BSP Eng | Phase 4A§5, 4B§8 |
| **L4** | Firmware Eng · Embedded SW Eng · Embedded Linux Eng · IoT Eng | Phase 2, 4B§4 |
| **L5** | AI Accelerator Architect · SoC Platform Eng | Phase 1§2, 4A, 5F |
| **L6** | RTL Design Eng · FPGA Eng · DV Eng | Phase 1§1, 4A |
| **L7** | *Theory: OpenROAD, GDS flow* | Phase 5F |
| **L8** | *Theory: chiplets, CoWoS, TinyTapeout* | Phase 5F |

**Cross-layer roles:**
- **Autonomous Vehicles HW/SW Engineer** — L1 through L4 (Phase 4B + Phase 5E)
- **AI Hardware Engineer (Full-Stack)** — L1 through L6 (the signature role this roadmap targets)

---

## Reference Projects

| Project | What it teaches | Used in |
|---------|----------------|---------|
| [**tinygrad**](https://github.com/tinygrad/tinygrad) | Minimal DL framework — IR, scheduler, BEAM, backends | Phase 3, 4C, 5E |
| [**openpilot**](https://github.com/commaai/openpilot) | Production ADAS — camera→ISP→modeld→planning→CAN | Phase 4B, 5E |

---

## Additional Resources

- [**Roles & Market Analysis**](Roles%20and%20Market%20Analysis.md) — 23 sub-layers, salary data, job postings, remote %, hiring priorities

---

<div align="center" markdown="1">

**A community-driven educational roadmap for AI hardware engineering.** · [Star ⭐](https://github.com/ai-hpc/ai-hardware-engineer-roadmap) if you find this useful

</div>
