<div align="center">

# AI Hardware Engineer Roadmap

**Design a custom AI inference chip. That is the goal.** <a href="https://github.com/ai-hpc/ai-hardware-engineer-roadmap/stargazers"><img src="https://img.shields.io/github/stars/ai-hpc/ai-hardware-engineer-roadmap" style="vertical-align: middle"/></a>

![AI Hardware Engineer Roadmap](ai-hardware-engineer.png)

</div>

A custom AI chip is an **8-layer vertical stack** — from the PyTorch model at the top to silicon fabrication at the bottom. Building one requires a team where every engineer has **background knowledge across all 8 layers** but **masters one layer deeply**. This roadmap builds that vertical literacy, then lets you specialize.

**Who is this for?** EE/ECE students, software ML engineers, embedded engineers, and career changers targeting AI accelerators, edge AI, or autonomous systems.

**Prerequisites:** Algebra/calculus · C or Python · Linux or WSL · FPGA board recommended for Phase 4 Track A

**Timeline:** ~2.5–5 years part-time (~10–15 hrs/week). Full-time learners move faster.

---

## The 8-Layer Stack

Every AI inference chip — from NVIDIA's H100 to Google's TPU to a startup's edge NPU — is this stack. Each layer builds on the one below it.

| Layer | What it does | Key technologies |
|:-----:|-------------|-----------------|
| **L1** | **AI Application & Framework** | PyTorch, TensorFlow, ONNX, quantization tools |
| **L2** | **Compiler & Graph Optimization** | MLIR dialects, TVM, LLVM, custom NPU lowering |
| **L3** | **Runtime & Driver** | C++ runtime, Linux kernel driver (PCIe/DMA), CUDA-like API |
| **L4** | **Firmware & OS** | C/Rust, FreeRTOS, bootloader, command processor |
| **L5** | **Hardware Architecture** | Systolic arrays, HBM controllers, NoC, UCIe, power domains |
| **L6** | **RTL & Logic Design** | SystemVerilog / Chisel, UVM verification, emulation, IP integration |
| **L7** | **Physical Implementation** | EDA tools (Innovus, PrimeTime), TSMC N3/N5 PDK, timing closure |
| **L8** | **Fabrication & Packaging** | Foundry process, CoWoS packaging, post-silicon bring-up |

**Layers 1–6** are built with hands-on projects throughout this curriculum. **Layers 7–8** are covered as theory and guided labs (OpenROAD, TinyTapeout) — they require foundry access and EDA licenses for full mastery.

---

## Background All, Master One

You don't need to master all 8 layers. But you need enough depth in each to collaborate across the stack. Then you **go deep in one layer** — that becomes your job.

| Master layer | Phase 4 track | Phase 5 specialization | Job titles |
|:------------:|:------------:|:---------------------:|------------|
| **L1** Application | Track B + Track C Part 2 | Edge Computing or HPC | ML Inference Optimization Engineer · Edge AI Deployment Engineer |
| **L2** Compiler | **Track C** | HPC or AI Chip Design | AI Compiler Engineer · DL Graph Optimization Engineer · ML Compiler Backend Engineer |
| **L3** Runtime | Track A §5 or Track B §8 | HPC | GPU/Accelerator Runtime Engineer · Inference Platform Engineer · Linux Kernel Engineer |
| **L4** Firmware | **Track B** (FSP, L4T) | Autonomous Driving | Firmware Engineer (AI/Edge SoC) · Embedded Software Engineer · Embedded Linux Engineer · IoT Engineer |
| **L5** Architecture | Track A + Track C | **AI Chip Design** | AI Accelerator Architect · SoC Platform Engineer |
| **L6** RTL | **Track A** (full) | **AI Chip Design** | RTL Design Engineer · FPGA Design Engineer · Design Verification Engineer |
| **L7–L8** Physical/Fab | Track A (background) | AI Chip Design (theory) | *Foundational lectures — not a mastery target in this curriculum* |

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

> *The language of hardware — from gates and Verilog to CUDA kernels.*
> Covers: **L5** (architecture), **L6** (RTL/HDL), **L3/L4** (OS basics), **L1** (CUDA entry point)

| Topic | Key Skills | Layer |
|-------|------------|:-----:|
| [**Digital Design and HDL**](Phase%201%20-%20Foundational%20Knowledge/1.%20Digital%20Design%20and%20Hardware%20Description%20Languages/Guide.md) | Number systems, logic, memory; Verilog, testbenches, synthesis | L6 |
| [**Computer Architecture**](Phase%201%20-%20Foundational%20Knowledge/2.%20Computer%20Architecture%20and%20Hardware/Guide.md) | ISA, pipelines, caches, OoO, coherence; modern CPUs/GPUs/memory | L5 |
| [**Operating Systems**](Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Guide.md) | Processes, threads, scheduling, memory management, drivers | L3/L4 |
| [**C++ and Parallel Computing**](Phase%201%20-%20Foundational%20Knowledge/4.%20C%2B%2B%20and%20Parallel%20Computing/Guide.md) | C++ & SIMD, OpenMP & OneTBB, CUDA & SIMT, OpenCL | L1/L3 |

**Build:** Breadboard calculator, FPGA digital clock, traffic light controller, UART module, basic RISC-V core; SIMD/OpenMP exercises; CUDA vector/SAXPY/matmul

---

### Phase 2: Embedded Systems (6–12 months)

> *The boards and buses that sit next to inference — MCUs, RTOS, and embedded Linux.*
> Covers: **L4** (firmware/OS), **L3** (drivers/BSP)

| Topic | Key Skills | Layer |
|-------|------------|:-----:|
| [**Embedded Software**](Phase%202%20-%20Embedded%20Systems/1.%20Embedded%20Software/Guide.md) | ARM Cortex-M, FreeRTOS, SPI/UART/I2C/CAN, power, OTA | L4 |
| [**Embedded Linux**](Phase%202%20-%20Embedded%20Systems/2.%20Embedded%20Linux/Guide.md) | Yocto, PetaLinux, kernel, rootfs | L3/L4 |

**Build:** FreeRTOS sensor pipeline, DMA UART, SPI IMU, CAN network, MCUboot, Yocto image

---

### Phase 3: Artificial Intelligence (6–12 months)

> *The workloads your hardware must run — and the AI applications that create inference demand.*
> Covers: **L1** (AI application & framework). Core + two tracks.
>
> *Hub:* [**Phase 3 — Artificial Intelligence**](Phase%203%20-%20Artificial%20Intelligence/Guide.md)

**Core (mandatory):**

| # | Topic | Key Skills | Layer |
|---|-------|------------|:-----:|
| 1 | [**Neural Networks**](Phase%203%20-%20Artificial%20Intelligence/1.%20Neural%20Networks/Guide.md) | MLPs, CNNs, training, backpropagation, loss functions | L1 |
| 2 | [**Deep Learning Frameworks**](Phase%203%20-%20Artificial%20Intelligence/2.%20Deep%20Learning%20Frameworks/Guide.md) | [micrograd](Phase%203%20-%20Artificial%20Intelligence/2.%20Deep%20Learning%20Frameworks/micrograd/Guide.md) → [PyTorch](Phase%203%20-%20Artificial%20Intelligence/2.%20Deep%20Learning%20Frameworks/PyTorch/Guide.md) → [tinygrad](Phase%203%20-%20Artificial%20Intelligence/2.%20Deep%20Learning%20Frameworks/tinygrad/Guide.md) | L1/L2 |

**Track A — Hardware & Edge AI** (→ Phase 4 FPGA/Jetson/Compiler):

| # | Topic | Key Skills | Layer |
|---|-------|------------|:-----:|
| 3 | [**Computer Vision**](Phase%203%20-%20Artificial%20Intelligence/Track%20A%20-%20Hardware%20and%20Edge%20AI/3.%20Computer%20Vision/Guide.md) | Detection, segmentation, 3D vision, OpenCV | L1 |
| 4 | [**Sensor Fusion**](Phase%203%20-%20Artificial%20Intelligence/Track%20A%20-%20Hardware%20and%20Edge%20AI/4.%20Sensor%20Fusion/Guide.md) | Camera/LiDAR/IMU, Kalman, BEVFusion, MOT | L1 |
| 5 | [**Edge AI & Model Optimization**](Phase%203%20-%20Artificial%20Intelligence/Track%20A%20-%20Hardware%20and%20Edge%20AI/5.%20Edge%20AI%20and%20Model%20Optimization/Guide.md) | Quantization, pruning, deployment pipeline | L1 |
| 6 | [**Voice AI**](Phase%203%20-%20Artificial%20Intelligence/Track%20A%20-%20Hardware%20and%20Edge%20AI/6.%20Voice%20AI/Guide.md) | STT (Whisper), TTS (VITS/Piper), VAD, keyword spotting | L1 |

**Track B — Agentic AI & ML Engineering** (→ Phase 5 HPC/GenAI):

| # | Topic | Key Skills | Layer |
|---|-------|------------|:-----:|
| 3 | [**Agentic AI & GenAI**](Phase%203%20-%20Artificial%20Intelligence/Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/3.%20Agentic%20AI%20and%20GenAI/Guide.md) | LLM agents, RAG, tool use, GenAI products | L1 |
| 4 | [**ML Engineering & MLOps**](Phase%203%20-%20Artificial%20Intelligence/Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/4.%20ML%20Engineering%20and%20MLOps/Guide.md) | Training pipelines, model serving, experiment tracking | L1 |
| 5 | [**LLM Application Development**](Phase%203%20-%20Artificial%20Intelligence/Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/5.%20LLM%20Application%20Development/Guide.md) | Fine-tuning, RAG architecture, production deployment | L1 |

---

### Phase 4: Hardware Deployment & Compilation (6–12 months each track)

> *Where it all comes together — deploy AI on real silicon and learn how compilers bridge models to hardware.*

Pick **Track A (Xilinx FPGA)**, **Track B (NVIDIA Jetson)**, **Track C (ML Compiler)**, or combine them. Track C complements both A and B.

#### Track A — Xilinx FPGA → L6 RTL + L5 Architecture + L3 Runtime

> *Prototype your own accelerator: Vivado, Zynq, HLS, and production FPGA design.*

| Topic | Key Skills | Layer |
|-------|------------|:-----:|
| [**Xilinx FPGA Development**](Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/1.%20Xilinx%20FPGA%20Development/Guide.md) | Vivado, IP, timing, ILA/VIO | L6 |
| [**Zynq UltraScale+ MPSoC**](Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/2.%20Zynq%20UltraScale%2B%20MPSoC/Guide.md) | PS/PL, Linux on Zynq | L5/L6 |
| [**Advanced FPGA Design**](Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/3.%20Advanced%20FPGA%20Design/Guide.md) | CDC, floorplanning, power, PR | L6 |
| [**HLS**](Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/4.%20High-Level%20Synthesis%20%28HLS%29/Guide.md) | C→RTL, dataflow, pipelining | L5/L6 |
| [**Runtime & Driver Development**](Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/5.%20Runtime%20and%20Driver%20Development/Guide.md) | XRT, DMA, kernel drivers, user-space runtime, Vitis AI/FINN | L3 |
| [**Projects**](Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/6.%20Projects/Wireless-Video-FPGA.md) | 1080p SDR PoC → 4K wireless video on Zynq UltraScale+ (VCU, MIPI, openwifi PHY, TDMA, ASIC path) | L3–L6 |

**Build:** Matmul/conv accelerators, NN on FPGA, XRT host app, DMA benchmark, platform driver, C++/Python runtime library, [1080p→4K wireless video link](Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/6.%20Projects/Wireless-Video-FPGA.md)

#### Track B — NVIDIA Jetson → L3 Runtime + L4 Firmware + L1 Application

> *Master the Jetson edge platform end-to-end: from JetPack and custom carrier boards to secure OTA and production manufacturing.*

| Topic | Key Skills | Layer |
|-------|------------|:-----:|
| [**Nvidia Jetson Platform**](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Guide.md) | Orin Nano, JetPack, L4T, CUDA | L3 |
| [**Custom carrier board**](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/2.%20Custom%20Carrier%20Board%20Design%20and%20Bring-Up/Guide.md) | P3768 reference, schematic, PCB, thermal, bring-up | L5 |
| [**L4T customization**](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/3.%20L4T%20Customization/Guide.md) | Rootfs, kernel/DT, OTA vs Yocto | L3/L4 |
| [**FSP / SPE firmware**](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/4.%20FSP%20%28Firmware%20Support%20Package%29%20Customization/Guide.md) | FreeRTOS on SPE/AON, peripherals, `spe-fw` flash | L4 |
| [**Application Development**](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/5.%20Application%20Development/Guide.md) | Peripherals, networking, multimedia, ML/AI, ROS 2 | L1/L3 |
| [**Security and OTA**](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/6.%20Security%20and%20OTA/Guide.md) | Secure boot, OP-TEE, encryption, A/B OTA | L4 |
| [**Compliance and manufacturing**](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/7.%20Compliance%20and%20Manufacturing/Guide.md) | FCC/CE, DFM, production flash, supply chain | L8 |
| [**Runtime & Driver Development**](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/8.%20Runtime%20and%20Driver%20Development/Guide.md) | CUDA runtime/driver API, TensorRT, DLA, nvgpu, DeepStream, Triton | L3 |

#### Track C — ML Compiler & Graph Optimization → L2 Compiler

> *The bridge between AI models and hardware — learn how compilers lower neural-network graphs to efficient, hardware-specific code, then apply it to real inference.*

| Part | Key Skills | Layer |
|------|------------|:-----:|
| [**Part 1 — Compiler Fundamentals**](Phase%204%20-%20Track%20C%20-%20ML%20Compiler%20and%20Graph%20Optimization/Guide.md) | ONNX/IR, graph optimization, LLVM, MLIR, TVM/tinygrad, custom backends | L2 |
| [**Part 2 — DL Inference Optimization**](Phase%204%20-%20Track%20C%20-%20ML%20Compiler%20and%20Graph%20Optimization/DL%20Inference%20Optimization/Guide.md) | Profiling, Triton/CUTLASS/Flash-Attention, quantization, inference runtimes | L1/L2 |

**Part 1 modules:** Graph IR · Graph optimization passes · LLVM fundamentals · MLIR (dialects, progressive lowering, custom NPU dialect) · ML-to-hardware pipelines (TVM, tinygrad BEAM, torch.compile, XLA, IREE) · Kernel fusion & tiling · Custom backend development

**Part 2 modules:** Graph & operator optimization + profiling · Kernel engineering (Triton, CUTLASS/CuTe, Flash-Attention, NCCL) · Compiler stack (IR, BEAM, codegen) · Quantization (PTQ, QAT, INT8/INT4) · Inference runtimes & deployment · tinygrad deep dive

**Build:** ONNX graph analysis, Conv+BN+ReLU fusion pass, custom LLVM pass, MLIR Toy tutorial + NPU dialect, TVM AutoTVM tuning, tinygrad BEAM study, TVM BYOC backend, Triton fused kernel, INT8 TensorRT engine, runtime benchmark report

---

### Phase 5: Specialization Tracks (ongoing)

> *Go deep in the direction that matches your career goal.*
>
> *Prerequisites:* **Phases 1–3**, and the **Phase 4** tracks noted per row.

| Track | Prerequisites | Focus | Guide |
|-------|--------------|-------|-------|
| **A: GPU Infrastructure** | Phase 4B (CUDA), Phase 4C (Compiler) | **[Nvidia GPU](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/1.%20GPU%20Infrastructure/Nvidia%20GPU/Guide.md):** NCCL, NVLink, clusters · **[AMD GPU](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/1.%20GPU%20Infrastructure/AMD%20GPU/Guide.md):** ROCm, HIP, MI300X | [Guide →](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/1.%20GPU%20Infrastructure/Guide.md) |
| **B: High Performance Computing** | Phase 5A (GPU Infra), Phase 4C (Compiler) | **[CUDA-X Libraries](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/2.%20High%20Performance%20Computing/CUDA-X%20Libraries/Guide.md):** cuBLAS, cuDNN, CUTLASS, TensorRT, NCCL, RAPIDS, and 40+ GPU-accelerated libraries | [Guide →](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/2.%20High%20Performance%20Computing/Guide.md) |
| **C: Edge AI** | Phases 1–2, Phase 4B (Jetson) | Efficient nets, quantization, Holoscan, real-time pipelines | [Guide →](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/3.%20Edge%20AI/Guide.md) |
| **D: Robotics** | Phase 3 (Fusion), Phase 4B (ROS2) | Nav2, MoveIt, planning | [Guide →](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/4.%20Robotics/Guide.md) |
| **E: Autonomous Vehicles** | Phase 3 (CV, Fusion), Phase 4B (Jetson) | 6 modules: fundamentals, openpilot, tinygrad, BEV perception, safety/ISO 26262, [Lauterbach](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/5.%20Autonomous%20Vehicles/6.%20Lauterbach%20TRACE32%20Debug/Guide.md) | [Guide →](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/5.%20Autonomous%20Vehicles/Guide.md) |
| **F: AI Chip Design** | Phase 4A (FPGA), Phase 4C (Compiler), Phase 3 (NN) | Systolic arrays, dataflow, tinygrad↔hardware, ASIC path | [Guide →](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/6.%20AI%20Chip%20Design/Guide.md) |

---

## Reference Projects

| Project | What it teaches | Used in |
|---------|----------------|---------|
| **[tinygrad](https://github.com/tinygrad/tinygrad)** | Minimal DL framework — IR, scheduler, BEAM, backends, compiler pipeline | Phase 3 (NN), Phase 4C (compiler), Phase 5E (openpilot inference) |
| **[openpilot](https://github.com/commaai/openpilot)** | Production ADAS — camera→ISP→modeld→planning→CAN | Phase 4B (Jetson deployment), Phase 5E (autonomous driving) |

---

## Layer → Job Title Quick Reference

| Layer | Job titles you can target | Primary phases |
|:-----:|--------------------------|---------------|
| **L1** Application | ML Inference Optimization Engineer · Edge AI Deployment Engineer | Phase 3, Phase 4B+C |
| **L2** Compiler | AI Compiler Engineer · DL Graph Optimization Engineer · ML Compiler Backend Engineer | Phase 4C |
| **L3** Runtime | GPU/Accelerator Runtime Engineer · Inference Platform Engineer · Linux Kernel Engineer · Embedded Linux BSP Engineer | Phase 4A§5, Phase 4B§8 |
| **L4** Firmware | Firmware Engineer (AI/Edge SoC) · Embedded Software Engineer · Embedded Linux Engineer · IoT Engineer | Phase 2, Phase 4B§4 |
| **L5** Architecture | AI Accelerator Architect · SoC Platform Engineer | Phase 1§2, Phase 4A, Phase 5F |
| **L6** RTL | RTL Design Engineer · FPGA Design Engineer · Design Verification Engineer | Phase 1§1, Phase 4A |
| **L7** Physical | *Theory: OpenROAD, GDS flow* | Phase 5F (AI Chip Design) |
| **L8** Fab/Package | *Theory: chiplets, CoWoS, TinyTapeout* | Phase 5F (AI Chip Design) |

**Cross-layer roles:**
- **Autonomous Vehicles HW/SW Engineer** — L1 through L4 (Phase 4B + Phase 5E)
- **AI Hardware Engineer (Full-Stack)** — L1 through L6 (the signature role this roadmap targets)

---

<div align="center">

**Built for the AI hardware community** · [Star ⭐](https://github.com/ai-hpc/ai-hardware-engineer-roadmap) if you find this useful

</div>
