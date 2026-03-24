<div align="center">

# AI Hardware Engineer Roadmap

**From Kernel-Level Parallel Programming to Custom AI Inference Accelerator Design — powered by NVIDIA GPUs, Jetson, and tinygrad** <a href="https://github.com/ai-hpc/ai-hardware-engineer-roadmap/stargazers"><img src="https://img.shields.io/github/stars/ai-hpc/ai-hardware-engineer-roadmap" style="vertical-align: middle"/></a>

![AI Hardware Engineer Roadmap](ai-hardware-engineer.png)

</div>

## The Four Career Steps

Each step is a **concrete role target** built on the **4-phase** curriculum below (Phase 3 splits into **Track A — FPGA** and **Track B — Jetson / edge AI**).

| Step | Role target | Common titles (same step) | Focus | Outcome |
|:----:|-------------|----------------------------|-------|---------|
| **1** | **Parallel Program Optimization Engineer** | — | CUDA/OpenCL kernels, memory hierarchy, warp/SM behavior, tinygrad backends | Read kernel traces, identify memory vs compute bottlenecks, optimize parallel programs on GPU/SoC |
| **2** | **DL Inference Optimization Engineer** | — | Model/operator optimization, TensorRT, tinygrad compiler (IR, scheduling, BEAM), quantization | Take a model from graph to optimized deployment with measurable latency/throughput improvement |
| **3** | **DL Inference for Edge / AV / Robotics** | **Embedded Software Engineer**, **Embedded Linux Engineer** | Power/latency-constrained deployment, sensor→actuation pipeline, openpilot/Jetson/DRIVE; MCU/RTOS + Linux BSP next to the ML stack | Own inference optimization for edge/AV/robotics; hit latency and power targets on real SoCs |
| **4** | **FPGA & Custom Chip for DL Inference** | **FPGA Engineer** (RTL/HLS/prototyping) | Mapping inference to hardware, HLS/RTL, accelerator architecture (systolic, dataflow), ASIC path | Design and implement FPGA accelerators for DL workloads; understand the custom-chip design path |

```mermaid
graph LR
  A["You (today)"] --> R1["Step 1<br/>Parallel Program<br/>Optimization Engineer"]
  R1 --> R2["Step 2<br/>DL Inference<br/>Optimization Engineer"]
  R2 --> R3["Step 3<br/>DL Inference for<br/>Edge / AV / Robotics"]
  R3 --> R4["Step 4<br/>FPGA & Custom Chip<br/>for DL Inference"]

  subgraph External Roles
    N1["GPU / Parallel Computing Engineer"]
    N2["Senior DL Inference Optimization Engineer<br/>(NVIDIA-style, Edge/AV/Robotics)"]
    N3["AI Accelerator / Chip Design Engineer"]
  end

  R1 --> N1
  R2 --> N2
  R3 --> N2
  R4 --> N3

  subgraph Phases
    P1["Phase 1<br/>(Digital, OS, C++/CUDA, AI)"]
    P2["Phase 2<br/>(Embedded Systems)"]
    P3A["Phase 3 · Track A<br/>(Xilinx FPGA)"]
    P3B["Phase 3 · Track B<br/>(Jetson, TensorRT, sensors, ROS2)"]
    P4["Phase 4<br/>(Specialization A–E)"]
  end

  R1 -- "mandatory" --> P1
  R1 -- "mandatory" --> P2
  R1 -- "optional" --> P3A
  R1 -- "mandatory (CUDA)" --> P3B

  R2 -- "mandatory" --> P1
  R2 -- "mandatory" --> P2
  R2 -- "optional" --> P3A
  R2 -- "mandatory" --> P3B

  R3 -- "mandatory" --> P1
  R3 -- "mandatory" --> P2
  R3 -- "mandatory" --> P3B
  R3 -- "optional" --> P3A
  R3 -- "optional" --> P4

  R4 -- "mandatory" --> P1
  R4 -- "mandatory" --> P2
  R4 -- "mandatory" --> P3A
  R4 -- "optional" --> P3B
  R4 -- "mandatory" --> P4
```

**Reference projects** used throughout all four steps:

| Project | Step 1 | Step 2 | Step 3 | Step 4 |
|---------|--------|--------|--------|--------|
| **[tinygrad](https://github.com/tinygrad/tinygrad)** | Trace ops→kernels, study backends | IR, scheduling, BEAM, quantization | On-device inference under edge constraints | Custom backend; workload for accelerator design |
| **[openpilot](https://github.com/commaai/openpilot)** | — | Why inference optimization matters in production | Full AV stack: camera→ISP→modeld→planning→CAN | Real workloads (vision, policy) for hardware design |

---

## 4-Phase Curriculum

### Phase 1: Digital Foundations (6–12 months)

| Topic | Key Skills | AI Connection |
|-------|------------|---------------|
| [**Digital Design Fundamentals**](Phase%201%20-%20Foundational%20Knowledge/1.%20Digital%20Design%20Fundamentals/Guide.md) | Number systems, Boolean algebra, combinational/sequential logic, memory (SRAM, DRAM, ROM) | *MAC units, memory bandwidth, and data types (INT8, FP16) that power AI inference start here* |
| [**Hardware Description Languages**](Phase%201%20-%20Foundational%20Knowledge/2.%20Hardware%20Description%20Languages%20(HDLs)/Guide.md) | Verilog syntax, behavioral/dataflow/structural modeling, testbenches, synthesis | *The language you will use to design AI accelerator datapaths* |
| [**Computer Architecture and Hardware**](Phase%201%20-%20Foundational%20Knowledge/3.%20Computer%20Architecture%20and%20Hardware/Guide.md) | ISA through microarchitecture (pipelines, caches, OoO, coherence); labs; modern CPUs/GPUs/memory/storage/I/O across form factors | *Same limits (bandwidth, latency, power) govern TinyML through data-center GPUs; know both theory and what ships* |
| [**Operating Systems**](Phase%201%20-%20Foundational%20Knowledge/4.%20Operating%20Systems/Guide.md) | Processes, threads, scheduling (CFS/EEVDF/RT), memory management, synchronization, device drivers, filesystems | *OS underpins Linux, RTOS, and all AI deployment targets; 24-lecture curriculum covering modern Linux internals* |
| [**Deep C++ and Parallel computing with CUDA**](Phase%201%20-%20Foundational%20Knowledge/5.%20Deep%20C%2B%2B%20and%20Parallel%20computing%20with%20CUDA/Guide.md) | Modern C++ (memory, STL), CPU threading, CUDA model (grids/blocks, memory spaces), streams; vector/matmul/reduction projects | *Host + kernel skills match real inference stacks; bridges OS threads to GPU parallelism before you study NN math* |
| [**AI Fundamentals**](Phase%201%20-%20Foundational%20Knowledge/6.%20AI%20Fundamentals%20-%20Neural%20Networks%20and%20Edge%20AI/Guide.md) | Neural networks, backpropagation, CNNs, tinygrad, PyTorch | *Understanding what the hardware must compute — the bridge from digital foundations to AI acceleration* |

**Projects:** Calculator on breadboard, FPGA digital clock, traffic light controller, UART module, basic RISC-V core, CUDA vector/SAXPY/matmul + CPU reference checks, micrograd implementation, CNN from scratch, tinygrad internals

---

### Phase 2: Embedded Systems (6–12 months)

| Topic | Key Skills | AI Connection |
|-------|------------|---------------|
| [**Embedded Software**](Phase%202%20-%20Embedded%20Systems/1.%20Embedded%20Software/Guide.md) | ARM Cortex-M architecture (CMSIS, MPU, TrustZone), FreeRTOS (tasks, queues, semaphores), SPI/UART/I2C/CAN drivers, power management, OTA updates | *Sensor pipelines (CAN, SPI, I2C) feed AI perception; FreeRTOS schedules real-time inference tasks; CAN/J1939 is how openpilot commands vehicle actuators* |
| [**Embedded Linux**](Phase%202%20-%20Embedded%20Systems/2.%20Embedded%20Linux/Guide.md) | Yocto, PetaLinux, kernel config, root filesystem | *Jetson, Qualcomm AI, and all edge AI platforms run embedded Linux* |

**Projects:** FreeRTOS sensor pipeline, DMA UART receiver, SPI IMU at max ODR, CAN two-node network, MCUboot secure bootloader, ultra-low-power IoT node, custom Yocto image

---

### Phase 3 — Track A: Xilinx FPGA (6–12 months)

| Topic | Key Skills | AI Connection |
|-------|------------|---------------|
| [**Xilinx FPGA Development**](Phase%203%20-%20Track%20A%20-%20Xilinx%20FPGA/1.%20Xilinx%20FPGA%20Development/Guide.md) | Vivado flow, IP cores, block design, timing closure, ILA/VIO debugging | *FPGAs are the prototyping platform for AI accelerators (FINN, Vitis AI)* |
| [**Zynq UltraScale+ MPSoC**](Phase%203%20-%20Track%20A%20-%20Xilinx%20FPGA/2.%20Zynq%20UltraScale%2B%20MPSoC/Guide.md) | PS/PL integration, embedded Linux on Zynq, device drivers | *Heterogeneous SoCs like Zynq are the template for AI chips (CPU + accelerator)* |
| [**Advanced FPGA Design**](Phase%203%20-%20Track%20A%20-%20Xilinx%20FPGA/3.%20Advanced%20FPGA%20Design/Guide.md) | CDC, floorplanning, power optimization, partial reconfiguration | *Production FPGA accelerators for AI require timing closure, power budgets, and reconfiguration* |
| [**High-Level Synthesis (HLS)**](Phase%203%20-%20Track%20A%20-%20Xilinx%20FPGA/4.%20High-Level%20Synthesis%20%28HLS%29/Guide.md) | C/C++ to RTL, dataflow, loop unrolling, pipelining | *HLS is how you build CNN accelerators (conv2d, matmul) on FPGAs without writing RTL by hand* |
| [**OpenCL**](Phase%203%20-%20Track%20A%20-%20Xilinx%20FPGA/5.%20OpenCL/Guide.md) | Kernels, work-groups, heterogeneous computing (CPU/GPU/FPGA) | *The programming model for deploying AI workloads across different hardware targets* |
| [**Computer Vision**](Phase%203%20-%20Track%20A%20-%20Xilinx%20FPGA/6.%20Computer%20Vision/Guide.md) | Image processing, object detection, OpenCV | *The primary AI workload you will deploy on hardware: perception from pixels* |

**Projects:** Matrix multiply accelerator, convolution engine, image processing pipeline, neural network acceleration, CPU vs GPU vs FPGA benchmarking

---

### Phase 3 — Track B: Nvidia Jetson & Edge AI (6–12 months)

> *Apply your AI and hardware foundations to real edge deployment: optimize and run models on Jetson, fuse sensors for perception, and build robotic systems with ROS2.*

| Topic | Key Skills | Projects |
|-------|------------|----------|
| [**Nvidia Jetson Platform**](Phase%203%20-%20Track%20B%20-%20Nvidia%20Jetson%20and%20Edge%20AI/1.%20Nvidia%20Jetson%20Platform/Guide.md) | Jetson Orin Nano, JetPack, L4T, CUDA, Nsight | Real-time object detection, custom model deployment, autonomous robot |
| [**Edge AI Optimization**](Phase%203%20-%20Track%20B%20-%20Nvidia%20Jetson%20and%20Edge%20AI/2.%20Edge%20AI%20Optimization/Guide.md) | Quantization, pruning, TensorRT, CUDA kernels | Optimized model on Orin Nano, video analytics, low-power AI pipeline |
| [**Sensor Fusion**](Phase%203%20-%20Track%20B%20-%20Nvidia%20Jetson%20and%20Edge%20AI/3.%20Sensor%20Fusion/Guide.md) | Camera + LiDAR + IMU, Kalman filtering, BEVFusion | Navigation robot, drone flight control, 3D mapping |
| [**ROS2**](Phase%203%20-%20Track%20B%20-%20Nvidia%20Jetson%20and%20Edge%20AI/4.%20ROS2/Guide.md) | ROS 2, DDS, nodes, topics, multi-robot systems | Robot navigation, multi-robot coordination, edge deployment |
| [**OrinClaw**](Phase%203%20-%20Track%20B%20-%20Nvidia%20Jetson%20and%20Edge%20AI/5.%20OpenClaw%20Assistant%20Box/Guide.md) (OpenClaw-based capstone) | Hardware-aware edge AI product design, low-power inference, on-device voice + automation, OTA, privacy/security | Jetson Orin Nano assistant box: **OrinClaw**, Alexa-level UX, offline-first |

---

### Phase 4: Specialization Tracks (Ongoing)

> *Choose one or more tracks based on your career goals. Tracks assume **Phases 1–2** plus the parts of **Phase 3** (Track A and/or B) noted in the prerequisites column.*

| Track | Prerequisites | Focus | Guide |
|-------|--------------|-------|-------|
| **A: Autonomous Driving** | Phase 3 **Track A** (Computer Vision), Phase 3 **Track B** (Sensor Fusion, Edge AI) | openpilot architecture (camerad, modeld, planning, control), tinygrad on-device inference, camera ISP pipelines, BEV perception | [Guide →](Phase%204%20-%20Advanced%20Topics%20and%20Specialization/4.%20Autonomous%20Driving/Guide.md) |
| **B: AI Chip Design** | Phase 3 **Track A** (HLS, Advanced FPGA), Phase 1 §6 (AI Fundamentals) | Systolic arrays, dataflow architectures, tinygrad as hardware-software interface, FPGA prototyping, ASIC flow overview | [Guide →](Phase%204%20-%20Advanced%20Topics%20and%20Specialization/5.%20AI%20Chip%20Design/Guide.md) |
| **C: HPC & GPU Infrastructure** | Phase 3 **Track B** (CUDA / Jetson stack) | Multi-GPU NCCL, NVLink/NVSwitch, InfiniBand, RDMA, GPUDirect; includes **DL Inference Optimization** (graph/ops, kernels, compiler, quantization, runtimes) | [Guide →](Phase%204%20-%20Advanced%20Topics%20and%20Specialization/1.%20HPC%20and%20DL%20with%20Nvidia%20GPU/Guide.md) |
| **D: Robotics** | Phase 3 **Track B** (ROS2, Sensor Fusion) | Nav2, MoveIt manipulation, motion planning, ROS-Industrial, sensor fusion for autonomous robots | [Guide →](Phase%204%20-%20Advanced%20Topics%20and%20Specialization/3.%20Robotics%20Application/Guide.md) |
| **E: Real Time Edge AI with Nvidia Jetson** | Phases 1–2, Phase 3 **Track B** (Jetson, TensorRT) | Efficient architectures (MobileNet, EfficientNet, YOLO), quantization, TinyML, edge inference runtimes, **NVIDIA Jetson Holoscan**, system integration | [Guide →](Phase%204%20-%20Advanced%20Topics%20and%20Specialization/2.%20Real%20Time%20Edge%20AI%20with%20Nvidia%20Jetson/Guide.md) |

---

## Career Paths

The **[four career steps](#the-four-career-steps)** above are the **progression** (what to optimize next). The tables below are **titles** you might hold on a job description, mapped to **where in the curriculum** that depth usually comes from. Phases are cumulative: later roles assume you can still read a schematic, reason about memory, and debug a Linux box when needed.

### By career step (1–4)

| Career step | Role titles (examples) | Phases you lean on most | Phase 4 specialization (if any) |
|:-----------:|------------------------|-------------------------|----------------------------------|
| **1** — Parallel Program Optimization | GPU / CUDA Engineer, Compute Kernel Engineer, Performance Engineer (GPU) | [1](Phase%201%20-%20Foundational%20Knowledge) (esp. architecture + §5 C++/CUDA), [3 · B](Phase%203%20-%20Track%20B%20-%20Nvidia%20Jetson%20and%20Edge%20AI) (Jetson CUDA stack) | [C: HPC & GPU](Phase%204%20-%20Advanced%20Topics%20and%20Specialization/1.%20HPC%20and%20DL%20with%20Nvidia%20GPU/Guide.md) |
| **2** — DL Inference Optimization | Inference Optimization Engineer, TensorRT / ONNX Runtime Engineer, Compiler Backend Engineer (ML) | [1](Phase%201%20-%20Foundational%20Knowledge) §5–§6, [3 · B](Phase%203%20-%20Track%20B%20-%20Nvidia%20Jetson%20and%20Edge%20AI) (TensorRT, Edge AI) | [C: HPC & GPU](Phase%204%20-%20Advanced%20Topics%20and%20Specialization/1.%20HPC%20and%20DL%20with%20Nvidia%20GPU/Guide.md) (graph, ops, quantization at scale) |
| **3** — Edge / AV / Robotics | Edge ML Engineer, Jetson Deployment Engineer, Perception Engineer, Robotics Integration Engineer; **Embedded Software Engineer**, **Embedded Linux Engineer** | [1](Phase%201%20-%20Foundational%20Knowledge)–[2](Phase%202%20-%20Embedded%20Systems), [3 · B](Phase%203%20-%20Track%20B%20-%20Nvidia%20Jetson%20and%20Edge%20AI) (Jetson, sensors, ROS2) | [A: Autonomous Driving](Phase%204%20-%20Advanced%20Topics%20and%20Specialization/4.%20Autonomous%20Driving/Guide.md), [D: Robotics](Phase%204%20-%20Advanced%20Topics%20and%20Specialization/3.%20Robotics%20Application/Guide.md), [E: Real-Time Edge AI](Phase%204%20-%20Advanced%20Topics%20and%20Specialization/2.%20Real%20Time%20Edge%20AI%20with%20Nvidia%20Jetson/Guide.md) |
| **4** — FPGA & custom silicon | FPGA ML Accelerator Engineer, RTL / Design Engineer (AI blocks), AI Silicon Architect (path to ASIC); **FPGA Engineer** | [1](Phase%201%20-%20Foundational%20Knowledge)–[3 · A](Phase%203%20-%20Track%20A%20-%20Xilinx%20FPGA), optional [3 · B](Phase%203%20-%20Track%20B%20-%20Nvidia%20Jetson%20and%20Edge%20AI) (workloads, power/latency targets) | [B: AI Chip Design](Phase%204%20-%20Advanced%20Topics%20and%20Specialization/5.%20AI%20Chip%20Design/Guide.md) |

### By phase depth (foundation → specialization)

| Phase | Typical roles unlocked | Notes |
|:-----:|------------------------|-------|
| **[1](Phase%201%20-%20Foundational%20Knowledge)** | Embedded-aware **Software Engineer**, **BSP / bring-up** adjacent (with Phase 2), **ML Engineer** who understands hardware limits | Digital + OS + C++/CUDA + NN fundamentals; everyone passes through here |
| **[2](Phase%202%20-%20Embedded%20Systems)** | **MCU / RTOS Firmware Engineer**, **Embedded Linux / Yocto Engineer**, **IoT Platform Engineer** | Sensor buses, real-time scheduling, custom images — underpins Jetson-style products |
| **[3 · A](Phase%203%20-%20Track%20A%20-%20Xilinx%20FPGA)** | **FPGA Design Engineer**, **RTL Engineer**, **HLS / Acceleration Prototyping Engineer**, **FPGA Engineer** | Often paired with **Track B** for benchmarking AI kernels vs GPU |
| **[3 · B](Phase%203%20-%20Track%20B%20-%20Nvidia%20Jetson%20and%20Edge%20AI)** | **Jetson / L4T Platform Engineer**, **Edge AI / TensorRT Engineer**, **Sensor Fusion / Perception Engineer**, **ROS2 Robotics Engineer**, **Embedded Linux Engineer**, **Embedded Software Engineer**, **Edge AI Product Engineer** (e.g. [OrinClaw](Phase%203%20-%20Track%20B%20-%20Nvidia%20Jetson%20and%20Edge%20AI/5.%20OpenClaw%20Assistant%20Box/Guide.md) capstone) | Main “shipping inference on a robot or device” milestone |
| **[4](Phase%204%20-%20Advanced%20Topics%20and%20Specialization)** | **ADAS / AV Stack Engineer**, **GPU Cluster / ML Infra Engineer**, **Robotics Autonomy Engineer**, **AI Accelerator Architect** | Pick tracks A–E from the Phase 4 table above |

### Quick lookup: role → phases & track

| Role | Primary phases | Typical career step | Phase 4 specialization |
|------|---------------|---------------------|------------------------|
| Parallel Program Optimization Engineer | [1](Phase%201%20-%20Foundational%20Knowledge), [3 · B](Phase%203%20-%20Track%20B%20-%20Nvidia%20Jetson%20and%20Edge%20AI) | 1 | [C](Phase%204%20-%20Advanced%20Topics%20and%20Specialization/1.%20HPC%20and%20DL%20with%20Nvidia%20GPU/Guide.md) (optional depth) |
| DL Inference Optimization Engineer | [1](Phase%201%20-%20Foundational%20Knowledge), [3 · B](Phase%203%20-%20Track%20B%20-%20Nvidia%20Jetson%20and%20Edge%20AI) | 2 | [C](Phase%204%20-%20Advanced%20Topics%20and%20Specialization/1.%20HPC%20and%20DL%20with%20Nvidia%20GPU/Guide.md) |
| Edge ML / Jetson Deployment Engineer | [1](Phase%201%20-%20Foundational%20Knowledge)–[2](Phase%202%20-%20Embedded%20Systems), [3 · B](Phase%203%20-%20Track%20B%20-%20Nvidia%20Jetson%20and%20Edge%20AI) | 3 | [E](Phase%204%20-%20Advanced%20Topics%20and%20Specialization/2.%20Real%20Time%20Edge%20AI%20with%20Nvidia%20Jetson/Guide.md) |
| MCU / RTOS Firmware Engineer (**Embedded Software Engineer**) | [1](Phase%201%20-%20Foundational%20Knowledge), [2](Phase%202%20-%20Embedded%20Systems) | — (supports 3) | — |
| Embedded Linux / BSP Engineer (**Embedded Linux Engineer**) | [1](Phase%201%20-%20Foundational%20Knowledge), [2](Phase%202%20-%20Embedded%20Systems) | — (supports 3) | — |
| FPGA / RTL Engineer (**FPGA Engineer**, ML acceleration) | [1](Phase%201%20-%20Foundational%20Knowledge)–[3 · A](Phase%203%20-%20Track%20A%20-%20Xilinx%20FPGA) | 4 | [B](Phase%204%20-%20Advanced%20Topics%20and%20Specialization/5.%20AI%20Chip%20Design/Guide.md) |
| AI Accelerator / Chip Design Engineer | [1](Phase%201%20-%20Foundational%20Knowledge)–[3 · A](Phase%203%20-%20Track%20A%20-%20Xilinx%20FPGA), optional [3 · B](Phase%203%20-%20Track%20B%20-%20Nvidia%20Jetson%20and%20Edge%20AI) | 4 | [B](Phase%204%20-%20Advanced%20Topics%20and%20Specialization/5.%20AI%20Chip%20Design/Guide.md) |
| Perception / Sensor Fusion Engineer | [1](Phase%201%20-%20Foundational%20Knowledge), [3 · A](Phase%203%20-%20Track%20A%20-%20Xilinx%20FPGA)–[3 · B](Phase%203%20-%20Track%20B%20-%20Nvidia%20Jetson%20and%20Edge%20AI) (CV + sensors) | 3 | [A](Phase%204%20-%20Advanced%20Topics%20and%20Specialization/4.%20Autonomous%20Driving/Guide.md) or [D](Phase%204%20-%20Advanced%20Topics%20and%20Specialization/3.%20Robotics%20Application/Guide.md) |
| ADAS / Autonomous Driving Engineer | [1](Phase%201%20-%20Foundational%20Knowledge)–[2](Phase%202%20-%20Embedded%20Systems), [3 · B](Phase%203%20-%20Track%20B%20-%20Nvidia%20Jetson%20and%20Edge%20AI) | 3 | [A](Phase%204%20-%20Advanced%20Topics%20and%20Specialization/4.%20Autonomous%20Driving/Guide.md) |
| Robotics Engineer (ROS2, autonomy) | [1](Phase%201%20-%20Foundational%20Knowledge), [2](Phase%202%20-%20Embedded%20Systems), [3 · B](Phase%203%20-%20Track%20B%20-%20Nvidia%20Jetson%20and%20Edge%20AI) | 3 | [D](Phase%204%20-%20Advanced%20Topics%20and%20Specialization/3.%20Robotics%20Application/Guide.md) |
| GPU / HPC / ML Infrastructure Engineer | [1](Phase%201%20-%20Foundational%20Knowledge), [3 · B](Phase%203%20-%20Track%20B%20-%20Nvidia%20Jetson%20and%20Edge%20AI) | 1–2 | [C](Phase%204%20-%20Advanced%20Topics%20and%20Specialization/1.%20HPC%20and%20DL%20with%20Nvidia%20GPU/Guide.md) |

---

## About

**Who is this for?** EE/ECE students, software ML engineers, embedded engineers, and career changers targeting AI accelerator design, edge AI, or autonomous systems. No prior AI/ML experience required — AI fundamentals are taught in Phase 1 after the hardware foundation.

**Prerequisites:** Basic algebra and calculus · C or Python · Linux or WSL · FPGA dev boards recommended from **Phase 3 · Track A**

**Estimated timeline:** ~2.5–5 years part-time (~10–15 hrs/week). Full-time learners move significantly faster.

---

<div align="center">

**Built for the AI hardware community** · [Star ⭐](https://github.com/ai-hpc/ai-hardware-engineer-roadmap) if you find this useful

</div>
