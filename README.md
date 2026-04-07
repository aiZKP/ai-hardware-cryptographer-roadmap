<div align="center" markdown="1">

# AI Hardware Engineer Roadmap

**Learn to build the hardware that runs AI — from writing your first CUDA kernel to designing a custom AI chip.**

![AI Hardware Engineer Roadmap](Assets/images/ai-hardware-engineer.png)

[📖 Read the full guide](https://ai-hpc.github.io/ai-hardware-engineer-roadmap/) · [⭐ Star this repo](https://github.com/ai-hpc/ai-hardware-engineer-roadmap)

</div>

---

## What is this?

Every AI model — GPT, Stable Diffusion, your self-driving car — runs on **specialized hardware**. Someone has to build that hardware, write the software that drives it, and make the two work together efficiently.

This is a **free, community-driven curriculum** that teaches you to do exactly that. It covers the full stack from the AI application at the top down to the chip design at the bottom — organized as a self-paced learning roadmap with guides, projects, and curated resources.

**You will learn to:**

- Write GPU kernels and parallel code that runs at hardware speed
- Deploy AI models on real embedded hardware (NVIDIA Jetson, Xilinx FPGA)
- Understand how ML compilers turn PyTorch into chip instructions
- Read and reason about chip architecture — the way AI accelerators are designed

---

## Who is this for?

| Background | What you'll get from this |
|------------|--------------------------|
| Software engineer wanting to go deeper into AI infrastructure | CUDA, parallel computing, ML compilers, GPU runtimes |
| ML / AI engineer who wants to understand the hardware | How chips work, why quantization matters, how to optimize inference |
| Embedded / firmware engineer moving into AI products | AI workloads, edge deployment, Jetson, sensor fusion |
| Computer science student aiming at AI hardware roles | A structured curriculum from foundations to specialization |
| Hardware engineer adding AI/software skills | Neural networks, CUDA, ML frameworks, model optimization |

---

## The AI Chip Stack — Explained Simply

A chip that runs AI isn't just silicon. It's **8 layers of technology** that must work together. Think of it like a building: the foundation (silicon) holds up the floors above it (firmware, OS, drivers), which hold up the penthouse (your AI application).

```
  ┌─────────────────────────────────────┐
  │  L1  AI App & Framework             │  ← PyTorch model, your code runs here
  │  L2  ML Compiler                    │  ← turns model into chip instructions
  │  L3  Runtime & Driver               │  ← OS talks to the GPU/chip
  │  L4  Firmware & OS                  │  ← boots the device, manages resources
  │  L5  Hardware Architecture          │  ← the chip's blueprint (systolic arrays, HBM)
  │  L6  RTL & Logic Design             │  ← describes the chip in hardware language
  │  L7  Physical Implementation        │  ← places transistors on silicon
  │  L8  Fabrication & Packaging        │  ← the foundry makes the physical chip
  └─────────────────────────────────────┘
```

| Layer | Plain English | Technologies |
|:-----:|---------------|-------------|
| **L1** | Where your AI model lives and runs | PyTorch, ONNX, TensorRT, MLOps |
| **L2** | Translates the model into efficient chip instructions | MLIR, TVM, LLVM, Triton |
| **L3** | The bridge between software and the chip | CUDA runtime, kernel drivers, APIs |
| **L4** | The firmware that boots and controls the device | FreeRTOS, embedded Linux, bootloaders |
| **L5** | How the chip is architected internally | Systolic arrays, HBM memory, NoC |
| **L6** | Writing the chip's logic in hardware code | SystemVerilog, FPGA, verification |
| **L7** | Physically placing circuits on a chip | Place & route, timing, EDA tools |
| **L8** | Sending to a foundry and getting chips back | TSMC process, CoWoS, packaging |

> **L1–L6:** Full hands-on projects throughout this curriculum.
> **L7–L8:** Conceptual with guided labs (OpenROAD, TinyTapeout).

---

## Where Do I Start?

**Pick your entry point based on where you are today:**

```
Coming from software / ML?
  → Start at Phase 1 (C++ and Parallel Computing) then Phase 3 (AI)

Coming from embedded / firmware?
  → Start at Phase 1 (Computer Architecture) then Phase 2 (Embedded Systems)

Already know CUDA and ML frameworks?
  → Jump to Phase 4 (your track: FPGA, Jetson, or ML Compiler)

Targeting chip design?
  → Follow Phase 1 → 2 → 4A → 5F in order
```

---

## The 5-Phase Curriculum

### Phase 1 — Digital Foundations
*Learn the language of hardware. Go from logic gates to writing GPU code.*

| Module | What you'll learn |
|--------|------------------|
| [Digital Design & HDL](Phase%201%20-%20Foundational%20Knowledge/1.%20Digital%20Design%20and%20Hardware%20Description%20Languages/Guide.md) | How digital logic works; write Verilog, simulate circuits |
| [Computer Architecture](Phase%201%20-%20Foundational%20Knowledge/2.%20Computer%20Architecture%20and%20Hardware/Guide.md) | How CPUs and GPUs work internally — pipelines, caches, memory |
| [Operating Systems](Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Guide.md) | Processes, memory, scheduling, device drivers |
| [C++ & Parallel Computing](Phase%201%20-%20Foundational%20Knowledge/4.%20C%2B%2B%20and%20Parallel%20Computing/Guide.md) | SIMD, OpenMP, oneTBB, **CUDA**, ROCm, OpenCL/SYCL |

---

### Phase 2 — Embedded Systems
*Get hands-on with real hardware: microcontrollers, sensors, and embedded Linux.*

| Module | What you'll learn |
|--------|------------------|
| [Embedded Software](Phase%202%20-%20Embedded%20Systems/2.%20Embedded%20Software/Guide.md) | ARM Cortex-M, FreeRTOS, communication buses (SPI/I2C/CAN), power management |
| [Embedded Linux](Phase%202%20-%20Embedded%20Systems/3.%20Embedded%20Linux/Guide.md) | Build custom Linux for embedded devices with Yocto and PetaLinux |

---

### Phase 3 — Artificial Intelligence
*Understand the AI workloads your hardware must run. Two tracks — pick one or both.*

**Core (everyone does these):**

| Module | What you'll learn |
|--------|------------------|
| [Neural Networks](Phase%203%20-%20Artificial%20Intelligence/1.%20Neural%20Networks/Guide.md) | How neural networks learn — backprop, CNNs, transformers from scratch |
| [Deep Learning Frameworks](Phase%203%20-%20Artificial%20Intelligence/2.%20Deep%20Learning%20Frameworks/Guide.md) | micrograd → PyTorch → tinygrad: understand what frameworks actually do |

**Track A — Hardware & Edge AI** *(leads to Phase 4A/B)*

| Module | What you'll learn |
|--------|------------------|
| [Computer Vision](Phase%203%20-%20Artificial%20Intelligence/Track%20A%20-%20Hardware%20and%20Edge%20AI/3.%20Computer%20Vision/Guide.md) | Object detection, segmentation, 3D vision, OpenCV |
| [Sensor Fusion](Phase%203%20-%20Artificial%20Intelligence/Track%20A%20-%20Hardware%20and%20Edge%20AI/4.%20Sensor%20Fusion/Guide.md) | Fuse camera + LiDAR + IMU; Kalman filters, BEVFusion |
| [Voice AI](Phase%203%20-%20Artificial%20Intelligence/Track%20A%20-%20Hardware%20and%20Edge%20AI/5.%20Voice%20AI/Guide.md) | Speech-to-text (Whisper), TTS, wake-word detection |
| [Edge AI & Optimization](Phase%203%20-%20Artificial%20Intelligence/Track%20A%20-%20Hardware%20and%20Edge%20AI/6.%20Edge%20AI%20and%20Model%20Optimization/Guide.md) | Quantization, pruning, deploying models on constrained devices |

**Track B — Agentic AI & ML Engineering** *(leads to Phase 4C / Phase 5)*

| Module | What you'll learn |
|--------|------------------|
| [Agentic AI & GenAI](Phase%203%20-%20Artificial%20Intelligence/Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/3.%20Agentic%20AI%20and%20GenAI/Guide.md) | Build LLM agents, RAG systems, tool-using AI |
| [ML Engineering & MLOps](Phase%203%20-%20Artificial%20Intelligence/Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/4.%20ML%20Engineering%20and%20MLOps/Guide.md) | Training pipelines, model serving, monitoring |
| [LLM Application Development](Phase%203%20-%20Artificial%20Intelligence/Track%20B%20-%20Agentic%20AI%20and%20ML%20Engineering/5.%20LLM%20Application%20Development/Guide.md) | Fine-tuning, RAG architecture, production LLM apps |

---

### Phase 4 — Hardware Deployment & Compilation
*Deploy AI on real chips. Three specialized tracks — choose based on your target role.*

#### Track A — Xilinx FPGA
*Design hardware accelerators and deploy AI on programmable chips.*

| Module | What you'll learn |
|--------|------------------|
| [FPGA Development](Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/1.%20Xilinx%20FPGA%20Development/Guide.md) | Vivado, IP cores, timing constraints, hardware debugging |
| [Zynq MPSoC](Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/2.%20Zynq%20UltraScale%2B%20MPSoC/Guide.md) | Combine ARM CPU + FPGA fabric on one chip |
| [Advanced FPGA Design](Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/3.%20Advanced%20FPGA%20Design/Guide.md) | Clock domain crossing, floorplanning, power |
| [HLS (High-Level Synthesis)](Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/4.%20High-Level%20Synthesis%20%28HLS%29/Guide.md) | Write C++ → get hardware automatically |
| [Runtime & Drivers](Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/5.%20Runtime%20and%20Driver%20Development/Guide.md) | Linux driver for your FPGA, DMA, Vitis AI |
| [Projects](Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/6.%20Projects/Wireless-Video-FPGA.md) | Build a 4K wireless video pipeline end-to-end |

#### Track B — NVIDIA Jetson
*Ship AI products on NVIDIA's embedded GPU platform.*

| Module | What you'll learn |
|--------|------------------|
| [Jetson Platform](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/1.%20Nvidia%20Jetson%20Platform/Guide.md) | JetPack, L4T, GPU on Orin — get up and running |
| [Carrier Board Design](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/2.%20Custom%20Carrier%20Board%20Design%20and%20Bring-Up/Guide.md) | Design your own PCB that hosts a Jetson module |
| [L4T Customization](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/3.%20L4T%20Customization/Guide.md) | Custom Linux kernel, device tree, OTA updates |
| [Firmware (FSP)](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/4.%20FSP%20%28Firmware%20Support%20Package%29%20Customization/Guide.md) | FreeRTOS on the safety co-processor |
| [AI Application Dev](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/5.%20Application%20Development/Guide.md) | ML inference, ROS 2, real-time video on Jetson |
| [Security & OTA](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/6.%20Security%20and%20OTA/Guide.md) | Secure boot, encrypted storage, over-the-air updates |
| [Manufacturing](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/7.%20Compliance%20and%20Manufacturing/Guide.md) | FCC/CE compliance, production flashing, DFM |
| [TensorRT & DLA](Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/8.%20Runtime%20and%20Driver%20Development/Guide.md) | Optimize models for Jetson's GPU and neural accelerator |

#### Track C — ML Compiler
*Learn how AI models are compiled and optimized into chip instructions.*

| Module | What you'll learn |
|--------|------------------|
| [Compiler Fundamentals](Phase%204%20-%20Track%20C%20-%20ML%20Compiler%20and%20Graph%20Optimization/Guide.md) | How MLIR, TVM, and LLVM work; build a custom backend |
| [DL Inference Optimization](Phase%204%20-%20Track%20C%20-%20ML%20Compiler%20and%20Graph%20Optimization/DL%20Inference%20Optimization/Guide.md) | Triton kernels, Flash-Attention, TensorRT-LLM, quantization |

---

### Phase 5 — Specialization
*Go deep in one area. These tracks are ongoing and expand continuously.*

| Track | What you'll specialize in | Guide |
|-------|--------------------------|-------|
| **GPU Infrastructure** | Multi-GPU systems, NVLink, NCCL, AMD ROCm/HIP, MI300X | [→](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/1.%20GPU%20Infrastructure/Guide.md) |
| **High-Performance Computing** | 40+ CUDA-X libraries: cuBLAS, cuDNN, NVSHMEM and more | [→](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/2.%20High%20Performance%20Computing/Guide.md) |
| **Edge AI** | Efficient model architectures, Holoscan, real-time pipelines | [→](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/3.%20Edge%20AI/Guide.md) |
| **Robotics** | ROS 2, Nav2, MoveIt, motion planning | [→](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/4.%20Robotics/Guide.md) |
| **Autonomous Vehicles** | openpilot, BEV perception, functional safety, hardware debug | [→](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/5.%20Autonomous%20Vehicles/Guide.md) |
| **AI Chip Design** | Systolic arrays, dataflow architectures, tinygrad↔hardware, ASIC flow | [→](Phase%205%20-%20Advanced%20Topics%20and%20Specialization/6.%20AI%20Chip%20Design/Guide.md) |

---

## What Jobs Does This Lead To?

| Target Role | Key Layers | Recommended Path |
|-------------|-----------|-----------------|
| **ML Inference Engineer** | L1 | Phase 3 → Phase 4C |
| **Edge AI Engineer** | L1 | Phase 3 Track A → Phase 4B |
| **AI Compiler Engineer** | L2 | Phase 1 → Phase 4C → Phase 5B |
| **GPU Runtime Engineer** | L3 | Phase 1 (CUDA) → Phase 4A/B §Runtime |
| **Firmware / Embedded Engineer** | L4 | Phase 1 → Phase 2 → Phase 4B |
| **AI Accelerator Architect** | L5 | Phase 1 → Phase 4A → Phase 5F |
| **RTL / FPGA Design Engineer** | L6 | Phase 1 (HDL) → Phase 4A |
| **Autonomous Vehicles Engineer** | L1–L4 | Phase 3 Track A → Phase 4B → Phase 5E |
| **AI Hardware Engineer (Full-Stack)** | L1–L6 | Full curriculum — the signature role this roadmap targets |

---

## Reference Projects Used Throughout

| Project | Why it's used |
|---------|--------------|
| [**tinygrad**](https://github.com/tinygrad/tinygrad) | A tiny DL framework (~2,500 lines) — shows exactly how frameworks, compilers, and hardware backends connect |
| [**openpilot**](https://github.com/commaai/openpilot) | Real-world ADAS software — shows how perception, ML, and hardware work together in production |

---

## Additional Resources

- [**Roles & Market Analysis**](Roles%20and%20Market%20Analysis.md) — 23 sub-roles, salary data, job postings, remote %, hiring priorities

---

<div align="center" markdown="1">

**A community-driven educational roadmap for AI hardware engineering.**

[⭐ Star this repo](https://github.com/ai-hpc/ai-hardware-engineer-roadmap) if you find it useful — it helps others discover it.

</div>
