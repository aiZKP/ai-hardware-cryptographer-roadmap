# 1. HPC and DL with Nvidia GPU (Phase 5 Track C)

**Timeline:** 12?24 months (fundamentals and deep dives); 24?48 months for advanced phase.

---

## Basic concepts: what "HPC and DL with Nvidia GPU" means


**HPC** = high-performance computing: solving problems that need massive compute and memory by using many machines (or many GPUs) working together. In AI, "HPC" usually means **large-scale training** and **high-throughput inference** on GPU clusters, not single workstations.

**Why Nvidia GPUs?** Nvidia GPUs are the dominant hardware for training and deploying large models. They offer the best-supported software stack: **CUDA** (programming model and runtime), **cuDNN** and **CUTLASS** (high-performance kernels ? cuDNN for conv/RNN/attention; CUTLASS for customizable GEMM, matrix multiply, and epilogue fusion used by frameworks and custom kernels), **TensorRT** (inference optimization and deployment), and **NCCL** (multi-GPU collectives). Add the fastest inter-GPU links (NVLink, NVSwitch) and the architectures (Hopper, Blackwell) that ML frameworks target first, and you see why AI infrastructure and kernel-level optimization work almost always involves Nvidia GPUs in data centers.

**What this track covers:**

* **Single GPU ? many GPUs** ? From one GPU (e.g. Jetson, which you saw in Phase 4) to **multi-GPU nodes** and **multi-node clusters**. You need to understand how jobs are placed, how data and gradients move, and how to avoid communication becoming the bottleneck.
* **Two main workloads:**
    * **Training** ? One big model, one big dataset; you split work across GPUs (data parallelism, model parallelism, pipeline parallelism). Performance is about throughput (samples/sec) and scaling to hundreds or thousands of GPUs.
    * **Inference** ? Many requests, one (or many) deployed model; you care about latency and throughput under load. At scale this means batching, KV-cache, and often multi-GPU or multi-node serving (e.g. TensorRT-LLM, vLLM).
* **The stack you must understand:**
    * **Hardware:** GPUs (A100, H100/H200, L40S, etc.), NVLink/NVSwitch inside a node, InfiniBand or Ethernet across nodes.
    * **Software:** CUDA, drivers, containers (NGC), orchestrators (Slurm, Kubernetes), and collective libraries (NCCL) for multi-GPU communication.
    * **Storage and I/O:** Getting data to GPUs fast (dataloaders, GPUDirect Storage, high-throughput disks) so the GPU is not waiting.

**Why "Setup" and "DL Inference Optimization" as two parts?**

* **HPC Setup** ? How to *run* and *operate* GPU clusters: provisioning, virtualization, networking, storage, job scheduling, and advanced CUDA/distributed training. You learn the environment and the performance model (where time is spent: compute vs communication vs I/O).
* **DL Inference Optimization** ? How to *optimize* inference itself: graph and operator optimization, writing and tuning kernels (Triton, CUTLASS, attention), compilers, quantization, and production runtimes (TensorRT-LLM, vLLM). This is where you push single-request latency and throughput to the limit.

Together they cover both **infrastructure** (getting the cluster and the workload running correctly and at scale) and **optimization** (getting the most out of each GPU and each request). Read the [Basic concepts in DL Inference Optimization](./DL%20Inference%20Optimization/Guide.md#basic-concepts-read-this-first) for the inference-side vocabulary (batching, KV-cache, speculative decoding, etc.) before or in parallel with the inference units.

### Key terms (used in this track)

*Order: basic ops ? attention ? distributed.*

| Term | Meaning |
|------|--------|
| **Matrix multiply** | Compute A?B for matrices A, B (often plus bias or activation). The core operation in linear layers and most heavy compute. "GEMM" is the standard name in libraries. |
| **GEMM** | **G**eneral **E**lement-wise **M**atrix **M**ultiply: C = ??(A?B) + ??C. The BLAS/cuBLAS/CUTLASS interface for matrix multiply. GPU GEMM kernels (tiling, tensor cores) dominate training and inference time. |
| **Epilogue fusion** | In a GEMM kernel, the **epilogue** is what you do with the result (add bias, ReLU/GELU, write to memory). **Fusion** = doing that in the *same* kernel as the multiply. Saves memory bandwidth and launch overhead; CUTLASS and similar libraries support it. |
| **Attention** | In transformers: each token has **Q, K, V** (from linear layers); *attention* = softmax(Q?K?/?d)?V. Lets the model focus on relevant tokens. The matmuls are heavy for long sequences; optimized attention kernels and KV-cache are critical. |
| **FlashAttention** | A family of **attention kernels** that reduce memory traffic by tiling and keeping Q,K,V in SRAM, and avoid materializing the full Q?K? matrix. Faster and more memory-efficient than naive attention; standard in LLM training and inference (e.g. FlashAttention-2, -3). |
| **KV-cache** | In transformer attention: keys and values for previous tokens are cached so you don't recompute them. **KV-cache** = that cache. Long context ? huge cache ? memory and bandwidth become the bottleneck; paging/sharding and efficient kernels matter. |
| **Data / model / pipeline parallelism** | **Data:** same model on every GPU, different data; sync gradients. **Model:** split the model across GPUs. **Pipeline:** different layers on different GPUs, pass activations in a pipeline. |
| **Collectives** | Multi-GPU operations: **AllReduce** (everyone gets the same sum), **AllGather** (everyone gets all pieces), **ReduceScatter**. Used to sync gradients (data parallel) or exchange activations (model/pipeline parallel). |
| **NCCL** | **N**vidia **C**ollective **C**ommunications **L**ibrary. Implements collectives (AllReduce, AllGather, etc.) for multi-GPU training and inference. Often the bottleneck at scale. |
| **NVLink / NVSwitch** | **NVLink:** high-bandwidth GPU?GPU (and GPU?CPU) link inside a node. **NVSwitch:** switch connecting many GPUs in one node. Much faster than PCIe for GPU?GPU traffic. |
---

## This track has two parts

| Part | Description | Guide |
|------|--------------|-------|
| **HPC Setup** | Fundamentals, virtualization, interconnects, advanced CUDA/distributed training/performance ? plus hardware-specific deep dives (8x H200, L40S, NCCL, CUDA Advanced, GPUDirect Storage) | [HPC Setup ?](./HPC%20Setup/Guide.md) |
| **DL Inference Optimization** | Graph/ops, kernel engineering (Triton, CUTLASS, Flash-Attention), compiler (IR, BEAM), quantization, runtimes. *MTS Kernels?style roles.* | [DL Inference Optimization ?](./DL%20Inference%20Optimization/Guide.md) |

---

## How to use this track

1. **Start with [HPC Setup](./HPC%20Setup/Guide.md)** ? Covers Nvidia GPU HPC fundamentals, virtualization (vGPU, KVM), interconnects and storage (InfiniBand, GDS, Slurm, Kubernetes), and Phase 2 advanced topics (advanced CUDA, distributed training, performance modeling). Use the deep dives (8x H200, L40S, NCCL, CUDA Advanced, GDS) for your target hardware and stack.
2. **Add [DL Inference Optimization](./DL%20Inference%20Optimization/Guide.md)** ? For kernel/inference optimization (e.g. MTS Kernels, DL Inference Optimization Engineer), work through the six units in order: graph/ops ? kernels ? compiler ? quantization ? runtimes ? tinygrad.

**Prerequisite:** Phase 4 (Jetson, TensorRT, CUDA) is assumed for both parts.
