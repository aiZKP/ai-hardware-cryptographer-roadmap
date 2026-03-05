# Orin Nano 8GB — Memory Architecture Deep Dive

> **Scope:** Production-level understanding of how memory works on Jetson Orin Nano 8GB (T234 SoC) — from DRAM initialization through SMMU translation, CMA internals, camera zero-copy pipelines, secure world isolation, and real production debugging.
>
> **Prerequisites:** You should be familiar with the [Orin Nano boot chain](../Guide.md#1-orin-nano-8gb--hardware--boot-chain-internals) and basic Linux memory concepts.

---

## Table of Contents

1. [T234 SoC Memory Architecture Overview](#1-t234-soc-memory-architecture-overview)
2. [Cache Hierarchy](#2-cache-hierarchy)
3. [Boot-Time Memory Initialization](#3-boot-time-memory-initialization)
4. [What the 8GB Really Looks Like](#4-what-the-8gb-really-looks-like)
5. [Linux Memory Zones](#5-linux-memory-zones)
6. [Buddy Allocator Internals](#6-buddy-allocator-internals)
7. [CMA — Contiguous Memory Allocator](#7-cma--contiguous-memory-allocator)
8. [How to Resize CMA](#8-how-to-resize-cma)
9. [SMMU (IOMMU) — Real Translation Path](#9-smmu-iommu--real-translation-path)
10. [Camera → ISP → CUDA Zero-Copy Path](#10-camera--isp--cuda-zero-copy-path)
11. [GPU Memory Management](#11-gpu-memory-management)
12. [DLA Memory Path](#12-dla-memory-path)
13. [OP-TEE and Secure Memory](#13-op-tee-and-secure-memory)
14. [Multi-Camera Memory Planning](#14-multi-camera-memory-planning)
15. [Performance Monitoring and Profiling](#15-performance-monitoring-and-profiling)
16. [Production Debug Checklist](#16-production-debug-checklist)
17. [Common Production Issues and Solutions](#17-common-production-issues-and-solutions)
18. [References](#18-references)

---

## 1. T234 SoC Memory Architecture Overview

The Tegra234 SoC in Orin Nano 8GB uses a **unified memory architecture** — CPU, GPU, DLA, and all accelerators share a single 8GB LPDDR5 pool. There is no discrete GPU memory.

```
                        8GB LPDDR5
                    ┌───────────────┐
                    │               │
       ┌────────────┤  Memory       ├────────────┐
       │            │  Controllers  │            │
       │            │  (MC)         │            │
       │            └───────┬───────┘            │
       │                    │                    │
  ┌────┴────┐         ┌────┴────┐         ┌─────┴────┐
  │ CPU     │         │ GPU     │         │ DLA/ISP  │
  │ A78AE   │         │ Ampere  │         │ NVENC    │
  │ cluster │         │ 1024    │         │ NVDEC    │
  │         │         │ cores   │         │ CSI/VI   │
  └─────────┘         └─────────┘         └──────────┘
```

### Memory Controllers

T234 has multiple memory controller (MC) channels connected to the LPDDR5 interface. The MC handles:

* Arbitration between CPU, GPU, and all accelerators
* Bandwidth partitioning (configurable via BPMP firmware)
* Quality-of-Service (QoS) priorities for latency-sensitive engines (e.g., display, camera)

### Unified Virtual Memory (UVM)

CUDA on Jetson supports unified virtual addressing (UVA). A pointer returned by `cudaMallocManaged()` is valid on both CPU and GPU — the runtime handles coherence via page faults and migration. However, for latency-critical paths (camera, inference), explicit allocation with `cudaMalloc()` or NVMM buffers avoids migration overhead.

---

## 2. Cache Hierarchy

Understanding caches is critical for optimizing memory-bound inference workloads.

```
CPU Core (A78AE)
  L1I: 64KB per core (instruction)
  L1D: 64KB per core (data)
  L2:  256KB per core
  L3:  2MB shared (cluster-level)

GPU (Ampere)
  L1 / Shared Memory: per-SM (configurable)
  L2: shared across all SMs
```

### Key Implications

* **CPU L3 is small (2MB)** — large working sets spill to DRAM quickly. This matters for pre/post-processing on CPU.
* **GPU L2** is shared — inference kernels compete for cache. Kernel tiling and occupancy directly affect L2 hit rates.
* **No hardware cache coherence between CPU and GPU** — this is why explicit synchronization (e.g., `cudaStreamSynchronize`) is required, and why zero-copy via NVMM/DMA-BUF is preferred over CPU-GPU memcpy.

---

## 3. Boot-Time Memory Initialization

On Orin Nano, memory is set up in stages before Linux ever runs.

### MB1 — DRAM Training

MB1 (loaded from QSPI NOR) performs LPDDR5 training:

* Calibrates timing parameters for each DRAM channel
* Stores training data for fast-boot on subsequent boots
* Configures ECC if enabled

If DRAM training fails, the board does not boot — no display, no serial output.

### MB2 — Carveout Setup

MB2 reads the device tree and reserves memory regions (carveouts) for firmware processors:

* **BPMP** — power management firmware
* **SPE** — safety processor
* **RCE** — camera real-time engine
* **OP-TEE** — secure world
* **VPR** — Video Protected Region (DRM)

These carveouts are removed from the Linux-visible memory map before the kernel starts.

### UEFI — Memory Map Handoff

UEFI constructs the EFI memory map describing which regions are:

* Usable by Linux (conventional memory)
* Reserved (firmware, carveouts)
* ACPI/runtime services

Linux receives this map via `efi_memmap` and initializes its memory subsystem accordingly.

### Device Tree Memory Nodes

The DTB loaded by UEFI defines:

```dts
memory@80000000 {
    device_type = "memory";
    reg = <0x0 0x80000000 0x0 0x70000000>,   /* Region 1 */
          <0x0 0xf0200000 0x0 0x0fe00000>;   /* Region 2 */
};
```

Carveouts are defined separately:

```dts
reserved-memory {
    #address-cells = <2>;
    #size-cells = <2>;
    ranges;

    bpmp_carveout: bpmp {
        compatible = "nvidia,bpmp-shmem";
        reg = <0x0 0x40000000 0x0 0x200000>;
        no-map;
    };
};
```

The `no-map` property means Linux cannot touch this memory.

If DTB carveouts are wrong, the system may crash before `start_kernel()` or firmware processors will malfunction.

---

## 4. What the 8GB Really Looks Like

On Orin Nano, DRAM is initialized by MB1 → MB2 → UEFI → Linux. Very early boot reserves memory **before Linux even starts**.

### Typical `/proc/iomem` Layout (Conceptual)

```
00000000-000fffff : Reserved (Boot ROM, vectors)
00100000-3fffffff : System RAM
40000000-4fffffff : Reserved (VPR, carveouts)
50000000-57ffffff : CMA
58000000-ffffffff : System RAM
```

### Reserved Regions

| Region                       | Purpose                         |
|------------------------------|---------------------------------|
| VPR (Video Protected Region) | DRM / secure video playback     |
| SPE carveout                 | Safety processor firmware       |
| BPMP carveout                | Power management firmware       |
| OP-TEE secure RAM            | Trusted execution environment   |
| CMA                          | Contiguous DMA for camera / GPU |

These are defined in the **Device Tree (DTB)** loaded by UEFI.

### Inspecting the Real Memory Map

```bash
# Full memory map
cat /proc/iomem

# Linux-visible memory summary
cat /proc/meminfo

# CMA-specific stats
grep Cma /proc/meminfo
```

Example `/proc/meminfo` output on Orin Nano 8GB:

```
MemTotal:        7633536 kB    ← Not 8GB! Carveouts took the rest
CmaTotal:         786432 kB    ← 768MB reserved for CMA
CmaFree:          524288 kB    ← Currently unused CMA
```

The ~400MB difference between 8GB and `MemTotal` is consumed by firmware carveouts, kernel code, and reserved regions.

---

## 5. Linux Memory Zones

Linux divides physical memory into zones:

| Zone        | Address Range    | Purpose                           |
|-------------|------------------|-----------------------------------|
| ZONE_DMA    | 0 – 16MB         | Legacy ISA DMA (mostly unused)    |
| ZONE_DMA32  | 0 – 4GB          | 32-bit addressable DMA            |
| ZONE_NORMAL | Above 4GB        | General-purpose allocations        |

Orin Nano mostly uses **ZONE_NORMAL** since LPDDR5 is mapped above the 4GB boundary for most of the address space.

You can inspect zone state:

```bash
cat /proc/zoneinfo
```

Key fields:

* `free` — pages currently free in this zone
* `min` / `low` / `high` — watermarks that trigger reclaim
* `nr_free_pages` — total free across all orders

When `free` drops below `min`, the kernel starts aggressive reclaim (kswapd, direct reclaim). On Jetson with large camera buffers, this can cause latency spikes.

---

## 6. Buddy Allocator Internals

The Linux buddy allocator manages free pages in powers of two:

```
Order 0 =   4KB  (1 page)
Order 1 =   8KB  (2 pages)
Order 2 =  16KB  (4 pages)
Order 3 =  32KB  (8 pages)
...
Order 9 =   2MB  (512 pages)
Order 10 =  4MB  (1024 pages)
```

### How Allocation Works

When a driver requests 64KB (order 4):

1. Check the order-4 free list
2. If empty, split a higher-order block (e.g., order 5 → two order 4 blocks)
3. Return one block, keep the other as free

### How Freeing Works

When a block is freed:

1. Check if the adjacent "buddy" block is also free
2. If yes, merge into a higher-order block
3. Repeat up the chain

### Inspecting Fragmentation

```bash
cat /proc/buddyinfo
```

Example output:

```
Node 0, zone   Normal  1024  512  256  128  64  32  16  8  4  2  1
```

Each number is the count of free blocks at that order (0 through 10). If high-order columns show `0`, the system is fragmented — large contiguous allocations will fail even if total free memory is sufficient.

### Why This Matters on Jetson

Camera and GPU drivers need large contiguous allocations. A fragmented system with plenty of free memory but no high-order blocks will fail to allocate buffers — this is why CMA exists.

---

## 7. CMA — Contiguous Memory Allocator

CMA is **not** separate memory. It is:

> A reserved movable region inside normal RAM where Linux can guarantee physically contiguous allocations.

Linux marks CMA pages as **MIGRATE_CMA**. When not used for DMA, these pages hold movable data (e.g., userspace pages). When a contiguous allocation is needed, the kernel migrates those pages elsewhere.

### Allocation Flow

When a driver calls `dma_alloc_contiguous()`:

1. Find free pages inside the CMA region
2. Use the buddy allocator to locate a contiguous block
3. Compact memory if needed (migrate movable pages out of the way)
4. Map the allocation into the SMMU for the requesting device
5. Return the DMA address (IOVA) and kernel virtual address

### Why Long-Running Systems Fail

Over time, memory fragmentation builds up:

```
CMA region:
[used][free][used][free][used][free][used]
```

Even if total free CMA is sufficient, no single contiguous chunk is large enough. The camera fails with:

```
Failed to allocate buffer
```

This is a classic embedded production issue. Mitigations:

* **Right-size CMA** for your workload (see next section)
* **Pre-allocate buffers at boot** and reuse them
* **Monitor CMA fragmentation** in production (see Section 16)
* **Use NVMM buffer pools** rather than allocating/freeing per-frame

---

## 8. How to Resize CMA

### Method 1: Kernel Command Line

Edit `/boot/extlinux/extlinux.conf`:

```
APPEND ... cma=1024M
```

This sets CMA to 1GB.

### Method 2: Device Tree

```dts
reserved-memory {
    linux,cma {
        compatible = "shared-dma-pool";
        reusable;
        size = <0x0 0x40000000>;   /* 1GB */
        linux,cma-default;
    };
};
```

Reflash DTB after modification.

### Sizing Guidelines

| Workload                      | Recommended CMA |
|-------------------------------|-----------------|
| Single camera, light inference | 256–512MB       |
| Single 4K camera + TensorRT   | 512–768MB       |
| Multi-camera AI pipeline       | 768MB–1GB       |
| Multi-camera + large models    | 1–1.5GB         |

Trade-offs:

* Too large CMA = less memory for userspace, model weights, and CUDA
* Too small CMA = camera buffer allocation failures under load
* 768MB–1GB is common for production multi-camera AI workloads

### Verify After Change

```bash
grep Cma /proc/meminfo
```

Confirm `CmaTotal` matches your configured size.

---

## 9. SMMU (IOMMU) — Real Translation Path

Orin Nano uses **ARM SMMU v2** (System Memory Management Unit). Every device that performs DMA goes through the SMMU.

### Translation Flow

```
Device (GPU, CSI, ISP, etc.)
   ↓
IOVA (I/O Virtual Address — what the device sees)
   ↓
SMMU page tables (owned by Linux kernel)
   ↓
Physical DRAM (actual memory location)
```

Devices **never see physical RAM directly**. The kernel:

1. Allocates memory (from CMA or normal pages)
2. Maps those physical pages into the SMMU page tables for the target device
3. Gives the device an IOVA
4. The device performs DMA using the IOVA

### Why This Architecture Is Powerful

* **Device isolation** — a misbehaving device cannot corrupt another device's memory
* **No rogue DMA** — unmapped accesses cause SMMU faults (caught and logged)
* **Shared buffers** — multiple devices can map the same physical pages at different IOVAs
* **Scatter-gather as contiguous** — physically scattered pages appear contiguous to the device

This is why GPU and camera can share buffers safely without copies.

### SMMU Page Table Structure

ARM SMMU uses multi-level page tables similar to CPU MMU:

```
Stream Table Entry (per device/stream ID)
   ↓
Context Descriptor
   ↓
Level 1 Page Table (covers large VA range)
   ↓
Level 2 Page Table
   ↓
Level 3 Page Table (4KB granule)
   ↓
Physical Page
```

Each device has its own stream ID and its own set of page tables, providing full isolation.

### SMMU Fault Debugging

When a device accesses an unmapped IOVA:

```bash
dmesg | grep smmu
```

Example fault:

```
arm-smmu 12000000.iommu: Unhandled context fault: iova=0x1234000, fsynr=0x11
```

This means the device tried to access IOVA `0x1234000` but no mapping existed. Common causes:

* Buffer freed while device still referencing it
* Incorrect DMA-BUF import
* Driver bug in IOVA mapping

---

## 10. Camera → ISP → CUDA Zero-Copy Path

This is the critical data path for real-time AI vision on Orin Nano.

### Step 1 — Sensor Capture

```
MIPI CSI-2 camera sensor
    ↓ (serial lanes)
NVCSI controller (deserializes)
    ↓
VI (Video Input — captures frames)
    ↓
ISP (Image Signal Processor — debayer, denoise, tone-map)
```

### Step 2 — ISP Output to CMA

ISP writes processed frames into **CMA memory**. CMA is required because ISP needs physically contiguous buffers for DMA. The ISP hardware has no scatter-gather capability.

### Step 3 — DMA-BUF Export

The V4L2 driver exports the buffer as a **DMA-BUF** file descriptor. Once exported, the same physical memory can be imported by any DMA-BUF-aware consumer:

* CUDA (via `cudaExternalMemory`)
* NvBufSurface (NVIDIA multimedia buffer API)
* GStreamer (via `nvv4l2camerasrc`)
* DeepStream (via source bin)

### Step 4 — GPU Access via SMMU

The GPU imports the DMA-BUF and maps it through its own SMMU context:

```
GPU IOVA → SMMU → Same physical CMA pages that ISP wrote to
```

**No memcpy.** This is true zero-copy. The GPU reads the exact same physical memory the ISP wrote, just mapped at a different virtual address through a different SMMU stream.

### Full Zero-Copy Pipeline

```
Sensor → NVCSI → VI → ISP → CMA buffer
                                ↓ (DMA-BUF export)
                          ┌─────┴─────┐
                          │           │
                     GPU (CUDA)   Display
                     via SMMU     via SMMU
                          │
                     TensorRT
                     inference
```

This is how real-time 4K vision runs efficiently on Orin Nano without saturating memory bandwidth.

### Memory Flow Summary

| Stage       | Memory Type        | Allocated By        |
|-------------|--------------------|---------------------|
| CSI/VI      | CMA (contiguous)   | V4L2 / nvargus      |
| ISP output  | CMA (same buffer)  | Reused from VI      |
| GPU input   | DMA-BUF import     | CUDA / NvBufSurface |
| GPU output  | CUDA device memory | cudaMalloc          |

---

## 11. GPU Memory Management

### NVMM (NVIDIA Multimedia Memory)

NVMM is NVIDIA's buffer management layer for multimedia pipelines (camera, encode, decode). Key properties:

* Buffers are allocated from CMA or carved-out memory
* Accessible by all NVIDIA engines (ISP, GPU, VIC, NVENC, NVDEC)
* Zero-copy between engines via DMA-BUF
* Managed by `libnvbufsurface`

NVMM is the foundation for GStreamer and DeepStream zero-copy pipelines.

### CUDA Memory on Jetson

On Jetson (unified memory architecture), CUDA memory types behave differently than on discrete GPUs:

| API                    | Where Memory Lives  | CPU Accessible? | GPU Accessible? | Notes                          |
|------------------------|---------------------|-----------------|-----------------|--------------------------------|
| `cudaMalloc`           | DRAM (pinned)       | No              | Yes             | Fastest for GPU-only access    |
| `cudaMallocManaged`    | DRAM (migrating)    | Yes             | Yes             | Page-fault based, higher latency |
| `cudaMallocHost`       | DRAM (pinned)       | Yes             | Yes             | Good for CPU↔GPU shared data   |
| DMA-BUF import         | CMA / NVMM          | Yes             | Yes             | Zero-copy from camera/decoder  |

For inference pipelines, prefer:

* **DMA-BUF import** for camera frames (zero-copy)
* **`cudaMalloc`** for model weights and scratch memory
* **`cudaMallocHost`** for CPU pre/post-processing buffers shared with GPU

Avoid `cudaMallocManaged` in latency-critical paths — page faults add unpredictable latency.

### GPU Memory Pressure

Monitor GPU memory usage:

```bash
# Total GPU memory (same as system memory on Jetson)
nvidia-smi  # or tegrastats

# Detailed CUDA memory
python3 -c "import torch; print(torch.cuda.memory_summary())"
```

On Orin Nano 8GB, GPU and CPU compete for the same 8GB. A large model loaded in CUDA reduces memory available for camera buffers, CMA, and OS. Memory planning is essential.

---

## 12. DLA Memory Path

Orin Nano 8GB includes **1 DLA (Deep Learning Accelerator)** capable of up to 10 TOPS (INT8).

### How DLA Accesses Memory

DLA uses the same unified DRAM but has its own SMMU stream:

```
DLA engine
   ↓
DLA IOVA
   ↓
SMMU (DLA stream context)
   ↓
Physical DRAM
```

### TensorRT DLA Execution

When TensorRT runs layers on DLA:

1. TensorRT allocates input/output buffers in DRAM
2. Maps them into DLA's SMMU context
3. Programs DLA registers (layer config, weights pointer, I/O pointers)
4. DLA executes the layer, reading weights and input from DRAM, writing output to DRAM
5. If next layer runs on GPU, the same output buffer is mapped into GPU's SMMU — zero-copy handoff

### DLA vs GPU Memory Trade-offs

| Aspect          | GPU                        | DLA                         |
|-----------------|----------------------------|-----------------------------|
| Memory bandwidth | High (shared with CPU)    | Lower (limited ports)       |
| Precision       | FP32, FP16, INT8           | FP16, INT8 only             |
| Layer support   | All TensorRT layers        | Subset (conv, pool, etc.)   |
| Power           | Higher                     | Much lower per TOPS         |

Running layers on DLA frees GPU for other tasks and reduces power consumption — critical for edge/battery systems.

---

## 13. OP-TEE and Secure Memory

Orin Nano supports ARM TrustZone with two execution worlds:

```
Normal World (Linux, CUDA, all userspace)
─────────────────────────────────────────
Secure World (OP-TEE Trusted OS)
```

### Secure Memory Carveout

A region of DRAM is carved out exclusively for the secure world:

* **Not mapped in Linux** — `cat /proc/iomem` will not show it
* **Protected by TrustZone** — hardware prevents normal-world access
* **Not accessible by SMMU** — even DMA from devices cannot reach it

Used for:

* DRM key storage and content decryption
* Secure boot chain validation at runtime
* Cryptographic services (hardware-accelerated)
* Secure storage (encrypted key material)

### Memory Impact

If the OP-TEE carveout is configured too large, Linux usable RAM shrinks. On a memory-constrained 8GB system, every MB counts. Default carveouts are typically 16–64MB.

### Interacting with OP-TEE

Linux communicates with OP-TEE via the TEE subsystem:

```bash
# Check if OP-TEE is running
ls /dev/tee*

# Typical devices
/dev/tee0        # TEE device
/dev/teepriv0    # Privileged TEE device
```

Applications use the OP-TEE client library (`libteec`) to call Trusted Applications (TAs) in the secure world.

---

## 14. Multi-Camera Memory Planning

Production Jetson systems often run 2–6 cameras simultaneously. Memory planning is critical.

### Per-Camera Memory Budget

For a single 1080p camera at 30 FPS:

| Component                  | Memory Per Frame | Frames Buffered | Total      |
|----------------------------|-----------------|-----------------|------------|
| CSI/VI capture buffer      | ~6MB (RAW10)    | 4 (ring)        | ~24MB      |
| ISP output (NV12)          | ~3MB            | 4 (ring)        | ~12MB      |
| CUDA inference input       | ~3MB            | 2               | ~6MB       |
| **Total per camera**       |                 |                 | **~42MB**  |

For 4K (3840x2160):

| Component                  | Memory Per Frame | Frames Buffered | Total      |
|----------------------------|-----------------|-----------------|------------|
| CSI/VI capture buffer      | ~24MB (RAW10)   | 4 (ring)        | ~96MB      |
| ISP output (NV12)          | ~12MB           | 4 (ring)        | ~48MB      |
| CUDA inference input       | ~12MB           | 2               | ~24MB      |
| **Total per camera**       |                 |                 | **~168MB** |

### System Memory Budget (4-camera 1080p Example)

| Component                    | Memory    |
|------------------------------|-----------|
| OS + kernel + services       | ~500MB    |
| Camera buffers (4 cameras)   | ~168MB    |
| CMA reserved                 | ~768MB    |
| TensorRT model (YOLOv8-S)    | ~100MB    |
| CUDA runtime + scratch       | ~200MB    |
| Firmware carveouts            | ~400MB    |
| **Remaining for userspace**  | **~5.5GB** |

With larger models or 4K cameras, this budget tightens significantly. Plan and measure before deployment.

### Buffer Pool Strategy

For production systems:

1. **Pre-allocate all buffers at startup** — avoid runtime allocation/free cycles
2. **Use fixed-size buffer pools** — NVMM pools or V4L2 REQBUFS with fixed count
3. **Pin buffers** — prevent kernel from swapping or migrating them
4. **Monitor fragmentation** — log CMA and buddy state periodically

---

## 15. Performance Monitoring and Profiling

### tegrastats

NVIDIA's real-time system monitor:

```bash
tegrastats --interval 1000
```

Output includes:

* RAM usage (used/total)
* CPU utilization per core
* GPU utilization percentage
* GPU frequency
* Temperature (CPU, GPU, board)
* Power consumption per rail

### /proc/meminfo

Key fields for Jetson memory analysis:

```bash
cat /proc/meminfo
```

| Field        | What It Tells You                              |
|--------------|------------------------------------------------|
| MemTotal     | Total Linux-visible RAM (after carveouts)      |
| MemAvailable | Estimated available memory for new allocations |
| CmaTotal     | Total CMA region size                          |
| CmaFree      | Free CMA memory                                |
| Slab         | Kernel slab allocator usage                    |
| Mapped       | Memory-mapped file pages                       |

### /proc/buddyinfo

Shows fragmentation per zone and order:

```bash
cat /proc/buddyinfo
```

Healthy system: numbers across all orders.
Fragmented system: high counts at order 0–2, zeros at order 6+.

### Nsight Systems

For profiling CUDA + camera + inference together:

```bash
nsys profile --trace=cuda,nvtx,osrt ./my_inference_app
```

Shows timeline of:

* CUDA kernel launches
* Memory allocations and transfers
* CPU/GPU synchronization points
* OS runtime events

### SMMU and DMA Debugging

```bash
# SMMU faults
dmesg | grep smmu

# IOMMU groups (which devices share SMMU context)
ls /sys/kernel/iommu_groups/*/devices/

# DMA-BUF usage
cat /sys/kernel/debug/dma_buf/bufinfo
```

---

## 16. Production Debug Checklist

When camera or inference randomly fails after hours of operation:

### Step 1 — Check Overall Memory

```bash
cat /proc/meminfo | grep -E "MemTotal|MemAvailable|CmaTotal|CmaFree"
```

If `MemAvailable` is very low, the system is under memory pressure. If `CmaFree` is low, camera buffers may fail to allocate.

### Step 2 — Check Fragmentation

```bash
cat /proc/buddyinfo
```

If high-order columns (order 6+) show `0`, memory is fragmented. Even with free memory, large contiguous allocations will fail.

### Step 3 — Check CMA Usage

```bash
grep Cma /proc/meminfo
```

Compare `CmaFree` to your expected per-frame allocation size. If `CmaFree` is less than the largest single allocation needed, allocation will fail.

### Step 4 — Check for SMMU Faults

```bash
dmesg | grep smmu
```

SMMU fault example:

```
arm-smmu 12000000.iommu: Unhandled context fault: iova=0x1234000, fsynr=0x11
```

This means a device accessed an unmapped IOVA — typically a use-after-free bug or incorrect DMA-BUF handling.

### Step 5 — Check for OOM Events

```bash
dmesg | grep -i "out of memory\|oom"
```

The OOM killer may have terminated a process. Check which process was killed and why.

### Step 6 — Check Thermal Throttling

```bash
cat /sys/devices/virtual/thermal/thermal_zone*/temp
cat /sys/devices/virtual/thermal/thermal_zone*/type
```

If temperature exceeds limits, the system throttles clocks. This can cause inference to miss deadlines, buffers to queue up, and memory to grow.

---

## 17. Common Production Issues and Solutions

### CMA Exhaustion

**Symptom:** `Failed to allocate buffer` after hours of operation.

**Cause:** CMA fragmentation from repeated alloc/free cycles.

**Solutions:**
* Pre-allocate buffer pools at startup; never free them
* Increase CMA size (see Section 8)
* Use NVMM buffer pools with fixed count
* Periodically trigger compaction: `echo 1 > /proc/sys/vm/compact_memory`

### SMMU Faults Under Load

**Symptom:** `Unhandled context fault` in dmesg, frames dropped.

**Cause:** Buffer freed while device still performing DMA, or race condition in DMA-BUF handling.

**Solutions:**
* Ensure proper synchronization before freeing DMA-BUF
* Check refcounting on shared buffers
* Use V4L2 QBUF/DQBUF properly (don't access buffer while queued)

### OOM Killer on Jetson

**Symptom:** Application killed unexpectedly; `dmesg` shows OOM.

**Cause:** Combined GPU + CPU + camera memory exceeds available RAM.

**Solutions:**
* Reduce model size (quantize to INT8, prune)
* Reduce camera buffer count or resolution
* Offload layers to DLA (frees GPU memory for other uses)
* Monitor memory budget and set process memory limits

### Thermal Throttling Causing Latency Spikes

**Symptom:** Inference FPS drops periodically; `tegrastats` shows clock reduction.

**Cause:** SoC temperature exceeds thermal limits.

**Solutions:**
* Add active cooling (fan, heatsink)
* Use `nvpmodel` to set appropriate power mode
* Reduce workload or duty cycle
* Check ambient temperature and enclosure ventilation

### Memory Leak in Long-Running Inference

**Symptom:** `MemAvailable` steadily decreases over hours/days.

**Cause:** CUDA memory not freed, DMA-BUF not closed, Python object accumulation.

**Solutions:**
* Use `torch.cuda.empty_cache()` periodically if using PyTorch
* Ensure every `cudaMalloc` has a matching `cudaFree`
* Close DMA-BUF file descriptors after use
* Profile with `cuda-memcheck` or Nsight Compute

---

## 18. References

* [NVIDIA Jetson Linux Developer Guide](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/) — official documentation covering boot, memory, drivers
* [Jetson Orin Nano Datasheet](https://developer.nvidia.com/embedded/jetson-orin-nano) — hardware specifications
* [ARM SMMU Architecture Specification](https://developer.arm.com/documentation/ihi0070/) — SMMU v3 reference (SMMU v2 subset)
* [Linux CMA Documentation](https://www.kernel.org/doc/html/latest/mm/cma.html) — kernel CMA subsystem
* [Linux Buddy Allocator](https://www.kernel.org/doc/html/latest/mm/page_allocator.html) — page allocator internals
* [OP-TEE Documentation](https://optee.readthedocs.io/) — Trusted Execution Environment
* [GStreamer on Jetson](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/Multimedia/AcceleratedGstreamer.html) — hardware-accelerated multimedia pipelines
* [TensorRT DLA Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/) — DLA integration with TensorRT
* Main guide: [Nvidia Jetson Platform Guide](../Guide.md)
