# Nvidia Jetson Platform — Practical Complete Guide

> **Primary hardware:** Jetson Orin Nano 8GB Developer Kit
> **Goal:** Go from unboxed hardware to a production-quality AI pipeline with ROS2 integration, sensor fusion, optimized inference, OTA updates, and hardened security.

---

## Table of Contents

1. [Jetson Hardware Overview](#1-jetson-hardware-overview)
2. [Installing JetPack — Step by Step](#2-installing-jetpack--step-by-step)
3. [Upgrading JetPack on Orin Nano 8GB](#3-upgrading-jetpack-on-orin-nano-8gb)
4. [Benefits of Higher JetPack Versions](#4-benefits-of-higher-jetpack-versions)
5. [Porting AI Models to Jetson](#5-porting-ai-models-to-jetson)
6. [ROS2 Integration](#6-ros2-integration)
7. [Optimizing AI Inference](#7-optimizing-ai-inference)
8. [End-to-End AI Pipeline: Sensors → Inference → Control](#8-end-to-end-ai-pipeline-sensors--inference--control)
9. [LiDAR, Camera, IMU — Practical Integration](#9-lidar-camera-imu--practical-integration)
10. [Power and Thermal Management in Practice](#10-power-and-thermal-management-in-practice)
11. [OTA Update Best Practices](#11-ota-update-best-practices)
12. [Security Hardening](#12-security-hardening)
13. [Projects](#13-projects)
14. [Resources](#14-resources)

---

## 1. Jetson Hardware Overview

### Orin Nano 8GB vs the Orin Family

| Module              | CPU           | GPU         | RAM   | AI TOPS | Power  | Use Case              |
|---------------------|---------------|-------------|-------|---------|--------|-----------------------|
| Orin Nano 4GB       | 6-core A78AE  | 512-core A  | 4GB   | 20      | 5–10W  | Light inference       |
| **Orin Nano 8GB**   | **6-core A78AE** | **1024-core A** | **8GB** | **40** | **5–15W** | **This guide** |
| Orin NX 8GB         | 6-core A78AE  | 1024-core A | 8GB   | 70      | 10–20W | Mid-range robotics    |
| Orin NX 16GB        | 8-core A78AE  | 1024-core A | 16GB  | 100     | 10–25W | Heavy inference       |
| AGX Orin 32GB       | 12-core A78AE | 2048-core A | 32GB  | 200     | 15–60W | Autonomous vehicles   |
| AGX Orin 64GB       | 12-core A78AE | 2048-core A | 64GB  | 275     | 15–60W | Full AV systems       |

### Orin Nano 8GB Key Hardware

```
CPU:  6× Arm Cortex-A78AE @ up to 1.5 GHz
GPU:  1024× CUDA cores (Ampere architecture)
      32× Tensor Cores
DLA:  1× Deep Learning Accelerator (up to 10 TOPS)
RAM:  8GB LPDDR5 (shared between CPU + GPU)
Storage: microSD + M.2 NVMe slot (2280)
I/O:
  1× USB 3.2 Gen2 Type-A
  1× USB 3.2 Gen2 Type-C (DisplayPort alt mode)
  1× Gigabit Ethernet
  40-pin GPIO header (I2C, SPI, UART, PWM, I2S)
  M.2 Key M (NVMe SSD)
  M.2 Key E (WiFi/BT)
  Camera connector (CSI-2, 2-lane)
```

### Memory Architecture (Critical for AI)

The Orin Nano uses **unified memory** — CPU and GPU share the same physical LPDDR5 pool:

```
CPU processes ←──────────────── 8GB LPDDR5 ────────────────→ GPU processes
                          No PCIe transfer needed!
                    Tensors live in unified address space
```

This means zero-copy GPU inference: camera frames captured by CPU stay in place and the GPU reads them directly without copying. This is a major advantage over discrete GPU systems.

---

## 2. Installing JetPack — Step by Step

### What is JetPack?

JetPack is NVIDIA's full SDK stack for Jetson. It bundles:
- **L4T (Linux for Tegra)**: Ubuntu-based OS with Jetson kernel + drivers
- **CUDA Toolkit**: GPU programming runtime
- **cuDNN**: GPU-accelerated deep learning primitives
- **TensorRT**: optimized inference engine
- **VPI** (Vision Programming Interface): hardware-accelerated CV
- **DeepStream**: video analytics pipeline SDK
- **Multimedia API**: camera/video capture

### JetPack Version → Software Stack

| JetPack | Ubuntu | CUDA  | cuDNN | TensorRT | L4T     |
|---------|--------|-------|-------|----------|---------|
| 5.1.4   | 20.04  | 11.4  | 8.6   | 8.6      | 35.6.x  |
| 6.0     | 22.04  | 12.2  | 9.0   | 10.0     | 36.3.x  |
| 6.1     | 22.04  | 12.6  | 9.3   | 10.3     | 36.4.x  |
| 6.2     | 22.04  | 12.8  | 9.7   | 10.7     | 36.5.x  |

### Method 1: SD Card Image (Easiest — Dev Kit Only)

The Orin Nano Developer Kit can boot from microSD. This is the fastest way to get started.

```bash
# Step 1: Download the SD card image from NVIDIA
# Go to: developer.nvidia.com/embedded/jetpack
# Select: Jetson Orin Nano Developer Kit → JetPack 6.x → SD Card Image
# File: jp6x-orin-nano-sd-card-image.zip  (~15GB)

# Step 2: Flash with Balena Etcher (GUI) or dd (CLI)
# Using dd (Linux):
unzip jp6x-orin-nano-sd-card-image.zip
sudo dd if=jp6x-orin-nano-sd-card-image.img of=/dev/sdX bs=1M status=progress
# Replace /dev/sdX with your SD card device (check with lsblk)

# Step 3: Insert SD card, connect HDMI + keyboard + power
# System boots into Ubuntu 22.04 setup wizard
```

### Method 2: SDK Manager (Full Control, Host PC Required)

SDK Manager gives you precise control over which JetPack components to install and enables NVMe boot configuration.

**Requirements:**
- Host PC running Ubuntu 20.04 or 22.04 (native, not VM recommended)
- USB-C cable (data, not charge-only) between host PC and Jetson USB-C port
- Jetson Orin Nano Dev Kit in **recovery mode**

```bash
# ── HOST PC SETUP ──────────────────────────────────────────

# Step 1: Download SDK Manager
# developer.nvidia.com/sdk-manager → download .deb
sudo dpkg -i sdkmanager_*.deb
sudo apt-get install -f   # fix any dependency issues

# Step 2: Launch SDK Manager
sdkmanager

# Step 3: Log in with NVIDIA Developer account

# ── JETSON: ENTER RECOVERY MODE ────────────────────────────
# On Orin Nano Dev Kit:
#   1. Power OFF the board
#   2. Hold the RECOVERY button (3-pin header, middle pin to GND)
#   3. While holding, press POWER button
#   4. Release RECOVERY button after 2 seconds
#   5. Connect USB-C from host PC to Jetson USB-C port

# Verify Jetson is detected in recovery mode:
lsusb | grep NVIDIA
# Should show: NVIDIA Corp. APX

# ── SDK MANAGER STEPS ──────────────────────────────────────
# 1. Select: Jetson Orin Nano [8GB module]
# 2. JetPack version: 6.x (latest)
# 3. Select target components:
#    ✓ Jetson Linux (L4T)         — required
#    ✓ CUDA Toolkit               — required for GPU
#    ✓ cuDNN                      — required for deep learning
#    ✓ TensorRT                   — required for optimized inference
#    ✓ VPI                        — camera/vision pipeline
#    ✓ DeepStream                 — (optional, video analytics)
# 4. Click Continue, accept licenses
# 5. Flashing starts — takes 10–20 minutes
```

### Method 3: NVMe Boot (Recommended for Production)

microSD is slow (max ~100 MB/s read) and has limited write endurance. NVMe SSD gives:
- Read: ~3500 MB/s (35× faster than SD)
- Much higher write endurance
- Faster model loading and dataset access

```bash
# After initial boot from SD card:

# Step 1: Install NVMe SSD in M.2 Key M slot (2280 form factor)
# (power off first, insert SSD, power on)

# Step 2: On Jetson, clone SD card to NVMe
sudo apt-get install pv
sudo dd if=/dev/mmcblk0 | pv | sudo dd of=/dev/nvme0n1 bs=4M status=progress

# Step 3: Expand NVMe partition
sudo parted /dev/nvme0n1 resizepart 1 100%
sudo resize2fs /dev/nvme0n1p1

# Step 4: Update UEFI boot order to prefer NVMe
# On Orin, use: sudo nvbootctrl set-active-boot-slot 0
# Or use NVIDIA's provided extlinux script:
sudo /opt/nvidia/jetson-io/config-by-hardware.py

# Step 5: Reboot and verify
df -h   # root should show NVMe size, not SD size
lsblk   # verify boot partition
```

### Post-Install Verification

```bash
# Check JetPack version
cat /etc/nv_tegra_release

# Check CUDA
nvcc --version
# Expected: Cuda compilation tools, release 12.x

# Check GPU
nvidia-smi
# Shows: Orin GPU, memory, driver version

# Check TensorRT
dpkg -l | grep tensorrt
python3 -c "import tensorrt; print(tensorrt.__version__)"

# Run NVIDIA system profiler
sudo tegrastats
# Shows: CPU%, GPU%, RAM, power, temperature — live
```

---

## 3. Upgrading JetPack on Orin Nano 8GB

### Can You Upgrade In-Place?

**Short answer:** Minor version upgrades (e.g., 6.0 → 6.1) can sometimes be done via `apt`. Major version upgrades (e.g., 5.x → 6.x) **always require reflashing**.

### Minor Version Upgrade via apt (Same Major Version)

```bash
# Add NVIDIA Jetson apt repository
sudo apt-get update
sudo apt-get upgrade   # upgrades L4T packages if available

# Check what version you'll get before upgrading:
apt-cache policy nvidia-l4t-core

# Upgrade specific JetPack components
sudo apt-get install --only-upgrade \
  cuda-toolkit-12-x \
  libcudnn9 \
  tensorrt

# After upgrade, reboot
sudo reboot

# Verify new version
cat /etc/nv_tegra_release
```

**Warning:** In-place apt upgrades can occasionally break dependencies. Always snapshot/backup before upgrading production systems.

### Major Version Upgrade (5.x → 6.x) — Reflash Required

```
JetPack 5.x (Ubuntu 20.04, CUDA 11.x)
             ↓  Cannot apt-upgrade across major versions
JetPack 6.x (Ubuntu 22.04, CUDA 12.x)

Why reflash?
  - Different Ubuntu base (20.04 → 22.04)
  - Different L4T kernel series (35.x → 36.x)
  - Different bootchain
  - Different partition layout
```

```bash
# Process for Orin Nano 8GB: 5.x → 6.x
# 1. Backup your application data and custom configs
rsync -av /home/user/ /media/backup/

# 2. Note all installed packages
dpkg --get-selections > /media/backup/installed_packages.txt

# 3. Reflash using SDK Manager (Method 2 above)
#    Select JetPack 6.x on host PC

# 4. After flash, reinstall your packages and restore data
# 5. Reinstall Python packages (new Python version on Ubuntu 22.04)
pip3 install -r /media/backup/requirements.txt
```

### Upgrade Decision Matrix

| Current → Target        | Method       | Time    | Risk | Data Loss |
|-------------------------|--------------|---------|------|-----------|
| JP6.0 → JP6.1           | apt upgrade  | 10 min  | Low  | No        |
| JP6.0 → JP6.2           | apt or reflash | 15 min | Med | No (apt) |
| JP5.1.x → JP6.x         | Reflash only | 30 min  | Med  | Yes*      |
| JP4.x → JP5.x/6.x       | Reflash only | 30 min  | High | Yes*      |

*Back up data to external storage before reflash.

---

## 4. Benefits of Higher JetPack Versions

### JetPack 6.x vs 5.x (Orin Nano Context)

```
JetPack 5.1.4                    JetPack 6.2
─────────────────                ─────────────────────────
Ubuntu 20.04                  →  Ubuntu 22.04 (LTS until 2027)
Python 3.8                    →  Python 3.10
CUDA 11.4                     →  CUDA 12.8
cuDNN 8.6                     →  cuDNN 9.7
TensorRT 8.6                  →  TensorRT 10.7
GCC 9                         →  GCC 11
OpenCV 4.5                    →  OpenCV 4.8
ROS2 Foxy (EOL)               →  ROS2 Humble/Jazzy (supported)
VPI 2.x                       →  VPI 3.x (CUDA graphs, better CPU)
DeepStream 6.x                →  DeepStream 7.x
```

### Concrete Benefits

**CUDA 12.x improvements:**
```
- CUDA Graph launch overhead: -40% vs CUDA 11.x
- INT8/FP8 Tensor Core utilization: improved Ampere support
- cudaMemcpyAsync improvements: better overlap with computation
- Cooperative Groups improvements
```

**TensorRT 10.x improvements:**
```
- Strongly Typed mode: explicit tensor types throughout the network
- Faster engine build times
- Better INT8 calibration
- New plugins for attention layers (Transformers on edge)
- Improved BF16 support
- Better quantization-aware training (QAT) support
```

**cuDNN 9.x improvements:**
```
- New graph-based API (replaces legacy API)
- Better memory reuse across operations
- Fused attention (FlashAttention-style) on Ampere
```

**Security improvements in JetPack 6:**
```
- Secure Boot with anti-rollback protection
- UEFI Secure Boot support
- Disk encryption (DM-Crypt) officially supported
- Measured Boot support
- OTA (Over-the-Air) update infrastructure built-in
```

**ROS2 compatibility:**
```
JetPack 5.x → Ubuntu 20.04 → ROS2 Foxy (EOL 2023) or Galactic (EOL 2022)
JetPack 6.x → Ubuntu 22.04 → ROS2 Humble (LTS until 2027) ← Use this
```

**Practical rule:** Always use the latest JetPack unless you have a specific reason not to (e.g., a dependency that hasn't been ported yet). Newer JetPack = better performance, better security, longer support.

---

## 5. Porting AI Models to Jetson

### The Porting Pipeline

```
Training Environment          Jetson Deployment
(Desktop/Cloud)               (Orin Nano 8GB)

PyTorch / TensorFlow          TensorRT Engine
     model.pt          →      model.engine
     model.pb          →      (compiled, optimized,
     model.onnx        →       quantized)

Steps:
  1. Train model (any framework)
  2. Export to ONNX (universal format)
  3. Convert ONNX → TensorRT engine
  4. Run inference with TensorRT runtime
```

### Step 1: Export PyTorch Model to ONNX

```python
# On training machine (or Jetson if RAM allows)
import torch
import torch.onnx

model = MyModel()
model.load_state_dict(torch.load("model.pt"))
model.eval()

# Define dummy input matching your model's expected input
dummy_input = torch.randn(1, 3, 640, 640)   # e.g., YOLO input

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=17,              # use latest stable
    input_names=["images"],
    output_names=["output"],
    dynamic_axes={
        "images": {0: "batch"},    # dynamic batch size
        "output": {0: "batch"}
    }
)

# Verify ONNX model
import onnx
model_onnx = onnx.load("model.onnx")
onnx.checker.check_model(model_onnx)
print("ONNX export successful")
```

### Step 2: Build TensorRT Engine on Jetson

**Always build TensorRT engines ON the target Jetson** — engines are hardware-specific.

```bash
# Method A: trtexec (command-line tool, easiest)

# FP32 (no quantization)
trtexec --onnx=model.onnx \
        --saveEngine=model_fp32.engine \
        --verbose

# FP16 (2× speedup, <1% accuracy loss typically)
trtexec --onnx=model.onnx \
        --saveEngine=model_fp16.engine \
        --fp16 \
        --verbose

# INT8 (4× speedup, needs calibration data)
trtexec --onnx=model.onnx \
        --saveEngine=model_int8.engine \
        --int8 \
        --calib=calib_data.cache \
        --verbose

# Dynamic batch size (1–16)
trtexec --onnx=model.onnx \
        --saveEngine=model_dynamic.engine \
        --fp16 \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:4x3x640x640 \
        --maxShapes=images:16x3x640x640
```

```python
# Method B: Python TensorRT API (more control)
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path, fp16=True):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser, \
         builder.create_builder_config() as config:

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB

        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError("ONNX parse failed")

        serialized = builder.build_serialized_network(network, config)
        with open(engine_path, 'wb') as f:
            f.write(serialized)
        print(f"Engine saved to {engine_path}")

build_engine("model.onnx", "model_fp16.engine", fp16=True)
```

### Step 3: Run TensorRT Inference

```python
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

class TRTInferencer:
    def __init__(self, engine_path):
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate device buffers
        self.inputs, self.outputs, self.bindings = [], [], []
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
        self.stream = cuda.Stream()

    def infer(self, input_data):
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host']

# Usage
inferencer = TRTInferencer("model_fp16.engine")
frame = np.random.randn(1, 3, 640, 640).astype(np.float32)
result = inferencer.infer(frame)
```

### DLA (Deep Learning Accelerator) — Extra Efficiency

The Orin Nano has 1 DLA engine. For supported layers, DLA runs at ~10 TOPS while freeing the GPU for other tasks.

```python
# Enable DLA in TensorRT engine build
config.default_device_type = trt.DeviceType.DLA
config.DLA_core = 0           # Orin Nano has DLA 0 only
config.set_flag(trt.BuilderFlag.GPU_FALLBACK)  # GPU fallback for unsupported layers
config.set_flag(trt.BuilderFlag.FP16)           # DLA requires FP16 or INT8

# Check which layers run on DLA vs GPU after building
# Use: trtexec --onnx=model.onnx --useDLACore=0 --verbose 2>&1 | grep "DLA"
```

---

## 6. ROS2 Integration

### Installing ROS2 Humble on JetPack 6 (Ubuntu 22.04)

```bash
# Add ROS2 apt repository
sudo apt install -y software-properties-common curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list

sudo apt update
sudo apt install -y ros-humble-desktop          # full install with RViz
# Or for minimal footprint:
sudo apt install -y ros-humble-ros-base         # no GUI tools

# Install build tools
sudo apt install -y python3-colcon-common-extensions python3-rosdep

# Initialize rosdep
sudo rosdep init
rosdep update

# Add to .bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### ROS2 + TensorRT: AI Node Architecture

```
Camera (CSI/USB) ─→ image_raw ─→ [preprocessing node] ─→ [TensorRT inference node]
                                                                ↓
LiDAR ──────────→ scan ─────────→ [pointcloud node] ──→ [sensor fusion node]
                                                                ↓
IMU ────────────→ imu ──────────→ [EKF node] ───────→ [object tracker]
                                                                ↓
                                                     [control output node]
                                                                ↓
                                                     cmd_vel / actuator commands
```

### Writing a TensorRT Inference ROS2 Node

```python
#!/usr/bin/env python3
# ros2_trt_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from cv_bridge import CvBridge
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TRTDetectionNode(Node):
    def __init__(self):
        super().__init__('trt_detection_node')

        # Parameters
        self.declare_parameter('engine_path', 'model_fp16.engine')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('input_size', [640, 640])

        engine_path = self.get_parameter('engine_path').value
        self.conf_thresh = self.get_parameter('confidence_threshold').value

        # Load TensorRT engine
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.allocate_buffers()

        self.bridge = CvBridge()

        # Subscribe to camera image
        self.sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publish detections
        self.pub = self.create_publisher(Detection2DArray, '/detections', 10)

        self.get_logger().info(f'TRT inference node started: {engine_path}')

    def load_engine(self, path):
        with open(path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self):
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            host = cuda.pagelocked_empty(size, np.float32)
            device = cuda.mem_alloc(host.nbytes)
            self.bindings.append(int(device))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host, 'device': device})
            else:
                self.outputs.append({'host': host, 'device': device})

    def preprocess(self, img):
        img = cv2.resize(img, (640, 640))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)               # add batch dim
        return np.ascontiguousarray(img)

    def infer(self, data):
        np.copyto(self.inputs[0]['host'], data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host'].copy()

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Preprocess
        input_data = self.preprocess(frame)

        # Run inference
        output = self.infer(input_data)

        # Parse output and publish
        detections = self.parse_detections(output, frame.shape, msg.header)
        self.pub.publish(detections)

    def parse_detections(self, output, img_shape, header):
        # Implement based on your model's output format
        # Example for YOLO-style output
        detections = Detection2DArray()
        detections.header = header
        # ... parse boxes, scores, classes from output ...
        return detections

def main(args=None):
    rclpy.init(args=args)
    node = TRTDetectionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### ROS2 Node Performance: Executor Strategy

```python
# Single-threaded executor (default): simple, no race conditions
rclpy.spin(node)

# Multi-threaded executor: callbacks run in parallel
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        # Camera callback: can overlap with LiDAR callback
        camera_group = MutuallyExclusiveCallbackGroup()
        lidar_group = MutuallyExclusiveCallbackGroup()

        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw',
            self.camera_cb, 10,
            callback_group=camera_group
        )
        self.lidar_sub = self.create_subscription(
            PointCloud2, '/lidar/points',
            self.lidar_cb, 10,
            callback_group=lidar_group
        )

executor = MultiThreadedExecutor(num_threads=4)
executor.add_node(node)
executor.spin()
```

### ROS2 QoS for Sensor Data

```python
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

# Sensor data (real-time, drop old frames, don't retry)
sensor_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,  # drop if can't deliver
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,                                        # only care about latest frame
    durability=QoSDurabilityPolicy.VOLATILE
)

# Control commands (must be delivered, no drops)
control_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10
)

self.camera_sub = self.create_subscription(
    Image, '/camera/image_raw',
    self.camera_cb, sensor_qos   # Use sensor QoS
)
self.cmd_pub = self.create_publisher(
    Twist, '/cmd_vel', control_qos  # Use reliable QoS
)
```

---

## 7. Optimizing AI Inference

### Precision Tradeoffs

```
Precision    Memory   Speed      Accuracy   Use Case
FP32         100%     1×         Baseline   Training, debug
FP16         50%      2–3×       ~0.1% drop General inference
INT8         25%      3–5×       ~1% drop   Production
INT4         12.5%    4–8×       ~3% drop   Very edge, LLMs

On Orin Nano (Ampere GPU):
  FP16 Tensor Cores: native support, excellent
  INT8 Tensor Cores: native support, fastest path
  FP32: no Tensor Core acceleration
```

### CUDA Streams: Overlapping Work

```python
import pycuda.driver as cuda

stream_inference = cuda.Stream()
stream_preprocess = cuda.Stream()

# Pipeline: while GPU is running inference on frame N,
#           CPU is preprocessing frame N+1

with cuda.Stream() as s1, cuda.Stream() as s2:
    # Frame N: copy to GPU on s1
    cuda.memcpy_htod_async(gpu_input_n, cpu_frame_n, s1)

    # Frame N: run inference on s1
    context.execute_async_v2(bindings, s1.handle)

    # Frame N+1: preprocessing on CPU happens concurrently
    cpu_frame_n1 = preprocess(next_raw_frame)

    # Sync s1, get result
    s1.synchronize()
    cuda.memcpy_dtoh_async(cpu_output_n, gpu_output_n, s1)
```

### CUDA Graphs: Reduce Launch Overhead

For a fixed inference workload (same input shape every call), CUDA Graphs eliminate kernel launch overhead:

```python
import torch

model = model.cuda().half()

# Warm up
dummy = torch.randn(1, 3, 640, 640, device='cuda', dtype=torch.float16)
for _ in range(3):
    _ = model(dummy)

# Capture CUDA graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(dummy)

# Replay graph (ultra-low overhead)
def fast_infer(x):
    dummy.copy_(x)    # update input in-place
    g.replay()        # replay captured graph
    return output.clone()
```

### Batch Inference Strategy

```
Single-frame inference (typical naive approach):
  Frame → [wait] → Model → Result → Frame → [wait] → ...
  Throughput: 1 frame / inference_time

Batched inference (correct approach):
  Accumulate frames → [batch=4] → Model → 4 results
  Throughput: 4 frames / inference_time (same GPU time!)
  Latency: slightly higher, but throughput multiplied

Dynamic batching: accept 1–N frames, fill timeout or max batch
```

### Profiling the Inference Pipeline

```bash
# 1. Quick throughput benchmark
trtexec --loadEngine=model_fp16.engine \
        --batch=1 \
        --iterations=100 \
        --warmUp=500 \
        --avgRuns=100

# 2. System-wide profiling with Nsight Systems
nsys profile \
    --trace=cuda,cudnn,tensorrt,osrt \
    --output=profile \
    python3 inference_script.py

# View report:
nsys-ui profile.qdrep

# 3. GPU kernel profiling with Nsight Compute
ncu --set full \
    --target-processes all \
    python3 inference_script.py

# 4. Jetson-specific: tegrastats
sudo tegrastats --interval 100 | tee tegrastats.log
```

### Latency Measurement

```python
import time
import numpy as np

def benchmark_inference(inferencer, input_data, n_runs=200, warmup=50):
    # Warmup (important: first runs include JIT compilation)
    for _ in range(warmup):
        inferencer.infer(input_data)

    # Measure
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        inferencer.infer(input_data)
        times.append((time.perf_counter() - t0) * 1000)  # ms

    times = np.array(times)
    print(f"Latency: mean={times.mean():.2f}ms  "
          f"p50={np.percentile(times,50):.2f}ms  "
          f"p95={np.percentile(times,95):.2f}ms  "
          f"p99={np.percentile(times,99):.2f}ms")
    print(f"Throughput: {1000/times.mean():.1f} FPS")

benchmark_inference(inferencer, dummy_input)
```

---

## 8. End-to-End AI Pipeline: Sensors → Inference → Control

### Full Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  SENSOR LAYER                                                        │
│  Camera (30fps) ──► CSI/USB → V4L2 → CUDA buffer                   │
│  LiDAR (10Hz)  ──► Ethernet/USB → PointCloud buffer                 │
│  IMU (100Hz)   ──► I2C/SPI → ring buffer                            │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  PREPROCESSING LAYER (GPU — CUDA/VPI)                                │
│  Image: resize, normalize, color convert (CUDA)                      │
│  PointCloud: voxelization, ground removal (CUDA)                     │
│  IMU: integration, bias correction (CPU)                             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  INFERENCE LAYER (GPU — TensorRT)                                    │
│  Object Detection: YOLO / SSD on camera frames                       │
│  PointCloud Detection: PointPillars on LiDAR scan                    │
│  Fusion: BEVFusion / simple box fusion                               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  FUSION & TRACKING LAYER (CPU + GPU)                                 │
│  EKF/UKF: fuse IMU + camera + LiDAR detections                      │
│  Object tracker: Hungarian algorithm + Kalman filter                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  CONTROL LAYER (CPU — real-time)                                     │
│  Path planner (A*, DWA, MPC)                                         │
│  Velocity/steering commands → actuators                              │
└─────────────────────────────────────────────────────────────────────┘
```

### Zero-Copy Camera Pipeline (Optimized)

```python
# Use GStreamer + nvarguscamerasrc for zero-copy CSI camera pipeline
import cv2

# CSI camera (native, zero-copy, hardware-accelerated)
gst_pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! "
    "nvvidconv ! "                                    # stays in GPU memory
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink drop=1"
)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
```

```python
# Better: use pycuda + nvargus for true zero-copy (frame stays in GPU)
# or use NVIDIA's Jetson.Utils library:
import jetson.utils as ju

camera = ju.videoSource("csi://0", argv=["--width=1920", "--height=1080", "--framerate=30"])
display = ju.videoOutput("display://0")

while True:
    img = camera.Capture()         # CUDA image, stays on GPU
    # img is a jetson.utils.cudaImage — pointer into shared GPU memory
    # pass directly to TensorRT without any CPU roundtrip
    result = detect(img)
    display.Render(img)
```

### Pipeline Timing Budget (Example: 30 FPS = 33ms per frame)

```
Budget: 33ms total for one pipeline cycle at 30 FPS

Camera capture:     2ms   (DMA from ISP to memory)
Preprocessing:      3ms   (GPU: resize, normalize)
TensorRT inference: 10ms  (GPU: FP16 detection model)
LiDAR processing:   5ms   (GPU: voxelization, PointPillars)
Sensor fusion:      4ms   (CPU: EKF update)
Object tracking:    2ms   (CPU: Hungarian + Kalman)
Path planning:      4ms   (CPU: DWA or A*)
Control output:     1ms   (CAN/UART command send)
Margin:             2ms
─────────────────────────
Total:             33ms = 30 FPS ✓
```

### Pipelining with Threads

```python
import threading
import queue
import time

# Thread-safe queues between pipeline stages
raw_frame_q = queue.Queue(maxsize=2)     # camera → preprocess
gpu_frame_q  = queue.Queue(maxsize=2)    # preprocess → inference
detection_q  = queue.Queue(maxsize=2)    # inference → fusion
control_q    = queue.Queue(maxsize=2)    # fusion → control

def camera_thread():
    """Runs at camera frequency (30 Hz)"""
    while True:
        ret, frame = cap.read()
        if not raw_frame_q.full():
            raw_frame_q.put_nowait(frame)

def preprocess_thread():
    """GPU preprocessing"""
    while True:
        frame = raw_frame_q.get()
        processed = gpu_preprocess(frame)  # CUDA kernel
        gpu_frame_q.put(processed)

def inference_thread():
    """TensorRT inference"""
    while True:
        processed = gpu_frame_q.get()
        detections = trt_infer(processed)
        detection_q.put(detections)

def control_thread():
    """Control loop — must run at fixed rate"""
    while True:
        if not detection_q.empty():
            detections = detection_q.get_nowait()
            cmd = compute_control(detections)
            send_command(cmd)
        time.sleep(0.01)  # 100 Hz control loop

# Start all threads
threads = [
    threading.Thread(target=camera_thread, daemon=True),
    threading.Thread(target=preprocess_thread, daemon=True),
    threading.Thread(target=inference_thread, daemon=True),
    threading.Thread(target=control_thread, daemon=True),
]
for t in threads:
    t.start()
```

---

## 9. LiDAR, Camera, IMU — Practical Integration

### Camera Integration

#### CSI Camera (Recommended for performance)

```bash
# Check if CSI camera is detected
sudo dmesg | grep imx          # IMX219, IMX477 modules
v4l2-ctl --list-devices

# Test: capture a frame
nvgstcapture-1.0 --sensor-id=0

# List available formats
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

```python
# Camera calibration with OpenCV (critical for accurate 3D projection)
import cv2
import numpy as np

# After capturing calibration images with checkerboard:
objpoints = []  # 3D world points
imgpoints = []  # 2D image points

# ... fill objpoints and imgpoints from checkerboard detection ...

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print(f"Camera matrix:\n{camera_matrix}")
print(f"Distortion coeffs: {dist_coeffs}")

# Save for use in ROS2 camera_info topic
np.save('camera_matrix.npy', camera_matrix)
np.save('dist_coeffs.npy', dist_coeffs)
```

#### USB Camera Setup

```bash
# Identify camera
ls /dev/video*
v4l2-ctl --device=/dev/video0 --list-formats-ext

# Check USB speed (should be USB 3.x for high-res)
lsusb -t

# Set optimal format
v4l2-ctl --device=/dev/video0 \
    --set-fmt-video=width=1280,height=720,pixelformat=MJPG
```

### LiDAR Integration

#### RPLIDAR (Common, low-cost)

```bash
# Install RPLIDAR ROS2 driver
sudo apt-get install ros-humble-rplidar-ros

# Connect via USB, find port
ls /dev/ttyUSB*
sudo chmod 666 /dev/ttyUSB0

# Launch
ros2 launch rplidar_ros rplidar_a2_launch.py \
    serial_port:=/dev/ttyUSB0 \
    serial_baudrate:=115200 \
    frame_id:=laser
```

#### Velodyne LiDAR (High-end)

```bash
sudo apt-get install ros-humble-velodyne

# Velodyne connects via Ethernet (192.168.1.201 by default)
sudo ip addr add 192.168.1.100/24 dev eth0

ros2 launch velodyne_driver velodyne_driver_node-VLP16-launch.py
```

#### PointCloud Processing

```python
# ROS2 PointCloud2 → numpy
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

def lidar_callback(self, msg):
    # Extract XYZ points
    points = np.array(list(pc2.read_points(
        msg, field_names=('x', 'y', 'z', 'intensity'),
        skip_nans=True
    )))

    if len(points) == 0:
        return

    xyz = points[:, :3]                    # shape [N, 3]
    intensity = points[:, 3:4]             # shape [N, 1]

    # Ground removal (simple height filter)
    mask = xyz[:, 2] > -0.3               # remove points below 30cm
    xyz_filtered = xyz[mask]

    # Pass to CUDA for voxelization
    self.process_pointcloud_gpu(xyz_filtered)
```

### IMU Integration

#### Reading IMU via I2C (MPU6050 / ICM42688)

```bash
# Enable I2C on Jetson GPIO header
sudo i2cdetect -y -r 1    # scan I2C bus 1
# Should show device address (0x68 for MPU6050)
```

```python
import smbus2
import struct
import time

class MPU6050:
    ADDR = 0x68
    PWR_MGMT_1 = 0x6B
    ACCEL_XOUT_H = 0x3B
    GYRO_XOUT_H = 0x43

    def __init__(self, bus_num=1):
        self.bus = smbus2.SMBus(bus_num)
        self.bus.write_byte_data(self.ADDR, self.PWR_MGMT_1, 0)  # wake up
        time.sleep(0.1)

    def _read_word(self, reg):
        high = self.bus.read_byte_data(self.ADDR, reg)
        low  = self.bus.read_byte_data(self.ADDR, reg + 1)
        val  = (high << 8) + low
        return val - 65536 if val >= 0x8000 else val

    def get_accel(self):
        ax = self._read_word(self.ACCEL_XOUT_H) / 16384.0     # g
        ay = self._read_word(self.ACCEL_XOUT_H + 2) / 16384.0
        az = self._read_word(self.ACCEL_XOUT_H + 4) / 16384.0
        return ax, ay, az

    def get_gyro(self):
        gx = self._read_word(self.GYRO_XOUT_H) / 131.0        # °/s
        gy = self._read_word(self.GYRO_XOUT_H + 2) / 131.0
        gz = self._read_word(self.GYRO_XOUT_H + 4) / 131.0
        return gx, gy, gz

# Publish as ROS2 Imu message
from sensor_msgs.msg import Imu

imu = Imu()
imu.header.stamp = self.get_clock().now().to_msg()
imu.header.frame_id = 'imu_link'
ax, ay, az = mpu.get_accel()
imu.linear_acceleration.x = ax * 9.81
imu.linear_acceleration.y = ay * 9.81
imu.linear_acceleration.z = az * 9.81
self.imu_pub.publish(imu)
```

### Extrinsic Calibration: Camera ↔ LiDAR

Extrinsic calibration finds the rigid transformation (rotation + translation) between sensor coordinate frames. This is critical for fusion.

```bash
# Install calibration tools
sudo apt-get install ros-humble-camera-calibration
pip3 install kalibr

# Kalibr calibration (recommended):
# 1. Print an AprilGrid target
# 2. Record a ROS2 bag with camera + IMU moving together
ros2 bag record -o calib_bag /camera/image_raw /imu/data

# 3. Run Kalibr
kalibr_calibrate_cameras \
    --bag calib_bag.bag \
    --topics /camera/image_raw \
    --models pinhole-equi \
    --target aprilgrid.yaml

# Camera-LiDAR extrinsic:
# Use: github.com/PJLab-ADLab/SensorsCalibration
```

```python
# Apply extrinsic transform: project LiDAR points into camera image
import numpy as np

# Extrinsic: 4×4 transform matrix (LiDAR frame → camera frame)
T_cam_lidar = np.array([
    [ 0.9998, -0.0052,  0.0191,  0.1500],   # example values
    [ 0.0049,  0.9999,  0.0104, -0.0050],
    [-0.0192, -0.0103,  0.9997,  0.0300],
    [ 0.0000,  0.0000,  0.0000,  1.0000]
])

# Camera intrinsic matrix
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

def lidar_to_image(points_lidar, T_cam_lidar, K):
    """Project LiDAR points onto camera image plane"""
    N = points_lidar.shape[0]
    pts_h = np.hstack([points_lidar[:, :3], np.ones((N, 1))])   # homogeneous
    pts_cam = (T_cam_lidar @ pts_h.T).T                          # camera frame

    # Keep only points in front of camera
    mask = pts_cam[:, 2] > 0
    pts_cam = pts_cam[mask]

    # Project to image
    pts_img = (K @ pts_cam[:, :3].T).T
    u = pts_img[:, 0] / pts_img[:, 2]
    v = pts_img[:, 1] / pts_img[:, 2]
    depth = pts_cam[:, 2]

    return u, v, depth, mask
```

### Time Synchronization Between Sensors

```python
# Use message_filters for approximate time synchronization
import message_filters
from sensor_msgs.msg import Image, PointCloud2

class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')

        self.camera_sub = message_filters.Subscriber(self, Image, '/camera/image_raw')
        self.lidar_sub  = message_filters.Subscriber(self, PointCloud2, '/lidar/points')

        # Synchronize: accept messages within 50ms of each other
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.camera_sub, self.lidar_sub],
            queue_size=10,
            slop=0.05    # 50ms tolerance
        )
        self.ts.registerCallback(self.fused_callback)

    def fused_callback(self, camera_msg, lidar_msg):
        # Both messages are time-synchronized
        # Process together for accurate fusion
        pass
```

---

## 10. Power and Thermal Management in Practice

### Power Modes

The Orin Nano 8GB has configurable TDP levels:

```bash
# Check current power mode
sudo nvpmodel -q

# Available modes for Orin Nano 8GB
# Mode 0: MAXN — full performance (15W TDP)
# Mode 1: 7W   — balanced (7W TDP)

# Set max performance mode
sudo nvpmodel -m 0

# Set max CPU and GPU clocks (within current power mode)
sudo jetson_clocks

# Verify
cat /sys/kernel/debug/bpmp/debug/clk/cpu0/rate   # CPU frequency
cat /sys/kernel/debug/bpmp/debug/clk/gpu/rate     # GPU frequency
```

### Real-Time Power and Temperature Monitoring

```bash
# tegrastats: comprehensive live view
sudo tegrastats --interval 100

# Example output:
# RAM 3045/7772MB (lfb 2x2MB) SWAP 0/3886MB
# CPU [35%@1510,28%@1510,12%@1510,9%@1510,15%@1510,8%@1510]
# EMC_FREQ 38% GR3D_FREQ 89%
# CPU@42C SOC0@40C SOC1@38C SOC2@38C GPU@44C tj@44C
# VDD_IN 6234mW VDD_CPU_GPU_CV 2901mW VDD_SOC 1158mW

# Parse tegrastats programmatically:
import subprocess
import re

def get_tegrastats():
    result = subprocess.run(['sudo', 'tegrastats', '--once'],
                            capture_output=True, text=True)
    line = result.stdout.strip()

    gpu_temp  = float(re.search(r'GPU@(\d+\.?\d*)C', line).group(1))
    cpu_temp  = float(re.search(r'CPU@(\d+\.?\d*)C', line).group(1))
    power_mw  = float(re.search(r'VDD_IN (\d+)mW', line).group(1))
    gpu_freq  = int(re.search(r'GR3D_FREQ (\d+)%', line).group(1))

    return {
        'gpu_temp_c': gpu_temp,
        'cpu_temp_c': cpu_temp,
        'power_mw': power_mw,
        'gpu_util_pct': gpu_freq
    }
```

### Thermal Throttling — Prevention

Thermal throttling kills performance silently. The CPU/GPU clocks drop without warning when the temperature exceeds the thermal budget.

```bash
# Check throttle temperature thresholds
cat /sys/devices/virtual/thermal/thermal_zone*/trip_point_*_temp

# Monitor for throttle events
dmesg | grep -i throttle
journalctl -f | grep -i thermal

# Practical thresholds for Orin Nano:
# CPU/GPU @ 85°C: begin throttling
# CPU/GPU @ 95°C: emergency shutdown
# Operating target: keep below 75°C
```

### Practical Cooling Solutions

```
Dev Kit enclosure: passive heatsink only → OK for 7W mode, marginal at 15W
Production fixes:
  1. Active cooling (5V fan on GPIO header)
  2. Thermal paste quality check (factory paste is mediocre)
  3. Heatsink + copper shim for better contact
  4. Enclosure with forced air flow (cut vents top and bottom)
  5. Avoid direct sunlight on enclosure
  6. Derate: run at 10W instead of 15W for reliable 24/7 operation
```

```python
# Automatic fan control via PWM (GPIO pin 33 = PWM0)
import Jetson.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setup(33, GPIO.OUT)

fan_pwm = GPIO.PWM(33, 25000)   # 25kHz PWM frequency
fan_pwm.start(0)                 # 0% duty cycle = off

def set_fan_speed(pct):
    """pct: 0-100"""
    fan_pwm.ChangeDutyCycle(pct)

def thermal_control():
    while True:
        stats = get_tegrastats()
        temp = max(stats['gpu_temp_c'], stats['cpu_temp_c'])

        if temp < 50:
            set_fan_speed(0)    # off
        elif temp < 60:
            set_fan_speed(30)   # 30%
        elif temp < 70:
            set_fan_speed(60)   # 60%
        elif temp < 80:
            set_fan_speed(80)   # 80%
        else:
            set_fan_speed(100)  # full blast

        time.sleep(2)
```

### Power Optimization for Battery-Powered Systems

```bash
# Disable unused hardware
sudo systemctl disable bluetooth    # if not needed
sudo systemctl disable cups         # printer service, never needed

# CPU frequency governor
echo powersave | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# or for max performance:
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable HDMI output when not needed (saves ~0.5W)
sudo systemctl disable gdm3          # disable GUI
sudo systemctl set-default multi-user.target

# Measure actual power draw per component
sudo cat /sys/bus/i2c/drivers/ina3221/*/iio:device*/in_power*_input
# Shows VDD_IN, VDD_CPU_GPU_CV, VDD_SOC individually
```

### Power Budget for Mobile Robot (Example: 5Ah @ 12V = 60Wh)

```
Component               Typical Draw    Peak
Jetson Orin Nano        8W              15W
Camera (USB)            2W              2.5W
LiDAR (RPLIDAR A2)      1.5W            2W
IMU                     0.1W            0.1W
Ethernet/WiFi           0.5W            1W
Drive motors            5–30W           50W
──────────────────────────────────────────
AI compute (no motors): ~12W typical
Runtime on 5Ah@12V: 60Wh / 12W = 5 hours
```

---

## 11. OTA Update Best Practices

### OTA Update Strategies on Jetson

#### Strategy 1: apt-based updates (Simple, for application updates)

```bash
# Unattended security updates (OS packages only)
sudo apt-get install unattended-upgrades
sudo dpkg-reconfigure unattended-upgrades

# Only enable security updates, not all upgrades
# Edit /etc/apt/apt.conf.d/50unattended-upgrades:
Unattended-Upgrade::Allowed-Origins {
    "Ubuntu:${distro_codename}-security";
    "NVIDIA:${distro_codename}";   // if NVIDIA updates are needed
};

# Test update process:
sudo unattended-upgrades --dry-run --debug
```

#### Strategy 2: NVIDIA Jetson OTA (Full JetPack updates)

NVIDIA provides official OTA infrastructure for L4T updates in JetPack 6.x:

```bash
# Check available OTA updates
sudo apt-get update
apt list --upgradable 2>/dev/null | grep nvidia-l4t

# Apply L4T OTA update (minor version, e.g., 36.3 → 36.4)
sudo apt-get upgrade nvidia-l4t-core nvidia-l4t-cuda

# Reboot to activate new kernel
sudo reboot
```

#### Strategy 3: A/B Partition Scheme (Production — Zero Downtime)

The Orin supports dual-boot partition (A/B slots). This is the industry standard for reliable OTA:

```
Slot A (active):  JetPack 6.1  ← currently running
Slot B (standby): JetPack 6.2  ← being written / updated

OTA process:
  1. Download new image
  2. Write to Slot B while Slot A keeps running
  3. Verify Slot B image integrity (SHA256)
  4. Atomically switch boot to Slot B
  5. Reboot into Slot B
  6. If boot fails: automatically fall back to Slot A
  7. If boot succeeds for N minutes: mark Slot B as permanent
```

```bash
# Check current boot slot
sudo nvbootctrl dump-slots-info

# Mark current slot as successful (call from application after health check)
sudo nvbootctrl mark-boot-successful

# Manually switch slot (for testing)
sudo nvbootctrl set-active-boot-slot 1   # switch to slot B
sudo reboot

# Check rollback info
sudo nvbootctrl get-current-slot
sudo nvbootctrl get-active-boot-slot
```

#### Strategy 4: Custom OTA with Mender.io or Balena

For fleet management (10+ devices):

```bash
# Install Mender client on Jetson
# Reference: docs.mender.io/get-started/preparation/prepare-a-raspberrypi-device
# (Jetson Orin support available in Mender 3.x+)

# Mender provides:
#  - Web dashboard for fleet management
#  - Staged rollouts (deploy to 10% first, then 100%)
#  - Rollback on failure
#  - Delta updates (only send changed files, saves bandwidth)
#  - Device authentication and authorization
```

### OTA Best Practices

```
1. ALWAYS use A/B partitioning for OS updates
   Never update the running system in place

2. Verify before applying:
   - Check SHA256/GPG signature of update package
   - Verify the update is from a trusted source
   - Test on a staging device before fleet rollout

3. Staged rollout:
   Fleet of 100 devices → deploy to 5 → monitor 24h → deploy to 100
   Never push to 100% simultaneously

4. Rollback conditions (auto-trigger):
   - Boot fails to complete in X seconds
   - Application fails health check after boot
   - Critical service fails to start

5. Bandwidth management:
   - Use delta updates (only changed files)
   - Schedule updates during off-hours (low activity)
   - Implement rate limiting to not saturate network

6. Application update vs OS update:
   - Application: can update via docker pull / Python package
   - OS/drivers/kernel: requires full OTA with A/B
   - Treat these separately with different cadences
```

### Application-Level Update (Docker)

```bash
# Package your AI application in a container
# docker-compose.yml on Jetson:

version: '3'
services:
  ai_pipeline:
    image: your-registry/jetson-ai:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - /dev:/dev
      - ./models:/models
    devices:
      - /dev/video0
    restart: unless-stopped

# Update application only (no reflash needed):
docker compose pull && docker compose up -d

# Rollback application:
docker compose down
docker tag your-registry/jetson-ai:previous your-registry/jetson-ai:latest
docker compose up -d
```

---

## 12. Security Hardening

### Threat Model for Jetson Edge Devices

```
Attack Vectors:
  Physical access:     device stolen, SD card extracted, JTAG debug
  Network:             SSH brute force, unencrypted MQTT, open ports
  Supply chain:        malicious model weights, tampered update packages
  Side channel:        power analysis, timing attacks on crypto
  Application:         input injection via camera/LiDAR data, model poisoning

Assets to Protect:
  Model weights (IP)
  Sensor data (privacy)
  Control authority (safety-critical)
  Device identity/credentials
```

### Secure Boot

```bash
# Jetson Orin supports UEFI Secure Boot
# Configure during flash with SDK Manager:
# "Secure Boot" option → requires signing key pair

# Or manually:
# 1. Generate RSA-2048 key pair (on air-gapped machine, store offline)
openssl genrsa -out secure_boot_key.pem 2048
openssl req -new -x509 -key secure_boot_key.pem -out secure_boot_cert.pem -days 3650

# 2. Enroll in UEFI (via SDK Manager or L4T flash scripts)
# 3. After enrollment: only signed bootloaders will run
# 4. This prevents booting modified L4T from attacker's SD card

# Enable anti-rollback (prevent downgrade attacks):
# Set fuse JTAG_DISABLE + BOOTROM_PRODUCTION_MODE
# WARNING: IRREVERSIBLE — test thoroughly before fusing production units
```

### Disk Encryption

```bash
# Enable LUKS encryption on data partition (not rootfs)
sudo cryptsetup luksFormat /dev/nvme0n1p3          # data partition
sudo cryptsetup luksOpen /dev/nvme0n1p3 data_enc
sudo mkfs.ext4 /dev/mapper/data_enc
sudo mount /dev/mapper/data_enc /data

# Store LUKS key in TPM or secure enclave (not on disk)
# Or derive from device-unique hardware ID:
sudo tpm2_createprimary -C e -c primary.ctx
sudo tpm2_create -C primary.ctx -u key.pub -r key.priv -a "fixedtpm|fixedparent|sensitivedataorigin|userwithauth|decrypt|sign"
```

### Network Hardening

```bash
# 1. Firewall (nftables/ufw)
sudo apt-get install ufw
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow from 192.168.1.0/24 to any port 22  # SSH: local network only
sudo ufw allow from 192.168.1.0/24 to any port 7400  # ROS2 DDS: local only
sudo ufw enable

# 2. Disable unnecessary services
sudo systemctl disable avahi-daemon     # mDNS (if not needed)
sudo systemctl disable cups             # printer
sudo systemctl disable ModemManager    # cellular (if no modem)
sudo systemctl list-units --type=service --state=active  # audit all

# 3. SSH hardening (/etc/ssh/sshd_config)
PermitRootLogin no
PasswordAuthentication no          # key-based auth only
PubkeyAuthentication yes
AllowUsers your_user               # whitelist specific user
Port 2222                          # non-standard port (minor obscurity)
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2

# 4. Change default NVIDIA credentials immediately after flash!
# Default: user=nvidia, password=nvidia  ← ALWAYS change this
passwd  # change current user password
sudo passwd -l root  # lock root account
```

### ROS2 Security (DDS Security)

```bash
# ROS2 uses DDS (Data Distribution Service) which by default is unencrypted
# Enable ROS2 Security (SROS2):

# 1. Install security tools
sudo apt-get install ros-humble-sros2

# 2. Create key infrastructure
ros2 security create_keystore ~/ros2_keystore
ros2 security create_key ~/ros2_keystore /trt_detection_node
ros2 security create_key ~/ros2_keystore /control_node

# 3. Set permissions policy (define who can publish/subscribe to what)
# Create a permissions file defining topic access per node

# 4. Launch with security enabled
export ROS_SECURITY_KEYSTORE=~/ros2_keystore
export ROS_SECURITY_ENABLE=true
export ROS_SECURITY_STRATEGY=Enforce

ros2 run your_package trt_detection_node
```

### Model Weight Protection

```python
# Encrypt TensorRT engine weights at rest
from cryptography.fernet import Fernet

# Generate key (store in hardware security module or TPM in production)
key = Fernet.generate_key()

def encrypt_engine(engine_path, encrypted_path, key):
    f = Fernet(key)
    with open(engine_path, 'rb') as fp:
        data = fp.read()
    with open(encrypted_path, 'wb') as fp:
        fp.write(f.encrypt(data))

def load_encrypted_engine(encrypted_path, key):
    f = Fernet(key)
    with open(encrypted_path, 'rb') as fp:
        data = f.decrypt(fp.read())
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    return runtime.deserialize_cuda_engine(data)
    # Engine is only in memory, never decrypted to disk
```

### Security Audit Checklist

```
Before deployment, verify:
□ Default passwords changed
□ SSH key-based auth only, no password auth
□ Firewall enabled with minimal open ports
□ Secure boot enabled and verified
□ Anti-rollback fuses set (if production)
□ Data partition encrypted (if sensitive data)
□ All services audited, unused ones disabled
□ Kernel patched to latest L4T version
□ ROS2 DDS security enabled (if network-connected)
□ Model weights encrypted at rest
□ OTA update channel uses TLS + signature verification
□ Physical ports secured (disable USB in production if not needed)
□ Log aggregation set up (centralized syslog for anomaly detection)
```

---

## 13. Projects

### Project 1: Fresh JetPack 6 Install + NVMe Boot
Flash Orin Nano 8GB to JetPack 6.x with SDK Manager, configure NVMe boot, and benchmark I/O speed vs SD card.

### Project 2: YOLO on Jetson with TensorRT
Train YOLOv8 on custom dataset, export to ONNX, convert to TensorRT FP16 engine, achieve >25 FPS on Orin Nano.

### Project 3: Camera + LiDAR Fusion Pipeline
Build a ROS2 node that fuses camera detection boxes with LiDAR point cloud depth, publish 3D bounding boxes.

### Project 4: Power vs Performance Profiling
Using `tegrastats`, plot FPS vs power consumption for: FP32 / FP16 / INT8 / DLA modes. Find the optimal operating point.

### Project 5: Thermal Stress Test + Cooling Comparison
Run inference at 100% load for 30 minutes. Compare: bare heatsink vs active cooling. Measure FPS drop due to throttling.

### Project 6: A/B OTA Update
Implement a simple OTA update service that fetches a new Docker image, tests it in a container, then switches production traffic to it with rollback capability.

### Project 7: Security Hardening
Start from a fresh Jetson install. Apply all items in the Security Audit Checklist. Verify each item. Run `nmap` from another machine to confirm attack surface.

---

## 14. Resources

### Official Documentation
- **JetPack SDK**: developer.nvidia.com/embedded/jetpack
- **NVIDIA SDK Manager**: developer.nvidia.com/sdk-manager
- **L4T Developer Guide**: docs.nvidia.com/jetson/archives/
- **TensorRT Developer Guide**: docs.nvidia.com/deeplearning/tensorrt/developer-guide/
- **VPI Documentation**: docs.nvidia.com/vpi/

### Community and Tools
- **Jetson Hacks** (jetsonhacks.com): Practical Jetson tutorials, GPIO, hardware setup
- **NVIDIA NGC**: ngc.nvidia.com — pre-built containers optimized for Jetson
- **DeepStream Getting Started**: docs.nvidia.com/metropolis/deepstream/

### Performance and Profiling
- `tegrastats` — always running in a side terminal during development
- Nsight Systems — system-level GPU+CPU trace
- `trtexec --verbose` — TensorRT engine build + benchmark

### Security
- **NVIDIA Jetson Security Guide**: docs.nvidia.com/jetson/archives/l4t-archived/ (Jetson Security section)
- **SROS2 Wiki**: wiki.ros.org/sros2

---

*Next: [3. Edge AI Optimization](../3. Edge AI Optimization/Guide.md)*
