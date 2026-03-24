**2. Real Time Edge AI with Nvidia Jetson (12–18 months)**

**Prerequisite:** Phases 1–2 (digital design, embedded software/Linux) and Phase 4 (Jetson, TensorRT, sensor fusion). This track deepens your ability to design, optimize, and deploy ML models on resource-constrained edge hardware — from microcontrollers to mobile SoCs to custom accelerators — with emphasis on low-latency, real-time streaming pipelines.

---

**1. ML Model Design for Edge Constraints**

* **Efficient Architectures:**
    * **MobileNets (V1–V4):** Depthwise separable convolutions, width/resolution multipliers, inverted residuals (MBConv), and universal inverted bottleneck (UIB).
    * **EfficientNet / EfficientDet:** Compound scaling (depth, width, resolution), BiFPN for detection, and EfficientViT for vision transformers at the edge.
    * **YOLO Family (v5–v11, YOLO-NAS):** Real-time detection trade-offs, anchor-free heads, NAS-optimized backbones, deployment on Jetson/mobile.
    * **Lightweight Transformers:** TinyViT, MobileViT, FastViT — attention on edge, hybrid CNN-Transformer designs.

* **Neural Architecture Search (NAS) for Edge:**
    * Hardware-aware NAS: latency/FLOPs-constrained search (MnasNet, Once-for-All, ProxylessNAS).
    * Supernet training and subnet extraction for target hardware profiles.
    * Multi-objective optimization: accuracy vs latency vs power vs memory.

* **Task-Specific Edge Models:**
    * Pose estimation (MoveNet, PoseNet), segmentation (FastSCNN, TopFormer), speech/keyword spotting (DS-CNN), anomaly detection.
    * On-device generative AI: distilled LLMs, speculative decoding, LoRA adapters on edge.

**Resources:**

* **"Efficient Deep Learning" by Menghani (MIT Press):** Comprehensive coverage of efficient model design, compression, and deployment.
* **[MIT HAN Lab — TinyML and Efficient AI](https://hanlab.mit.edu/courses):** Lecture series on efficient inference and NAS.
* **[MLPerf Tiny Benchmark](https://mlcommons.org/en/inference-tiny/):** Industry-standard edge inference benchmarks.

**Projects:**

* Compare MobileNetV3 vs EfficientNet-Lite vs FastViT on Jetson Orin Nano: accuracy, latency, power.
* Run a hardware-aware NAS search targeting a Cortex-M7 using MCUNetV2 or Once-for-All.
* Deploy a keyword spotting model (DS-CNN) on an STM32 with latency < 20 ms.

---

**2. Model Compression & Optimization**

* **Quantization:**
    * **Post-Training Quantization (PTQ):** INT8, INT4, FP16, mixed-precision calibration, per-channel vs per-tensor.
    * **Quantization-Aware Training (QAT):** Fake quantization nodes, straight-through estimator, fine-tuning for accuracy recovery.
    * **Advanced Techniques:** GPTQ, AWQ, SmoothQuant for LLMs; binary/ternary networks for extreme compression.
    * **Tooling:** TensorRT, ONNX Runtime quantization, AI Edge Torch, tinygrad quantization passes.

* **Pruning:**
    * **Structured Pruning:** Filter/channel pruning for hardware-friendly speedup; magnitude-based, Taylor, FPGM criteria.
    * **Unstructured Pruning:** Weight-level sparsity, lottery ticket hypothesis, sparse tensor acceleration.
    * **Pruning + Quantization Pipelines:** Combined compression with joint fine-tuning.

* **Knowledge Distillation:**
    * Teacher-student frameworks, feature-level vs logit-level distillation.
    * Self-distillation and online distillation for edge scenarios.
    * Distilling large models (LLMs, large vision models) into edge-deployable students.

* **Operator & Graph Optimization:**
    * Layer fusion (Conv-BN-ReLU), constant folding, dead code elimination.
    * Custom operator implementation for target hardware.
    * ONNX graph optimization and custom passes.

**Resources:**

* **[TensorRT Documentation & Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/):** NVIDIA's inference optimization guide.
* **"Neural Network Distillation" survey papers:** Comprehensive overview of distillation methods.
* **[ONNX Runtime Optimization](https://onnxruntime.ai/):** Cross-platform inference optimization.

**Projects:**

* Quantize a ResNet-50 to INT8 using TensorRT PTQ; measure accuracy drop and speedup on Jetson.
* Apply QAT to a detection model; compare accuracy vs PTQ on the same target.
* Build a pruning + distillation pipeline: prune a BERT model, distill into TinyBERT, deploy on edge.

---

**3. Edge Inference Runtimes & Deployment**

* **NVIDIA TensorRT:**
    * Engine building, layer fusion, dynamic shapes, plugin API.
    * INT8 calibration workflows, DLA (Deep Learning Accelerator) offloading on Jetson.
    * Profiling with Nsight Systems and `trtexec`.

* **TensorFlow Lite / LiteRT:**
    * Converter workflow, delegate system (GPU, NNAPI, EdgeTPU, Hexagon DSP).
    * Custom operators, model metadata, on-device training.
    * Coral EdgeTPU: compiler, co-compilation, pipelining across multiple TPUs.

* **ONNX Runtime & Other Runtimes:**
    * Execution providers (CUDA, TensorRT, CoreML, QNN, XNNPACK).
    * Mobile/embedded deployment patterns.
    * Apache TVM: auto-tuning, Relay IR, microTVM for bare-metal targets.
    * tinygrad: understand the stack from tensor ops → linearized IR → optimized kernels → backend codegen.

* **Bare-Metal & RTOS Inference:**
    * TensorFlow Lite Micro (TFLM): interpreter on Cortex-M, arena allocation, custom kernels.
    * CMSIS-NN / CMSIS-DSP: ARM's optimized neural network and DSP kernels for Cortex-M.
    * NNoM, TinyMaix, microTVM for alternative MCU inference paths.

**Resources:**

* **[TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers):** Official guide for MCU deployment.
* **[Apache TVM Documentation](https://tvm.apache.org/docs/):** Compiler framework for deep learning.
* **[tinygrad Documentation](https://github.com/tinygrad/tinygrad):** Minimalist ML framework with hardware backend focus.

**Projects:**

* Deploy a YOLOv8 model on Jetson Orin Nano via TensorRT with DLA offloading; profile with Nsight.
* Run a person detection model on STM32 using TFLM + CMSIS-NN; optimize arena size.
* Port a tinygrad model to a custom backend; compare generated kernels across CUDA/OpenCL/Metal.

---

**4. TinyML & Microcontroller AI**

* **TinyML Foundations:**
    * Hardware landscape: Cortex-M0/M4/M7/M55, RISC-V with vector extensions, specialized ML accelerators (Ethos-U55/U65, MAX78000).
    * Memory constraints: SRAM/Flash budgets, model-data co-optimization, double buffering.
    * Power profiling: active/sleep power, duty cycling, energy harvesting for always-on ML.

* **On-Device Training & Adaptation:**
    * Transfer learning on MCU: frozen backbone + trainable head.
    * Federated learning at the edge: communication-efficient updates.
    * Continual/incremental learning for drift adaptation.

* **Sensor-ML Pipelines:**
    * Audio: feature extraction (MFCC, mel spectrogram) → keyword spotting / sound classification.
    * IMU: accelerometer/gyroscope → activity recognition, vibration anomaly detection.
    * Vision: low-resolution cameras → person detection, gesture recognition.
    * Time-series: predictive maintenance, environmental monitoring.

* **MLOps for Edge:**
    * Model versioning and A/B testing on device fleets.
    * OTA model updates with rollback.
    * On-device telemetry, drift detection, and feedback loops.
    * Edge-cloud hybrid inference: when to offload vs compute locally.

**Resources:**

* **"TinyML" by Pete Warden & Daniel Situnayake (O'Reilly):** Foundational book for ML on microcontrollers.
* **[Edge Impulse](https://www.edgeimpulse.com/):** End-to-end TinyML platform with data collection, training, and deployment.
* **[Harvard CS249r: Tiny Machine Learning](https://sites.google.com/g.harvard.edu/tinyml):** Academic course on TinyML systems.
* **[Arm Ethos-U Documentation](https://developer.arm.com/ip-products/processors/machine-learning/arm-ethos-u):** NPU for Cortex-M.

**Projects:**

* Build an always-on keyword spotter on Arduino Nano 33 BLE Sense with Edge Impulse; measure power.
* Implement vibration-based anomaly detection on STM32 + IMU; deploy with TFLM.
* Design an OTA model update pipeline for an ESP32 fleet with rollback capability.
* Run a person detection model on MAX78000 or Ethos-U55 evaluation kit; compare vs Cortex-M inference.

---

**5. Edge AI System Design & Integration**

* **Hardware Selection & Benchmarking:**
    * Comparing platforms: MCU (STM32, nRF), MPU (Jetson, Raspberry Pi), accelerators (Coral, Hailo, Qualcomm QCS).
    * Metrics: TOPS, TOPS/W, latency, cost, thermal envelope.
    * MLPerf benchmarking methodology and result interpretation.

* **Camera & Vision Pipelines:**
    * ISP (Image Signal Processor) pipelines: RAW → debayer → denoise → tone-map → AI input.
    * Camera interfaces: MIPI CSI-2, USB, GigE Vision.
    * Video analytics: multi-stream inference, tracker integration (DeepSORT, ByteTrack).
    * GStreamer / DeepStream for building accelerated video pipelines.

* **Multi-Model & Multi-Accelerator Systems:**
    * Pipeline parallelism: camera → detection → classification → tracking across CPU/GPU/DLA/NPU.
    * Model orchestration: scheduling, priority, preemption on shared accelerators.
    * Heterogeneous computing: CPU + GPU + DSP + NPU co-execution.

* **Power, Thermal & Reliability:**
    * Thermal management: heatsinks, throttling policies, dynamic frequency scaling.
    * Battery-powered AI: duty cycling, wake-on-event, adaptive inference (early exit networks).
    * Safety and certification: functional safety (IEC 61508), automotive (ISO 26262), medical (IEC 62304).

**Resources:**

* **[NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk):** Video analytics framework for edge AI.
* **[Hailo Developer Zone](https://hailo.ai/developer-zone/):** Edge AI processor documentation and tools.
* **[Qualcomm AI Hub](https://aihub.qualcomm.com/):** Model optimization and deployment for Snapdragon.

**Projects:**

* Build a multi-camera video analytics system on Jetson with DeepStream: detection + tracking + counting.
* Design a battery-powered wildlife monitoring camera with duty-cycled inference; target 30-day battery life.
* Benchmark the same model across Jetson Orin Nano, Coral Edge TPU, Hailo-8, and Raspberry Pi 5 + AI HAT; compare latency, power, cost.
* Build a heterogeneous pipeline: preprocessing on CPU, detection on GPU, classification on DLA, post-processing on CPU.

---

**6. NVIDIA Jetson Holoscan for Real-Time Streaming**

* **Holoscan SDK Overview:**
    * Domain-agnostic, multimodal AI sensor processing platform for **real-time streaming** at the edge or in the cloud.
    * Combines low-latency sensor/network connectivity, optimized data-processing and AI libraries, and microservices for streaming and imaging applications on embedded, edge, and cloud.

* **Core Architecture:**
    * **Applications:** Top-level container for the pipeline.
    * **Fragments:** Logical groupings that run independently (e.g., capture, inference, visualization).
    * **Operators:** Individual units of work (IO, inference, visualization) that process streaming data.
    * Enables building sensor → preprocessing → AI inference → output pipelines with predictable latency.

* **Hardware & Stack:**
    * Supported on Jetson AGX Orin (32GB/64GB), Orin NX (16GB), Orin Nano (8GB); requires JetPack 6 (L4T r36.x).
    * **Holoscan Sensor Bridge:** High-bandwidth sensor data over Ethernet with FPGA interface support.
    * Built-in operators: HoloInfer (TensorRT integration), HoloViz (visualization), and IO operators for streaming.

* **Use Cases:**
    * Medical imaging and surgical video workflows, endoscopy tool tracking.
    * Industrial inspection, robotics, and any application requiring low-latency, multi-sensor AI pipelines.

**Resources:**

* **[NVIDIA Holoscan SDK](https://developer.nvidia.com/holoscan-sdk):** Official SDK and documentation.
* **[Holoscan SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/overview.html):** Overview, concepts, and APIs.
* **[HoloHub](https://nvidia-holoscan.github.io/holohub/):** Reference applications, operators, tutorials, and benchmarks.
* **[Getting Started with Holoscan Sensor Bridge](https://docs.nvidia.com/holoscan/sensor-bridge/latest/getting_started.html):** High-bandwidth sensor ingestion.

**Projects:**

* Run a HoloHub reference application (e.g., endoscopy tool tracking) on Jetson Orin Nano; measure end-to-end latency.
* Build a custom Holoscan fragment: camera input → TensorRT detection → HoloViz overlay; compare latency vs a standalone GStreamer/DeepStream pipeline.
* Integrate Holoscan Sensor Bridge with a high-bandwidth sensor (e.g., multi-camera or FPGA source) and run real-time inference.
