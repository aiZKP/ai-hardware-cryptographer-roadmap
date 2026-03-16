# Non-Contact Multi-Sensor Monitoring on Edge — Project Guide

> **Goal:** Build a non-contact monitoring device that fuses RGB/Depth and thermal cameras, extracts micro-fluctuation signals (0.8–3 Hz) from thermal data, and runs the full pipeline in real time on an edge device (e.g. Raspberry Pi or Jetson), with optional IoT integration (BLE, MQTT).

---

## Overview (3 sentences)

**How to achieve the goal:** Calibrate and align a dual-camera setup (RGB/Depth + thermal) so ROIs can be mapped between streams with sub-pixel accuracy; run an EVM- or FFT-based DSP pipeline to extract and bandpass-filter thermal micro-fluctuations in the 0.8–3 Hz band and validate SNR on pre-recorded, synchronized datasets. Deploy the pipeline on edge hardware with optimized NumPy/SciPy (or equivalent) so all processing runs locally in real time without cloud offload. In Phase 2, add synchronized audio capture and IoT (BLE provisioning, MQTT streaming) for live multi-modal monitoring.

---

## Table of Contents

1. [Device and Sensors](#1-device-and-sensors)
2. [Key Technical Challenges](#2-key-technical-challenges)
3. [Phase 1: Offline Pipeline](#3-phase-1-offline-pipeline)
4. [Phase 2: Live and IoT Integration](#4-phase-2-live-and-iot-integration)
5. [Why It’s Advanced](#5-why-its-advanced)
6. [Main Application](#6-main-application)
7. [Other Possible Applications](#7-other-possible-applications)
8. [Resources](#8-resources)

---

## 1. Device and Sensors

The system is a **non-contact monitoring device**: it observes and analyzes the subject remotely without physical contact.

### Sensor array

| Component | Role |
|----------|------|
| **RGB/Depth camera** | Color images + depth (distance). Used to reconstruct 3D positions and define ROIs (e.g. hand, face). |
| **Thermal camera** | Temperature across the scene. Target: detect **small, localized temperature fluctuations** that may correspond to physiological signals (blood flow, micro-movements). |
| **Audio capture** | Synchronized with video/thermal for multi-modal analysis and event alignment. |

- **Heterogeneity:** RGB/Depth and thermal have **different FOVs, resolutions, and physical alignment**. A core task is mapping a region of interest from one feed to the other (e.g. “this hand in RGB” → “these pixels in thermal”).

---

## 2. Key Technical Challenges

### A. Multi-camera sensor fusion

- **Problem:** Two heterogeneous sensors (RGB/Depth + thermal) with different fields of view, resolutions, and mounting.
- **Goal:** Map an ROI from one camera to the other. Example: detect a hand in RGB/Depth and know the **exact corresponding pixels** in the thermal image so thermal analysis is done on the right region.
- **Requirements:**
  - **Extrinsic calibration:** Rotation and translation between the two camera frames (rig or hand–eye style).
  - **Intrinsic calibration:** Per-camera lens distortion and intrinsics (e.g. with a checkerboard or dedicated calibration target).
  - **Sub-pixel accuracy:** Physiological thermal signals are subtle; alignment must be precise enough that ROI boundaries and sampling are stable.

### B. Micro-fluctuation detection in thermal data

- **Target band:** **0.8 Hz – 3 Hz** — very small, sub-pixel-level temporal variations. May correspond to:
  - Pulses or blood flow
  - Minor muscle movements
  - Minute environmental or sensor noise
- **Requirements:**
  - **Eulerian Video Magnification (EVM) or similar:** Amplify small temporal changes in the thermal video so they become measurable.
  - **Bandpass filtering:** Restrict analysis to 0.8–3 Hz to reject DC drift and higher-frequency noise.
  - **SNR optimization:** Thermal cameras are noisy; extracting such small fluctuations needs careful pipeline design (filtering, windowing, averaging, possibly spatial pooling over ROI).

### C. Edge computing constraints

- **All processing must run locally** on a small device (e.g. Raspberry Pi, Jetson Nano).
- Implications:
  - **No cloud offload** for heavy compute.
  - **Real-time capable** algorithms (bounded latency per frame or per buffer).
  - **Efficient implementations:** Optimize NumPy/SciPy (or C/Cython) so FFT, filtering, and EVM run within the frame budget; consider fixed-point or reduced precision if needed.

---

## 3. Phase 1: Offline Pipeline

Validate the full signal chain on **pre-recorded, synchronized** datasets before moving to live hardware.

1. **Input:** Pre-recorded, time-synchronized videos from RGB/Depth and thermal cameras (and optionally audio).
2. **Calibration and alignment:**
   - Compute intrinsic and extrinsic matrices for both cameras.
   - Implement ROI mapping: given an ROI in RGB/Depth (e.g. from detection or 3D), project or warp to the thermal image so the same physical region is selected in thermal.
3. **Thermal micro-fluctuation analysis:**
   - For each ROI in thermal space, run a temporal pipeline:
     - EVM (or similar) to amplify small changes.
     - Bandpass filter (0.8–3 Hz).
     - Optional: FFT or power spectral density to confirm energy in band.
   - **Validate SNR:** Ensure the hardware and pipeline yield a sufficient signal-to-noise ratio to reliably detect the target micro-signals.
4. **Output:** Clean frequency-domain (or filtered time-domain) data demonstrating that 0.8–3 Hz micro-fluctuations can be extracted and that the setup is suitable for Phase 2.

---

## 4. Phase 2: Live and IoT Integration

If Phase 1 shows reliable micro-signal extraction:

1. **Deploy on live hardware:** Run the same calibration and DSP pipeline on the edge device in real time (live RGB/Depth + thermal streams).
2. **Audio:** Add synchronized audio capture so events can be aligned across vision, thermal, and sound.
3. **IoT integration:**
   - **BLE provisioning:** Pair and configure the device with other systems (e.g. phone app, gateway) over Bluetooth Low Energy.
   - **MQTT:** Stream extracted signals or summaries in real time for dashboards, logging, or cloud backup.
4. **Goal:** A fully operational device that captures and processes multi-modal signals (RGB, depth, thermal, audio) with precise alignment and optional real-time streaming, all on the edge.

---

## 5. Why It’s Advanced

- **Real-time multi-sensor fusion:** Few systems combine RGB/Depth and thermal with live, aligned processing on small devices.
- **Micro-signal extraction:** Sub-pixel thermal fluctuation detection in the 0.8–3 Hz band is at the intersection of physiology, signal processing, and computer vision.
- **Edge + IoT:** Combines CV, calibration, DSP, and embedded/IoT (BLE, MQTT) under strict latency and resource limits.

---

## 6. Main Application

**Contactless baby / child vital monitoring** — the primary target use case for this project. Devices in this category (e.g. [iBaby Labs i20](https://ibabylabs.com/)) provide **contactless breathing and heart rate monitoring** so caregivers can track a baby’s vitals while they sleep—without wearables, clips, or skin contact.

- **Breathing rate:** Inferred from chest/abdomen micro-motion (e.g. optical flow or subtle intensity changes in RGB/thermal over an ROI). The 0.8–3 Hz band and EVM-style amplification align with respiratory rates.
- **Heart rate:** Inferred from **remote PPG (rPPG)** or analogous optical sensing: tiny, periodic changes in skin tone or thermal signature caused by blood pulsation are captured by the camera and processed (bandpass, phase analysis) to derive pulse. No contact required; works with night vision in low light.
- **Edge + AI:** An on-device NPU or CPU runs real-time contactless analysis of heart rate, breathing, and micro-movements; alerts (e.g. safe zone, face cover) and insights (sleep quality, mood cues) can be sent to a phone app. BLE/Wi‑Fi and optional cloud sync support parental dashboards and peace of mind.
- **Why it fits this project:** The same technical stack—multi-sensor (RGB/Depth + thermal) fusion, ROI alignment, micro-fluctuation extraction (0.8–3 Hz), and edge deployment—directly supports building a baby/child monitor that offers “invisible care, visible peace”: continuous vital monitoring without disrupting sleep or requiring wearables.

---

## 7. Other Possible Applications

Other uses of a non-contact, multi-sensor (RGB/Depth + thermal) device that extracts micro-fluctuations (0.8–3 Hz) on the edge:

| Domain | Application |
|--------|-------------|
| **Vital signs (adult)** | Contactless heart rate / pulse and respiration from facial or body ROIs; care homes, hospitals, or home monitoring where skin sensors are undesirable. |
| **Stress & arousal** | Thermal and subtle motion in face/body; 0.8–3 Hz band for physiological rhythms. Driver drowsiness, workplace wellness, mental load. |
| **Sleep & rest** | Non-contact breathing and movement during sleep; thermal in low light; edge + MQTT for dashboards or caregivers. |
| **Fall & activity** | RGB/Depth for pose and fall detection; thermal for robustness in dark/clutter; edge alerts (e.g. elderly living alone). |
| **Industrial & security** | Thermal for presence, overheating, or anomalies; RGB/Depth for occupancy and 3D; edge for privacy and latency. |
| **Research & physiology** | Lab/field studies with synced RGB, depth, thermal, audio; export or MQTT for analysis. |
| **Accessibility & assistive tech** | Hands-free vital/state monitoring for users who cannot wear sensors; BLE/MQTT to apps or caregivers. |

---

## 8. Resources

- **Eulerian Video Magnification (EVM)** — original paper and implementations for amplifying subtle temporal changes in video.
- **Multi-camera calibration:** OpenCV camera calibration, stereo/rig calibration; thermal–RGB alignment (e.g. FLIR/opencv thermal examples).
- **Bandpass filtering / FFT:** SciPy signal processing (butter, filtfilt, fft, welch PSD) for 0.8–3 Hz and SNR estimation.
- **Edge:** NumPy/SciPy optimization; Raspberry Pi or Jetson performance tuning; optional Cython or C for hot loops.
- **IoT:** BLE (e.g. BlueZ, bleak); MQTT (e.g. paho-mqtt) for streaming.

---

*Back to [Edge AI Optimization — Projects](../Guide.md#14-projects).*
