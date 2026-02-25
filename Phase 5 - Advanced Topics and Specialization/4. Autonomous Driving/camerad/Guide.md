# camerad — Openpilot Camera Pipeline

> **Goal:** Understand how openpilot captures, processes, and delivers camera frames to the perception stack. camerad is the first process in the pipeline: raw sensor → ISP → VisionIpc → modeld.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Camera Configuration](#3-camera-configuration)
4. [Hardware Stack](#4-hardware-stack)
5. [ISP Pipeline](#5-isp-pipeline)
6. [Auto Exposure (AE)](#6-auto-exposure-ae)
7. [VisionIpc](#7-visionipc)
8. [Data Flow](#8-data-flow)
9. [Source Map](#9-source-map)
10. [Key Concepts](#10-key-concepts)

---

## 1. Overview

**camerad** is a native C++ process that:

- Captures frames from up to **three cameras** (wide road, road, driver)
- Runs them through the **Qualcomm Spectra ISP** (Image Signal Processor)
- Publishes **YUV frames** via **VisionIpc** (shared memory IPC)
- Publishes **FrameData** (metadata) via cereal messaging
- Implements **auto exposure (AE)** to adapt to lighting

**Location:** `openpilot/system/camerad/`

**Entry point:** `main.cc` → `camerad_thread()` (pinned to CPU core 6)

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              camerad                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  Sensor (I2C)     │  CSI/MIPI    │  IFE (Image Front End)  │  BPS (optional)  │
│  OX03C10/OS04C10 │  → RAW       │  Demosaic, CCM, Gamma   │  Downscale        │
│                  │              │  Vignetting correction  │  (driver cam)     │
└─────────────────────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
   Exposure control                    YUV output
   (AE algorithm)                      VisionIpc buffers
         │                                    │
         └────────────────┬───────────────────┘
                          ▼
                   FrameData (cereal)
                   frame_id, timestamps, gain, exposure
```

**Three cameras (comma 3X):**

| Camera | Stream | Role | Focal length |
|--------|--------|------|--------------|
| **Wide road** | `VISION_STREAM_WIDE_ROAD` | Wide FoV, peripheral | 1.71 mm |
| **Road** | `VISION_STREAM_ROAD` | Main driving view | 8.0 mm |
| **Driver** | `VISION_STREAM_DRIVER` | Driver monitoring (DMS) | 1.71 mm |

---

## 3. Camera Configuration

Defined in `cameras/hw.h`:

```cpp
// Wide: fisheye, peripheral
WIDE_ROAD_CAMERA_CONFIG = {
  .camera_num = 0,
  .stream_type = VISION_STREAM_WIDE_ROAD,
  .focal_len = 1.71,
  .publish_name = "wideRoadCameraState",
  .output_type = ISP_IFE_PROCESSED,
};

// Road: main forward-facing, narrow FoV
ROAD_CAMERA_CONFIG = {
  .camera_num = 1,
  .stream_type = VISION_STREAM_ROAD,
  .focal_len = 8.0,
  .publish_name = "roadCameraState",
  .vignetting_correction = true,
  .output_type = ISP_IFE_PROCESSED,
};

// Driver: cabin-facing
DRIVER_CAMERA_CONFIG = {
  .camera_num = 2,
  .stream_type = VISION_STREAM_DRIVER,
  .focal_len = 1.71,
  .publish_name = "driverCameraState",
  .output_type = ISP_BPS_PROCESSED,  // BPS for extra processing
};
```

**Python intrinsics** (for modeld warp, calibration): `common/transformations/camera.py`

- Road: 1928×1208, focal 2648 px (OX) or 1344×760, focal ~1142 px (OS)
- Wide: 1928×1208, focal 567 px
- Driver: same as wide

---

## 4. Hardware Stack

### Sensors

| Sensor | Device | Resolution | Notes |
|--------|--------|------------|-------|
| **OX03C10** | comma 3X (tici/tizi) | 1928×1208 | OmniVision, 3 MP |
| **OS04C10** | comma 3X (mici) | 2688×1520 | OmniVision, 4 MP |
| **AR0231** | Legacy (neo) | 1164×874 | Aptina |

**Sensor interface:** I2C for register control (exposure, gain, init). MIPI CSI for image data.

**Files:** `sensors/ox03c10.cc`, `sensors/os04c10.cc`, `sensors/sensor.h`

### Qualcomm Spectra ISP

- **IFE** (Image Front End): demosaic, color correction (CCM), gamma, vignetting
- **BPS** (Bayer Processing Segment): used for driver camera (extra downscale/processing)
- **CDM** (Camera Data Mover): DMA, buffer management
- **CSIPHY**: MIPI CSI physical layer

**Kernel:** Linux V4L2 (Video4Linux2), `CAM_REQ_MGR` (Request Manager) for frame synchronization.

---

## 5. ISP Pipeline

```
Sensor (RAW Bayer) → CSI → IFE → [BPS for driver] → YUV (NV12)
```

**Output format:** NV12 (Y plane + interleaved UV plane). Standard format for vision/ML.

**Buffer flow:**

1. `SpectraMaster` initializes `/dev/video0`, sync, ISP, ICP
2. `SpectraCamera` per camera: sensor init, IFE config, BPS config (driver only), link devices
3. `CameraBuf` allocates raw (optional) and YUV buffers
4. `VisionIpcServer` creates shared-memory buffers for consumers (modeld, etc.)
5. On V4L2 frame event: `handle_camera_event` → `processFrame` → `sendFrameToVipc` + publish FrameData

---

## 6. Auto Exposure (AE)

camerad implements **software AE** (no sensor AEC). Goal: keep a target region at ~12.5% grey (median luminance).

**Algorithm** (`camera_qcom2.cc`):

1. **Measure:** `calculate_exposure_value()` bins luminance in an AE rectangle, returns median grey fraction
2. **Target:** `target_grey_fraction` ≈ 0.125, scaled by scene brightness (darker → lower target)
3. **Control loop:** PI-like update with ~3-frame latency (sensor register buffering)
4. **Optimizer:** Brute-force over `(exposure_time, analog_gain)` to minimize `getExposureScore` (EV error)
5. **DC gain:** High conversion gain for low light; hysteresis to avoid flicker

**AE rectangles** (per camera, in pixel coords):

- Wide: `{96, 400, 1734, 524}` — lower part of frame
- Road: `{96, 160, 1734, 986}` — most of frame
- Driver: `{96, 242, 1736, 906}` — face region

**Exposure params:** `exposure_time` (µs), `analog_gain`, `dc_gain_enabled`. Sent to sensor via I2C.

---

## 7. VisionIpc

**VisionIpc** = shared-memory IPC for video frames. Avoids copies; modeld reads directly from shared buffers.

**Server (camerad):**

```cpp
VisionIpcServer v("camerad");
v.create_buffers_with_sizes(stream_type, VIPC_BUFFER_COUNT, width, height, yuv_size, stride, uv_offset);
// ...
vipc_server->send(cur_yuv_buf, &extra);  // on each frame
```

**Client (modeld):**

```python
from msgq.visionipc import VisionIpcClient, VisionStreamType
vipc = VisionIpcClient("camerad", 0, VisionStreamType.VISION_STREAM_ROAD)
# vipc.recv() → frame buffer
```

**Stream types:** `VISION_STREAM_WIDE_ROAD`, `VISION_STREAM_ROAD`, `VISION_STREAM_DRIVER`

**Buffer count:** 18 (`VIPC_BUFFER_COUNT`). Allows producer/consumer to run at different rates.

---

## 8. Data Flow

```
V4L2 poll(POLLPRI) on video0
    │
    ▼
VIDIOC_DQEVENT (frame done)
    │
    ├─► handle_camera_event()
    │       │
    │       ├─► processFrame() → YUV ready
    │       │
    │       ├─► sendFrameToVipc() → VisionIpc send
    │       │
    │       ├─► calculate_exposure_value() → grey_frac
    │       │
    │       ├─► set_camera_exposure() → I2C exposure registers
    │       │
    │       └─► pm->send("roadCameraState", FrameData)
    │
    └─► (next frame)
```

**FrameData** (cereal): `frameId`, `timestampSof`, `timestampEof`, `integLines`, `gain`, `measuredGreyFraction`, `targetGreyFraction`, `exposureValPercent`, `sensor`, `processingTime`.

---

## 9. Source Map

| Component | Path |
|-----------|------|
| Main loop | `system/camerad/main.cc` |
| Camera state, AE | `system/camerad/cameras/camera_qcom2.cc` |
| Camera buf, VIPC send | `system/camerad/cameras/camera_common.cc` |
| ISP (IFE, BPS, CDM) | `system/camerad/cameras/spectra.cc`, `cdm.cc` |
| Sensor drivers | `system/camerad/sensors/ox03c10.cc`, `os04c10.cc` |
| Camera config | `system/camerad/cameras/hw.h` |
| Intrinsics (Python) | `common/transformations/camera.py` |
| VisionIpc | `msgq/visionipc/` |

---

## 10. Key Concepts

| Concept | Description |
|---------|-------------|
| **NV12** | YUV 4:2:0: Y plane full res, UV interleaved half res. Common for ISP output and ML. |
| **VisionIpc** | Zero-copy frame sharing via shared memory. Server creates buffers; clients subscribe by stream type. |
| **FrameData** | Cereal message with frame metadata. Used for sync (frame_id, timestamps) and AE debugging. |
| **ISP** | Image Signal Processor. Converts RAW Bayer → demosaic → color correct → gamma → YUV. |
| **AE** | Auto exposure. Software loop: measure luminance → compute desired EV → set sensor exposure/gain. |
| **Request Manager** | V4L2 CAM_REQ_MGR: ties sensor capture to ISP pipeline. One request = one frame through the pipeline. |

---

## Further Reading

- **Trace the pipeline:** Start at `camerad_thread()` in `camera_qcom2.cc`, follow `handle_camera_event` → `processFrame` → `sendFrameToVipc`
- **modeld consumption:** `selfdrive/modeld/modeld.py` — `VisionIpcClient` for `VISION_STREAM_ROAD` (and wide if used)
- **Calibration:** `common/transformations/camera.py`, `get_warp_matrix` for model input warp
