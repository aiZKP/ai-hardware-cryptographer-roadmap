# Project: 1080p → 4K Wireless Video on FPGA

**Parent:** [Phase 4 Track A — Xilinx FPGA](../1.%20Xilinx%20FPGA%20Development/Guide.md)

**Layers touched:** L3 (runtime/driver), L4 (firmware), L5 (hardware architecture), L6 (RTL/HLS)

**Prerequisites:** Track A §1–§4 (Vivado, Zynq, Advanced FPGA, HLS), Phase 2 (Embedded Linux, FreeRTOS).

---

## Overview

Two-phase project that builds a wireless video link on FPGA — starting with a rapid 1080p proof-of-concept using SDR + Raspberry Pi, then evolving into a fully integrated 4K system on Zynq UltraScale+ with custom PHY, bare-metal firmware, and an ASIC transition plan.

This project exercises nearly every Track A skill: Vivado IP integration, Zynq PS/PL co-design, HLS for video processing, Linux and bare-metal firmware, and runtime/driver development for DMA and streaming.

---

## Plan 1: 1080p Wireless Video Proof-of-Concept — Detailed Execution Plan

**Timeline:** 2–3 weeks (14–18 working days)

**Goal:** Build a functional 1080p60 wireless video link using SDR (antsdr + openwifi) with external Raspberry Pi for H.264 encode/decode.

---

### 1.1 Bill of Materials (Order Day 1)

| Component | Qty | Part Number | Est. Cost | Shipping |
|-----------|-----|-------------|-----------|----------|
| ANTSDR E310 (Zynq 7020 + AD9361) | 2 | MicroPhase ANTSDR E310 | $300–350 each | 2–3 days |
| Raspberry Pi 4 (4GB RAM) | 2 | RPi4B-4GB | $75 each | Same-day local |
| Raspberry Pi Camera Module v2 | 1 | RPi Camera v2 | $30 | Same-day |
| HDMI Cable | 1 | Standard HDMI | $10 | Local |
| Ethernet Cables (Cat5e, 2m) | 2 | Generic | $5 each | Local |
| Antennas (2.4/5 GHz SMA) | 2 | 5 dBi dual-band | $15 each | 2–3 days |
| MicroSD Cards (32GB Class 10) | 4 | SanDisk Ultra | $12 each | Local |
| Power Supplies (5V/3A USB-C) | 2 | For RPi4 | $10 each | Local |

**Total hardware cost: ~$1,000–1,100**

### 1.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TRANSMITTER SIDE                              │
│                                                                         │
│  ┌──────────┐    ┌────────────────┐    ┌────────────┐    ┌──────────┐  │
│  │  Camera   │──▶│ Raspberry Pi 4 │──▶│  ANTSDR TX  │──▶│ Antenna  │  │
│  │ Module v2 │   │ (H.264 Encode) │   │ (openwifi)  │   │          │  │
│  └──────────┘    └────────────────┘    └────────────┘    └────┬─────┘  │
│                    192.168.10.50        192.168.10.122         │        │
│                         Ethernet (UDP)      │            5 GHz RF      │
└─────────────────────────────────────────────┼─────────────────┼────────┘
                                              │                 │
                                              │    Wireless     │
                                              │   (802.11a/n)   │
┌─────────────────────────────────────────────┼─────────────────┼────────┐
│                           RECEIVER SIDE     │                 │        │
│                                             │                 │        │
│  ┌──────────┐    ┌────────────────┐    ┌────────────┐    ┌──────────┐  │
│  │ Display  │◀──│ Raspberry Pi 4 │◀──│  ANTSDR RX  │◀──│ Antenna  │  │
│  │ Monitor  │   │ (H.264 Decode) │   │ (openwifi)  │   │          │  │
│  └──────────┘    └────────────────┘    └────────────┘    └──────────┘  │
│                    192.168.10.51        192.168.10.123                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 IP Address Plan

| Device | Interface | IP Address | Role |
|--------|-----------|------------|------|
| ANTSDR TX | eth0 | 192.168.10.122 | SDR transmitter |
| ANTSDR RX | eth0 | 192.168.10.123 | SDR receiver |
| RPi TX | eth0 | 192.168.10.50 | Video capture/encode |
| RPi RX | eth0 | 192.168.10.51 | Video decode/display |
| ANTSDR TX | wlan0 | 192.168.13.1 | Ad-hoc network (TX) |
| ANTSDR RX | wlan0 | 192.168.13.2 | Ad-hoc network (RX) |

### 1.4 MicroSD Card Preparation (Day 1, Evening)

| Card | Device | Image | Purpose |
|------|--------|-------|---------|
| Card 1 | ANTSDR TX | openwifi-1.4.0-notter.img | Transmitter SDR |
| Card 2 | ANTSDR RX | openwifi-1.4.0-notter.img | Receiver SDR |
| Card 3 | RPi TX | Raspberry Pi OS Lite (64-bit) | Video capture + encode |
| Card 4 | RPi RX | Raspberry Pi OS Lite (64-bit) | Video decode + display |

```bash
# ── ANTSDR Images ──────────────────────────────────────────
wget https://github.com/open-sdr/openwifi-hw-img/releases/download/v1.4.0/openwifi-1.4.0-notter.img.xz
xz -d openwifi-1.4.0-notter.img.xz
sudo dd if=openwifi-1.4.0-notter.img of=/dev/sdb bs=1M status=progress
sync

# ── Raspberry Pi Images ────────────────────────────────────
# Use Raspberry Pi Imager (GUI) or:
wget https://downloads.raspberrypi.org/raspios_lite_arm64_latest
# Flash, then enable SSH:
touch /boot/ssh
```

**Hardware assembly checklist:**
- [ ] Insert pre-flashed SD card into ANTSDR TX and RX boards
- [ ] Attach antennas to both ANTSDR boards (SMA connectors)
- [ ] Connect Ethernet: ANTSDR TX ↔ RPi TX, ANTSDR RX ↔ RPi RX
- [ ] Connect camera module to RPi TX (ribbon cable)
- [ ] Connect HDMI from RPi RX to display monitor
- [ ] Power on all devices (ANTSDR first, then RPi)

---

### 1.5 Week 1: Foundation & Parallel Setup

#### Day 1–2: SDR Link Setup (Stream A)

**Boot and configure ANTSDR TX:**
```bash
ssh root@192.168.10.122
# Password: analog (changes to openwifi after post_config)

cd ~/openwifi && ./post_config.sh

# Verify openwifi loaded
lsmod | grep xpu
cat /sys/kernel/debug/adi/axi_ad9361/status
# Expected: "AD9361 Rev 2 initialized"

# Configure ad-hoc network
ifconfig wlan0 down
iwconfig wlan0 mode ad-hoc essid openwifi-adhoc channel 36
ifconfig wlan0 192.168.13.1 netmask 255.255.255.0 up
```

**Boot and configure ANTSDR RX (same process):**
```bash
ssh root@192.168.10.123
cd ~/openwifi && ./post_config.sh
ifconfig wlan0 down
iwconfig wlan0 mode ad-hoc essid openwifi-adhoc channel 36
ifconfig wlan0 192.168.13.2 netmask 255.255.255.0 up
```

**Test wireless link:**
```bash
# On RX:
iperf -s -i 1

# On TX:
iperf -c 192.168.13.2 -i 1 -t 30
# Target: >50 Mbps stable (sufficient for 1080p60 H.264 at 20-30 Mbps)

# Latency test (on TX):
ping -i 0.1 192.168.13.1
# Expected: 0.6-0.7 ms (openwifi reference)
```

#### Day 1–2: Raspberry Pi Video Setup (Stream B — Parallel)

**On both RPi TX and RPi RX:**
```bash
ssh pi@192.168.10.50   # or .51

sudo raspi-config
# Expand Filesystem, Enable Camera (TX only)

sudo apt update && sudo apt upgrade -y
sudo apt install -y gstreamer1.0-tools \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    gstreamer1.0-omx gstreamer1.0-omx-rpi-config
```

**Verify camera (RPi TX only):**
```bash
vcgencmd get_camera
# Expected: supported=1 detected=1

# Test hardware encoding (10 seconds)
gst-launch-1.0 libcamerasrc ! \
    capsfilter caps=video/x-raw,width=1920,height=1080,framerate=30/1 ! \
    v4l2h264enc extra-controls="controls,repeat_sequence_header=1,video_bitrate=20000000" ! \
    h264parse ! filesink location=test.h264

ls -lh test.h264
# Expected: ~25MB for 10 seconds at 20Mbps
```

#### Day 2–3: Routing Configuration (Stream C — Parallel)

**On both ANTSDR boards — enable IP forwarding:**
```bash
echo 1 > /proc/sys/net/ipv4/ip_forward
echo "net.ipv4.ip_forward=1" >> /etc/sysctl.conf

iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
iptables -A FORWARD -i wlan0 -o eth0 -j ACCEPT
iptables -A FORWARD -i eth0 -o wlan0 -j ACCEPT
iptables-save > /etc/iptables.rules
```

**On RPi TX:**
```bash
sudo ip route add 192.168.13.0/24 via 192.168.10.122
```

**On RPi RX:**
```bash
sudo ip route add 192.168.13.0/24 via 192.168.10.123
```

**Test end-to-end connectivity:**
```bash
# From RPi TX:
ping 192.168.10.51
# Should succeed with ~2-3ms additional latency
```

---

### 1.6 Week 2: Integration & Streaming

#### Day 4–5: TX Video Pipeline

Create `/home/pi/stream_tx.sh` on RPi TX:

```bash
#!/bin/bash
BITRATE=20000000    # 20 Mbps
FPS=60
WIDTH=1920
HEIGHT=1080
TARGET_IP=192.168.10.51
TARGET_PORT=5000

if ! vcgencmd get_camera | grep -q "detected=1"; then
    echo "Camera not detected!"; exit 1
fi

gst-launch-1.0 -v \
    libcamerasrc ! \
    video/x-raw,width=$WIDTH,height=$HEIGHT,framerate=$FPS/1,format=NV12 ! \
    v4l2convert ! \
    v4l2h264enc \
        extra-controls="controls,repeat_sequence_header=1,video_bitrate=$BITRATE,video_bitrate_mode=0" \
        ! 'video/x-h264,level=(string)4.2,profile=high' ! \
    h264parse ! \
    rtph264pay config-interval=1 pt=96 ! \
    udpsink host=$TARGET_IP port=$TARGET_PORT sync=0
```

Key parameters:
- `v4l2h264enc` — RPi hardware H.264 encoder
- `repeat_sequence_header=1` — sends SPS/PPS every I-frame (critical for decoder sync)
- `sync=0` — disables clock sync for lowest latency

#### Day 5–6: RX Video Pipeline

Create `/home/pi/stream_rx.sh` on RPi RX:

```bash
#!/bin/bash
LISTEN_PORT=5000

gst-launch-1.0 -v \
    udpsrc port=$LISTEN_PORT buffer-size=1048576 ! \
    application/x-rtp,media=video,encoding-name=H264,payload=96 ! \
    rtph264depay ! \
    h264parse ! \
    v4l2h264dec ! \
    video/x-raw,width=1920,height=1080 ! \
    autovideosink sync=0
```

Key parameters:
- `buffer-size=1048576` — 1MB buffer to handle network jitter
- `v4l2h264dec` — RPi hardware decoder
- `sync=0` — lowest latency display

#### Day 6–7: End-to-End Streaming Test

**Startup order:**
1. ANTSDR RX → `ifconfig wlan0 up`
2. RPi RX → `./stream_rx.sh`
3. ANTSDR TX → `ifconfig wlan0 up`
4. RPi TX → `./stream_tx.sh`

**Expected:** Video appears on HDMI display within 2–3 seconds. Smooth 1080p60 playback.

**Latency measurement:**
```bash
# On RPi TX — add timestamp overlay:
gst-launch-1.0 libcamerasrc ! \
    video/x-raw,width=1920,height=1080,framerate=60/1 ! \
    timeoverlay halignment=left valignment=top ! \
    v4l2h264enc extra-controls="controls,video_bitrate=20000000" ! \
    rtph264pay ! udpsink host=192.168.10.51 port=5000
```
Photograph both screens with phone camera to measure offset. Target: < 100 ms.

---

### 1.7 Week 3: Validation & Documentation

#### Day 8–9: Performance Characterization

**Throughput test matrix:**

| Configuration | Expected Throughput | Measured | Notes |
|---------------|---------------------|----------|-------|
| 20 MHz channel, QPSK | 50 Mbps | | Baseline |
| 20 MHz channel, 16-QAM | 100 Mbps | | Default openwifi |
| Distance 1m LOS | | | |
| Distance 10m LOS | | | |
| Through wall | | | |

**Latency breakdown target:**

| Stage | Target | Measurement method |
|-------|--------|-------------------|
| Capture | < 5 ms | Camera exposure time |
| Encode | < 10 ms | v4l2h264enc profiling |
| Network TX | < 2 ms | Wire capture |
| Wireless | < 5 ms | Ping from ANTSDR |
| Network RX | < 2 ms | Wire capture |
| Decode | < 10 ms | v4l2h264dec profiling |
| Display | < 10 ms | HDMI output delay |

#### Day 10–11: Low-Latency Optimization

**Reduce encoding latency (RPi TX):**
```bash
# Use intra-refresh instead of periodic I-frames
v4l2h264enc extra-controls="controls,repeat_sequence_header=1,\
    video_bitrate=20000000,h264_i_frame_period=0"
```

**Optimize PHY (ANTSDR):**
```bash
# Set fixed rate (no rate adaptation)
sdrctl dev sdr0 set reg rate 0 54000000

# Increase TX power (check local regulations)
iwconfig wlan0 txpower 20

# Set fixed rate
iwconfig wlan0 rate 54M fixed
```

#### Day 12–14: Documentation

**Deliverables checklist:**
- [ ] Hardware configuration guide with photos and wiring diagrams
- [ ] Automated setup scripts for all 4 devices
- [ ] GStreamer pipeline library with performance notes
- [ ] Performance report: throughput, latency, range data
- [ ] Troubleshooting guide
- [ ] Video demonstration recording

**Repository structure:**
```
wireless-video-prototype/
├── README.md
├── hardware/
│   ├── wiring_diagram.png
│   └── bom.csv
├── software/
│   ├── antsdr/
│   │   ├── setup.sh
│   │   ├── network_config.sh
│   │   └── firewall_rules.sh
│   ├── rpi_tx/
│   │   ├── setup.sh
│   │   ├── stream_tx.sh
│   │   └── measure_latency.sh
│   └── rpi_rx/
│       ├── setup.sh
│       └── stream_rx.sh
├── pipelines/
│   ├── tx_baseline.txt
│   ├── tx_low_latency.txt
│   ├── rx_baseline.txt
│   └── rx_low_latency.txt
├── docs/
│   ├── performance_report.pdf
│   └── troubleshooting.md
└── demo/
    └── demo_video.mp4
```

---

### 1.8 Troubleshooting Guide

**ANTSDR issues:**

| Symptom | Cause | Solution |
|---------|-------|---------|
| SSH refused | Board not booted | Check LEDs: green solid = booted |
| `iwconfig` no wireless | Driver not loaded | `modprobe xpu`; check `lsmod | grep xpu` |
| Low throughput (<20 Mbps) | Interference | Change channel: `iwconfig wlan0 channel 40` |
| High ping (>2 ms) | Retransmissions | `iwconfig wlan0 retry 0` |

**Raspberry Pi issues:**

| Symptom | Cause | Solution |
|---------|-------|---------|
| Camera not detected | Not enabled | `sudo raspi-config` → Interface → Camera |
| Encoding 5 fps only | Wrong encoder | Use `v4l2h264enc`, not software encoder |
| `v4l2h264enc` not found | Missing plugin | `sudo apt install gstreamer1.0-omx` |
| No video on display | HDMI not detected | Check cable; `tvservice -p` to force |

**Network issues:**

| Symptom | Cause | Solution |
|---------|-------|---------|
| Cannot ping across link | IP forwarding off | `echo 1 > /proc/sys/net/ipv4/ip_forward` |
| UDP not arriving | Firewall | `iptables -F` to flush rules temporarily |

**Quick recovery:**
```bash
# Reset ANTSDR wireless
ifconfig wlan0 down
iwconfig wlan0 mode ad-hoc essid openwifi-adhoc channel 36
ifconfig wlan0 192.168.13.1 netmask 255.255.255.0 up

# Reset routing
iptables -F && iptables -t nat -F
echo 1 > /proc/sys/net/ipv4/ip_forward
```

---

### 1.9 Success Criteria

**MVP (must have):**
- [ ] 1080p60 video captured from camera
- [ ] Hardware H.264 encoding at 20 Mbps
- [ ] UDP transmission from RPi TX → ANTSDR TX → wireless → ANTSDR RX → RPi RX
- [ ] Hardware H.264 decoding and display on HDMI monitor
- [ ] End-to-end latency < 150 ms

**Stretch goals:**
- [ ] Latency < 80 ms
- [ ] Distance > 20 meters LOS
- [ ] 40 MHz channel operation
- [ ] Automatic reconnection after disconnect

---

## Plan 2: Integrated 4K Wireless Chip Prototype — Detailed Execution Plan

**Timeline:** 8 weeks (2 months)

**Goal:** Build a fully integrated 4Kp60 wireless transmitter/receiver on Zynq UltraScale+ EV (ZCU106) with hardened VCU, MIPI CSI/DSI interfaces, custom low-latency wireless PHY/MAC, and ASIC transition plan. Aggressive parallelization across 3 work streams compresses the schedule.

---

### 2.1 Bill of Materials

| Component | Qty | Part Number | Est. Cost | Lead Time |
|-----------|-----|-------------|-----------|-----------|
| ZCU106 Evaluation Kit (ZU7EV) | 2 | Xilinx EK-U1-ZCU106-G | $3,500 each | 3–5 days |
| FMCOMMS5 (AD9361 2x2 MIMO) | 2 | Analog Devices FMCOMMS5 | $1,500 each | 5–7 days |
| MIPI CSI-2 Camera Module | 2 | LI-IMX274MIPI-FMC | $400 each | 5–7 days |
| MIPI DSI Display Panel | 1 | AUO B101UAN01.7 or similar | $200 | 5–7 days |
| HDMI Monitor (4K60) | 1 | Any 4K60 HDMI | $300 | Local |
| FMC-to-MIPI Adapter | 2 | Digilent or custom | $150 each | 7–10 days |
| Antennas (5.8 GHz SMA) | 4 | 5 dBi dual-band | $20 each | 2–3 days |
| MicroSD Cards (32GB) | 4 | SanDisk Ultra | $12 each | Local |

**Total hardware cost: ~$12,000–13,000** (two complete TX/RX sets)

### 2.2 Hardware Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                 TRANSMITTER — ZCU106 + FMCOMMS5                   │
│                                                                   │
│  ┌────────────┐   ┌─────────────────────────────────────────┐    │
│  │ MIPI CSI-2 │   │         Zynq UltraScale+ EV (ZU7EV)    │    │
│  │ Camera     │   │                                         │    │
│  │ (IMX274)   │   │  PS: Cortex-A53 (4x) + DDR4            │    │
│  └─────┬──────┘   │  ┌─────────────────────────────────┐   │    │
│        │ D-PHY    │  │     Programmable Logic (PL)      │   │    │
│        ▼          │  │                                   │   │    │
│  ┌──────────┐     │  │  MIPI CSI-2 RX → Demosaic       │   │    │
│  │ MIPI RX  │────▶│  │       ↓                          │   │    │
│  │ Subsystem│     │  │  VCU Encoder (Hardened, 4K60)    │   │    │
│  └──────────┘     │  │       ↓                          │   │    │
│                   │  │  Custom Packetizer                │   │    │
│                   │  │       ↓                          │   │    │
│                   │  │  Wireless PHY (OFDM TX)          │   │    │
│                   │  │       ↓                          │   │    │
│                   │  │  TDMA MAC Controller              │   │    │
│                   │  └──────────┬────────────────────────┘   │    │
│                   └─────────────┼────────────────────────────┘    │
│                                 │ FMC                             │
│                   ┌─────────────▼──────────────────┐             │
│                   │  FMCOMMS5 (AD9361)             │             │
│                   │  2x2 MIMO · 5.8 GHz · 40 MHz  │──▶ Antenna │
│                   └────────────────────────────────┘             │
└───────────────────────────────────────────────────────────────────┘

Receiver: Antenna → AD9361 → PHY RX → Depacketizer → VCU Decode → Display
```

### 2.3 Development Tools & Licenses

| Tool | Version | License | Purpose |
|------|---------|---------|---------|
| Vivado Design Suite | 2020.2+ | **Enterprise** (for ZU7EV) | FPGA design, synthesis, implementation |
| Vitis (SDK) | 2020.2+ | Included | Bare-metal/FreeRTOS firmware |
| PetaLinux | 2020.2+ | Included | Linux BSP (initial validation only) |
| MIPI CSI-2 RX Subsystem | v4.0+ | **Eval (60 day)** or purchased | Camera input |
| MIPI DSI TX Subsystem | v4.0+ | **Eval (60 day)** or purchased | Display output |

**License cost estimate:** ~$5,000–8,000 for permanent, or use 60-day eval for prototype.

### 2.4 Software Architecture

```
┌───────────────────────────────────────────────────────────────┐
│  APPLICATION — Custom Bare-Metal / FreeRTOS                   │
│  Video pipeline control · TDMA scheduling · System init       │
├───────────────────────────────────────────────────────────────┤
│  MIDDLEWARE — Xilinx VCS Libraries (VCU Control)              │
│  Encoder/decoder config · Buffer management · DMA sharing     │
├───────────┬──────────────┬──────────────┬─────────────────────┤
│ VCU Driver│ MIPI CSI-2   │ Wireless PHY │ AD9361 SPI Driver   │
│           │ Driver       │ Driver       │                     │
├───────────┼──────────────┼──────────────┼─────────────────────┤
│ VCU (HW)  │ MIPI D-PHY   │ Wireless PHY │ AD9361 RF           │
│ Enc/Dec   │ CSI/DSI      │ (PL)         │ Front-end           │
└───────────┴──────────────┴──────────────┴─────────────────────┘
```

---

### 2.5 Phase 1 — Platform + VCU + Wireless (Weeks 1–4, All Parallel)

All three streams run simultaneously from day 1. By week 4 you have VCU encoding, MIPI camera capture, and wireless link — all validated independently, ready to merge.

#### Stream A (Week 1–2): Development Environment & Base Platform

**Day 1–2: Install tools**
```bash
source /tools/Xilinx/Vivado/2020.2/settings64.sh
source /tools/Xilinx/Vitis/2020.2/settings64.sh
```

**Day 3–4: Create base hardware design**
```tcl
create_project zcu106_vcu_base ./zcu106_vcu_base -part xczu7ev-ffvc1156-2-e
create_bd_design "base_design"

# Add Zynq UltraScale+ MPSoC with ZCU106 board preset
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.3 zynq_ultra_ps_e_0
apply_board_preset -board xilinx.com:zcu106:part0:1.1 [get_bd_cells zynq_ultra_ps_e_0]

# Add VCU encoder and decoder IPs
create_bd_cell -type ip -vlnv xilinx.com:ip:vcuenc:1.0 vcuenc_0
create_bd_cell -type ip -vlnv xilinx.com:ip:vcudec:1.0 vcudec_0
# Connect VCU to PS via AXI, configure for 4K60
```

**Day 5: Generate bitstream**
```bash
make_wrapper -files [get_files .../base_design.bd] -top
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
write_hw_platform -include_bit -force -file ./zcu106_vcu_base.xsa
```

**Day 6–7: Bare-metal VCU test**
```c
// main.c — Minimal VCU initialization test
#include "xvcuenc.h"
#include "xvcudec.h"

int main() {
    XVcuenc encoder;
    XVcuenc_Config *enc_cfg = XVcuenc_LookupConfig(XPAR_XVCUENC_0_DEVICE_ID);
    XVcuenc_CfgInitialize(&encoder, enc_cfg, enc_cfg->BaseAddress);
    xil_printf("VCU Encoder initialized\r\n");

    // Configure for 4K60 H.265
    XVcuenc_EncConfig enc_config = {
        .width = 3840, .height = 2160, .frame_rate = 60,
        .bitrate = 40000000,  // 40 Mbps
        .codec_type = XVCUENC_CODEC_TYPE_HEVC,
        .gop_length = 60
    };
    XVcuenc_Configure(&encoder, &enc_config);
    xil_printf("VCU configured for 4K60 H.265\r\n");
    return 0;
}
```

#### Stream A (Week 2–3): VCU Encoding/Decoding Validation

**Add test pattern generator to PL:**
```tcl
create_bd_cell -type ip -vlnv xilinx.com:ip:v_tpg:1.0 v_tpg_0
set_property -dict [list CONFIG.C_MAX_COLS {3840} CONFIG.C_MAX_ROWS {2160}] [get_bd_cells v_tpg_0]
# Connect TPG → VCU encoder input via AXI4-Stream
```

**Validate with GStreamer on Linux (before bare-metal port):**
```bash
# Encode test pattern
gst-launch-1.0 videotestsrc ! video/x-raw,width=3840,height=2160,framerate=60/1 ! \
    omxh265enc target-bitrate=40000000 ! filesink location=test_4k60.h265

# Decode and display
gst-launch-1.0 filesrc location=test_4k60.h265 ! h265parse ! omxh265dec ! kmssink
```

**VCU performance targets:**

| Metric | Target |
|--------|--------|
| 4K60 H.265 encode latency | < 10 ms |
| 4K60 H.265 decode latency | < 8 ms |
| Simultaneous encode + decode | Yes |
| Bitrate range | 20–100 Mbps |

#### Stream A (Week 3–4): MIPI Camera Input

**MIPI D-PHY pin constraints (ZCU106 HP I/O bank):**
```tcl
set_property PACKAGE_PIN AK30 [get_ports {mipi_clk_p}]
set_property IOSTANDARD LVDS [get_ports {mipi_clk_p}]
set_property PACKAGE_PIN AL30 [get_ports {mipi_clk_n}]
set_property IOSTANDARD LVDS [get_ports {mipi_clk_n}]
# Data lanes similar — UltraScale+ supports MIPI D-PHY natively on HP I/O
set_property DIFF_TERM FALSE [get_ports {mipi_*_p}]  # External 100 ohm termination
```

**Add MIPI CSI-2 RX Subsystem:**
```tcl
create_bd_cell -type ip -vlnv xilinx.com:ip:mipi_csi2_rx_subsystem:4.0 mipi_csi2_rx_0
set_property -dict [list CONFIG.C_NUM_LANES {4} CONFIG.C_LANE_RATE {1440} \
    CONFIG.C_DATA_TYPE {RAW10}] [get_bd_cells mipi_csi2_rx_0]
# Connect MIPI RX → demosaic → VCU encoder
```

**I2C sensor configuration (IMX274):**
```c
XIicPs i2c;
XIicPs_Config *cfg = XIicPs_LookupConfig(XPAR_XIICPS_0_DEVICE_ID);
XIicPs_CfgInitialize(&i2c, cfg, cfg->BaseAddress);
// Initialize IMX274 for 4K60 mode via I2C register writes
```

#### Stream B (Week 1–3, parallel with Stream A): Wireless PHY + TDMA

**Port openwifi to ZCU106:**
```bash
git clone https://github.com/open-sdr/openwifi-hw.git
cd openwifi-hw
# Key modifications for UltraScale+:
#   - Update clocking (MMCM → PLL)
#   - Adjust AXI interface widths
#   - Update address maps for ZCU106
```

**Replace CSMA/CA with TDMA controller:**
```verilog
module tdma_controller (
    input  wire        clk,
    input  wire        rst,
    input  wire [31:0] slot_duration,   // microseconds
    input  wire [7:0]  my_slot_id,
    input  wire [7:0]  total_slots,
    output reg         tx_enable,
    output reg         rx_enable
);
    reg [31:0] slot_timer;
    reg [7:0]  current_slot;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            slot_timer   <= 0;
            current_slot <= 0;
            tx_enable    <= 0;
            rx_enable    <= 0;
        end else begin
            if (slot_timer >= slot_duration) begin
                slot_timer   <= 0;
                current_slot <= (current_slot + 1) % total_slots;
            end else begin
                slot_timer <= slot_timer + 1;
            end
            tx_enable <= (current_slot == my_slot_id);
            rx_enable <= (current_slot != my_slot_id);
        end
    end
endmodule
```

**TDMA synchronization (beacon-based):**
```c
typedef struct {
    uint64_t master_time;
    int64_t  offset;
} sync_state_t;

void sync_receive_beacon(uint64_t master_timestamp) {
    uint64_t local = get_timer_64();
    int64_t new_offset = (int64_t)master_timestamp - (int64_t)local;
    sync.offset = (sync.offset * 7 + new_offset) / 8;  // Low-pass filter
    adjust_timer(sync.offset);
}
```

#### Stream C (Week 1–3, parallel): AD9361 RF Front-End

**Bare-metal AD9361 driver via SPI:**
```c
int ad9361_init(void) {
    XSPi_Config *spi_cfg = XSpi_LookupConfig(XPAR_XSPI_0_DEVICE_ID);
    XSpi_CfgInitialize(&spi, spi_cfg, spi_cfg->BaseAddress);
    XSpi_SetOptions(&spi, XSP_MASTER_OPTION);

    ad9361_soft_reset();
    ad9361_set_frequency(5800);       // 5.8 GHz
    ad9361_set_bandwidth(40000000);   // 40 MHz
    ad9361_set_tx_gain(10);           // 10 dB
    ad9361_set_rx_gain(20);           // 20 dB
    ad9361_enable();
    return XST_SUCCESS;
}
```

**AD9361 key parameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Frequency | 5.8 GHz | ISM band, less interference |
| Bandwidth | 40 MHz | Supports compressed 4K bitrate |
| Sample Rate | 61.44 MSPS | AD9361 standard |
| TX Gain | 5–15 dB | Adjust for range |
| RX Gain | 15–25 dB | Auto or manual |

#### Week 4: Merge — Video + Wireless Integration

**Custom packetizer (video frames → wireless packets):**
```verilog
module video_packetizer (
    input  wire         clk, rst,
    // Video input from VCU (AXI4-Stream)
    input  wire [255:0] video_tdata,
    input  wire         video_tvalid, video_tlast,
    output wire         video_tready,
    // Packet output to wireless PHY
    output reg  [31:0]  pkt_tdata,
    output reg          pkt_tvalid, pkt_tlast,
    input  wire         pkt_tready
);
    // Fragment video NALUs into 1500-byte wireless packets
    // Add sequence number + timestamp + frame type header
    // ...
endmodule
```

**End-to-end test between two ZCU106 boards:**
1. Power on TX board (camera + encode + packetize + PHY TX)
2. Power on RX board (PHY RX + depacketize + decode + display)
3. Configure TX as TDMA master (slot 0), RX as slave (slot 1)
4. Monitor display for 4K video output
5. Measure latency with GPIO toggles on oscilloscope

**Phase 1 deliverables (end of week 4):**
- [ ] VCU 4K60 encode/decode validated (Linux + bare-metal init)
- [ ] MIPI camera capture at 4K60
- [ ] openwifi PHY ported to ZCU106 with TDMA MAC
- [ ] AD9361 bare-metal driver working
- [ ] Packetizer/depacketizer integrated
- [ ] First 4K60 wireless video stream between two boards

---

### 2.6 Phase 2 — Integration, Optimization & ASIC Roadmap (Weeks 5–8)

#### Week 5–6: FreeRTOS Integration + Latency Optimization

```c
// FreeRTOS task allocation across quad-core A53
// Core 0: System control + TDMA scheduling
void vSystemControlTask(void *p) {
    for (;;) { tdma_schedule(); vTaskDelay(pdMS_TO_TICKS(1)); }
}

// Core 1: Video capture + encoding
void vVideoTxTask(void *p) {
    for (;;) { capture_frame(); vcu_encode(); packetize_and_send(); vTaskDelay(pdMS_TO_TICKS(16)); }
}

// Core 2: Wireless receive + decoding
void vVideoRxTask(void *p) {
    for (;;) { receive_packet(); depacketize(); vcu_decode(); display_frame(); }
}

// Core 3: RF control + monitoring
void vRFControlTask(void *p) {
    for (;;) { monitor_rssi(); adjust_gain_if_needed(); vTaskDelay(pdMS_TO_TICKS(100)); }
}
```

FreeRTOS and latency optimization run in the same sprint — assign tasks to cores on day 1 of week 5, then tune latency for the rest of the sprint.

#### Week 5–6 continued: Latency Optimization

**VCU low-latency encoder settings:**
```c
enc_config.gop_mode     = XVCUENC_GOP_MODE_LOW_DELAY_P;  // IPPPPP...
enc_config.gop_length   = 1;      // Intra-refresh, no periodic I-frames
enc_config.slice_size   = 8;      // More slices for better parallelism
enc_config.low_latency  = 1;
enc_config.rc_mode      = XVCUENC_RC_MODE_HW;
enc_config.qp_mode      = XVCUENC_QP_MODE_AUTO;
```

**Latency budget:**

| Stage | Target | Method |
|-------|--------|--------|
| Camera capture | < 5 ms | Minimal buffering |
| VCU encode | < 8 ms | Low-delay RC, intra-refresh |
| Packetization | < 2 ms | DMA direct to PHY |
| Wireless PHY | < 5 ms | TDMA fixed slots |
| VCU decode | < 6 ms | Low-latency decode mode |
| Display output | < 2 ms | Direct memory to DP |
| **Total** | **< 30 ms** | |

**GPIO-based latency measurement:**
```c
#define LATENCY_GPIO 0xA0000000
Xil_Out32(LATENCY_GPIO, 0x1);  // High at capture start
// ... pipeline ...
Xil_Out32(LATENCY_GPIO, 0x0);  // Low at display output
// Measure with oscilloscope between TX and RX GPIO pins
```

#### Week 7: Bare-Metal Transition + Validation

**Replace FreeRTOS with minimal cooperative scheduler:**
```c
typedef struct { void (*func)(void); uint32_t period_ms; uint32_t last_run; } task_t;

task_t tasks[] = {
    { tdma_schedule,     1,   0 },
    { video_encode_task, 16,  0 },
    { video_decode_task, 16,  0 },
    { rf_monitor_task,   100, 0 }
};

void main_loop(void) {
    while (1) {
        uint32_t now = get_timer_ms();
        for (int i = 0; i < ARRAY_SIZE(tasks); i++) {
            if ((now - tasks[i].last_run) >= tasks[i].period_ms) {
                tasks[i].func();
                tasks[i].last_run = now;
            }
        }
        __asm__("wfi");  // Wait for interrupt — save power
    }
}
```

**Interrupt-driven pipeline:**
```c
void vcu_encode_irq(void)    { XVcuenc_InterruptClear(&enc); packetizer_start_dma(); }
void pkt_dma_irq(void)       { wireless_tx_start(); }
void wireless_rx_irq(void)   { depacketizer_start(); }
void vcu_decode_irq(void)    { display_update(); }
```

#### Week 8: Documentation & ASIC Roadmap

#### IP Block Inventory

| IP Block | Source | ASIC Ready | Notes |
|----------|--------|:----------:|-------|
| VCU (H.265) | Xilinx hardened | No | Replace with CEVA/Chips&Media licensed IP |
| MIPI D-PHY | Xilinx PL | No | License from MIPI Alliance member |
| CSI-2 RX Controller | Xilinx IP | Yes | RTL available, port to ASIC process |
| Wireless PHY (OFDM) | Custom | Yes | Synthesizable Verilog |
| TDMA MAC | Custom | Yes | Synthesizable Verilog |
| Packetizer | Custom | Yes | Synthesizable Verilog |
| AD9361 Interface | Custom | Yes | Adapt to ASIC RF interface |

#### ASIC Target Specifications

| Parameter | Target |
|-----------|--------|
| Process node | 12nm or 16nm |
| Die size | < 50 mm² |
| Power | < 5W |
| Package | FCBGA |
| MIPI interface | D-PHY 2.5 Gbps |
| RF | External RFIC or integrated |

#### ASIC Integration Plan

| Quarter | Milestone |
|---------|-----------|
| Q1 | Select IP vendors (H.265 codec, MIPI PHY) |
| Q2 | Chip architecture, floorplan |
| Q3 | RTL integration, simulation |
| Q4 | Synthesis, place & route |
| Y2 Q1 | Tape-out, prototype bring-up |

**Phase 2 deliverables (end of week 8):**
- [ ] FreeRTOS running on quad-core A53 with per-core task allocation
- [ ] Latency optimized to < 30 ms end-to-end
- [ ] Bare-metal scheduler implemented, interrupt-driven pipeline working
- [ ] Stable 4K60 streaming over 10+ meters
- [ ] Complete design documentation (block diagrams, data flow, timing, register maps)
- [ ] IP inventory with ASIC readiness assessment
- [ ] ASIC roadmap with timeline and cost estimates
- [ ] Demonstration video of working prototype

---

### 2.7 Timeline Summary

| Phase | Weeks | Key deliverables |
|-------|-------|------------------|
| **1 — Platform + VCU + Wireless** | 1–4 | VCU validated, MIPI camera working, wireless link up, first 4K stream (3 parallel streams) |
| **2 — Integration + Optimization + ASIC** | 5–8 | FreeRTOS, <30ms latency, bare-metal transition, documentation, ASIC roadmap |

**Total: 8 weeks (2 months).** Achieved by running VCU/MIPI (Stream A), wireless PHY (Stream B), and RF front-end (Stream C) fully in parallel during Phase 1, then merging in week 4.

---

### 2.8 Troubleshooting Guide

**VCU:**

| Symptom | Cause | Solution |
|---------|-------|---------|
| Encoder not starting | Missing clock | Check VCU clock config in PS |
| Frame drops | Bandwidth insufficient | Reduce bitrate or increase buffer |
| High latency | Large GOP | Set gop_mode to low-delay-P |
| Corrupted output | Wrong format | Verify input is NV12 |

**MIPI:**

| Symptom | Cause | Solution |
|---------|-------|---------|
| No camera detection | Termination | Verify external 100 ohm resistors |
| CRC errors | Lane rate too high | Reduce lane rate, check SI |
| I2C fails | Missing pull-ups | Add 4.7k ohm on I2C lines |

**Wireless:**

| Symptom | Cause | Solution |
|---------|-------|---------|
| No RF output | AD9361 not init | Check SPI communication |
| Low throughput | Interference | Change channel, use directional antennas |
| No sync | Timing offset | Verify TDMA beacon reception |

**Bare-metal:**

| Symptom | Cause | Solution |
|---------|-------|---------|
| Watchdog reset | Task starvation | Add yield points, increase stack |
| Cache coherency | Uncached DMA bufs | Mark buffers non-cacheable |
| Interrupt lost | Priority issues | Configure interrupt priorities |

---

## Skills Exercised (Track A Modules)

| Track A Module | How this project uses it |
|---------------|------------------------|
| §1 Vivado | IP Integrator block design, bitstream generation, ILA debugging |
| §2 Zynq MPSoC | PS/PL co-design, VCU integration, AXI interconnect, Linux + bare-metal |
| §3 Advanced FPGA | CDC for multi-clock video/RF domains, floorplanning around VCU/PHY |
| §4 HLS | Video preprocessing (color convert, scale), packetizer/depacketizer |
| §5 Runtime & Driver | DMA transfers, Linux V4L2 drivers, bare-metal register access, GStreamer integration |
