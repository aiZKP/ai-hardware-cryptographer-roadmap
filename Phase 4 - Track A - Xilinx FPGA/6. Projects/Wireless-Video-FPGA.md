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

## Plan 2: Integrated 4K Wireless Chip Prototype

**Timeline:** 3.5 months (14 weeks)

**Goal:** Build a fully integrated 4Kp60 wireless transmitter/receiver on Zynq UltraScale+ EV (ZCU106) with bare-metal control and low-latency custom PHY.

### Accelerated Approach

| Original step | Accelerated method |
|---------------|--------------------|
| Set up Vivado, VCU bare-metal | Use Xilinx Vitis pre-built examples for VCU on ZCU106. Start with Linux to validate VCU, then port to bare-metal. |
| MIPI I/O IP development | Use **Xilinx MIPI D-PHY RX/TX IP** (available in Vivado). No custom PHY development. |
| Wireless PHY design | Start from **openwifi PHY** already ported to Zynq UltraScale+. Modify only MAC for TDMA. |
| RF front-end integration | Use **FMCOMMS5** with AD9361 — existing Linux drivers and bare-metal examples exist. |
| Bare-metal scheduling | Use **FreeRTOS** as stepping stone — faster than bare-metal scheduler from scratch. |

### Hardware

- ZCU106 evaluation kit (includes ZU7EV)
- FMCOMMS5 (or ADRV-CRR) FMC card
- MIPI DSI camera module with FMC adapter
- HDMI display (via ZCU106 onboard HDMI TX)

### Phase 1 — Platform & VCU Validation (4 weeks, parallel)

| Team | Work |
|------|------|
| **A** | Set up Vivado/Vitis; build Linux image with VCU (use Xilinx VCU TRD). Validate 4Kp60 encode/decode with sample file. |
| **B** | Set up openwifi on ZCU106 + FMCOMMS5 (use pre-built openwifi for ZCU106). Verify link with iperf. |
| **C** | Develop bare-metal hello world on ZCU106; bring up UART, DDR, clocks. |

### Phase 2 — MIPI & Wireless Integration (4 weeks, parallel)

| Team | Work |
|------|------|
| **A** | Integrate MIPI D-PHY RX IP in PL, connect to VCU input. Capture live camera video under Linux, verify VCU encodes it. |
| **B** | Modify openwifi PHY for TDMA (remove CSMA/CA). Implement simple TDMA controller in PL/PS. Test with two boards. |
| **C** | Port VCU driver to FreeRTOS (using Xilinx VCU library). Get simple encode/decode loop running without Linux. |

### Phase 3 — System Integration & Bare-Metal (4 weeks)

- Merge all components into single design: MIPI → VCU → wireless packetizer → PHY → RF.
- Use FreeRTOS to manage data flow; implement TDMA slot scheduling.
- Validate full 4Kp60 wireless link between two ZCU106 boards (one TX, one RX).
- Measure latency (use GPIO toggles).
- Transition from FreeRTOS to minimal bare-metal (optional — FreeRTOS is acceptable for ASIC firmware).

### Phase 4 — Documentation & ASIC Roadmap (2 weeks)

- Document all IP blocks, interfaces, and firmware.
- Identify components to be hardened for custom ASIC.

### Timeline Summary

| Phase | Duration | Key deliverables |
|-------|----------|------------------|
| 1 — Platform & VCU | 4 weeks | Linux with VCU working, openwifi link, bare-metal base |
| 2 — MIPI & Wireless | 4 weeks | Live camera → VCU encode, TDMA PHY tested |
| 3 — Integration | 4 weeks | Full 4K wireless link, latency measured |
| 4 — Documentation | 2 weeks | Final report, ASIC plan |

### Risk & Mitigation

| Risk | Mitigation |
|------|------------|
| VCU bare-metal support missing | Use FreeRTOS as target; acceptable for ASIC firmware and faster to develop |
| MIPI I/O not working | Use Xilinx MIPI IP (validated). Include loopback test early |
| Wireless PHY complexity | Start with openwifi (already works on ZCU106); only modify MAC |
| TDMA synchronization | Use simple beacon-based synchronization; start with one-way video only |

### Path to ASIC

- The prototype produces synthesizable Verilog for MIPI I/O, wireless PHY/MAC, and packetizer.
- VCU will be replaced with a licensed H.265 IP or custom design.
- Firmware (FreeRTOS or bare-metal) ports to the ASIC's embedded processor.
- RF front-end integration follows the same interface as AD9361 (or direct RF if using integrated RF ASIC).

---

## Skills Exercised (Track A Modules)

| Track A Module | How this project uses it |
|---------------|------------------------|
| §1 Vivado | IP Integrator block design, bitstream generation, ILA debugging |
| §2 Zynq MPSoC | PS/PL co-design, VCU integration, AXI interconnect, Linux + bare-metal |
| §3 Advanced FPGA | CDC for multi-clock video/RF domains, floorplanning around VCU/PHY |
| §4 HLS | Video preprocessing (color convert, scale), packetizer/depacketizer |
| §5 Runtime & Driver | DMA transfers, Linux V4L2 drivers, bare-metal register access, GStreamer integration |
