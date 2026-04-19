# ESP32-C6 ESP-Hosted over SPI on Jetson Orin Nano - Project Guide

> **Goal:** Bring up an **ESP32-C6** as an external Wi-Fi coprocessor on the **Jetson Orin Nano 8GB Developer Kit** using **ESP-Hosted-NG over SPI**, so the Jetson gets a normal Linux wireless interface driven through the 40-pin header.

**Hub:** [Network and Connectivity](Guide.md)  
**Related local guides:** [Peripheral Access](../1.%20Peripheral%20Access/Guide.md) · [Orin Nano GPIO/SPI/I2C/CAN deep-dive](../../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-GPIO-SPI-I2C-CAN/Guide.md)

---

## Table of Contents

1. [Why this project matters](#1-why-this-project-matters)
2. [Target architecture](#2-target-architecture)
3. [Hardware and software prerequisites](#3-hardware-and-software-prerequisites)
4. [Wiring the Jetson header to ESP32-C6](#4-wiring-the-jetson-header-to-esp32-c6)
5. [Prepare the Jetson SPI host side](#5-prepare-the-jetson-spi-host-side)
6. [Build and flash ESP-Hosted-NG for ESP32-C6](#6-build-and-flash-esp-hosted-ng-for-esp32-c6)
7. [Port the Linux host side from Raspberry Pi to Jetson](#7-port-the-linux-host-side-from-raspberry-pi-to-jetson)
8. [Validation checklist](#8-validation-checklist)
9. [Common failure modes](#9-common-failure-modes)
10. [Stretch goals](#10-stretch-goals)
11. [References](#11-references)

---

## 1. Why this project matters

The Jetson Orin Nano developer kit does not include onboard Wi-Fi. In many edge builds you solve that with a USB dongle or an M.2 Key E card, but an **ESP32-C6 + ESP-Hosted-NG** path teaches a different and more embedded-friendly pattern:

- the Jetson stays the main Linux application processor
- the ESP32-C6 acts as a dedicated connectivity peripheral
- SPI and GPIOs become part of your system integration work
- the final result still looks like a standard Linux network interface on Jetson

This makes it a strong project for anyone learning how Linux, GPIO interrupts, board wiring, and wireless connectivity fit together on an AI edge device.

**Important scope note:** this is not just a firmware flash. Espressif documents Raspberry Pi as the reference Linux SPI host. On Jetson, the **ESP firmware stays upstream**, but the **Linux host integration needs a small platform port** for SPI bus selection, GPIO mapping, and driver loading.

---

## 2. Target architecture

```text
Jetson Orin Nano 8GB dev kit
  |
  |-- SPI1: MOSI / MISO / SCLK / CS0
  |-- GPIO: Handshake / Data Ready / Reset
  |
  +--> ESP32-C6 running ESP-Hosted-NG (SPI peripheral)
         |
         +--> 2.4 GHz Wi-Fi connection to the network

Linux on Jetson then uses wlanX through the normal stack:
NetworkManager / nmcli / wpa_supplicant / hostapd / ip / iw
```

**Target outcome**

- Jetson exposes a usable `wlanX` interface through ESP-Hosted
- `nmcli dev wifi list` works on the Jetson
- Jetson can join an AP and pass normal traffic
- the transport is stable enough to measure with `ping` and `iperf3`

---

## 3. Hardware and software prerequisites

### Hardware

- Jetson Orin Nano 8GB Developer Kit
- ESP32-C6 development board with USB access for flashing and power
- 8-10 short jumper wires, ideally under 10 cm
- common ground between Jetson and ESP board
- optional logic analyzer for SPI timing/debug

### Software

- JetPack 6.x / L4T 36.x on Jetson
- Linux kernel headers installed on Jetson
- `gpiod`, `spi-tools`, `NetworkManager`, `iperf3`
- Espressif [esp-hosted](https://github.com/espressif/esp-hosted) repository
- ESP-IDF set up through `esp_hosted_ng/esp/esp_driver/setup.sh`

### Power and signal safety

- The Jetson 40-pin header uses **3.3 V logic**
- Keep the ESP32-C6 on its own USB power during bring-up
- Share **ground** between boards
- Do **not** assume the Jetson 3.3 V header pin should power the full ESP dev board during early testing

---

## 4. Wiring the Jetson header to ESP32-C6

Espressif's SPI setup guide documents the signal roles for ESP32-C6 on a Raspberry Pi host. On Jetson, keep the same signal roles and remap them onto the Orin Nano's SPI1 header and spare GPIOs.

| Function | Jetson pin | Jetson signal | ESP32-C6 pin | Direction |
|----------|------------|---------------|--------------|-----------|
| MOSI | 19 | `SPI1_MOSI` | `IO7` | Jetson -> ESP |
| MISO | 21 | `SPI1_MISO` | `IO2` | ESP -> Jetson |
| SCLK | 23 | `SPI1_SCLK` | `IO6` | Jetson -> ESP |
| CS0 | 24 | `SPI1_CS0` | `IO10` | Jetson -> ESP |
| Handshake | 22 | `GPIO22` | `IO3` | ESP -> Jetson |
| Data Ready | 15 | `GPIO15` | `IO4` | ESP -> Jetson |
| Reset | 18 | `GPIO18` | `RST` or `EN` | Jetson -> ESP |
| Ground | 20 or 25 | `GND` | `GND` | common reference |

### Why the extra GPIOs matter

ESP-Hosted SPI is not only a 4-wire SPI bus:

- **Handshake** tells the host the ESP side is ready
- **Data Ready** tells the host that the ESP side has data pending
- **Reset** is required so the host can force the ESP side into a known state

If those three GPIOs are wrong, the SPI transport may partially initialize or fail after the first event.

### Recommended bring-up practice

- keep wires short and similar in length
- start on a bench, not inside an enclosure
- label the three non-SPI GPIOs clearly before first power-up
- if you have a logic analyzer, probe `CS`, `SCLK`, `Handshake`, and `Data Ready`

---

## 5. Prepare the Jetson SPI host side

### 5.1 Enable SPI1 on the 40-pin header

Use Jetson-IO to enable the SPI controller behind header pins 19/21/23/24:

```bash
sudo /opt/nvidia/jetson-io/jetson-io.py
```

Enable **SPI1 (1 device)**, save, and reboot.

After reboot:

```bash
ls -l /dev/spidev0.0
```

On the developer kit, the header's SPI1 controller typically appears to Linux as **`/dev/spidev0.0`**.

### 5.2 Install useful tools

```bash
sudo apt update
sudo apt install -y \
  linux-headers-$(uname -r) \
  libgpiod-dev gpiod \
  spi-tools \
  network-manager \
  iperf3
```

### 5.3 Verify the basic SPI path first

Before loading the ESP-Hosted driver, make sure the controller is alive and the header is configured correctly.

- confirm `/dev/spidev0.0` exists
- confirm your chosen GPIO lines are free for use
- verify there is no conflicting device already bound to the same bus/chip-select

If you plan to use the Espressif kernel module directly, be aware that **generic `spidev` may need to be disabled** once you move from raw SPI sanity checks to the real host driver path.

### 5.4 Start conservatively

During first bring-up:

- use short wires
- start with **1 MHz** SPI clock if timing is unstable
- avoid changing multiple variables at once

---

## 6. Build and flash ESP-Hosted-NG for ESP32-C6

Clone the repo and prepare the ESP side exactly from the upstream ESP-Hosted-NG tree:

```bash
git clone https://github.com/espressif/esp-hosted.git
cd esp-hosted/esp_hosted_ng/esp/esp_driver
./setup.sh
cd esp-idf
. ./export.sh
cd ../network_adapter
```

Set the target to ESP32-C6:

```bash
idf.py set-target esp32c6
```

Open configuration:

```bash
idf.py menuconfig
```

Select the SPI transport:

- `Example Configuration`
- `Transport layer`
- `SPI interface`

Then build and flash:

```bash
idf.py -p /dev/ttyACM0 build flash
idf.py -p /dev/ttyACM0 monitor
```

Replace `/dev/ttyACM0` with the serial device for your ESP32-C6 board.

### What you should see

- successful build and flash
- ESP boot logs on the serial monitor
- the firmware configured for the same transport you intend to use on the Jetson host side

**Do not mix transports.** A host built for SPI and an ESP firmware built for SDIO or UART will fail in ways that look like board or timing bugs.

---

## 7. Port the Linux host side from Raspberry Pi to Jetson

Espressif ships Raspberry Pi as the reference Linux host. On Jetson, use that host implementation as the starting point rather than trying to invent a new stack.

### 7.1 Start from the upstream host tree

Relevant upstream locations:

- `esp_hosted_ng/host/`
- `esp_hosted_ng/host/spi/`
- `esp_hosted_ng/docs/setup.md`
- `esp_hosted_ng/docs/porting_guide.md`

The Raspberry Pi helper script is useful as a reference:

```bash
cd esp-hosted/esp_hosted_ng/host
```

On Raspberry Pi the documented entry point is:

```bash
bash rpi_init.sh spi
```

On Jetson, treat that script as **reference logic**, not as a guaranteed drop-in command.

### 7.2 Jetson-specific porting work

Port these pieces to Jetson:

1. **Reset GPIO**
   Use a free Jetson GPIO, such as header pin 18 (`GPIO18`), and update the host reset handling accordingly.

2. **Handshake and Data Ready GPIOs**
   Update the host SPI definitions so they match your chosen Jetson GPIO header pins:
   - handshake on pin 22 (`GPIO22`)
   - data ready on pin 15 (`GPIO15`)

3. **SPI bus and chip select selection**
   Set the host-side SPI bus and chip-select values to match the Linux-visible device behind the header. On the dev kit, that is usually the controller exposed as `spidev0.0`.

4. **Device tree and pinmux**
   Make sure:
   - SPI1 is enabled on the 40-pin header
   - the three extra GPIOs are free and configured for GPIO use
   - no conflicting overlay or driver claims the same pins

5. **`spidev` conflict handling**
   Once you move to the real ESP-Hosted kernel module path, disable generic `spidev` if it prevents the Espressif driver from claiming the bus.

6. **Build environment**
   If you build on-device, point the host build at Jetson kernel headers. If you cross-compile, update `ARCH`, `CROSS_COMPILE`, and kernel paths for `aarch64`.

### 7.3 Good bring-up order

Use this order. It reduces ambiguity.

1. Enable SPI1 and verify `/dev/spidev0.0`
2. Confirm Jetson GPIO choices physically match your wiring
3. Flash the ESP32-C6 with SPI-enabled ESP-Hosted firmware
4. Port the host-side GPIO mapping and bus selection
5. Start at low SPI frequency
6. Watch `dmesg` for the first init event from ESP to host
7. Only then move to Wi-Fi association and throughput tests

---

## 8. Validation checklist

### 8.1 Transport-level validation

```bash
sudo dmesg -w
```

You want to see:

- the Jetson-side driver loads cleanly
- the ESP reset sequence completes
- the host receives the first event from the ESP side

### 8.2 Interface-level validation

```bash
ip link show
nmcli device status
nmcli dev wifi list
```

Expected result:

- a new wireless interface such as `wlan0` or `wlan1`
- scan results visible through NetworkManager

### 8.3 Join a network

```bash
sudo nmcli dev wifi connect "SSID" password "password"
ip addr show
ping -c 4 8.8.8.8
```

### 8.4 Basic throughput smoke test

On another machine:

```bash
iperf3 -s
```

On the Jetson:

```bash
iperf3 -c <server-ip>
```

Do not optimize too early. First prove:

- stable scan
- stable association
- clean packet flow
- no repeated transport resets

---

## 9. Common failure modes

### No first event in `dmesg`

Usually one of these:

- reset GPIO is wrong
- handshake or data-ready GPIO is mapped incorrectly
- SPI mode or clock is too aggressive
- ESP firmware is not actually built for SPI

### First event appears, then traffic stalls

Common causes:

- handshake/data-ready interrupts are not configured correctly on the host
- GPIO edge polarity is wrong
- jumper wires are too long or noisy
- the host driver and ESP firmware are from inconsistent revisions

### Bus exists, but the ESP-Hosted driver cannot claim it

Likely cause:

- generic `spidev` still owns the device tree path that the kernel module needs

### Wi-Fi interface appears but is unstable

Work through this order:

1. Lower SPI frequency to 1 MHz
2. Shorten the wires
3. Re-check shared ground
4. Confirm the ESP board power is clean
5. Re-test with a simple AP close to the bench

### Transport works, but performance is weak

That is normal early on. Once the path is stable:

- increase SPI clock stepwise
- measure each change
- keep notes on which frequency starts to fail

---

## 10. Stretch goals

- add AP mode support and use the ESP32-C6 as the Jetson's provisioning radio
- evaluate Bluetooth support from the same ESP-Hosted stack revision
- replace jumper wires with a small adapter board or custom carrier interconnect
- add a device-tree overlay and scripts so the Jetson setup becomes reproducible
- measure power, latency, and throughput against a USB Wi-Fi dongle baseline

---

## 11. References

### Official upstream references

- [Espressif esp-hosted repository](https://github.com/espressif/esp-hosted)
- [ESP-Hosted-NG setup guide](https://github.com/espressif/esp-hosted/blob/master/esp_hosted_ng/docs/setup.md)
- [ESP-Hosted-NG SPI protocol notes](https://github.com/espressif/esp-hosted/blob/master/esp_hosted_ng/docs/spi_protocol.md)
- [ESP-Hosted-NG Linux porting guide](https://github.com/espressif/esp-hosted/blob/master/esp_hosted_ng/docs/porting_guide.md)

### Local roadmap references

- [Network and Connectivity](Guide.md)
- [Peripheral Access](../1.%20Peripheral%20Access/Guide.md)
- [Orin Nano GPIO/SPI/I2C/CAN deep-dive](../../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-GPIO-SPI-I2C-CAN/Guide.md)
