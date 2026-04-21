# ESP32-C6 Zigbee NCP on Jetson Orin Nano - Project Guide

> **Goal:** Use a **second ESP32-C6** as a **Zigbee Network Co-Processor (NCP)** for the **Jetson Orin Nano 8GB Developer Kit**, so the Jetson can act as the higher-level Zigbee host while keeping your first ESP32-C6 dedicated to **ESP-Hosted Wi-Fi/BLE**.

**Hub:** [Network and Connectivity](Guide.md)  
**Related local guides:** [ESP32-C6 ESP-Hosted over SPI on Jetson Orin Nano](ESP32-C6-ESP-Hosted-SPI-Jetson-Orin-Nano.md) · [ESP32-C6 OpenThread RCP on Jetson Orin Nano](ESP32-C6-OpenThread-RCP-Jetson-Orin-Nano.md) · [Peripheral Access](../1.%20Peripheral%20Access/Guide.md)

---

## 1. Why this project matters

This is the clean next experiment after your current Jetson + ESP32-C6 networking work:

- the first ESP32-C6 already gives Jetson `wlan0` and BLE through **ESP-Hosted**
- the second ESP32-C6 already proved it can act as an **802.15.4 coprocessor**
- your current **Thread** path is not blocked at the radio or UART level, but **full OTBR** is blocked on the present Jetson kernel because `CONFIG_IP_MROUTE` and `CONFIG_IPV6_MROUTE` are missing

Zigbee is a good next step because it still uses **IEEE 802.15.4**, but it does **not** require OTBR's IPv6 border-routing path. That means the specific Linux multicast-routing blocker you hit with OTBR should not be the main issue for a Zigbee host/NCP design. That last point is an engineering inference from the protocol architecture, not a claim that the whole Jetson Zigbee stack is already turnkey.

What this project teaches:

- how a Linux host talks to a Zigbee coprocessor instead of a Thread RCP
- how Zigbee differs from Thread even though both use 802.15.4
- how to keep Wi-Fi/BLE and Zigbee on separate radios for a cleaner system design
- where Jetson integration becomes a host-application problem instead of a Linux-network-interface problem

---

## 2. Current Status and Scope

On your current Jetson image, the OpenThread RCP path has already been validated up to the point where the node can become `leader`. The failure happens later when `otbr-agent` tries to initialize border-routing support and the kernel reports:

```text
InitMulticastRouterSock() ... Protocol not available
```

You also confirmed directly that the kernel is missing:

```text
# CONFIG_IP_MROUTE is not set
# CONFIG_IPV6_MROUTE is not set
```

This guide is therefore scoped as:

- **current practical next path:** Zigbee host + coprocessor work on the existing Jetson image
- **deferred path:** rebuild or replace JetPack / kernel later for full OTBR

One expectation must be stated clearly: Zigbee on Jetson will not look like Thread on Jetson.

- Thread + OTBR creates a Linux-facing interface such as `wpan0`
- Zigbee NCP does **not** create a Linux IP interface
- instead, the Jetson host application speaks a Zigbee coprocessor protocol and manages Zigbee roles, joins, clusters, and device state itself

---

## 3. Target Architecture

```text
Jetson Orin Nano
  |
  |-- SPI -> ESP32-C6 #1 -> ESP-Hosted -> wlan0 + hci0
  |
  |-- UART (recommended first) -> ESP32-C6 #2 -> Zigbee NCP
  |                                 ^
  |                                 |
  |                         ESP ZNSP over SLIP
  |
  +--> Linux host side
        |-- Zigbee host application / gateway logic
        |-- Coordinator or Router role management
        |
        +--> Wi-Fi / Ethernet / USB network uplink
```

**Target outcome**

- Jetson keeps using `wlan0` from the first ESP32-C6 as its normal Wi-Fi path
- Jetson talks to the second ESP32-C6 over UART or SPI as a Zigbee coprocessor
- the Zigbee side can form or join a Zigbee network
- joined Zigbee devices are managed from the host side through Zigbee commands, not through a Linux `wpan0` interface

---

## 4. Why Use a Second ESP32-C6

This guide intentionally keeps Zigbee off the first ESP32-C6.

Why:

- **ESP-Hosted** already uses the first chip as a Jetson Wi-Fi/BLE coprocessor
- Zigbee also uses the ESP32-C6's **802.15.4** radio
- Espressif documents that ESP32-C6 has **one shared 2.4 GHz RF module** for Wi-Fi, Bluetooth LE, and IEEE 802.15.4, with time-division multiplexing managing access
- combining all of that into one custom firmware stack is possible in theory, but much harder to stabilize than using two radios

So the safer split remains:

- **ESP32-C6 #1** for Jetson Wi-Fi/BLE through ESP-Hosted
- **ESP32-C6 #2** for Zigbee coprocessor work

---

## 5. Hardware and Software Prerequisites

### Hardware

- Jetson Orin Nano 8GB Developer Kit
- your existing **ESP32-C6 #1** already working with ESP-Hosted over SPI
- a **second ESP32-C6 dev board**
- jumper wires for a direct UART link, or optionally a USB cable if you first validate over the board's USB-UART bridge
- USB cable for flashing and monitoring the second ESP32-C6
- one or more Zigbee devices for later validation:
  - another ESP32-C6 or ESP32-H2 running a Zigbee end-device example
  - a commercial Zigbee end device

### Software

- JetPack 6.x / L4T 36.x on Jetson
- ESP-IDF installed on a Linux build machine
- Espressif [ESP Zigbee SDK](https://github.com/espressif/esp-zigbee-sdk)
- a host-side application plan on Jetson:
  - your own Zigbee host tool in C/C++/Python
  - or a future host example based on Espressif's documented NCP protocol

### Recommended first transport

Espressif documents the Zigbee NCP protocol over either **UART** or **SPI**. On your Jetson, **UART is the better first transport**.

Why:

- it avoids colliding with the SPI bus already used by ESP-Hosted
- your Jetson header UART path is already electrically understood from the Thread/RCP work
- the NCP protocol is explicitly documented over UART
- debugging framed serial traffic is simpler than debugging a new SPI host stack and a new Zigbee stack at the same time

For the fastest lab bring-up, a board USB-UART path to Jetson is acceptable. For the cleaner embedded path, move to the Jetson header UART once the host logic is understood.

---

## 6. The Official Models You Are Implementing

Espressif's current Zigbee material exposes **two related but distinct host-side patterns**:

### Zigbee NCP model

The official Zigbee NCP guide defines **ESP ZNSP** as the protocol a **host application processor** uses to interact with the Zigbee stack on a **Network Co-Processor**. Those frames are carried over **SLIP**, and the transport can be **UART** or **SPI**.

That is the conceptual model that best fits Jetson:

- Jetson is the host processor
- ESP32-C6 runs the Zigbee stack as the coprocessor
- Jetson sends network-management and application commands over the NCP link

The official NCP API also exposes host connection modes for:

- `NCP_HOST_CONNECTION_MODE_UART`
- `NCP_HOST_CONNECTION_MODE_SPI`

### Zigbee gateway + RCP model

Espressif's current example tree also contains a **`zigbee_gateway`** example. That example runs on a Wi-Fi-capable ESP host SoC and uses an **802.15.4 RCP** running `ot_rcp` on another chip.

That matters for Jetson for two reasons:

- it shows Espressif already treats Zigbee gateway designs as a **multi-chip host + 802.15.4 radio** problem
- it also shows that the latest official example emphasis is stronger on an **ESP host SoC gateway** than on a turnkey Linux host daemon

As of **April 21, 2026**, the official `esp_zigbee_ncp` example README still says a separate host example will be provided later as a reference. So the right expectation on Jetson is:

- the **wire protocol and NCP device side are officially documented**
- the **Linux host side is still more custom than OTBR**

---

## 7. Zigbee Roles on Jetson: What Makes Sense

Espressif documents support for:

- **Coordinator**
- **Router**
- **(Sleepy) End Device**

For a Jetson-hosted gateway, the most useful roles are:

### Coordinator

Use this if the Jetson should create and manage the Zigbee network itself. This is the most natural role for a Linux host acting as a gateway, bridge, or coordinator appliance.

### Router

Use this if the Jetson should join an existing Zigbee network instead of forming its own.

### End Device

This is generally **not** the role you want for a Jetson-class host. End-device behavior is more appropriate for battery devices or compact application nodes, not for a Linux gateway.

---

## 8. Wire the Jetson UART to the Zigbee NCP

The official `esp_zigbee_ncp` example shows a simple UART connection where the host TX goes to the NCP's `GPIO4` RX pin and the host RX comes from the NCP's `GPIO5` TX pin. That aligns well with the UART wiring pattern you already used on Jetson.

For the Jetson Orin Nano header path on your current image:

| Function | Jetson pin | Linux device | Direction |
|---|---:|---|---|
| UART1_TXD | `8` | `/dev/ttyTHS1` | Jetson -> ESP |
| UART1_RXD | `10` | `/dev/ttyTHS1` | ESP -> Jetson |
| Ground | `6` or `9` or `14` | -- | common reference |

For the ESP32-C6 NCP side:

| Function | ESP32-C6 pin | Direction |
|---|---:|---|
| NCP RX | `GPIO4` | Jetson TX -> ESP RX |
| NCP TX | `GPIO5` | ESP TX -> Jetson RX |
| Ground | `GND` | common reference |

So the direct wiring is:

```text
Jetson pin 8  (UART1_TXD) -> ESP32-C6 GPIO4
Jetson pin 10 (UART1_RXD) <- ESP32-C6 GPIO5
Jetson GND                -> ESP32-C6 GND
```

The official NCP example README also notes that these UART pins can be changed in:

```text
idf.py menuconfig -> Component config -> Zigbee Network Co-processor
```

So if your board routing or chosen transport differs, treat the example's pin settings and your bench wiring as one matched pair.

### Jetson UART prerequisites

The same Jetson-side UART precautions from the Thread guide still apply:

- disable `nvgetty` if it owns the header UART
- keep the port on **3.3 V logic**
- do not let multiple processes fight over `/dev/ttyTHS1`

Basic port setup example:

```bash
sudo systemctl stop nvgetty
sudo systemctl disable nvgetty
sudo stty -F /dev/ttyTHS1 115200 cs8 -cstopb -parenb raw -echo
```

Use the actual baud configured on both sides. The exact Zigbee NCP example baud must match the host application's serial configuration.

---

## 9. Build and Flash the ESP32-C6 Zigbee NCP Firmware

The official Techpedia Zigbee solution page points to:

```text
esp-zigbee-sdk/examples/esp_zigbee_ncp
```

and the current official GitHub example page documents it as the NCP device example for ESP32-C6.

On your Linux build host:

```bash
git clone --depth=1 https://github.com/espressif/esp-zigbee-sdk.git
cd esp-zigbee-sdk/examples/esp_zigbee_ncp

# Use a compatible ESP-IDF version for the SDK
# The current SDK README recommends ESP-IDF v5.5.4 for new work.

idf.py set-target esp32c6
idf.py menuconfig
idf.py build
```

Flash it:

```bash
# Use the actual USB serial port for the second ESP32-C6 board
idf.py -p /dev/ttyUSB0 flash monitor
```

The example README documents:

- the board acts as a Zigbee NCP
- it works together with a host via UART
- `GPIO4` / `GPIO5` are the example RX/TX mapping
- host-side operations such as `NETWORK_INIT`, `NETWORK_PRIMARY_CHANNEL_SET`, `NETWORK_FORMNETWORK`, and `START` trigger network formation and coordinator behavior

### Port naming note

As with the Thread guide:

- do not hardcode `/dev/ttyACM0`
- do not assume every board shows up as `/dev/ttyUSB0`

Use the actual device path exposed by the second ESP32-C6 board on your Linux machine.

---

## 10. What the Jetson Host Actually Has to Do

This is the biggest conceptual difference from OTBR.

With Thread + OTBR, the Jetson host launches an existing daemon and gets a Linux-facing control path such as `ot-ctl`. With Zigbee NCP, the host must drive the Zigbee control flow more directly.

The official Zigbee NCP frame list shows host operations such as:

- `NETWORK_INIT`
- `NETWORK_START`
- `NETWORK_STATE`
- `NETWORK_FORM`
- `NETWORK_JOIN`
- `NETWORK_PERMIT_JOINING`
- `NETWORK_ROLE_GET` / `NETWORK_ROLE_SET`
- `NETWORK_CHANNEL_SET`
- `NETWORK_PAN_ID_SET`
- `NETWORK_PRIMARY_KEY_SET`
- `ZCL_ATTR_READ`
- `ZCL_ATTR_WRITE`
- `APS_DATA_REQUEST`

So the host side on Jetson is responsible for:

- opening the UART or SPI transport
- SLIP-framing and parsing **ESP ZNSP** packets
- forming or joining a Zigbee network
- keeping track of role and stack status
- sending ZDO, APS, and ZCL commands to devices
- reacting to join, leave, attribute-report, and indication events

That is why Zigbee on Jetson is **not blocked by the same kernel issue as OTBR**, but is also **not yet as turnkey as `otbr-agent`**.

---

## 11. Recommended Bring-Up Milestones on Jetson

Treat this as a staged host-integration problem.

### Milestone 1: NCP transport only

Prove the Jetson can open the NCP serial path and exchange framed traffic without resets or dropped bytes.

Success looks like:

- no contention on `/dev/ttyTHS1`
- stable NCP startup logs on the ESP side
- successful host/NCP framing exchange

### Milestone 2: basic network control

Implement or validate:

- `NETWORK_INIT`
- `NETWORK_STATE`
- `NETWORK_ROLE_SET`
- channel / PAN / key configuration commands

Success looks like:

- the host can read stack state
- the host can set the intended role
- the NCP persists or reports the configured network settings correctly

### Milestone 3: form a coordinator network

If Jetson is the gateway:

- set Coordinator role
- form a new network
- open joining for a test window

Success looks like:

- the network forms successfully
- permit-join window opens
- a test Zigbee end device can join

### Milestone 4: real device control

Move from stack bring-up to application behavior:

- discover joined devices
- read or write attributes
- test cluster commands
- verify device notifications or reports reach the host

At that point, Zigbee integration is no longer a serial-demo exercise. It becomes a real coordinator/gateway design.

---

## 12. Why This Path Is Different from the Current Thread Blocker

Your current OTBR failure is specifically tied to **Linux border routing**, not to the ESP32-C6 radio or 802.15.4 itself.

The proof from the Thread work is:

- Jetson could already talk to the ESP32-C6 RCP
- the Thread node could already become `leader`
- the crash happened later when OTBR tried to initialize Linux multicast-routing support

Zigbee NCP changes the problem shape:

- no OTBR
- no `wpan0`
- no IPv6 border-routing socket
- host logic speaks Zigbee control messages over serial instead

So the missing Jetson kernel options:

- `CONFIG_IP_MROUTE`
- `CONFIG_IPV6_MROUTE`

should not be the main blocker for Zigbee host/NCP work.

That is the architectural reason this is the right next experiment while you defer the JetPack/kernel rebuild for later Thread OTBR work.

---

## 13. Common Failure Modes

### The host expects a Linux radio interface like `wpan0`

That is the wrong model for Zigbee NCP. Success is not measured by a new Linux network interface. It is measured by host/NCP command exchange, network formation or join, and actual Zigbee device control.

### UART transport is unstable

Typical causes:

- wrong baud rate
- multiple processes opening `/dev/ttyTHS1`
- `nvgetty` still attached
- TX/RX swapped
- ESP UART pin config does not match the bench wiring

### You assume the current SDK repo only supports the gateway/RCP path

The official docs expose both:

- a Zigbee gateway / RCP example
- a Zigbee NCP example and NCP protocol documentation

But the Linux host side is still more manual than the Thread OTBR flow, so read the docs as an API/protocol contract, not as a promise of a one-command Jetson daemon.

### Trying to merge Zigbee onto the first ESP32-C6 immediately

Avoid it for now.

You already have:

- a working ESP-Hosted split
- a validated second-radio concept

Preserve that clean architecture until the separate Zigbee path is stable.

---

## 14. Good Next Steps

- flash `esp_zigbee_ncp` onto the second ESP32-C6
- start with UART before attempting SPI
- treat the Jetson work as a host-protocol bring-up, not a Linux netdev bring-up
- choose **Coordinator** first if Jetson is meant to be the gateway
- keep the Thread OTBR kernel rebuild as a separate later milestone

---

## 15. References

### Official upstream references

- [ESP Zigbee SDK introduction](https://docs.espressif.com/projects/esp-zigbee-sdk/en/latest/esp32c6/introduction.html)
- [ESP Zigbee NCP guide](https://docs.espressif.com/projects/esp-zigbee-sdk/en/latest/esp32c6/user-guide/ncp.html)
- [ESP Zigbee NCP API reference](https://docs.espressif.com/projects/esp-zigbee-sdk/en/latest/esp32c6/api-reference/esp_zigbee_ncp.html)
- [ESP Zigbee SDK repository](https://github.com/espressif/esp-zigbee-sdk)
- [ESP Zigbee NCP example](https://github.com/espressif/esp-zigbee-sdk/tree/main/examples/esp_zigbee_ncp)
- [ESP Zigbee gateway example](https://github.com/espressif/esp-zigbee-sdk/tree/main/examples/zigbee_gateway)
- [ESP-Techpedia Zigbee solution introduction](https://docs.espressif.com/projects/esp-techpedia/en/latest/esp-friends/solution-introduction/zigbee/zigbee-solution.html)
- [ESP32-C6 RF coexistence](https://docs.espressif.com/projects/esp-idf/en/v5.1.3/esp32c6/api-guides/coexist.html)

### Local roadmap references

- [ESP32-C6 ESP-Hosted over SPI on Jetson Orin Nano](ESP32-C6-ESP-Hosted-SPI-Jetson-Orin-Nano.md)
- [ESP32-C6 OpenThread RCP on Jetson Orin Nano](ESP32-C6-OpenThread-RCP-Jetson-Orin-Nano.md)
- [Network and Connectivity hub](Guide.md)
