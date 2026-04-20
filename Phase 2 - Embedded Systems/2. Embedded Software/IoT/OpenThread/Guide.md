# OpenThread

*Follows [**IoT Networking and Device Connectivity**](../Guide.md) and extends the Phase 2 embedded-software path from local peripheral buses to low-power IP networking. Use this guide to understand what Thread is, how OpenThread implements it, and how the same stack can run either directly on an MCU or through a Linux host plus radio co-processor.*

---

## 1. Why OpenThread Matters

OpenThread is Google's open-source implementation of the **Thread** protocol. It is designed for low-power, IPv6-based mesh networking on top of **IEEE 802.15.4**, which makes it a good fit for battery-powered sensors, smart-home endpoints, industrial nodes, and border-router designs.  
Official sources: [OpenThread overview](https://openthread.io/), [ESP-IDF Thread guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/network/esp_openthread.html)

For roadmap learners, OpenThread is valuable because it sits exactly at the boundary between **MCU firmware** and **networked systems**. You cannot treat it as "just another driver" or "just another protocol" because it forces you to think about radio timing, low-power behavior, secure commissioning, IPv6 networking, and Linux-host integration at the same time.

---

## 2. What Thread Actually Is

Thread is an **IP-based mesh protocol**. At the radio layer it uses IEEE 802.15.4, but at the network layer it is still an IPv6 system, which is why a Thread node can fit into normal IP-based tooling much more naturally than many older embedded fieldbuses.

That distinction is important. BLE is excellent for short-range peripheral links, and Wi-Fi is excellent when you need bandwidth, but Thread is built for **low-power, many-node, always-available networking** where the network can repair itself as routers join or disappear.

OpenThread's official platform documentation describes the stack as implementing the full Thread networking layers, including **IPv6, 6LoWPAN, IEEE 802.15.4 with MAC security, Mesh Link Establishment, and Mesh Routing**. In practice, that means the stack handles both the radio-facing mechanics and the higher routing behavior that turns many small devices into one stable network.  
Official source: [OpenThread features](https://openthread.io/)

---

## 3. Protocol Stack: From Radio Frames to IP Packets

OpenThread is easiest to understand if you read it as a layered stack:

| Layer | What it does |
|---|---|
| IEEE 802.15.4 PHY/MAC | Defines the low-power radio link, frame format, channel use, acknowledgements, and MAC-layer security |
| 6LoWPAN | Compresses IPv6 headers so IPv6 packets fit efficiently over a tiny 802.15.4 frame budget |
| IPv6 | Gives each node a real IP identity instead of a proprietary application-only address model |
| UDP / ICMPv6 / higher services | Supports messaging, discovery, and management above the network layer |
| Thread control plane | Handles network formation, partition recovery, roles, commissioning, and mesh routing |

The important engineering idea is that Thread is not a "small custom sensor protocol." It is a constrained IP network. That is why it fits so naturally with border routers and Linux hosts, and why host tools such as `ot-ctl` and OTBR can manage it without inventing an entirely separate networking world.

6LoWPAN matters here because raw IPv6 packets are too large and verbose for a small 802.15.4 frame budget. Thread stays IP-native by compressing and fragmenting traffic where needed instead of abandoning IPv6.

---

## 4. Device Roles and Why They Matter

A Thread network is not flat. Nodes take on different roles depending on whether they route traffic, sleep aggressively, or coordinate parts of the mesh.

### Leader

The Leader manages key network-wide state such as partition information and configuration data. It is not a permanent boss node; if it disappears, another eligible node can take over.

### Router

Routers forward packets for other devices and keep the mesh connected. They are the infrastructure of the Thread network, so they are normally mains-powered or at least less aggressively power-constrained than sleepy endpoints.

### Router-Eligible End Device (REED)

A REED is not routing yet, but it can become a router if the network needs more routing capacity. This makes the network adaptive instead of requiring every node to be manually assigned a role up front.

### End Device / Sleepy End Device

End devices attach through a parent router instead of forwarding mesh traffic themselves. Sleepy end devices go further and spend most of their time asleep, which is how Thread supports long battery life without giving up network reachability.

### Border Router and Commissioner

A Border Router connects the Thread mesh to other IP networks such as Ethernet, Wi-Fi, or USB-backed Linux interfaces. A Commissioner is responsible for onboarding and authorizing new devices, which is why Thread setup is a network-management problem, not just a radio problem.

---

## 5. OpenThread Software Architectures

One reason OpenThread is so widely used is that it supports multiple deployment models instead of forcing every product into the same software split. The OpenThread platform documentation explicitly calls out both **SoC** and **co-processor** designs.  
Official source: [OpenThread platforms](https://openthread.io/platforms)

### SoC Design

In a System-on-Chip design, the radio, OpenThread stack, and application all run on the same MCU or wireless SoC. This is the most common approach for end devices because it is low-cost, low-power, and direct: your firmware calls the OpenThread APIs locally and the same chip drives the radio.

This is the model you use when the device itself is the product, such as a battery sensor, light switch, or smart plug. In ESP-IDF, the `openthread/ot_cli` example is a good mental model for this path.

### NCP Design

In a Network Co-Processor design, the host application talks to a co-processor using the **Spinel** protocol. OpenThread can run partly on the host side, with the network-facing device behaving like a dedicated networking component rather than the main application processor.

This split is useful when a stronger host CPU exists already and you want the network function separated from the application logic. It also makes host-side development easier because large application changes do not require rebuilding the radio firmware every time.

### RCP Design

In a Radio Co-Processor design, the host processor runs the core OpenThread stack while the attached device handles the minimal radio-facing work. OpenThread's official co-processor documentation describes this as the host keeping the protocol logic while the device acts as the Thread radio controller, usually connected over **SPI** or **UART** using **Spinel**.  
Official sources: [Co-Processor Designs](https://openthread.io/platforms/co-processor), [OT Daemon](https://openthread.io/platforms/co-processor/ot-daemon)

This is the model that matters most for Linux gateways and Jetson-class systems. It lets a Linux host run OTBR or `ot-daemon`, while a small MCU or radio SoC provides the 802.15.4 link.

---

## 6. OpenThread Internals: Why the Stack Is Portable

OpenThread is intentionally written so the networking core is portable across many chips and operating systems. The OpenThread platform guide describes it as portable C/C++ with a **narrow Platform Abstraction Layer (PAL)**, which is why the same core can run on bare-metal systems, FreeRTOS, Zephyr, Linux, and macOS.  
Official source: [OpenThread platforms](https://openthread.io/platforms)

The PAL is the contract between the OpenThread core and the underlying hardware port. The OpenThread porting guide lists the major PAL areas as:

* alarm/timer services
* bus interfaces such as UART or SPI
* IEEE 802.15.4 radio interface
* entropy / random source
* settings storage
* logging
* system-specific initialization  
Official source: [Platform Abstraction Layer APIs](https://openthread.io/guides/porting/implement-platform-abstraction-layer-apis)

This matters because it tells you what "porting OpenThread" really means. You are not rewriting the mesh stack from scratch. You are implementing the hardware-facing interfaces correctly so the portable upper layers can trust your radio, timer, storage, and transport behavior.

---

## 7. OpenThread on ESP32-C6

ESP-IDF exposes Thread support through its `openthread` component and documents several official examples:

* `openthread/ot_cli`
* `openthread/ot_rcp`
* `openthread/ot_br`
* `openthread/ot_trel`
* sleepy-device examples  
Official source: [ESP-IDF Thread guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/network/esp_openthread.html)

This is the key practical point for the roadmap: **ESP32-C6 can be used either as a full Thread node or as an RCP attached to a Linux host**. That makes it unusually good for study, because the same silicon can teach you both MCU-side Thread firmware and host-plus-radio architectures.

ESP-IDF also exposes the OpenThread lifecycle through functions such as:

* `esp_openthread_init(...)`
* `esp_openthread_auto_start(...)`
* `esp_openthread_launch_mainloop(...)`

These APIs show the embedded control flow clearly. First you initialize the platform and stack, then optionally start with a dataset, then enter the main loop that services radio events, timers, and protocol work.

---

## 8. CLI, Spinel, and Host Control

The OpenThread CLI is not just a demo shell. It is one of the fastest ways to understand what state the stack is in, how a node forms a network, and what role it currently holds.

For SoC designs, the CLI usually runs directly on the device. For host-plus-RCP designs, the host talks to the radio through **Spinel**, which OpenThread describes as the standard host-controller protocol used by both RCP and NCP designs.  
Official source: [Co-Processor Designs](https://openthread.io/platforms/co-processor)

When the host side is POSIX or Linux, **OpenThread Daemon (`ot-daemon`)** is the lightweight service wrapper around this model. The OpenThread docs show that it runs as a service, exposes a UNIX socket, and can be controlled with `ot-ctl`. This is the cleanest way to understand a Linux host talking to a serial Thread radio without bringing in the full Border Router stack immediately.  
Official source: [OT Daemon](https://openthread.io/platforms/co-processor/ot-daemon)

---

## 9. Security and Reliability

Thread is designed for real deployments, not one-off lab links. OpenThread's public description emphasizes reliable, secure, low-power device-to-device communication, and the Thread stack includes MAC security plus commissioning and border-router support.  
Official source: [OpenThread overview](https://openthread.io/)

From an embedded-software perspective, the interesting part is that security is not "added later in the cloud." Device identity, network admission, credential storage, and persistent settings are part of the firmware design from the beginning. That is why the PAL includes non-volatile settings and entropy sources as first-class platform requirements.

Reliability comes from the mesh behavior itself. A Thread network can survive node loss, promote eligible devices into routing roles, and keep sleepy endpoints attached through parents, which is very different from a simple point-to-point radio link.

---

## 10. Why OpenThread Belongs in Embedded Software

OpenThread belongs in **Embedded Software**, not only in Linux or networking sections, because the protocol is tightly coupled to firmware concerns:

* radio driver correctness
* timer precision
* ISR and task scheduling behavior
* persistent settings
* power states and sleepy-device timing
* serial buses used for CLI or Spinel

If your UART path drops bytes, your RCP host design fails. If your timer or alarm implementation is wrong, the mesh behavior becomes unstable. If your power model is wrong, your sleepy node either burns too much current or falls off the network.

This is exactly the type of protocol that turns "I can write drivers" into "I can build a networked embedded product."

---

## 11. Connection to the Rest of the Roadmap

| Roadmap area | How OpenThread connects |
|---|---|
| ARM MCU + CMSIS / HAL | You need working radio, timer, UART/SPI, entropy, and settings drivers underneath the stack |
| FreeRTOS | Thread nodes often run in RTOS-based firmware, so task structure, event loops, queues, and power-aware scheduling matter |
| UART / SPI | These buses are used not just for sensors, but also for CLI and Spinel host-controller links |
| Embedded Linux | OTBR and `ot-daemon` are Linux-host examples of the RCP architecture |
| Jetson deployment | A Jetson can act as the stronger host while an ESP32-C6 acts as the Thread radio |

OpenThread is therefore a natural bridge between pure MCU work and mixed MCU-plus-Linux systems. If you understand it well, you are better prepared for gateway design, border routers, Matter-style stacks, and multi-processor embedded products.

---

## 12. Suggested Projects

### Project 1: Single-board CLI node

Flash `ot_cli` onto an ESP32-C6 and use the CLI to form a one-node Thread network. Practice `ifconfig up`, `thread start`, `state`, and dataset commands until you are comfortable reading the network state directly.

### Project 2: Two-node mesh

Bring up two Thread-capable boards and verify that one becomes Leader while the other joins as a child or router. Watch how the routing roles and attach behavior change as you reset or power-cycle one node.

### Project 3: ESP32-C6 as RCP for a Linux host

Flash `ot_rcp` and connect it to a Linux host over UART or SPI. Use `ot-daemon` or OTBR and confirm that the host creates `wpan0`, can query `ot-ctl state`, and can start a Thread dataset.

### Project 4: Border router with external backbone

Use OTBR to bridge the Thread mesh to Ethernet, Wi-Fi, or a USB-backed Linux bridge. This is where OpenThread stops being just a radio experiment and becomes a real infrastructure component.

---

## References

### Official

* [OpenThread overview](https://openthread.io/)
* [OpenThread platforms](https://openthread.io/platforms)
* [OpenThread co-processor designs](https://openthread.io/platforms/co-processor)
* [OpenThread daemon](https://openthread.io/platforms/co-processor/ot-daemon)
* [OpenThread platform abstraction layer guide](https://openthread.io/guides/porting/implement-platform-abstraction-layer-apis)
* [ESP-IDF Thread / OpenThread guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/network/esp_openthread.html)

### Roadmap follow-ons

* [ARM MCU, FreeRTOS, and Communication Protocols](../../Guide.md)
* [Jetson ESP-Hosted Host Code](../../../3.%20Embedded%20Linux/Jetson%20ESP-Hosted%20Host%20Code/Guide.md)
* [ESP32-C6 OpenThread RCP on Jetson Orin Nano](../../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/5.%20Application%20Development/2.%20Network%20and%20Connectivity/ESP32-C6-OpenThread-RCP-Jetson-Orin-Nano.md)
