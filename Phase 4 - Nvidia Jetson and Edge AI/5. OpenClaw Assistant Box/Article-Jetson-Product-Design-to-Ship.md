# How to bring up an NVIDIA Jetson product: from design to ship

**Audience:** Engineers and program leads taking a **Jetson compute-module** design from **architecture** through **first units** to a **repeatable product** (custom carrier, enclosure, software image, and factory path).

**Scope:** A **phase-oriented** checklist—not a replacement for NVIDIA’s **Orin Nano/NX Module Design Guide**, **JetPack** release notes, or your CM’s DFM rules. Use it to **sequence work** and **avoid common gaps** between “schematic done” and “we can flash 100 units.”

**Related material on this roadmap:** [OrinClaw-Hardware-Design-Requirements.md](OrinClaw-Hardware-Design-Requirements.md) (subsystem decisions), [Guide.md](Guide.md) (software, inference, security, OTA), [Job-Post-PCB-Contractor-1wk.md](Job-Post-PCB-Contractor-1wk.md) (contractor deliverables).

---

## 1. What “product” means here

A **shippable Jetson product** is not a **dev kit in a box**. It usually means:

| Layer | You own it |
|-------|------------|
| **Mechanical** | Enclosure, thermal path, connector panel, labels |
| **Electrical** | **Custom carrier** (or SOM partner baseboard you customize), **BOM**, **ESD**, **test points** |
| **Firmware / BSP** | **Pinmux**, **device tree**, **kernel modules** that match **your** wiring (WiFi bridge, audio codec, PHY) |
| **Software image** | **Pinned JetPack** line, **partitioning**, **Docker/compose** or equivalent, **OTA** strategy |
| **Quality** | **Bring-up script**, **factory test**, **failure analysis**, revision control on **PCB + image** |

If any of these stops at “works on my desk once,” you do not yet have a **product**—you have a **prototype**.

---

## 2. End-to-end phase map (design → ship)

Think in **gates**. Names vary (EVT/DVT/PVT); the **dependencies** do not.

| Phase | Focus | Typical exit criteria |
|-------|--------|------------------------|
| **A — Architecture** | SoM choice, block diagram, power budget, risky interfaces (RGMII, **M.2**, **HDMI**, **SPI** to RF MCU), RF/antenna | Written requirements; **no** layout start without **connector** and **thermal** envelope |
| **B — Schematic + pinmux** | Full carrier schematic, **NVIDIA pinmux workbook** started, **power sequencing**, **strap** review | **MPN** on symbols; **pinmux** rows for every SoM ball you use |
| **C — Layout + fab** | Stack-up, **SI/PI**, **RF keepout**, **assembly** drawing, Gerbers, **BOM** with **DNP** variants | **DFM** sign-off from fab; **v0** boards ordered |
| **D — Hardware bring-up** | First power, **UART**, **NVMe**, **Ethernet**, **USB**, display if any, **peripherals** (e.g. **I2S**, **SPI** co-processor) | **Known-good** hardware revision note; **minimal** DTB boots |
| **E — BSP integration** | Merge **DT** fragments, **out-of-tree** drivers (e.g. **ESP-Hosted**), **udev**, **audio** devices | **Flashed** image **reproducible** from docs + artifacts |
| **F — Product software** | **Inference** stack, **orchestrator**, **health**, **security** baseline | Acceptance tests pass on **golden** units |
| **G — Manufacturing + OTA** | **ICT** or **fixture**, **serial** scheme, **signed** updates, **rollback** | Pilot build **yield** known; **field update** proven |

Phases **overlap** in real programs (software starts before layout freezes), but **dependencies** are strict: **wrong pinmux** cannot be fixed in software without **PCB spin** or **bodge**.

---

## 3. Architecture and requirements (before CAD)

- **Pick the module** (e.g. **Orin Nano 8GB**) and **freeze** **JetPack** target line for **first silicon** (e.g. **5.x bring-up**, **6.x production**—your program’s [Guide.md §3](Guide.md) strategy).
- **Power:** Size input (**USB-C PD**, barrel, **PoE** if used), **Orin sequencing**, **worst-case** W—fill a **power budget table** (see [OrinClaw-Hardware-Design-Requirements.md §4](OrinClaw-Hardware-Design-Requirements.md)).
- **High-speed list:** **RGMII**, **PCIe/NVMe**, **USB3/2**, **HDMI** or **DP**—each needs **length/match**, **ESD**, and **connector** choice locked early.
- **RF:** If you use a **WiFi/BT** module or **ESP32** + **ESP-Hosted**, **antenna**, **keepout**, and **coexistence** are **layout** constraints, not afterthoughts.
- **Regulatory:** Intentional radiators (**WiFi**, **BT**, **Thread**) drive **FCC/CE** scope—note **which** antennas and modules are **certified** vs **integration** responsibility.

Deliverable: **one** hardware requirements doc + **block diagram** signed off for **scope**.

---

## 4. Schematic design (carrier)

- **Module socket** and **straps:** Boot device (**NVMe**/eMMC), **recovery** access—errors here waste **weeks**.
- **Power tree:** **PD controller**, **buck** tree, **PMIC** if used; **inrush**, **UVLO**, **brownout** on **USB attach**.
- **Ethernet:** **PHY** + **magnetics** (or integrated **RJ45**); **RGMII** **MDIO**/MDC; **clocks**.
- **Storage:** **M.2** **2280** (mechanical keepout vs **heatsink**); **3.3 V** and **perst** handling.
- **USB:** **Host** ports (**Type-A**, **Type-C**); **ESD**; **CC** logic for **C**; separate **PD sink** from **data** if required—document user-facing roles ([OrinClaw-Hardware-Design-Requirements.md §2.10](OrinClaw-Hardware-Design-Requirements.md)).
- **Audio:** **Codec** **I2S**, **mic** bias, **amp**; **anti-pop**, **filter**.
- **Debug:** **UART** header (gated in production **image policy**); **test points** for rails and **critical** clocks.

Deliverable: **schematic PDF** + **BOM CSV** + **design notes** (SI, power, RF).

---

## 5. Pinmux and device tree (NVIDIA workflow)

- **Source of truth:** NVIDIA **Orin Nano / NX pinmux** **Excel** (JetPack-aligned)—**not** hand-written `.dtsi` as primary.
- **Cross-reference:** **Ball ↔ net ↔ spreadsheet row** for **every** used SoM pin ([Job-Post-PCB-Contractor-1wk.md](Job-Post-PCB-Contractor-1wk.md) expectation).
- **Generate** `*pinmux*.dtsi` / pad configs via **NVIDIA’s documented flow**; **BSP owner** merges into **board DTS**.
- **Validate** against **schematic** before fab: **GPIO** conflicts and **boot-critical** pins cause **silent** or **hard** failures.

Deliverable: **versioned** `carrier_pinmux_JPx_vYYYYMMDD.xlsm` + **generated** fragments + **integration** notes.

---

## 6. PCB layout, DFM, and fab package

- **Stack-up** agreed with fab; **impedance** for **RGMII**, **PCIe**, **USB**, **HDMI** (or SoM routing per **Design Guide**).
- **Placement:** **PHY** near **RJ45**; **M.2** clear of **module** **heatsink** and **shell**; **RF** module at **edge** with **keepout**.
- **DFM/DFA:** **assembly** polarity, **fiducials**, **panelization**, **test coupon** if needed.
- **Outputs:** Gerbers, **IPC-356** netlist if required, **pick-and-place**, **BOM** with **revision**.

Deliverable: **fab-ready** package **versioned** with **PCB** revision (e.g. `PCB_R1A`).

---

## 7. First article hardware bring-up (board on bench)

Use a **written** checklist; **one** person **records** results.

1. **Visual / continuity** (where safe): shorts on **PD** rail, **wrong** **FET** orientation.
2. **Limited power:** **current-limited** supply or **bench** **PD** source; measure **rails** in **sequencing** order per NVIDIA.
3. **UART console:** **Boot logs** to **U-Boot**/kernel; fix **DT** if **hangs** before **kernel**.
4. **Storage:** **NVMe** detected; **flash** **rootfs**; **reboot** loop test.
5. **Ethernet:** **link**, **DHCP**, **iperf** sanity.
6. **USB host:** **enumerate** keyboard, storage, **FTDI**-class device if in scope.
7. **Display** (if fitted): **EDID**, **known-good** mode.
8. **Audio:** **capture** / **playback** loopback.
9. **Co-processor link** (e.g. **SPI** to **ESP32**): **loopback** or **firmware** **version** read.
10. **Thermal:** **stress** with **`tegrastats`**; **no** throttle surprise under **enclosure**—**heatsink** **torque** per NVIDIA.

Deliverable: **Bring-up report** (revision, date, **pass/fail**, **photos** of **mods**).

---

## 8. BSP and Linux integration

- **Kernel**: Match **JetPack**; rebuild **out-of-tree** drivers (**ESP-Hosted**, etc.) for **that** **kernel**.
- **Rootfs**: **Ubuntu** + L4T; decide **`/`** vs **`/data`** layout early ([Guide.md §4](Guide.md)).
- **Device permissions**: **`udev`** for **audio**, **video**, **serial** bridges.
- **Secure baseline**: **firewall**, **ssh** policy, **no** default passwords—align with your **security** section ([Guide.md §8](Guide.md)).

Deliverable: **Flashable** image or **documented** **flash** + **post-install** script; **manifest** of **versions** (JetPack, **DTB** hash, **firmware** blobs).

---

## 9. Product application stack (what users experience)

- **Services**: **STT**, **LLM**, **TTS**, **gateway**, **UI**—**Compose** or **systemd**; **health** endpoint.
- **Inference**: **Pinned** runtimes (**TensorRT** / **ONNX Runtime**); **engines** built for **this** JetPack ([Guide.md §5](Guide.md)).
- **OTA**: **Signed** artifacts, **A/B** or **compose** swap, **rollback**—do not **SSH-only** updates for **fleet**.

Deliverable: **Release** **notes** + **acceptance** **tests** on **golden** hardware.

---

## 10. Manufacturing and quality

- **Serial numbers** and **revision** traceability (**PCB** + **image**).
- **Factory test**: **short** **automated** script (**Ethernet**, **storage**, **audio** loop, **RF** **smoke**)—**not** only **visual**.
- **Yield**: Track **first-pass**; **RMA** **root-cause** to **schematic** vs **assembly** vs **software**.
- **Compliance samples**: **pre-scan** **EMI** before **big** **build** if schedule is tight.

---

## 11. Common failure modes (save your schedule)

| Symptom | Often actually |
|---------|----------------|
| **No boot from NVMe** | **Straps**, **DT** **pcie**/**nvme** nodes, **M.2** **power**, **bad** **SSD** lot |
| **Ethernet up, no link** | **PHY** **mode** **straps**, **magnetics**, **REFCLK**, **wrong** **phy** **address** |
| **USB works on dev kit, not carrier** | **USB** **role**, **mux**, **CC**, **SS** **lanes** **swap** |
| **WiFi** **missing** | **ESP-Hosted** **.ko**/**firmware** **mismatch**, **SPI** **wiring**, **regulator** **brownout** |
| **Audio** **hiss**/dropouts | **ground**, **I2S** **word** **clock**, **codec** **register** **init**, **DMA** **coherency** |
| **Random** **OOM** under **load** | **8GB** **UMA** **budget**—**too** **many** **services** + **model** ([Guide.md §5](Guide.md)) |

---

## 12. Summary

Bringing a **Jetson product** to ship is **systems integration**: **hardware**, **BSP**, and **application** **must** **version** together. The **shortest** path is **early** **pinmux**/**power**/**RF** **correctness**, **disciplined** **bring-up** **notes**, and a **software** **line** that **pins** **JetPack** and **tests** **rollback** before you **scale** **builds**.

---

*Educational roadmap article. Follow NVIDIA official documentation for module-specific electrical, thermal, and software requirements.*
