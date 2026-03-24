# Jetson Orin Nano carrier board — hardware design requirements

**Document role:** Hardware-facing specification for a custom **NVIDIA Jetson Orin Nano 8GB** compute-module carrier: subsystem scope, resolved architecture decisions, checklist, power budget template, and bring-up expectations. For use in schematic, pinmux, PCB layout, and manufacturing handoff.

**Product class (generic):** Edge AI / voice-assistant appliance carrier — Ethernet, **ESP32-C6** on-carrier (ESP-Hosted + Matter/Thread/Zigbee), NVMe, I2S audio, status LED ring, **USB-C PD** (system power input), HDMI for external monitor, **USB host: 2× Type-A + 1× Type-C** (standard **USB 2.0 / USB 3.x** only — **not Thunderbolt / USB4**) — **§2.10**. **Developer expansion (no extra carrier ICs):** optional **USB FT232H** kit on any host port — **§2.9**; OS/docs only.

---

## 1. Project summary (scope)

Design a **custom carrier** (not a developer-kit clone) integrating:

| Subsystem | Requirement |
|-----------|-------------|
| **SoM** | NVIDIA Jetson **Orin Nano 8GB module** — module socket, power sequencing per NVIDIA / module datasheet |
| **Power** | **USB-C PD** input; power tree sized for **peak SoC + NVMe + RF + audio + display** + **USB VBUS** loads (see §4, §2.10) |
| **USB host** | **2× USB Type-A** + **1× USB Type-C** **downstream host** per **§2.10** — **USB 2.0** and/or **USB 3.x SuperSpeed** per SoM pinout and stack; **not Thunderbolt / USB4** |
| **USB power in** | **USB-C PD** receptacle(s) per **§2.2** — **power negotiation**; **may** be a **separate** USB-C from the **host** USB-C (recommended for UX) or combined via **PD + USB mux** (document in schematic notes) |
| **Ethernet** | **Gigabit** — RGMII PHY + magnetics (or integrated RJ45 per BOM choice) |
| **Storage** | **M.2 NVMe 2280** — default mass-storage class **512 GB**; socket, 3.3 V rails, length keepout |
| **Display out** | **HDMI** — **external monitor only** (setup, debug, optional local UI); **not** an integrated panel on the appliance; implement per **NVIDIA Orin Nano module design guide** and **JetPack-aligned pinmux** for the chosen **display / SOR** mapping |
| **RF coprocessor** | **ESP32-C6 module** — **ESP-Hosted** on **SPI** to Jetson; antenna + keepout; coexistence per vendor guidance |
| **Audio** | **I2S** — codec, **mic array** interface, **Class-D** (or suitable) amplifier to **full-range** speaker; layout suitable for **acoustic echo cancellation** |
| **UX I/O** | **Programmable LED ring** (room-glanceable status), **hardware mute**, **action button(s)** |
| **Battery** | **Optional** program variant — see §2.1 (default first spin = **AC / USB-C PD only** unless the program funds battery on the same tape-out) |
| **Pinmux** | **NVIDIA Excel → device tree** workflow only; **JetPack-aligned**; every used ball documented **ball ↔ net ↔ spreadsheet row** |
| **Documentation** | Block diagram, power budget, pinmux cross-reference, bring-up checklist, layout handoff notes |

---

## 2. Resolved architecture questions

### 2.1 Battery subsystem

| Question | **Decision** |
|----------|----------------|
| Required in first revision? | **No** for **base bring-up** spin. First validated PCB targets **USB-C PD + mains** only. |
| Future optional variant? | **Yes.** Program may add **battery SKU** later. Hardware plan: **(A)** tape out **AC-only** first, then **battery re-spin**, or **(B)** one PCB with **DNP / populate** battery BOM line — choose in **statement of work**; default recommendation **(A)** to reduce risk. |
| Cell configuration | **1S Li-ion / LiPo** preferred for volume, BMS simplicity, and typical small-tower enclosure fit. **2S** only if mechanical and PD budget explicitly allow — document deviation. |
| Runtime target (when battery ships) | **Target 30–60 minutes** at nominal voice + automation load with optional **low-power** behavior (dim LED, capped GPU) — **not** full sustained ML stress at maximum clocks. |
| Full load while charging? | **Design goal:** on PD, system runs from adapter and charges battery when present. If PD budget is insufficient, firmware **may** throttle GPU or LED while charging. Hardware **shall** support **≥ 30 W** negotiated input for the base configuration (see §4). Peak worst-case **may** require throttle or higher PD contract — document in power budget. |

**Hardware action:** For AC-only spins, **document** placement for future **BMS, OR-ing, and path FETs** so a battery variant minimizes schematic rip-up.

---

### 2.2 Power requirements

| Question | **Decision** |
|----------|----------------|
| Peak system power (planning) | Size input and PMIC rails for **~35 W peak** (short transient), **~15–25 W typical sustained** under voice, inference, and WiFi. Include margin for converter efficiency and RF + audio peaks. |
| USB-C PD profile (base configuration) | **Minimum:** negotiate **≥ 15 V @ 2 A (30 W)** or **9 V @ 3 A (27 W)** — choose the minimum set that satisfies **Orin Nano module + NVMe + peripherals** after buck efficiency. **Preferred:** **15 V @ 3 A (45 W)** or **20 V @ 2.25 A** class **USB PD 3.0** (or **PPS** if the controller supports) for headroom and future battery charge + run. **5 V-only profiles are not sufficient** for this class. |
| Sequencing | Obey **module power-up order** per NVIDIA / module documentation; USB-C PD controller **must** fail safely (no brownout on attach/detach). |

---

### 2.3 Mechanical constraints

| Question | **Decision** |
|----------|----------------|
| Envelope reference? | Use the program’s **industrial design package** (envelope drawing, reference renders, CMF notes) as **directional** for connector placement and keepouts until **CAD freeze**. Example form factor often cited: **~130 mm** tall cylindrical / soft tower, fabric grille, **top LED ring**, **side mute**, **base:** **USB-C PD**, **2× USB-A + 1× USB-C host**, Gigabit Ethernet, vents, foot ring. |
| Connector placement (electrical intent) | **RJ45** + **USB-C PD** (power) + **2× USB-A host** + **1× USB-C host** (data, **no TB**) per **§2.10** on **base / rear facet**; **HDMI** on same facet or adjacent **service panel**. **Silk / manual:** distinguish **power** USB-C vs **data** USB-C if both are Type-C. **No user connectors on top** (LED + microphones only). **M.2 2280** placement **must** clear **module heatsink** and **enclosure shell**. **ESP32-C6 antenna** on **RF-friendly** edge per module datasheet; **HDMI shell** and ESD **must not** violate **antenna keepout**. |
| CAD freeze | Deliver **connector keepout drawing** (DXF, STEP, or dimensioned PDF) **before PCB placement freeze**; until then use **maximum envelope** from industrial design. |

---

### 2.4 LED system

| Question | **Decision** |
|----------|----------------|
| Host: Jetson or ESP32-C6? | **Primary: Jetson-side control** for **deterministic** status timing (wake acknowledgement on the order of **~150 ms** class). Implement with dedicated **LED driver IC(s)** (e.g. I2C/SPI RGB controller or constant-current shift-register chain) on **GPIO / I2C / SPI** from pinmux. **ESP32-C6 shall not** be the sole owner of the LED ring on the first revision (cross-SoC sync and failure modes). Optional C6 duplication for factory test only if explicitly required — default **no**. |
| Brightness / current | **Visible at ~3 m** in normal room lighting. Use **diffused ring** (light guide); **PWM dimming** for idle and low-power modes. Itemize **peak segment current** and **total LED power** in the power budget; note **thermal** impact on the enclosure. |

---

### 2.5 Audio system

| Question | **Decision** |
|----------|----------------|
| Voice vs hi-fi? | **Voice-first:** far-field capture, **barge-in**, low-latency text-to-speech playback. **Playback quality:** adequate for TTS and light media — **not** a hi-fi stereo target. **Mic array** with **AEC-friendly** mechanical spacing from the speaker; **speaker 4–8 Ω, 3–10 W RMS** class typical; enclosure sealing / bracing is a **shared** mechanical + acoustic task — carrier provides **clean I2S, low-noise mic bias, sensible amp layout**. |

---

### 2.6 HDMI / external video out

| Question | **Decision** |
|----------|----------------|
| Role | **Default UX** remains **voice + LED + LAN web UI** — no **integrated** display panel on the appliance. **HDMI** supports **bring-up, factory test, first-boot on a monitor, debug, demos, optional desktop** when a monitor is attached. |
| Electrical | Route **Jetson Orin Nano** display signals per **NVIDIA module datasheet and design guide** for **HDMI** (SOR / pinmux for target JetPack). **ESD** on connector; **controlled-impedance** differential pairs and length matching per NVIDIA guidance and any **retimer / level-shifter** datasheet if used. |
| Resolution | Target **at least 1080p60** where SoM and stack support the chosen HDMI circuit; document **tested modes** in bring-up. |
| Software | **Device tree** enables the display pipeline (pinmux workbook + board DTS integration). **Normal shipping use** does **not** require a connected monitor. |
| BOM | HDMI connector **in** base carrier BOM unless a **cost-down variant** explicitly **DNPs** it (separate variant ID and assembly drawing). |

---

### 2.7 BSP integration (split of work)

| Question | **Decision** |
|----------|----------------|
| Pinmux / DT artifacts | **Hardware contractor minimum:** JetPack-aligned **pinmux workbook**, documented run of NVIDIA’s **official device-tree generation flow**, **reference `*pinmux*.dtsi` outputs** (regenerable; spreadsheet is source of truth, not hand-edited DTSI). **Ball ↔ net ↔ spreadsheet row** cross-reference table. |
| Integration and validation | **Program BSP owner** merges fragments into **board DTS**, builds **DTB**, validates **boot and peripherals** on hardware. **Optional paid:** contractor smoke test on module + fabbed board. |

---

### 2.8 Layout scope and schedule

| Question | **Decision** |
|----------|----------------|
| Fab-ready in one week? | **Default:** **Phase A** — schematic + pinmux + documentation + layout constraints in first milestone. **Phase B** — layout + DRC-clean fab package as second milestone unless a single bid covers full layout. **Phased delivery** is acceptable when risk is non-trivial (**RGMII, M.2, HDMI, SPI to C6, RF**). |

---

### 2.9 Developer expansion: USB FT232H kit (optional — **no FT232H on carrier**)

| Question | **Decision** |
|----------|----------------|
| Scope | **This stage:** keep **ESP32-C6** on the carrier for integrated RF and IoT. **Do not** add FT232H or similar bridge ICs to the **main PCB BOM** for this feature. |
| What ships? | An **optional retail / developer expansion kit**: commodity **FT232H** (or FT2232H-class) **USB module** + flying leads or small breakout; user connects **DUT** to the module’s **I2C / SPI / UART / GPIO** pins per vendor wiring. |
| How it attaches | **USB only** — plug into any **exposed USB host** port (**Type-A** or **Type-C** with **C-to-A** adapter/cable as needed). **No new silicon** on OrinClaw for FT232H; **§2.10** defines **2× A + 1× C** host. |
| Software / OS | **Image responsibility (not schematic):** kernel **`ftdi_sio`** (or documented **D2XX** policy), stable **`/dev/ttyUSB*`** or `libftdi`/`pyftdi`, **`udev` rules** (VID/PID, permissions, `dialout`), optional **Docker device passthrough** for OpenClaw services, and a **short user guide** for hardware-oriented skills (e.g. ClawHub-published skill folders) that depend on the kit. |
| vs ESP32-C6 | C6 = **always-on radios + on-product buses**. FT232H kit = **bench / lab** multi-protocol USB bridge for **skills** that need **MPSSE-style** or **pyftdi** workflows — complementary, not duplicate. |

---

### 2.10 USB host and USB-C PD (fixed: **2× Type-A + 1× Type-C host**; **not Thunderbolt**)

**Roles**

| Connector | Role |
|-----------|------|
| **USB-C PD** | **System power input** only (per **§2.2**) — **USB Power Delivery** negotiation. **Not** marketed or wired as a **Thunderbolt** or **USB4** dock port. |
| **2× USB Type-A** | **Downstream USB host** — keyboard, mouse, FT232H-class dongle, thumb drive, **USB-A** peripherals. |
| **1× USB Type-C** | **Downstream USB host** — same as Type-A but for **USB-C** devices and **C-to-A** cables/adapters; **USB 2.0** and/or **USB 3.x SuperSpeed** per **Orin Nano** USB lane budget and pinmux (**document** speed class per revision). |

**Thunderbolt / USB4**

- **Do not** implement **Thunderbolt 3/4** or **USB4** **PCIe / DisplayPort tunneling** on the **user** USB-C host port. This is a **normal USB-C host** (high-speed USB only), **not** a laptop-style **Thunderbolt** port. **No TB retimer**, **no** TB controller — BOM and enclosure copy stay **USB / SuperSpeed USB** only unless the program explicitly changes scope.

**USB-C PD vs host USB-C**

- **Recommended UX:** **two** USB-C receptacles on the enclosure when both PD and a **Type-C host** are present: one **labeled for power** (PD sink), one **labeled for data** (USB host). **Alternate:** one USB-C with **PD + USB mux** — document **CC** behavior and user-facing limitations (e.g. **no SS** while certain cables attached).
- **Power:** Size **VBUS** / inrush / polyfuse for **all** downstream ports loaded (e.g. **SSD enclosure** on **C**, keyboard + dongle on **A**). State **per-port** and **aggregate** limits on silk or manual.

**Use cases**

| Role | Typical device |
|------|----------------|
| Local console / setup | **USB keyboard** (often **Type-A**) |
| Pointing | **USB mouse** *or* **Bluetooth mouse** (BT via **ESP32-C6**) |
| Developer / lab | **USB FT232H-class** dongle (**§2.9**) — **A** or **C** (with adapter) |

**Program decision (baseline SKU)**

- **Ship:** **2× USB-A host** + **1× USB-C host** + **USB-C PD input** (implementation: **separate** PD receptacle vs muxed — **§2.2** + schematic notes).
- **User guide:** **Bluetooth mouse** still reduces **wired** congestion; **powered hub** only if more than **three** downstream devices are needed at once.

---

## 3. Technical requirements checklist

- [ ] **Pinmux:** NVIDIA **Excel only** as source of truth; no invalid combinations; regenerable `.dtsi`; versioned workbook filename e.g. `carrier_pinmux_JPx_vYYYYMMDD.xlsm` (match actual JetPack).
- [ ] **Signal mapping:** Every SoM ball used on carrier: **ball ↔ net ↔ function ↔ spreadsheet row**.
- [ ] **Power:** Orin Nano sequencing satisfied; negotiated **USB-C PD** documented; fill §4 power table.
- [ ] **High-speed:** **RGMII** length/match; **M.2** impedance; **SPI to C6** per Espressif + NVIDIA guidance; **HDMI TMDS** (or SoM display lanes) per **Orin Nano** design guide and chosen silicon.
- [ ] **RF:** C6 **antenna keepout**, ground reference, **no copper under antenna** per module datasheet; **ESD** on external interfaces.
- [ ] **Audio:** Short **I2S** paths, clean returns; mic vs speaker placement input for mechanical.
- [ ] **DFM/DFA:** **MPN** on every schematic symbol before layout freeze; assembly polarity; **test points** for first power-on.
- [ ] **Secure boot / production:** Strapping and storage compatible with **NVIDIA Jetson secure boot** and program signing policy — do not block secure-boot with wrong straps or fuse assumptions.
- [ ] **USB host (§2.10):** **2× Type-A** + **1× Type-C** host implemented; **ESD** on **A** and **C**; **VBUS** / aggregate current policy documented; **no Thunderbolt** on user USB-C.

---

## 4. Power budget template (deliverable)

Fill and maintain:

| Rail / domain | Typical (W) | Peak (W) | Note |
|---------------|-------------|----------|------|
| Orin Nano module (incl. memory) | | | Per NVIDIA guidance + measured later |
| NVMe SSD | | | 2280 active read/write |
| Ethernet PHY | | | |
| ESP32-C6 (WiFi/BT + radios) | | | Peak TX |
| HDMI / display path (if applicable) | | | Active output increases SoC display power |
| Audio codec + amplifier | | | Max SPL scenario |
| LED ring (max) | | | All segments on |
| USB / misc (VBUS to Type-A ports) | | | Keyboard + FT232H + hub margin per **§2.10** |
| **Total input (post converter)** | | | Compare to **negotiated USB-C PD** |

---

## 5. Engineering phases

| Phase | Content |
|-------|---------|
| **1 — Architecture and risk** | Block diagram; Jetson ↔ peripheral map; **2–3 alternates** per risky block; power tree; start pinmux workbook |
| **2 — Pinmux and schematic** | Freeze workbook for JetPack line; complete schematic; design notes (SI, PI, RF, audio) |
| **3 — Layout** | Stack-up; placement; route **RGMII, M.2, SPI, HDMI, USB** first; DRC; fab outputs |
| **4 — Documentation and bring-up** | Power budget; pinmux ↔ schematic; bring-up checklist (first power, UART, NVMe, Ethernet, **HDMI detect and known-good video mode**, SPI/C6, I2S loopback); **optional:** USB host enumerates **FT232H-class** dongle + `udev` note (§2.9); open issues |

---

## 6. Revision history

| Version | Date | Change |
|---------|------|--------|
| 0.1 | 2026-03-18 | Initial carrier hardware requirements |
| 0.2 | 2026-03-18 | Added HDMI external display |
| 0.3 | 2026-03-18 | Generic hardware-only text: removed product names, external links, and program-doc traceability |
| 0.4 | 2026-03-18 | Added optional **IEEE 802.3bt PoE++** PD on Ethernet (up to **60 W** budget, isolated PoE DC/DC); USB-C vs PoE priority; checklist, power table, bring-up |
| 0.5 | 2026-03-18 | Removed **PoE**; power input is **USB-C PD** only again |
| 0.6 | 2026-03-19 | Added optional **§6 post-ship developer expansion** for add-on hardware modules and hardware-oriented skill development |
| 0.7 | 2026-03-19 | **§6 rework (vNext):** optional **bottom expansion plate** docking to main unit; exposes **GPIO, SPI, I2C**; normal users omit; mechanical/electrical contract for plate + main dock |
| 0.8 | 2026-03-19 | Removed **bottom expansion plate** and **§6**; no dock/expansion hardware in this spec |
| 0.9 | 2026-03-19 | **§2.9** optional **USB FT232H developer kit** — no on-board FT232H; ESP32-C6 unchanged; OS/docs/USB host support |
| 1.0 | 2026-03-19 | **§2.10** **2× vs 3× USB host** ports; default **2** + BT mouse; optional **3rd** SKU |
| 1.1 | 2026-03-19 | **§2.10** locked to **2× USB-A + 1× USB-C host** (USB 2/3 only); **USB-C PD** power input separate from **non-TB** host **C**; dropped **3rd Type-A** SKU option |

---

*End of hardware design requirements — carrier schematic, pinmux, layout, and bring-up.*
