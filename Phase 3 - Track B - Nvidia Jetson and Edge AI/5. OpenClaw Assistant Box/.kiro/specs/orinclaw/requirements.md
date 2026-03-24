# Requirements Document

## Introduction

OrinClaw is an always-on local AI assistant appliance built on Jetson Orin Nano 8GB, orchestrated by OpenClaw. It delivers offline-first voice interaction, smart-home control, browser automation, and optional BYOK cloud AI — all from a purpose-built custom PCB product with strong at-rest security, layered OTA, and deterministic low-latency UX. The product targets better usability than cloud-dependent smart speakers by being offline-first, faster in perceived response, more reliable, and more capable for power users.

**Scope note:** The **V1 shippable product** is a **single-room** assistant on the minimal stack (see repo `Guide.md`). **Multi-room / media-hub / person-aware routing** is **Milestone 10** and **post–Milestone-9** R&D, not a V1 gate.

**Normative technical source of truth:** [Guide.md](../../../Guide.md) (hardware, software, security, OTA, milestones). **Non-normative business / GTM context:** [OrinClaw-Product-Business-Plan.md](../../../OrinClaw-Product-Business-Plan.md) (market, investment, team—does not override product requirements unless adopted here).

---

## Glossary

- **OrinClaw**: The product and project name for this capstone AI assistant appliance.
- **OpenClaw**: The open-source orchestration platform providing the Gateway, multi-channel inbox, skills runtime, and browser automation used as OrinClaw's control plane.
- **Gateway**: The OpenClaw control-plane process (Node.js, port 18789) that routes sessions, tools, and channels.
- **DeviceService**: The OrinClaw service responsible for LED ring, hardware buttons, mute switch, and power/thermal state.
- **WakeService**: The always-on, low-power wake-word detection service.
- **STTService**: The streaming speech-to-text inference service.
- **LLMService**: The local large-language-model inference service providing chat and tool-calling.
- **TTSService**: The text-to-speech synthesis service producing audio output.
- **SkillsRuntime**: The sandboxed plugin execution environment for home automation, browser control, and other tools.
- **BrowserTool**: The Playwright-based browser automation container/tool.
- **ESP32-C6**: The co-processor module providing WiFi, Bluetooth/BLE, Thread, Zigbee, and Matter radios, plus LED/button I/O.
- **ESP-Hosted-NG**: The Espressif host driver that exposes the ESP32-C6 WiFi/BT as a standard Linux cfg80211 interface over SPI.
- **L4T**: NVIDIA Linux for Tegra — the Ubuntu-based OS shipped as part of JetPack for Jetson modules.
- **JetPack**: NVIDIA's software stack for Jetson (L4T + CUDA + TensorRT + cuDNN).
- **UMA**: Unified Memory Architecture — the single LPDDR pool shared by CPU and GPU on Jetson Orin Nano.
- **TTFT**: Time-to-first-token — latency from end of user speech to first LLM output token.
- **FDE**: Full-disk encryption (LUKS/dm-crypt or TCG Opal SED) protecting the 512 GB NVMe at rest.
- **LUKS**: Linux Unified Key Setup — the standard Linux block-device encryption layer.
- **BMS**: Battery Management System — protection and charge/discharge control for the optional battery.
- **OTA**: Over-the-air software update.
- **P0/P1/P2/P3**: OTA planes — Boot/rootfs, Host glue, Compose stack, Data plane respectively.
- **BYOK**: Bring Your Own Key — optional user-supplied cloud LLM API keys (Claude/GPT/Gemini).
- **AEC**: Acoustic Echo Cancellation — software removal of speaker output from microphone capture.
- **Matter**: The CSA smart-home interoperability protocol supported over Thread and WiFi.
- **mDNS**: Multicast DNS — used to resolve `orinclaw.local` on the LAN.
- **TIM**: Thermal Interface Material (paste or pad) between SoC and heatsink.
- **SMART**: Self-Monitoring, Analysis and Reporting Technology — NVMe drive health telemetry.
- **WER**: Word Error Rate — accuracy metric for speech-to-text transcription.
- **TensorRT**: NVIDIA's inference optimization and runtime library for GPU-accelerated models.
- **ONNX**: Open Neural Network Exchange — portable model interchange format.
- **DLA**: Deep Learning Accelerator — dedicated inference engine on Jetson SoCs.
- **KV cache**: Key-value cache used by transformer LLMs to avoid recomputing attention over prior context.
- **NVMe Sanitize**: NVMe-standard cryptographic or block-erase command that renders drive data unrecoverable.
- **tegrastats**: NVIDIA utility reporting Jetson CPU/GPU clocks, temperatures, and memory usage in real time.
- **SPL**: Sound Pressure Level — acoustic output loudness measured in dB.
- **PD**: USB Power Delivery — the USB-C charging/power negotiation protocol.
- **CC/CV**: Constant-current / constant-voltage — standard Li-ion battery charging profile.
- **TPM2**: Trusted Platform Module version 2 — hardware security chip used for key sealing.
- **fTPM**: Firmware TPM — TPM2 functionality implemented in firmware (e.g. ARM TrustZone).
- **TCG Opal**: Trusted Computing Group Opal — hardware self-encrypting drive standard.
- **SED**: Self-Encrypting Drive — NVMe drive with hardware-native encryption per TCG Opal.
- **DKMS**: Dynamic Kernel Module Support — framework for building out-of-tree kernel modules.
- **CMA**: Contiguous Memory Allocator — kernel region reserved for GPU/multimedia on Jetson.
- **RAG**: Retrieval-Augmented Generation — LLM technique that queries a local document index at inference time.
- **RSSI**: Received Signal Strength Indicator — WiFi link quality metric.
- **BOM**: Bill of Materials — the complete list of components for the custom PCB.
- **DFM/DFA**: Design for Manufacturability / Design for Assembly — PCB production readiness checklists.
- **MPN**: Manufacturer Part Number — unique identifier for a specific component from a specific vendor.
- **DNP**: Do Not Place — BOM flag for components omitted in a given build variant.
- **NRE**: Non-Recurring Engineering — one-time cost for custom PCB design and tooling.
- **EMI**: Electromagnetic Interference.
- **ESD**: Electrostatic Discharge protection.
- **I2S**: Inter-IC Sound — digital audio bus used for mic array and speaker amplifier.
- **SPI**: Serial Peripheral Interface — bus used for ESP-Hosted between Jetson and ESP32-C6.
- **UART**: Universal Asynchronous Receiver-Transmitter — optional secondary bus for LED/button DeviceService.
- **MQTT**: Message Queuing Telemetry Transport — lightweight IoT messaging protocol.
- **CSA**: Connectivity Standards Alliance — the standards body governing Matter certification.
- **UN38.3**: UN transport testing standard for lithium batteries.
- **p95 / p99**: 95th / 99th percentile latency — statistical tail-latency metrics.
- **Tailscale** (or equivalent mesh VPN): Optional private connectivity so trusted clients reach OrinClaw on the LAN without WAN port-forwarding; must follow least-privilege ACLs and authenticated Gateway exposure (see product `Guide.md` §8).
- **V1**: The first shippable OrinClaw release — **single-room** voice assistant, offline-first core, **Core OrinClaw SKU** only unless the program explicitly expands scope.
- **Core OrinClaw SKU**: V1 launch configuration per `Guide.md` §3 — Jetson Orin Nano 8GB-class, 512 GB NVMe, voice + LED + LAN web UI, **no integrated main display** (HDMI for **external** monitor is allowed for bring-up/debug); optional Tailscale; optional Matter-prefer path as specified in compatibility requirements.


---

## Requirements

### Requirement 1: Product Identity and Naming

**User Story:** As a developer or end user, I want the product to have a consistent identity across all surfaces, so that hostname, URLs, OTA channels, and documentation are unambiguous and traceable.

#### Acceptance Criteria

1. THE OrinClaw SHALL use the hostname `orinclaw` (or `orinclaw-<room>` for multi-unit deployments) as the system hostname set at first boot.
2. THE OrinClaw SHALL advertise itself via mDNS so that `http://orinclaw.local` resolves to the device web UI on the local network.
3. THE OrinClaw OTA system SHALL use channel identifiers `orinclaw-stable` and `orinclaw-beta` for P2 manifest distribution.
4. THE OrinClaw stack orchestration SHALL be defined in `orinclaw-deploy/docker-compose.yml` with the Compose project name `orinclaw`.
5. WHEN multiple OrinClaw units are present on the same LAN, THE OrinClaw SHALL append a unique serial suffix to the mDNS hostname to avoid collisions.

---

### Requirement 2: User Promises — Always Ready

**User Story:** As a user, I want the device to be ready to respond immediately after power-on, so that I never experience boot-lag surprises when I try to use it.

#### Acceptance Criteria

1. WHEN the device is powered on, THE WakeService SHALL be in a listening state and able to detect the wake word within 30 seconds of power application.
2. WHEN the wake word is detected, THE DeviceService SHALL produce an audible chime and LED acknowledgement within 200 ms of wake detection.
3. THE OrinClaw SHALL complete the full boot sequence from power-on to wake-word-ready state in under 30 seconds under normal operating conditions.
4. IF the boot sequence exceeds 60 seconds, THEN THE OrinClaw SHALL illuminate the Error LED pattern and log a boot-time diagnostic entry.


---

### Requirement 3: User Promises — Instant Feedback

**User Story:** As a user, I want immediate audible and visual acknowledgement when the device hears me, so that I know it is responding and not ignoring me.

#### Acceptance Criteria

1. WHEN the WakeService detects the wake word, THE DeviceService SHALL transition the LED ring to the Listening state (blue pulse) within 150 ms.
2. WHEN the WakeService detects the wake word, THE OrinClaw SHALL play a short confirmation chime or tone within 250 ms on the speaker output path (MAY be implemented via TTSService, a dedicated audio asset, or DeviceService-coordinated playback — implementation SHALL meet the latency budget).
3. WHILE the STTService is processing speech, THE DeviceService SHALL maintain the Listening LED state continuously without interruption.
4. WHEN the LLMService begins processing, THE DeviceService SHALL transition the LED ring to the Processing state (amber spin) within 100 ms of the STT result being delivered.

---

### Requirement 4: User Promises — Offline-First Operation

**User Story:** As a user, I want the core voice assistant and local automations to work without internet access, so that my assistant is reliable regardless of WAN availability.

#### Acceptance Criteria

1. WHILE WAN connectivity is unavailable, THE OrinClaw SHALL complete the full voice pipeline (wake word → STT → local LLM → TTS) using only on-device resources.
2. WHILE WAN connectivity is unavailable, THE Gateway web UI SHALL remain reachable on the LAN at `http://orinclaw.local`.
3. WHILE WAN connectivity is unavailable, THE SkillsRuntime SHALL execute local skills including MQTT/Home Assistant control and browser automation to LAN targets.
4. WHEN an OTA update is requested and WAN is unavailable, THE OrinClaw SHALL display an explicit "update unavailable" status in the web UI and via the LED state — no silent failure.
5. WHEN WAN connectivity is restored after an outage, THE OrinClaw SHALL resume normal operation including OTA checks within one user turn without requiring a restart.
6. WHEN a soak test of 24 hours or more is run with WAN disconnected, THE OrinClaw SHALL show no unbounded memory growth and SHALL NOT trigger an OOM condition in the voice pipeline.


---

### Requirement 5: User Promises — Privacy by Default

**User Story:** As a user, I want my voice and transcribed text to stay on the device by default, so that I can trust the assistant with private conversations.

#### Acceptance Criteria

1. THE OrinClaw SHALL NOT transmit microphone audio or transcribed text to any external server unless a cloud connector has been explicitly enabled by the user.
2. THE OrinClaw SHALL NOT enable any vendor cloud logging by default; any future diagnostics upload SHALL require explicit opt-in and SHALL be documented in the user-facing privacy policy.
3. WHEN a factory reset is performed, THE OrinClaw SHALL wipe all cloud connector keys and return to offline-first defaults with no cloud connectors active.
4. WHERE a BYOK cloud connector is enabled, THE OrinClaw SHALL display a visible indicator in the web UI identifying which replies used a cloud model.
5. IF a BYOK cloud API is unreachable, THEN THE OrinClaw SHALL fall back to the local LLM or present a clear spoken and UI error — no indefinite hang.

---

### Requirement 6: User Promises — Predictable Performance

**User Story:** As a user, I want stable response times under load and graceful degradation when the device is hot or memory-constrained, so that the assistant feels reliable.

#### Acceptance Criteria

1. THE OrinClaw SHALL maintain end-to-end voice response latency at p95 below 2.5 seconds under normal operating conditions.
2. WHILE the SoC junction temperature exceeds 75°C, THE OrinClaw SHALL announce "Performance mode reduced" via TTS and display the thermal LED state.
3. WHILE the SoC junction temperature exceeds 80°C, THE OrinClaw SHALL reduce GPU clock frequency to prevent thermal damage and SHALL log the throttle event.
4. WHILE available unified memory headroom falls below 500 MB, THE OrinClaw SHALL reject new LLM context extensions and SHALL log a memory pressure warning.
5. THE OrinClaw SHALL log p50, p95, and p99 latency for each pipeline stage (wake ack, STT first partial, TTFT, TTS first chunk, end-to-end) to `/data/logs` for benchmark reporting.


---

### Requirement 7: Hardware Platform — Compute and Storage

**User Story:** As a hardware engineer, I want the compute and storage platform to be clearly specified, so that bring-up, software development, and production PCB design are unambiguous.

#### Acceptance Criteria

1. THE OrinClaw bring-up platform SHALL use a Jetson Orin Nano 8GB module (dev kit for Phase 1, custom carrier PCB for production).
2. THE OrinClaw SHALL boot from and use a 512 GB NVMe M.2 2280 SSD as the primary storage for rootfs, models, logs, and OTA staging.
3. THE OrinClaw SHALL NOT use eMMC as the primary system storage in this program.
4. THE OrinClaw SHALL provide Gigabit Ethernet connectivity directly from the Jetson carrier.
5. THE OrinClaw SHALL use JetPack 5.1.2 for Phase 1 bring-up, JetPack 6.2.1 for the production-aligned stack, and SHALL adopt JetPack 7.x only when NVIDIA officially supports the target module and carrier.
6. WHEN a JetPack version upgrade is performed, THE OrinClaw build system SHALL rebuild all TensorRT/ONNX engines, the ESP-Hosted host driver, and all container base images against the new L4T version before deployment.
7. THE OrinClaw SHALL record the active JetPack version in release notes, factory configuration, and the `/health` endpoint.

---

### Requirement 8: Hardware Platform — Custom PCB

**User Story:** As a hardware engineer, I want the final product to use a purpose-built custom carrier PCB, so that unnecessary components are eliminated and product-grade reliability, cost, and certification readiness are achieved.

#### Acceptance Criteria

1. THE OrinClaw production hardware SHALL use a custom carrier PCB with a Jetson compute module socket rather than the developer kit PCB.
2. THE OrinClaw custom PCB SHALL include only circuits required for the defined UX: module socket, USB-C PD controller, Ethernet PHY, M.2 NVMe socket, **HDMI output for an external monitor** (bring-up, debug, and optional local UI — not an integrated on-device main display), I2S audio codec, mic array interface, speaker amplifier, ESP32-C6 with SPI for ESP-Hosted, LED ring drivers, hardware buttons, mute switch, **two externally user-accessible USB Type-A downstream host interfaces** and **one externally user-accessible USB Type-C downstream host interface** for peripherals (per `OrinClaw-Hardware-Design-Requirements.md` §2.10), ESD protection, and factory test points.
3. THE OrinClaw USB Type-C **host** receptacle SHALL support **USB 2.0 and/or USB 3.x SuperSpeed** only; THE OrinClaw SHALL **not** implement **Thunderbolt 3/4** or **USB4** tunneling on that user-facing connector unless the program explicitly revises hardware scope. THE user documentation SHALL distinguish **USB-C PD power input** from the **USB-C data host** port where both use Type-C (per §2.10).
4. THE OrinClaw custom PCB BOM SHALL assign a manufacturer part number (MPN) to every component symbol in the schematic before PCB layout freeze.
5. THE OrinClaw custom PCB BOM SHALL mark any unpopulated component with a DNP flag and a variant identifier.
6. THE OrinClaw hardware deliverables SHALL include schematics, PCB layout source and release Gerbers, stackup, BOM, assembly drawing, pick-and-place file, and a DFM/DFA checklist.
7. WHEN a PCB revision is released, THE OrinClaw BOM file SHALL be versioned to match the PCB revision (e.g. `BOM_OrinClaw_R1A.csv` matches `PCB_R1A` Gerbers).


---

### Requirement 9: Hardware Platform — Connectivity

**User Story:** As a user, I want reliable WiFi, Bluetooth, and smart-home radio connectivity, so that the device integrates seamlessly into my home network and IoT ecosystem.

#### Acceptance Criteria

1. THE OrinClaw SHALL provide WiFi and Bluetooth/BLE connectivity to the Linux host via an ESP32-C6 module connected over SPI using the ESP-Hosted-NG driver.
2. THE OrinClaw ESP-Hosted-NG integration SHALL expose a standard Linux wireless interface (cfg80211 / wpa_supplicant / NetworkManager) on the Jetson host.
3. THE OrinClaw SHALL support Thread, Zigbee, and Matter protocols on the ESP32-C6 co-processor alongside the ESP-Hosted WiFi/BT stack.
4. WHEN Gigabit Ethernet is connected, THE OrinClaw SHALL prefer Ethernet for heavy traffic (OTA, large downloads) and SHALL maintain WiFi for IoT and convenience paths.
5. THE OrinClaw ESP32-C6 firmware SHALL implement auto-reconnect with exponential backoff and keep-alive pings to the access point.
6. THE OrinClaw SHALL expose WiFi RSSI and link state from the ESP-Hosted interface in the `/health` endpoint.
7. IF the ESP-Hosted SPI link drops, THEN THE OrinClaw SHALL surface the No Network LED state, log the event, and continue operating on Ethernet if available.

---

### Requirement 10: Hardware Platform — Physical UX Controls

**User Story:** As a user, I want physical controls for muting the microphone, triggering actions, and reading device state, so that I can interact with the device without a screen or phone.

#### Acceptance Criteria

1. THE OrinClaw SHALL include a hardware microphone mute switch that physically disconnects the microphone signal path when engaged.
2. WHEN the hardware mute switch is engaged, THE DeviceService SHALL illuminate the Muted LED state (red dim solid) within 100 ms.
3. THE OrinClaw SHALL include a single hardware action button that the user can press to trigger a configurable action (e.g. repeat last reply, cancel, or wake).
4. THE OrinClaw SHALL include an LED ring as the primary out-of-band status signal visible from across the room.
5. THE OrinClaw LED ring SHALL implement all states defined in the LED State Table (Requirement 20) as the single source of truth for firmware and DeviceService.


---

### Requirement 11: Audio Subsystem

**User Story:** As a user, I want clear, room-filling voice output and accurate far-field microphone capture, so that I can interact naturally from anywhere in the room.

#### Acceptance Criteria

1. THE OrinClaw speaker subsystem SHALL use a full-range driver or woofer-plus-tweeter configuration with 4 Ω or 8 Ω impedance and 3–10 W RMS power handling.
2. THE OrinClaw speaker SHALL achieve a maximum SPL of 80–85 dB at 1 m with clear voice-band reproduction from 300 Hz to 3.4 kHz.
3. THE OrinClaw SHALL use a Class-D amplifier driven from I2S with a low-latency path from TTSService to the speaker.
4. THE OrinClaw SHALL use a 2-microphone array with far-field DSP connected via I2S.
5. THE OrinClaw SHALL implement software acoustic echo cancellation (AEC) in the audio pipeline to prevent speaker output from corrupting microphone capture.
6. THE OrinClaw PCB layout SHALL place the microphone array with sufficient physical separation and mechanical isolation from the speaker to support effective AEC.
7. WHEN the TTSService produces the first audio chunk, THE OrinClaw SHALL begin speaker playback within 300 ms of receiving that chunk.
8. THE OrinClaw speaker enclosure SHALL be sealed or ported with internal bracing to prevent buzzes and rattles at maximum SPL.

---

### Requirement 12: Thermal Management

**User Story:** As a hardware engineer, I want the thermal design to sustain full inference load without throttling, so that the user experience remains consistent under prolonged use.

#### Acceptance Criteria

1. THE OrinClaw thermal design SHALL maintain SoC junction temperature below 80°C under sustained 15 W inference load with ambient temperature up to 35°C.
2. THE OrinClaw heatsink SHALL provide sufficient fin area and airflow (passive for 7–10 W, active 5 V fan for 15–25 W) to achieve a junction-to-ambient thermal resistance below 3°C/W with fan.
3. THE OrinClaw SHALL use quality thermal interface material (TIM) between the SoC and heatsink with even contact pressure per NVIDIA mounting specifications.
4. THE OrinClaw enclosure SHALL provide a clear airflow path (inlet low, outlet high or side-to-side) with no hot spots near the PCB or optional battery.
5. WHEN a 30-minute inference stress test is run, THE OrinClaw SHALL show no CPU or GPU clock reduction (throttle) and SHALL maintain junction temperature within the target range.
6. IF the SoC junction temperature exceeds 80°C during normal operation, THEN THE OrinClaw SHALL log the thermal event and reduce GPU clock frequency to protect the hardware.


---

### Requirement 13: Battery and Power (Optional)

**User Story:** As a user, I want the device to continue operating during a power outage and optionally run on battery, so that my assistant is available even when mains power is interrupted.

#### Acceptance Criteria

1. WHERE the optional battery is included, THE OrinClaw SHALL accept USB-C PD input at 15–25 W and charge the battery using a CC/CV profile matched to the cell count.
2. WHERE the optional battery is included, WHEN AC power is present, THE OrinClaw SHALL power the device from USB-C PD and charge the battery simultaneously.
3. WHERE the optional battery is included, WHEN AC power is removed, THE OrinClaw SHALL switch to battery power with no brownout or reset of the Jetson module.
4. WHERE the optional battery is included, THE OrinClaw BMS SHALL protect against overcharge, overdischarge, overcurrent, short-circuit, and thermal runaway.
5. WHERE the optional battery is included, THE OrinClaw SHALL expose battery state-of-charge and charging status in the `/health` endpoint and optionally on the LED ring.
6. WHERE the optional battery is included and the device is running on battery, THE OrinClaw SHALL optionally enter a low-power mode that reduces GPU maximum frequency and dims the LED ring to extend runtime — configurable from the web UI.
7. WHERE the optional battery is included, THE OrinClaw battery design SHALL comply with UN38.3 and applicable regional safety regulations for lithium cells.

---

### Requirement 14: Software Stack and Data Layout

**User Story:** As a developer, I want the software stack to be clearly structured with defined data paths and service boundaries, so that deployment, updates, and debugging are straightforward.

#### Acceptance Criteria

1. THE OrinClaw SHALL use OpenClaw as the orchestrator providing the Gateway, multi-channel inbox, skills runtime, and browser automation.
2. THE OrinClaw Gateway SHALL run on Node.js version 22 or later and SHALL be started with `openclaw gateway --port 18789`.
3. THE OrinClaw SHALL install the Gateway as a persistent daemon using `openclaw onboard --install-daemon`.
4. THE OrinClaw SHALL organize persistent data under `/data/models`, `/data/skills`, `/data/logs`, and `/data/config` with `/data/config` permissions set to 0700.
5. THE OrinClaw SHALL package all application services (Gateway, STT, LLM, TTS) as Docker Compose services defined in `orinclaw-deploy/docker-compose.yml` with pinned image digests.
6. THE OrinClaw L4T host SHALL disable unused desktop and JetPack daemons to minimize boot time and memory footprint.
7. THE OrinClaw SHALL implement log rotation under `/data/logs` to prevent unbounded NVMe write amplification.
8. THE OrinClaw SHALL expose a `/health` endpoint reporting the status of Gateway, STTService, LLMService, TTSService, WiFi interface, ESP32-C6 side-channel, and optional battery.
9. THE OrinClaw program MAY support an **optional developer expansion** consisting of a **USB-connected FT232H-class** module (not mounted on the main carrier) for **I2C, SPI, UART, and GPIO** access from OpenClaw skills; WHEN this path is offered, THE OrinClaw SHALL document **kernel driver support** (`ftdi_sio` or equivalent), **`udev` device permissions**, optional **container device passthrough**, and a **reference skill** or README — **without** requiring a carrier PCB change solely for FT232H integration (per `OrinClaw-Hardware-Design-Requirements.md` §2.9).


---

### Requirement 15: Inference Optimization

**User Story:** As a developer, I want the inference pipeline to be optimized for low latency and efficient memory use on Jetson's unified memory architecture, so that the voice assistant feels fast and does not OOM under real workloads.

#### Acceptance Criteria

1. THE OrinClaw SHALL keep the resident LLM model loaded in unified memory at all times after first load — no repeated loads during normal inference.
2. THE OrinClaw SHALL run exactly one resident LLM at a time and SHALL NOT keep multiple large model weight sets mapped simultaneously.
3. THE OrinClaw LLMService SHALL enforce explicit maximum context length and KV cache size limits to prevent unbounded unified memory growth.
4. THE OrinClaw SHALL use pre-built TensorRT or ONNX Runtime engines (built at factory or delivered via P3 OTA) rather than rebuilding engines on every first boot in the field.
5. THE OrinClaw SHALL apply INT4 or INT8 quantization for the LLM and FP16 or INT8 for STT and TTS to reduce model footprint and improve inference latency.
6. THE OrinClaw STTService SHALL stream partial transcription results to the Gateway before the user finishes speaking.
7. THE OrinClaw LLMService SHALL stream output tokens to the Gateway as they are generated rather than waiting for the full response.
8. THE OrinClaw TTSService SHALL begin producing audio chunks from the first LLM token stream and SHALL NOT wait for the complete LLM response before starting synthesis.
9. THE OrinClaw audio pipeline SHALL use zero-copy shared memory buffers between the microphone capture and STTService where the platform supports it.
10. THE OrinClaw SHALL isolate dedicated CPU cores for the audio pipeline and Gateway orchestrator to provide deterministic scheduling.
11. WHEN profiling the full stack (STT + LLM + TTS + Gateway) with `tegrastats`, THE OrinClaw SHALL maintain at least 500 MB of free unified memory headroom under peak load.

---

### Requirement 16: Latency Budget

**User Story:** As a user, I want the assistant to respond faster than cloud-dependent smart speakers, so that the interaction feels natural and immediate.

#### Acceptance Criteria

1. WHEN the wake word is detected, THE DeviceService SHALL produce the LED and chime acknowledgement within 200 ms.
2. WHEN the user begins speaking after wake, THE STTService SHALL deliver the first partial transcription result within 500 ms of speech onset.
3. WHEN the user finishes speaking, THE LLMService SHALL produce the first output token within 1 second (TTFT < 1 s).
4. WHEN the LLMService produces the first output token, THE TTSService SHALL deliver the first audio chunk to the speaker within 300 ms.
5. THE OrinClaw SHALL achieve end-to-end latency (user stops speaking to first audible TTS output) at p95 below 2.5 seconds under normal operating conditions.
6. THE OrinClaw benchmark report SHALL record p50, p95, and p99 values for each latency stage listed above on pinned hardware and JetPack version.


---

### Requirement 17: Memory Budget

**User Story:** As a developer, I want the memory allocation across all services to be planned and enforced, so that the 8 GB unified memory pool is never exhausted under real workloads.

#### Acceptance Criteria

1. THE OrinClaw memory budget SHALL allocate no more than 1.5 GB to L4T and baseline daemons after L4T optimization.
2. THE OrinClaw memory budget SHALL allocate no more than 1 GB to Docker and the OpenClaw Gateway combined.
3. THE OrinClaw memory budget SHALL allocate no more than 200 MB to the WakeService (upper bound; typical resident wake engines are often ≪ 200 MB — validate on target).
4. THE OrinClaw memory budget SHALL allocate no more than 1.5 GB to the STTService including GPU activations.
5. THE OrinClaw memory budget SHALL allocate no more than 4 GB to the single resident LLM including KV cache.
6. THE OrinClaw memory budget SHALL allocate no more than 600 MB to the TTSService.
7. THE OrinClaw memory budget SHALL allocate no more than 300 MB to ESP-Hosted and small daemons.
8. THE OrinClaw SHALL maintain at least 500 MB of free unified memory headroom at all times during normal operation.
9. WHEN the full stack is running, THE OrinClaw SHALL be validated against the memory budget using `tegrastats` and `/proc/meminfo` measurements — not datasheet figures alone.

---

### Requirement 18: Recommended Default Models

**User Story:** As a developer, I want a defined default model stack that fits the 8 GB unified memory budget and meets latency targets, so that the product ships with a validated baseline configuration.

#### Acceptance Criteria

1. THE OrinClaw default LLM SHALL be one of: Phi-3 mini (Q4), Qwen2.5-7B (Q4), or TinyLlama — selected based on latency and quality validation under full unified-memory load.
2. THE OrinClaw default STTService SHALL use Whisper small or a distilled equivalent with streaming support, exported to ONNX or TensorRT for GPU acceleration.
3. THE OrinClaw default TTSService SHALL use Piper or Coqui TTS with first-audio-chunk latency validated against the 300 ms budget.
4. THE OrinClaw model selection SHALL be validated under full stack load (STT + LLM + TTS + OS + Docker running simultaneously) — a model that fits in isolation is not sufficient.
5. THE OrinClaw SHALL document the model format (ONNX/GGUF/TensorRT engine), source or build procedure, and storage path in the project repository.


---

### Requirement 19: UX Design — Setup and Onboarding

**User Story:** As a new user, I want to set up the device in under 5 minutes without a terminal, so that the product is accessible to non-technical users.

#### Acceptance Criteria

1. THE OrinClaw SHALL provide a setup flow accessible via captive portal or local web UI at `http://orinclaw.local` that completes WiFi onboarding and basic configuration in under 5 minutes.
2. THE OrinClaw setup flow SHALL include a WiFi onboarding step and an optional Ethernet-first flow for users who prefer wired connectivity.
3. THE OrinClaw setup flow SHALL include a voice calibration wizard that configures microphone gain, noise profile, and wake word sensitivity.
4. THE OrinClaw web UI SHALL provide a one-click update button and a one-click rollback button for P2 OTA operations.
5. THE OrinClaw web UI SHALL be accessible from any browser on the LAN without installing additional software.
6. WHEN the device is in factory-default state, THE OrinClaw SHALL present the setup flow automatically on first access to `http://orinclaw.local`.

---

### Requirement 20: LED State Table (Single Source of Truth)

**User Story:** As a user, I want the LED ring to communicate device state clearly and consistently, so that I always know what the device is doing without needing a screen or phone.

#### Acceptance Criteria

1. THE DeviceService SHALL implement the following LED states as the single source of truth for both firmware and software:
   - Idle: Off or dim solid — device ready, no activity (optional neutral/warm white at low brightness if CMF requires).
   - Listening: Blue, pulse — wake word detected, capturing speech.
   - Processing: Amber, spin — STT/LLM/TTS pipeline running.
   - Success: Green, 1 short flash — command completed successfully.
   - Error: Red, 2 blinks (or blink count equals error code) — failure occurred.
   - Updating: Amber and blue alternating — OTA update in progress.
   - Muted: Red, dim solid — hardware microphone mute switch engaged.
   - Weak WiFi: Amber, slow pulse — low RSSI or reconnecting.
   - No Network: Red, slow blink — WiFi down or no network available.
2. THE DeviceService SHALL transition to the correct LED state within 100 ms of the triggering event.
3. THE DeviceService LED state table SHALL be documented in the user guide and SHALL match the firmware implementation exactly — no undocumented states.
4. WHEN two state conditions apply simultaneously, THE DeviceService SHALL prioritize states in this order: Muted > Error > Updating > No Network > Weak WiFi > Processing > Listening > Success > Idle.
5. THE OrinClaw LED patterns SHALL use distinct combinations of color and animation so that states are distinguishable at 3 m distance in normal room lighting.


---

### Requirement 21: UX Design — Interaction and Accessibility

**User Story:** As a user, I want natural interaction patterns including barge-in, repeat, and accessibility options, so that the assistant is usable for a wide range of users and situations.

#### Acceptance Criteria

1. THE OrinClaw SHALL support barge-in: WHEN the user speaks while TTS audio is playing, THE OrinClaw SHALL stop playback and begin processing the new utterance within 300 ms.
2. THE OrinClaw SHALL support a "repeat" command or action button press that replays the last TTS response.
3. WHERE the accessibility option is enabled, THE TTSService SHALL produce speech at a reduced rate and higher clarity profile configurable from the web UI.
4. THE OrinClaw SHALL provide hardware microphone mute via a physical switch that cuts the microphone signal path independently of software.
5. THE OrinClaw web UI SHALL provide per-skill permission prompts on first use of each skill.
6. THE OrinClaw web UI SHALL provide UI toggles to enable or disable each cloud connector independently.

---

### Requirement 22: UX Design — Reliability and Failure Modes

**User Story:** As a user, I want the device to handle failures gracefully with clear feedback, so that I am never left confused about what went wrong.

#### Acceptance Criteria

1. WHEN a component (STTService, LLMService, or TTSService) times out or crashes, THE OrinClaw SHALL retry the operation once, then illuminate the Error LED state and optionally speak "I'm having trouble."
2. WHEN the ESP-Hosted SPI link drops, THE OrinClaw SHALL surface the No Network LED state, log the event, and continue operating on Ethernet if available without requiring a restart.
3. IF the optional battery is low, THEN THE OrinClaw SHALL produce a voice warning and illuminate a low-battery LED pattern before the device shuts down.
4. WHEN an OTA update fails the health gate, THE OrinClaw SHALL automatically roll back to the previous compose manifest and illuminate the Error LED state.
5. WHILE the device is operating in offline mode (WAN unavailable), THE OrinClaw SHALL NOT display error messages to the user — it SHALL operate silently in local mode and queue any cloud actions until WAN is restored.
6. WHEN the SoC is thermally throttled, THE OrinClaw SHALL announce "Performance mode reduced" via TTS and provide an estimated recovery time if available.


---

### Requirement 23: Smart Home and Compatibility

**User Story:** As a user, I want the device to control my smart-home devices using standard protocols, so that it works with my existing ecosystem without vendor lock-in.

#### Acceptance Criteria

1. THE OrinClaw SHALL support the Matter protocol as a controller and/or bridge, enabling commissioning and control of Matter devices over Thread and WiFi via the ESP32-C6.
2. THE OrinClaw SHALL integrate with Home Assistant via REST API and MQTT for device control and automation.
3. THE OrinClaw SHALL support Zigbee2MQTT as an optional integration for Zigbee device control.
4. THE OrinClaw SHALL support Telegram and WhatsApp messaging via opt-in connectors configurable from the web UI.
5. THE OrinClaw SHALL provide browser automation via a Playwright-based tool running in a container.
6. THE OrinClaw SHALL use ONNX as the model interchange format and SHALL build TensorRT engines per device for production inference.
7. WHERE an optional separate ESP32 smart-home controller satellite is used, THE OrinClaw SHALL communicate with it over LAN-local MQTT, HTTP, or WebSocket — no cloud intermediary required.
8. THE OrinClaw SHALL expose Matter commissioning and device control in the web UI and via voice commands.

---

### Requirement 24: BYOK Cloud Connectors (Optional)

**User Story:** As a power user, I want to optionally connect my own cloud LLM API keys, so that I can access larger models for complex tasks while maintaining local-first defaults.

#### Acceptance Criteria

1. THE OrinClaw SHALL store BYOK API keys only under `/data/config` with file permissions set to 0600.
2. THE OrinClaw SHALL NOT bake API keys into container images or commit them to version control.
3. THE OrinClaw SHALL support at least one of the following routing policies, configurable from the web UI: per-session explicit selection, local-default with cloud on request, or heuristic split.
4. WHERE a BYOK cloud connector is enabled, THE OrinClaw web UI SHALL display a visible indicator identifying which replies used a cloud model.
5. THE OrinClaw web UI SHALL provide a key rotation interface for updating BYOK API keys without restarting the stack.
6. WHEN a cloud connector is disabled from the web UI, THE OrinClaw SHALL immediately stop all outbound API calls to that cloud provider.
7. IF no BYOK keys are configured, THEN THE OrinClaw SHALL operate in strictly local mode with no outbound LLM API calls.
8. IF a configured BYOK cloud API is unreachable, THEN THE OrinClaw SHALL fall back to the local LLM or present a clear spoken and UI error within 5 seconds — no indefinite hang.


---

### Requirement 25: Security — Boot Integrity and Rootfs

**User Story:** As a security engineer, I want the boot chain to be verified and the rootfs to be protected, so that only signed, authorized software runs on the device.

#### Acceptance Criteria

1. THE OrinClaw production firmware SHALL use NVIDIA Jetson Secure Boot to verify the boot firmware, kernel, and device tree chain before execution.
2. THE OrinClaw SHALL use A/B boot partitions so that a failed boot on the new slot automatically falls back to the previous known-good slot via boot counter.
3. THE OrinClaw rootfs strategy SHALL use either an immutable root with writable `/data` overlay or dm-verity on the system partition — the chosen strategy SHALL be documented and regression-tested against the Jetson GPU stack.
4. THE OrinClaw production image SHALL disable the UART console or gate it behind a factory-only image to prevent root shell access on exposed pins.
5. WHEN a P0 rootfs update is applied, THE OrinClaw SHALL verify the signed image before switching the active boot slot.

---

### Requirement 26: Security — Kernel and Host Hardening

**User Story:** As a security engineer, I want the kernel and host OS to be hardened against information leaks and privilege escalation, so that a compromised container or LAN attacker cannot easily pivot to the host.

#### Acceptance Criteria

1. THE OrinClaw host SHALL apply the following sysctl settings and SHALL validate that they do not break audio, Docker, or GPU inference: `kernel.kptr_restrict=2`, `kernel.dmesg_restrict=1`, `net.ipv4.conf.all.rp_filter=1`, `net.ipv4.conf.default.rp_filter=1`, `fs.protected_symlinks=1`, `fs.protected_hardlinks=1`.
2. THE OrinClaw host SHALL disable IP forwarding (`net.ipv4.ip_forward=0`) unless explicitly required for routing.
3. THE OrinClaw host SHALL set `kernel.yama.ptrace_scope` to 1 or higher on production images.
4. THE OrinClaw host SHALL use SSH with ed25519 keys only, `PermitRootLogin no`, and no password authentication.
5. THE OrinClaw host SHALL apply a default-deny firewall (nftables or ufw) allowing only required ports.
6. THE OrinClaw host SHALL use systemd-timesyncd or chrony for time synchronization to ensure TLS validity and log correlation.
7. WHEN any sysctl or LSM hardening change is applied, THE OrinClaw CI SHALL run regression tests confirming that ESP-Hosted WiFi, Docker GPU inference, and audio pipeline remain functional.


---

### Requirement 27: Security — Full-Disk Encryption and Physical Theft

**User Story:** As a security engineer, I want the 512 GB NVMe to be encrypted with keys stored off the drive, so that a stolen drive cannot be mounted and read by an attacker.

#### Acceptance Criteria

1. THE OrinClaw production units SHALL encrypt the 512 GB NVMe using LUKS/dm-crypt (full-disk) or a qualified TCG Opal SED locking configuration — "no encryption" is not an acceptable ship configuration.
2. THE OrinClaw SHALL NOT store the LUKS volume key or any equivalent unwrap secret on an unencrypted partition of the NVMe drive.
3. THE OrinClaw SHALL unwrap the volume key from TPM2/fTPM/secure element, a documented user passphrase flow, or TCG Opal host authentication — the chosen strategy SHALL be documented and tested end-to-end.
4. WHEN a factory reset is performed, THE OrinClaw SHALL execute a cryptographic wipe and NVMe Sanitize (or vendor-equivalent) command on user data namespaces.
5. THE OrinClaw ship gate SHALL include an offline drive extraction test: remove the 512 GB NVMe (or take a full `dd` image), attach to a Linux host, and verify that no plaintext data is mountable without the wrapping secret.
6. THE OrinClaw custom PCB SHALL place the M.2 slot under an RF shield with security screws and a tamper-evident label to raise the cost of physical extraction.
7. THE OrinClaw user-facing documentation SHALL state that a lost or stolen device with weak or no encryption exposes data stored on disk.

---

### Requirement 28: Security — Container Isolation

**User Story:** As a security engineer, I want inference and skill containers to be isolated from the host, so that a compromised container cannot escalate to host privileges or read secrets.

#### Acceptance Criteria

1. THE OrinClaw Docker Compose services SHALL use read-only root filesystems with tmpfs mounts for writable scratch space where possible.
2. THE OrinClaw Docker Compose services SHALL drop all Linux capabilities and add back only those explicitly required (e.g. `CAP_SYS_NICE` if needed for scheduling).
3. THE OrinClaw Docker Compose services SHALL set `no-new-privileges: true` in security options.
4. THE OrinClaw Docker Compose services SHALL apply the default Docker seccomp profile or a stricter custom profile tested against Playwright and GPU hooks.
5. THE OrinClaw Docker Compose services SHALL NOT use `privileged: true` except for documented one-offs that are explicitly justified and reviewed.
6. THE OrinClaw SHALL inject secrets into containers via runtime environment variables or tmpfs mounts from host-staged files — never baked into image layers.
7. THE OrinClaw OpenClaw Gateway SHALL bind to loopback or authenticated interfaces — raw WebSocket SHALL NOT be exposed to `0.0.0.0` without authentication.


---

### Requirement 29: OTA Strategy — Layered Update Planes

**User Story:** As a developer, I want a layered OTA system that separates application updates from OS updates, so that the OpenClaw stack can be updated frequently without reflashing Linux.

#### Acceptance Criteria

1. THE OrinClaw OTA system SHALL implement four update planes: P0 (JetPack/rootfs/kernel), P1 (host glue: nvidia-container-toolkit, systemd units, ESP-Hosted .ko and C6 firmware), P2 (Compose stack: Gateway, STT, LLM, TTS), and P3 (data plane: models, TensorRT engines, skill packs).
2. THE OrinClaw P2 update SHALL use a signed manifest declaring `target_jetpack`/`min_l4t`, per-service image digests (SHA256), compose revision, and manifest SHA256 — the device SHALL reject a P2 manifest whose `min_l4t` does not match the running L4T version.
3. THE OrinClaw P2 update pipeline SHALL: download and verify the manifest signature, pre-pull images by digest, stage the new compose file, perform an atomic switch, run a health gate (3–5 minute timeout), and commit or roll back automatically.
4. WHEN the P2 health gate fails, THE OrinClaw SHALL automatically restore the previous compose manifest and image digests and SHALL illuminate the Error LED state.
5. THE OrinClaw P0 update SHALL use A/B boot slots with a boot counter so that a failed boot on the new slot automatically falls back to the previous slot.
6. THE OrinClaw SHALL maintain the last-known-good P2 manifest and image digests on disk (N−1 release) to enable rollback without re-downloading.
7. THE OrinClaw OTA staging area SHALL enforce a size quota to limit NVMe write amplification and SHALL monitor NVMe SMART health.
8. THE OrinClaw SHALL offer `orinclaw-stable` and `orinclaw-beta` channels for P2 manifests; P0 updates SHALL use a separate conservative channel.
9. WHEN WAN is unavailable during an OTA check, THE OrinClaw SHALL display "update unavailable" in the web UI and via LED — no silent failure or indefinite retry loop.

---

### Requirement 30: OTA Strategy — Signed Artifacts and Integrity

**User Story:** As a security engineer, I want all OTA artifacts to be signed and integrity-verified before application, so that a supply-chain or network attacker cannot deliver malicious updates.

#### Acceptance Criteria

1. THE OrinClaw OTA system SHALL verify the publisher signature on every P2 manifest before pulling any container images.
2. THE OrinClaw OTA system SHALL pull container images by digest only (image@sha256) — `latest` tags SHALL NOT be used on production devices.
3. THE OrinClaw OTA system SHALL verify the SHA256 digest of each pulled image against the manifest before staging the new compose file.
4. THE OrinClaw P0 slot images SHALL be signed per NVIDIA Jetson Secure Boot requirements and SHALL be verified before the active boot slot is switched.
5. THE OrinClaw SHALL record the applied release ID, timestamp, and manifest SHA256 in `/data/config/update-state.json` after each successful P2 update.


---

### Requirement 31: Risk Mitigations — Unified Memory OOM (R1)

**User Story:** As a developer, I want hard memory caps and profiling gates to prevent OOM crashes, so that the voice pipeline remains stable under real workloads on the 8 GB unified memory platform.

#### Acceptance Criteria

1. THE OrinClaw SHALL run exactly one resident LLM at a time and SHALL NOT load a second large model while the first is mapped.
2. THE OrinClaw LLMService SHALL enforce a hard maximum context length and KV cache size that keeps total unified memory usage within the budget defined in Requirement 17.
3. THE OrinClaw SHALL be profiled with `tegrastats` under full stack load before each milestone release; the profiling results SHALL be recorded in the benchmark report.
4. IF unified memory headroom falls below 500 MB during operation, THEN THE OrinClaw SHALL log a memory pressure warning and SHALL reject new context extensions until headroom recovers.

---

### Requirement 32: Risk Mitigations — ESP-Hosted Fragility (R2)

**User Story:** As a developer, I want the ESP-Hosted integration to be pinned and tested against each JetPack version, so that WiFi does not break after a kernel or firmware update.

#### Acceptance Criteria

1. THE OrinClaw SHALL pin the ESP-Hosted host driver version and the ESP32-C6 slave firmware version together in the P1 manifest and SHALL document the rebuild procedure for each JetPack upgrade.
2. THE OrinClaw SPI layout on the custom PCB SHALL follow Espressif's length-matching and signal-integrity guidelines for ESP-Hosted.
3. WHEN Gigabit Ethernet is connected and the ESP-Hosted WiFi link drops, THE OrinClaw SHALL continue operating on Ethernet without requiring a restart.
4. WHEN a JetPack upgrade is applied, THE OrinClaw CI SHALL rebuild the ESP-Hosted host driver for the new L4T kernel and run a WiFi connectivity regression test before releasing the update.

---

### Requirement 33: Risk Mitigations — NVMe Endurance (R4)

**User Story:** As a developer, I want NVMe write amplification to be minimized and monitored, so that the 512 GB drive lasts the product lifetime without premature wear-out.

#### Acceptance Criteria

1. THE OrinClaw SHALL rotate logs under `/data/logs` with a maximum total size cap to prevent unbounded NVMe writes.
2. THE OrinClaw OTA staging area SHALL enforce a maximum size quota and SHALL clean up staging artifacts after a successful or failed update.
3. THE OrinClaw SHALL minimize swap usage on the NVMe in production; swap SHALL NOT be used as a substitute for proper memory budgeting.
4. THE OrinClaw SHALL monitor NVMe SMART health metrics and SHALL expose drive health status in the `/health` endpoint.
5. THE OrinClaw SHALL use an endurance-rated 512 GB M.2 2280 NVMe SSD specified for the expected write workload.

---

### Requirement 34: Risk Mitigations — RF Coexistence (R5)

**User Story:** As a hardware engineer, I want WiFi, Thread, Zigbee, and Matter to coexist reliably on the ESP32-C6, so that enabling smart-home radios does not degrade WiFi performance or cause certification failures.

#### Acceptance Criteria

1. THE OrinClaw ESP32-C6 firmware SHALL implement coexistence between ESP-Hosted (WiFi/BT) and Thread/Zigbee/Matter stacks per Espressif's documented coexistence guidelines.
2. THE OrinClaw custom PCB antenna layout SHALL follow the ESP32-C6 module vendor's keepout and routing guidelines to minimize RF desense.
3. THE OrinClaw SHALL perform coexistence testing (WiFi throughput + Thread/Zigbee/Matter operation simultaneously) before PCB tape-out and before each firmware release.
4. IF RF coexistence testing reveals unacceptable degradation, THEN THE OrinClaw SHALL implement firmware feature flags or a combined firmware strategy to resolve the conflict before shipping.


---

### Requirement 35: Risk Mitigations — OTA Brick Prevention (R10)

**User Story:** As a developer, I want the OTA system to be resilient to failures, so that a bad update never permanently bricks a device in the field.

#### Acceptance Criteria

1. THE OrinClaw P0 OTA SHALL use A/B boot slots with boot counters so that a failed boot on the new slot automatically falls back to the previous slot without user intervention.
2. THE OrinClaw P2 OTA SHALL automatically roll back to the N−1 manifest if the health gate fails within the timeout window.
3. THE OrinClaw SHALL document a UART and USB recovery procedure for restoring a device that cannot boot from either A/B slot.
4. THE OrinClaw SHALL NOT promote a new P0 or P2 release to the stable channel until it has passed the health gate on reference hardware.
5. WHEN an OTA update is in progress, THE DeviceService SHALL illuminate the Updating LED state and THE OrinClaw web UI SHALL warn the user not to power-cycle the device.

---

### Requirement 36: Risk Mitigations — Physical Theft and NVMe Removal (R14)

**User Story:** As a security engineer, I want the device to be hardened against physical theft and drive extraction, so that a stolen NVMe cannot expose user data.

#### Acceptance Criteria

1. THE OrinClaw SHALL implement mandatory FDE on the 512 GB NVMe per Requirement 27 — this is a non-negotiable ship gate.
2. THE OrinClaw custom PCB SHALL place the M.2 slot under an RF shield secured with security screws and covered with a tamper-evident label.
3. THE OrinClaw SHALL NOT expose production UART or JTAG debug interfaces on accessible pins of the shipping product.
4. THE OrinClaw ship gate SHALL include the offline drive extraction QA test defined in Requirement 27 acceptance criterion 5.
5. THE OrinClaw factory reset procedure SHALL execute NVMe Sanitize (or vendor-equivalent) and SHALL verify via `strings` on a raw image that only high-entropy (encrypted) data remains.

---

### Requirement 37: Deliverables — Hardware Package

**User Story:** As a hardware engineer, I want a complete set of hardware deliverables, so that the custom PCB can be fabricated, assembled, and brought up by a contract manufacturer.

#### Acceptance Criteria

1. THE OrinClaw hardware deliverables SHALL include: NVIDIA Orin Nano / Orin NX **pinmux workbook** (`.xlsm`/`.xlsx`) as **source of truth** for carrier pinmux, JetPack-aligned, with **ball ↔ net ↔ spreadsheet row** cross-reference, ready for NVIDIA’s documented device-tree generation flow (see `Guide.md` §10 and [Job-Post-PCB-Contractor-1wk.md](../../../Job-Post-PCB-Contractor-1wk.md)).
2. THE OrinClaw hardware deliverables SHALL include: schematics, PCB layout source files, release Gerbers, stackup documentation, optimized BOM (no unnecessary parts, all necessary parts documented with MPN), assembly drawing, pick-and-place file, and a DFM/DFA checklist.
3. THE OrinClaw hardware deliverables SHALL include a module carrier bring-up document covering power sequencing, strap options, first-boot test procedure, and comparison to dev-kit behavior.
4. THE OrinClaw hardware deliverables SHALL include a thermal plan demonstrating no throttle in a 30-minute stress test on the shipping mechanical and PCB configuration.
5. THE OrinClaw hardware deliverables SHALL include an audio subsystem plan documenting mic array placement, echo cancellation strategy, and speaker integration in the enclosure and PCB.

---

### Requirement 38: Deliverables — Software and Security Package

**User Story:** As a developer, I want a complete set of software and security deliverables, so that the product can be deployed, updated, and audited in the field.

#### Acceptance Criteria

1. THE OrinClaw software deliverables SHALL include a documented P2 manifest schema, signing key practice, apply/rollback procedure, and health gate specification per the OTA strategy.
2. THE OrinClaw software deliverables SHALL include a versioned host security profile artifact listing firewall rules, sysctl values, SSH policy, and container hardening flags that passed the §8 verification checklist on the pinned JetPack version.
3. THE OrinClaw software deliverables SHALL include documented LUKS/Opal choice, key custody strategy, factory-reset sanitize procedure, and offline drive-extraction QA results.
4. THE OrinClaw software deliverables SHALL include a benchmark report recording p50/p95/p99 latency for each pipeline stage, sustained tokens/sec, STT WER, thermal measurements, and reliability test results on pinned hardware and JetPack version.
5. THE OrinClaw software deliverables SHALL include a demo script covering: lights control via MQTT, PDF summarization via local RAG, browser automation on a LAN test page, offline mode with WAN disconnected, BYOK cloud indicator with disable verification, and **optional** verified private remote access (e.g. Tailscale) without public port-forwarding.


---

### Requirement 39: Milestone Plan

**User Story:** As a project manager, I want a defined milestone sequence, so that hardware and software development proceed in a validated order with clear acceptance gates.

#### Acceptance Criteria

1. THE OrinClaw project SHALL complete Milestone 1 (Bring-up) before proceeding to Milestone 2: flash JetPack 5.1.2 to 512 GB NVMe, confirm `tegrastats` monitoring, and plan FDE validation on real hardware.
2. THE OrinClaw project SHALL complete Milestone 2 (Audio I/O) before proceeding to Milestone 3: verify microphone capture and speaker playback at ALSA/PipeWire level with echo control baseline.
3. THE OrinClaw project SHALL complete Milestone 3 (Wake Word) before proceeding to Milestone 4: demonstrate reliable far-field wake word detection and confirm hardware mute switch cuts the mic signal.
4. THE OrinClaw project SHALL complete Milestone 4 (STT) before proceeding to Milestone 5: demonstrate streaming transcription and record latency and WER measurements in the room environment.
5. THE OrinClaw project SHALL complete Milestone 5 (LLM) before proceeding to Milestone 6: demonstrate streaming local chat with tool calling and validate memory budget under full stack load.
6. THE OrinClaw project SHALL complete Milestone 6 (TTS) before proceeding to Milestone 7: demonstrate natural voice output with barge-in support and validate first-audio-chunk latency.
7. THE OrinClaw project SHALL complete Milestone 7 (Skills) before proceeding to Milestone 8: demonstrate MQTT/Home Assistant control, browser automation, and optional BYOK cloud connector with visible indicator.
8. THE OrinClaw project SHALL complete Milestone 8 (OTA + Hardening) before proceeding to Milestone 9: demonstrate signed P2 update, automatic rollback on health gate failure, and pass the attack-surface audit checklist.
9. THE OrinClaw project SHALL complete Milestone 9 (Custom PCB + Product Bring-up) as the **final core hardware** milestone before optional Milestone 10: complete carrier schematic and layout, fabricate and assemble the board, bring up the Jetson module on the custom carrier, run full regression against the dev-kit software baseline, and freeze the production BOM.
10. THE OrinClaw custom PCB layout and fabrication MAY proceed in parallel with Milestones 6–8 provided that module bring-up (Milestone 9) does not begin until a stable software baseline from **Milestone 8** (OTA + hardening) or later is available unless the program explicitly accepts higher integration risk.
11. THE OrinClaw project SHALL treat **Milestone 10 (Multi-room social assistant + media hub)** as **post–Milestone-9** work: optional room endpoints (Wi‑Fi preferred; BLE where appropriate), mobile as speaker/mic node, occupancy-aware activation, optional camera-assisted person-aware routing — only after single-room voice is stable; SHALL require explicit user/household **opt-in**, privacy disclosures, and secure pairing/transport per `Guide.md` §2 and §11.

---

### Requirement 40: Compliance and Certification Readiness

**User Story:** As a product manager, I want the hardware and software to be designed with certification requirements in mind, so that regulatory approval does not block the product launch.

#### Acceptance Criteria

1. THE OrinClaw custom PCB SHALL be designed with FCC and CE RF certification in mind for each intentional radiator (ESP32-C6 on-board and any optional satellite SKU).
2. THE OrinClaw SHALL use a certified ESP32-C6 module with a compliant antenna to simplify FCC/CE modular approval.
3. WHERE the optional battery is included, THE OrinClaw battery design SHALL comply with UN38.3 and IEC 62133 and applicable regional regulations before shipping.
4. WHERE Matter support is shipped, THE OrinClaw SHALL pursue CSA Matter certification for the controller/bridge role.
5. THE OrinClaw project SHALL engage an EMC test lab for pre-scan testing before final PCB tape-out to identify and resolve emissions issues early.
6. THE OrinClaw project SHALL document which certifications apply to each SKU and the planned timeline for engaging test labs in the project repository.

---

### Requirement 41: Remote Private Access (Optional)

**User Story:** As a remote user or admin, I want to reach my home OrinClaw over an encrypted private network without exposing router ports, so that I can use the Web UI or Gateway from office/travel safely.

#### Acceptance Criteria

1. WHERE Tailscale (or equivalent mesh VPN) is enabled, THE OrinClaw SHALL join only as a **private** tailnet/VPN node; THE OrinClaw SHALL NOT require public WAN port-forwarding for that access path.
2. WHERE remote access is used, THE OrinClaw SHALL enforce **authenticated** access to Gateway and Web UI (follow OpenClaw upstream binding/security guidance — no raw unauthenticated control plane on `0.0.0.0`).
3. THE OrinClaw deployment documentation SHALL describe tailnet ACL/tag policy, MFA/SSO expectations for human users, and device revoke/rotation on loss or role change.
4. THE OrinClaw SHALL verify remote access with a checklist: approved client reaches `orinclaw` via VPN hostname; unauthorized clients and revoked devices are denied; WAN port scan does not expose OrinClaw services directly.

---

### Requirement 42: V1 Product Scope, Target Users, and SKU Strategy

**User Story:** As a product owner, I want V1 scope and SKUs frozen so that engineering, certification, and support do not chase unbounded features before first ship.

#### Acceptance Criteria

1. THE OrinClaw V1 product SHALL target **single-room** voice assistant operation with reliable wake, STT, local LLM, and TTS as defined in `Guide.md` §2 *V1 launch scope*.
2. THE OrinClaw V1 SHALL remain **offline-first** for the core voice path with LAN web UI, MQTT/Home Assistant integration, and optional Matter-prefer behavior per compatibility requirements — not **cloud-dependent** as the default mode.
3. THE OrinClaw V1 SHALL **NOT** treat the following as ship gates: general-purpose tablet-style on-device UI; whole-home synchronized audio; person-aware routing by default; enterprise fleet/MDM; public multi-tenant hosting — per `Guide.md` §2 *Explicit non-goals for V1*.
4. THE OrinClaw program SHALL ship **one primary hero SKU** for V1 — **Core OrinClaw** (Jetson Orin Nano 8GB-class, 512 GB NVMe, voice + LED + LAN web UI, no main display) — and SHALL document optional SKUs (display, battery, satellite controller) as **post-V1** unless explicitly approved — per `Guide.md` §3 *Recommended SKU strategy*.
5. THE OrinClaw project SHALL maintain a **SKU matrix** document listing features, radios, and certification scope per variant before any variant is offered for sale.

---

### Requirement 43: Privacy and Data Lifecycle

**User Story:** As a user and regulator-facing product, I want data handling policies to be explicit and testable, not implied by “local AI.”

#### Acceptance Criteria

1. THE OrinClaw SHALL keep **default-local** handling: microphone audio, transcripts, and device actions SHALL NOT leave the device unless the user explicitly enables a cloud connector — consistent with Requirements 4–5 and `Guide.md` §6 *Privacy and data lifecycle*.
2. THE OrinClaw SHALL define and document **retention windows** for logs, optional transcripts, and diagnostics; retention SHALL be **minimal** for operation and support unless the user opts into extended history.
3. WHEN data crosses the **cloud boundary**, THE OrinClaw UI SHALL make that boundary **visible** (badge, settings, or last-reply indicator) and user-facing docs SHALL state **what** is sent.
4. WHERE **occupancy, camera, or person-aware** features exist (e.g. Milestone 10), THE OrinClaw SHALL require **opt-in**, clear labeling, reversibility, and updated privacy disclosures before enablement.
5. WHEN factory reset is executed, THE OrinClaw SHALL remove user accounts, pairings, WiFi credentials, cloud keys, and local history per the published policy — in addition to cryptographic wipe requirements in Requirement 27.

---

### Requirement 44: Ship Readiness Gate

**User Story:** As a program lead, I want release approval to depend on integrated product readiness, not checklist theater on a single subsystem.

#### Acceptance Criteria

1. THE OrinClaw release process SHALL **NOT** approve general availability based on milestone completion alone — per `Guide.md` §10 *Ship / no-ship gate*.
2. BEFORE GA, THE OrinClaw SHALL verify: **Product** — V1 scope met without depending on roadmap-only (e.g. Milestone 10) features.
3. BEFORE GA, THE OrinClaw SHALL verify: **Reliability** — benchmark and soak tests from `Guide.md` §10 pass on representative production or pilot hardware.
4. BEFORE GA, THE OrinClaw SHALL verify: **Security** — `Guide.md` §8 verification and **drive-extraction** tests pass on the actual shipping SKU.
5. BEFORE GA, THE OrinClaw SHALL verify: **Operations** — OTA rollback and factory reset demonstrated end-to-end on a production-like image.
6. BEFORE GA, THE OrinClaw SHALL verify: **Supportability** — a **non-developer** can complete setup, recovery, and reset using shipped documentation alone.

---

### Requirement 45: Serviceability, Manufacturing, and Support Deliverables

**User Story:** As operations and support, I want factory, service, and customer-facing artifacts so that units can be built, repaired, and recovered predictably.

#### Acceptance Criteria

1. THE OrinClaw program SHALL deliver a **factory provisioning flow** covering serial number, device identity, key enrollment (if applicable), burn-in, and first-boot validation — per `Guide.md` §10 *Serviceability, manufacturing, and support deliverables*.
2. THE OrinClaw program SHALL deliver a **service manual**: safe enclosure access, SSD replacement policy, tamper policy, and post-repair verification — aligned with FDE and physical security requirements.
3. THE OrinClaw program SHALL deliver a **support package**: recovery image path, rollback instructions, LED error reference, and `/health` interpretation for support staff.
4. THE OrinClaw program SHALL deliver a **release checklist** with sign-off slots for hardware, software, security, OTA, documentation, and **user-facing privacy** disclosures before shipment.

---

### Requirement 46: Documentation Traceability

**User Story:** As an auditor or new team member, I want to know which document is authoritative and how this spec relates to the rest of the repo.

#### Acceptance Criteria

1. THE OrinClaw requirements SHALL remain **traceable** to [Guide.md](../../../Guide.md); when `Guide.md` changes scope or ship gates, THIS document SHALL be updated in the same change set or within one agreed review cycle.
2. THE OrinClaw program MAY use [OrinClaw-Product-Business-Plan.md](../../../OrinClaw-Product-Business-Plan.md) for market, investment, and team planning; those sections SHALL NOT contradict normative requirements here unless this document is amended.
3. THE OrinClaw spec path `.kiro/specs/orinclaw/` MAY retain the folder name `orinclaw` for tooling; product naming in requirements SHALL remain **OrinClaw** / **orinclaw** as in Requirement 1.
