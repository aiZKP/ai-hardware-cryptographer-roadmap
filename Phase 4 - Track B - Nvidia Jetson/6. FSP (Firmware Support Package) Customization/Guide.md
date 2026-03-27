# FSP (Firmware Support Package) and SPE firmware

**Phase 4 — Track B — Nvidia Jetson** · Module 6 of 6

> **Focus:** Customize firmware that runs on the Jetson **Sensor Processing Engine (SPE)**—the **Cortex-R5** in the **always-on (AON)** domain—using NVIDIA’s **Firmware Support Package (FSP)** on **FreeRTOS**. This is the path for **low-level I/O**, **wake** scenarios, and **time-critical** tasks that should not live on the main Linux **CCPLEX**.
>
> **Scope:** The **step-by-step demo sections** below target **Jetson Orin Nano (T234)** using **r35.6** SPE guide paths (**p3767** MB1 BCT, **p3768** kernel DTS where NVIDIA names them). Other Jetson models use different files—see the same SPE guide for **AGX Xavier** / **AGX Orin** variants.

**Previous:** [5. OpenClaw Assistant Box (OrinClaw capstone)](../5.%20OpenClaw%20Assistant%20Box/Guide.md) · **Next (curriculum):** [Phase 5 — Real Time Edge AI with Nvidia Jetson](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/2.%20Real%20Time%20Edge%20AI%20with%20Nvidia%20Jetson/Guide.md) · **Companion:** [2. L4T customization](../2.%20L4T%20Customization/Guide.md) (flash layout, `Linux_for_Tegra`, BCT/pinmux)

---

## Why this sits next to L4T

SPE firmware is a **separate binary** (`spe_t194.bin` / `spe_t234.bin`) flashed via the same **Jetson Linux** toolkit as the rest of the board. Changing SPE behavior almost always touches **production concerns** you already practice in L4T work: **pinmux**, **GPIO init**, **firewall (SCR)**, **device tree** (including **BPMP** DT for clocks), and **partition-only** flashes. Treat SPE images as **versioned artifacts** alongside kernel and DTB.

---

## Official documentation (pin to your JetPack / L4T line)

The material below is aligned with the **Jetson SPE Developer Guide** packaged with the **r35.6** documentation set. For other releases, open the matching archive under [Jetson documentation](https://docs.nvidia.com/jetson/).

| Topic | Link |
|--------|------|
| **SPE guide (welcome, BSP layout, feature matrix)** | [Jetson SPE Developer Guide — r35.6](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/index.html) |
| **FSP architecture (OSA / CPL / drivers)** | [FSP (Firmware Support Package)](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_fsp.html) |
| **Build, artifacts, flash** | [Compiling and Flashing](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/rt-compiling.html) |
| **AODMIC demo (DMIC5, wake)** | [AODMIC Application](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_aodmic_app.html) |
| **Inter-processor channels** | [IVC](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_ivc.html) |
| **GTE (timestamped GPIO events)** | [GTE Application](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_gte_app.html) |
| **AON GPIO from SPE** | [GPIO Application](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_gpio.html) |
| **AON I2C from SPE** | [I2C application](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_i2c_app.html) |
| **AON SPI from SPE** | [SPI application](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_spi_app.html) |
| **Timer driver demo** | [Timer application](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_timer_app.html) |

---

## SPE / BSP layout (mental model)

From the SPE BSP package:

- **`fsp/source`** — Common **drivers**, **OSA** (OS abstraction), **CPL** (CPU abstraction), and **`soc/<soc>/...`** port/ID data.
- **`rt-aux-cpu-demo-fsp`** — Demo apps (`app/`), build system (`Makefile`, `soc/t19x` / `soc/t23x` **target_specific.mk**), platform code, and **FreeRTOS** integration.
- **`FreeRTOSV10.4.3/FreeRTOS/Source`** — FreeRTOS sources (version as shipped in that BSP drop).

NVIDIA’s welcome page lists **which demos are supported on which platforms** (e.g. **IVC**, **GTE**, **GPIO**, **I2C**, **SPI**, **Timer** on **Orin Nano** in that matrix; **AODMIC** is **not** listed for Orin Nano there). Always confirm against your **SoC** and **guide revision**.

**Orin Nano builds:** In `soc/t23x/target_specific.mk`, set **`ENABLE_SPE_FOR_ORIN_NANO := 1`** for demos that target the Nano module (as in the **GTE**, **GPIO**, and **SPI** recipes below). Use the same file for per-app flags such as **`ENABLE_I2C_APP`**, **`ENABLE_SPI_APP`**, **`ENABLE_TIMER_APP`**.

---

## FSP architecture (short)

### OSA (Operating System Abstraction)

Abstracts **FreeRTOS v10** so drivers and apps can be ported to another RTOS with minimal churn. Headers live under `fsp/source/include/osa/freertosv10/osa/`—for example:

| Concern | Header (API entry point) |
|---------|----------------------------|
| Semaphore | `osa-semaphore.h` |
| Mutex | `osa-mutex.h` |
| Event group | `osa-event-group.h` |
| Queue | `osa-queue.h` |
| Tasks / scheduling | `osa-task.h` |
| Software timer | `osa-timer.h` |

### CPL (CPU / platform abstraction)

CPU primitives shared across auxiliary processors—cache ops, register access, barriers, interrupt controller, chip ID. Examples (paths as in NVIDIA’s FSP page):

| Concern | Header |
|---------|--------|
| Cache | `fsp/source/include/cpu/arm/common/cpu/cache.h` |
| Register access (Cortex-R5) | `fsp/source/include/cpu/arm/armv7/cortex-r5/reg-access/reg-access.h` |
| Barriers | `fsp/source/include/cpu/arm/common/cpu/barriers.h` |
| VIC (interrupts) | `fsp/source/include/cpu/arm/common/cpu/arm-vic.h` |
| Chip ID / SKU | `fsp/source/include/chipid/chip-id.h` |

### Drivers

Peripheral drivers sit above OSA/CPL. NVIDIA points to headers such as **`fsp/source/include/gpio/tegra-gpio.h`** for GPIO. **SoC-specific** pieces use `fsp/source/soc/<soc>/port/aon` and `fsp/source/soc/<soc>/ids/aon` for **instance IDs**, **base addresses**, and **IRQ** data.

---

## Toolchain, build, and flash

### Toolchain

NVIDIA expects an external **GNU Arm Embedded** toolchain (documentation references **`gcc-arm-none-eabi-7-2018-q2-update`**). NVIDIA does not redistribute it; download from Arm’s archive for your host OS. On **Windows**, use **WSL2** (Ubuntu) for the same flow as Jetson flashing.

### Environment and build

```bash
export SPE_FREERTOS_BSP=<root containing rt-aux-cpu-demo-fsp, fsp, FreeRTOS>
export CROSS_COMPILE=<toolchain>/bin/arm-none-eabi-

cd "${SPE_FREERTOS_BSP}/rt-aux-cpu-demo-fsp"
make -j"$(nproc)" bin_t19x    # T194-class
make -j"$(nproc)" bin_t23x    # T234-class
```

Other useful targets (see [Compiling and Flashing](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/rt-compiling.html)):

- **`make docs`** — Doxygen output under `out/docs/index.html`
- **`make` / `make all`** — All SOC binaries + docs
- **`make clean`**, **`make clean_t19x`**, **`make clean_t23x`**, **`make clean_docs`**

**Artifact:** `out/<soc>/spe.bin`

### Install into `Linux_for_Tegra` and flash

1. **Back up** the stock files in `Linux_for_Tegra/bootloader/`:
   - **`spe_t194.bin`** (T194)
   - **`spe_t234.bin`** (T234)
2. Copy your build:
   - T194 → `Linux_for_Tegra/bootloader/spe_t194.bin`
   - T234 → `Linux_for_Tegra/bootloader/spe_t234.bin`
3. Flash **only** the SPE partition when iterating:

```bash
# T194 (partition name spe-fw)
sudo ./flash.sh -k spe-fw <board-name> mmcblk0p1

# T234 (partition name A_spe-fw)
sudo ./flash.sh -k A_spe-fw <board-name> mmcblk0p1
```

Use the same **`<board-name>`** conventions as in the Jetson Linux Developer Guide for your carrier/module.

---

## Orin Nano — IVC echo channel (CCPLEX ↔ AON)

**IVC** uses a mailbox-backed **memory channel**. The stock SPE distribution documents an **echo** channel: Linux sends a string, SPE echoes it back.

NVIDIA’s r35.6 **IVC** page labels the **device tree** recipe under **AGX Orin**; **Orin Nano** is the same **T234** line, so you use the same **`tegra234-aon.dtsi`** pattern after syncing sources.

1. Run **`source_sync.sh`** (per Jetson Linux documentation) so kernel DTS is available under `Linux_for_Tegra/sources/`.
2. Edit **`Linux_for_Tegra/sources/hardware/nvidia/soc/t23x/kernel-dts/tegra234-soc/tegra234-aon.dtsi`**. Either add an **`aon_echo`** node or, if it already exists disabled, set **`status = "okay"`** and match mailboxes to the guide (**`mboxes = <&aon 1>;`** for the AGX Orin example):

```dts
aon_echo {
    compatible = "nvidia,tegra186-aon-ivc-echo";
    mboxes = <&aon 1>;
    status = "okay";
};
```

3. **Compile** the device tree, install the **DTB** into **`Linux_for_Tegra/kernel/dtb/`** or **`Linux_for_Tegra/dtb/`** (whichever your Jetson Linux layout uses), and **reflash** using the **Building the kernel / device tree** flow from the Jetson Linux Developer Guide.
4. Build the **SPE** firmware that includes the **IVC echo** task, then copy **`out/t23x/spe.bin`** → **`${L4T}/bootloader/spe_t234.bin`** and flash (**`-k A_spe-fw`** when you are only updating SPE).

**Firmware sources (reference):**

- `rt-aux-cpu-demo-fsp/app/ivc-echo-task.c`
- `rt-aux-cpu-demo-fsp/platform/ivc-channel-ids.c`

**Test from Linux** (after the echo channel is enabled):

```bash
sudo su -c 'echo tegra > /sys/devices/platform/aon_echo/data_channel'
cat /sys/devices/platform/aon_echo/data_channel
# expect: tegra
```

Full detail: [IVC](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_ivc.html).

---

## Orin Nano — GTE application (`app/gte-app.c`)

The **Generic Timestamping Engine (GTE)** snoops signals and exposes **timestamped events** via FIFOs (three **32-bit** slices for AON/SPE—see **`gte-tegra-hw.h`** for signal mapping). The demo pairs with **GPIO** activity; pin ↔ GTE bit mapping comes from the **pinmux spreadsheet** (**hidden “GTE”** column for slice 2), as in NVIDIA’s guide.

### Software steps (Orin Nano)

1. In **`soc/t23x/target_specific.mk`**: **`ENABLE_GTE_APP := 1`** and **`ENABLE_SPE_FOR_ORIN_NANO := 1`**. Rebuild **`bin_t23x`**, copy **`spe.bin`** → **`${L4T}/bootloader/spe_t234.bin`**.
2. After **`source_sync.sh`**, disable the Linux **GTE** driver so it does not own the block while SPE uses it. File from NVIDIA (Orin Nano / CVB path):  
   **`Linux_for_Tegra/sources/hardware/nvidia/platform/t23x/p3768/kernel-dts/cvb/tegra234-p3768-0000-a0.dtsi`**

```dts
gte@c1e0000 {
    status = "disabled";
};
```

3. Rebuild the **kernel device tree**, copy DTBs to **`Linux_for_Tegra/kernel/dtb/`**, and adjust **firewall** so SPE can reach **GTE** registers. For Orin Nano, NVIDIA patches **`${L4T}/bootloader/tegra234-firewall-config-base.dtsi`**: change **`reg@1359`** (**`GTE_GPIO_SCR_TESCR_0`**) to allow SPE read/write:

```dts
reg@1359 { /* GTE_GPIO_SCR_TESCR_0 */
    exclusion-info = <3>;
    value = <0x38001232>;
};
```

4. **Full flash** so **DTB + firewall** land on the device (not SPE-only the first time you change these).

### Expected log (abbreviated)

When the FIFO crosses **`GTE_FIFO_OCCUPANCY`**, the GTE ISR runs. NVIDIA’s guide shows lines such as **`gpio_app_task - Setting GPIO_APP_OUT to 1`**, **`Slice Id` / `Event Id` / `Timestamp in nanosec`**, and **`can_gpio_irq_handler`** toggling the output—pair with the **GPIO** demo wiring for a closed loop.

Full detail: [GTE Application](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_gte_app.html).

---

## Orin Nano — GPIO application (`app/gpio-app.c`)

Demo: drive one **AON GPIO** from SPE and receive an **interrupt** on another. **MB1** pinmux and **GPIO int map** must match the Orin Nano carrier (**p3767** BCT in the r35.6 guide).

### Hardware (Orin Nano dev kit)

On **J12**, tie **Pin 5** (**PDD1**, output) to **Pin 3** (**PDD2**, input).

### Software steps

1. **`soc/t23x/target_specific.mk`**: **`ENABLE_GPIO_APP := 1`**, **`ENABLE_SPE_FOR_ORIN_NANO := 1`**. Rebuild, copy **`spe.bin`** → **`${L4T}/bootloader/spe_t234.bin`**.
2. **GPIO interrupt map** — **`${L4T}/bootloader/t186ref/BCT/tegra234-mb1-bct-gpioint-p3767-0000.dts`**: under **`port@DD`**, route **DD2** to interrupt line **2**:

```dts
port@DD {
    pin-0-int-line = <4>; // GPIO DD0 to INT0
    pin-1-int-line = <4>; // GPIO DD1 to INT0
    pin-2-int-line = <2>; // GPIO DD2 to INT2
};
```

3. **AON GPIO ownership (MB1 GPIO DTSI)** — **`${L4T}/bootloader/tegra234-mb1-bct-gpio-p3767-dp-a03.dtsi`**: add **DD2** as input and **DD1** as output-low (example from NVIDIA):

```dts
gpio-input = <
    TEGRA234_AON_GPIO(EE, 2)
    TEGRA234_AON_GPIO(EE, 4)
    TEGRA234_AON_GPIO(DD, 2)
>;
gpio-output-low = <
    TEGRA234_AON_GPIO(DD, 1)
    TEGRA234_AON_GPIO(CC, 0)
>;
```

4. **Pinmux** — **`${L4T}/bootloader/t186ref/BCT/tegra234-mb1-bct-pinmux-p3767-dp-a03.dtsi`**: repurpose **gen8_i2c_scl_pdd1** / **gen8_i2c_sda_pdd2** from **I2C8** to **`rsvd1`** with the pull / tristate / input enables NVIDIA specifies (so the pins behave as GPIO for the demo). See the [GPIO Application](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_gpio.html) page for the exact property values.
5. **Full flash** so **MB1 BCT** and **GPIO** settings take effect.

### Expected serial output

```text
gpio_app_task - Setting GPIO_APP_OUT to 1 - IRQ should trigger
can_gpio_irq_handler - gpio irq triggered - setting GPIO_APP_OUT to 0
```

Full detail: [GPIO Application](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_gpio.html).

---

## Orin Nano — I2C application (`app/i2c-app.c`)

NVIDIA groups **Jetson AGX Orin** and **Orin Nano** under one **software** recipe: give **Linux** ownership of **AON I2C8** back to **SPE**, open the **firewall**, then run the demo firmware.

AON has multiple **I2C** instances (**2**, **8**, **10**); which ones are available depends on the platform. This demo uses **I2C bus 8** and reads a **BMI160** (6-axis) **WHO_AM_I**-style ID over the bus.

### Hardware

Wire a **BMI160** module (or equivalent) to **40-pin header J30** (NVIDIA’s map):

| J30 pin | Signal |
|---------|--------|
| 1 | 3.3V |
| 3 | SDA |
| 5 | SCL |
| 34 | SAO (tie per module; sets **7-bit address 0x68** in the demo) |
| 39 | GND |

Reference board (example cited in NVIDIA’s guide): [BMI160 breakout](https://hackspark.fr/en/electronics/1341-6dof-bosch-6-axis-acceleration-gyro-gravity-sensor-gy-bmi160.html).

### Software steps

1. Run **`source_sync.sh`** so kernel DTS is under **`Linux_for_Tegra/sources/`**.
2. In **`Linux_for_Tegra/sources/hardware/nvidia/soc/t23x/kernel-dts/tegra234-soc/tegra234-soc-cvm.dtsi`**, disable the **I2C8** controller so the kernel does not claim it:

```dts
i2c@c250000 {
    status = "disabled";
};
```

3. **Compile** DTBs, copy into **`Linux_for_Tegra/kernel/dtb/`** or **`Linux_for_Tegra/dtb/`**, per your Jetson Linux layout.
4. In **`${L4T}/bootloader/tegra234-firewall-config-base.dtsi`**, allow SPE to access **I2C8** clocks/resets (**`CLK_RST_CONTROLLER_AON_SCR_I2C8_0`**):

```dts
reg@2130 { /* CLK_RST_CONTROLLER_AON_SCR_I2C8_0 */
    exclusion-info = <3>;
    value = <0x30001610>;
};
```

5. **`soc/t23x/target_specific.mk`**: **`ENABLE_I2C_APP := 1`** (and **`ENABLE_SPE_FOR_ORIN_NANO := 1`** if you build the Nano SPE image). Rebuild **`bin_t23x`**, copy **`spe.bin`** → **`${L4T}/bootloader/spe_t234.bin`**.
6. **Full flash** so **kernel DTB + firewall + SPE** stay consistent.

### Success output

```text
I2C test successful
```

Full detail: [I2C application](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_i2c_app.html).

---

## Orin Nano — SPI application (`app/spi-app.c`)

**SPI2** lives in the **AON** domain. The demo is a **loopback**: **MISO** and **MOSI** must be shorted; firmware sends fixed bytes and checks the readback.

> **BCT conflict:** The SPI demo’s **MB1 GPIO** edits **remove** several **`TEGRA234_AON_GPIO(CC, …)`** entries from output lists in **`tegra234-mb1-bct-gpio-p3767-dp-a03.dtsi`**. If you merged **GPIO** or **GTE** recipes that rely on those lines, reconcile one coherent BCT before flashing.

### Hardware (Orin Nano dev kit)

On **J2**, short **MISO** ↔ **MOSI**, then connect:

| J2 pin | SPI2 signal |
|--------|-------------|
| 126 | CLK |
| 127 | MISO |
| 128 | MOSI |
| 130 | CS0 |

### Software steps

1. **Pinmux** — **`${L4T}/bootloader/t186ref/BCT/tegra234-mb1-bct-pinmux-p3767-dp-a03.dtsi`**: set **`spi2_sck_pcc0`**, **`spi2_miso_pcc1`**, **`spi2_mosi_pcc2`**, **`spi2_cs0_pcc3`** to **`nvidia,function = "spi2"`** with pull/tristate/input enables per [SPI application — Orin Nano](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_spi_app.html) (e.g. **MISO** keeps **tristate/input enabled** for sampling).
2. **MB1 GPIO** — **`${L4T}/bootloader/tegra234-mb1-bct-gpio-p3767-dp-a03.dtsi`**: drop **CC0–CC3** from **`gpio-output-low`** / **`gpio-output-high`** as in NVIDIA’s diff so **SPI2** balls are not driven as GPIO.
3. **Firewall** — **`${L4T}/bootloader/tegra234-mb2-bct-scr-p3767-0000.dts`**: add **`reg@2135`** for **`CLK_RST_CONTROLLER_AON_SCR_SPI2_0`**:

```dts
reg@2135 { /* CLK_RST_CONTROLLER_AON_SCR_SPI2_0 */
    exclusion-info = <3>;
    value = <0x30001410>;
};
```

4. **`soc/t23x/target_specific.mk`**: **`ENABLE_SPI_APP := 1`**, **`ENABLE_SPE_FOR_ORIN_NANO := 1`**. Rebuild, install **`spe_t234.bin`**, **full flash**.

### Success output

```text
SPI test successful
```

Full detail: [SPI application](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_spi_app.html).

---

## Orin Nano — Timer application (`app/timer-app.c`)

Uses **timer2** in **periodic** mode (NVIDIA’s example: **5 s** period, **`TIMER2_PTV`**). The demo prints on each IRQ and stops after **`STOP_TIMER`** counts.

1. **`soc/t23x/target_specific.mk`**: **`ENABLE_TIMER_APP := 1`** (set other **`ENABLE_*_APP`** flags to **0** unless you intentionally combine demos). For **Orin Nano** module images, keep **`ENABLE_SPE_FOR_ORIN_NANO := 1`** if your BSP requires it for **`bin_t23x`** on Nano.
2. Rebuild **`bin_t23x`**, copy **`spe.bin`** → **`spe_t234.bin`**, flash (**`-k A_spe-fw`** is enough if you did not change kernel or BCT).

### Expected serial output

```text
Timer2 irq triggered
```

(repeats each period until the stop count is reached.)

Full detail: [Timer application](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/md__home_jenkins_workspace_Utilities_rt_aux_cpu_demo_fsp_docs_work_rt_aux_cpu_demo_fsp_doc_timer_app.html).

---

## Example: AODMIC application (AGX-oriented in r35.6 matrix)

The r35.6 **supported features** table in the SPE welcome page lists **AODMIC** for **AGX Xavier** and **AGX Orin**, not **Orin Nano**. The following is still useful if you work on those carriers or a **custom** T234 board that exposes **DMIC5** to AON.

The **AODMIC** demo shows **DMIC5** capture from SPE, **GPCDMA**, optional **system wake** from volume threshold, and the interaction with **suspend** and **BPMP** clocks. NVIDIA documents:

- **Sample rates** — `TEGRA_AODMIC_RATE_8KHZ`, `_16KHZ` (default in `app/aodmic-app.c`), `_44KHZ`, `_48KHZ`.
- **Channels** — `TEGRA_AODMIC_CHANNEL_STEREO` (default), `MONO_LEFT`, `MONO_RIGHT`.
- **`num_periods`** — typically **2** (double-buffered); max defined in `fsp/source/include/aodmic/tegra-aodmic.h` as `AODMIC_MAX_NUM_PERIODS` (extend if needed).
- **Oversampling** — AODMIC uses **64×** oversampling; effective PDM clock is **64 × sample_rate**—stay within your microphone’s allowed bit clock range.

**Hardware:** PDM microphone on the platform’s **AODMIC** pins (NVIDIA gives **40-pin** mappings for **AGX Xavier** and **AGX Orin**—DAT/CLK pins and power/GND).

**Software integration** (high level—exact edits change by release and carrier):

- **Pinmux** — Xavier: MB1 CFG edits (e.g. `tegra19x-mb1-pinmux-...cfg`). Orin: MB1 **pinmux DTSI** (e.g. `tegra234-mb1-bct-pinmux-...dtsi`) to route **CAN1**/GPIO balls to **dmic5**.
- **GPIO** — Remove conflicting **AON GPIO** claims in the MB1 GPIO DTSI where pins become DMIC.
- **Firewall** — Orin example: **SCR** override DTS allows SPE to touch **AODMIC clock** (`CLK_RST_CONTROLLER_AON_SCR_DMIC5_0`).
- **Enable the app** — In `soc/t19x/target_specific.mk` or `soc/t23x/target_specific.mk`, set **`ENABLE_AODMIC_APP := 1`**, rebuild, copy `spe.bin` to the correct **`spe_t194.bin` / `spe_t234.bin`**, then flash (full flash when MB1/SCR change; **SPE-only** once those are stable).

**Suspend caveat:** After resume, **BPMP** may gate **dmic5** clock and break capture. Mitigations in the guide: runtime `debugfs` clock force-on, or patch **BPMP DTB** `dmic5` **lateinit** clock tuple (third argument = **sample_rate × 64**), then flash **`bpmp-fw-dtb`** or full image.

**Wake path:** For **voice wake** demos, **wake83** must appear in BPMP UART logs in both **wake mask** and **Tier2** routing (NVIDIA gives example mask snippets).

---

## Practice checklist

- [ ] Match **SPE BSP** and **L4T** major lines to your shipping **JetPack** (avoid mixing undocumented combinations).
- [ ] Store **original** `spe_t194.bin` / `spe_t234.bin` and every **custom** build in **Git** or artifact storage with **board + L4T** metadata.
- [ ] When changing **pinmux / SCR / BPMP DT**, plan a **full flash** once, then use **`-k spe-fw`** / **`-k A_spe-fw`** for firmware iteration.
- [ ] Read the **SoC TRM** for **SPE/AON** peripheral instances and **firewall** rules before relying on demos in production.

---

*Primary reference: [Jetson Sensor Processing Engine (SPE) Developer Guide — r35.6](https://docs.nvidia.com/jetson/archives/r35.6.0/spe/index.html) (NVIDIA). Align commands and file names with the archive that matches your Jetson Linux release.*
