# L4T customization (production)

**Phase 4 — Track B — Nvidia Jetson** · Module 2 of 5

> **Focus:** Master **Linux for Tegra (L4T)** with **JetPack** as a **production** stack: reproducible images, a minimal root filesystem, kernel and device-tree integration, reliable boot and updates, and hardening—so you ship products instead of fighting the platform.

**Previous:** [1. Nvidia Jetson Platform](../1.%20Nvidia%20Jetson%20Platform/Guide.md) · **Next:** [3. Edge AI Optimization](../3.%20Edge%20AI%20Optimization/Guide.md)

---

## Why L4T + JetPack for real products

For most small teams and solo developers, **L4T with JetPack is the pragmatic default**: NVIDIA-supported drivers, CUDA and multimedia stacks, familiar Debian/Ubuntu tooling, and fast iteration. Production work here means **freezing baselines**, **automating flashes and rootfs**, and **treating the image as an artifact**—not ad-hoc `apt` on every device.

---

## Production practices (what to master)

| Area | Production goal |
|------|-----------------|
| **Minimal root filesystem** | Start from the **L4T sample rootfs**, then remove GUI and packages you do not need—smaller images, fewer moving parts, smaller attack surface. Document every package delta. |
| **Kernel & device tree** | Use the supported **`flash.sh`** workflow to bake in **custom device trees**, **overlays**, **drivers**, and **patches**. Keep kernel changes in a **small, versioned Git repo** so flashes are repeatable. |
| **Systemd** | Run your application under **systemd units** with restart policies, dependencies, and logging—reliable bring-up and recovery on embedded hardware. |
| **Containers** | Run the product stack in **Docker** (well supported on JetPack) for **isolation**, **reproducible runtime**, and easier rollout across a **fleet** of devices. |
| **Security hardening** | Disable unused services, prefer **SSH keys** over passwords, and layer **firewall** (`ufw` or equivalent) and **mandatory access control** (for example **AppArmor**) where your threat model requires it. |
| **Boot & updates** | Align **partition layout**, **A/B** if you use it, and **OTA** flows with how you build and sign images—see Platform deep dives below. |

---

## Practical workflow (high level)

1. **Freeze a JetPack / L4T baseline**—board SKU, carrier, exact JetPack version. Pin **apt** and NVIDIA repo configuration in scripts or docs.
2. **List product deltas**: packages, kernel config, DTB or overlay changes, **systemd** units, services to **mask/disable**.
3. **Build a minimal, reproducible rootfs** from the sample rootfs path: strip desktop stacks unless the product needs them; snapshot with your own flash or image recipe.
4. **Prefer supported integration paths** first—`flash.sh`, `extlinux.conf`, **`nvpmodel`**, **`jetson-io`** for DT overlays—before maintaining a heavy forked kernel tree.
5. **Automate validation**: smoke tests after flash (GPU, camera, network), plus **OTA** dry-runs if you ship image-based updates.

---

## Official NVIDIA documentation

Use the **Jetson Linux Developer Guide** for the same **major JetPack / L4T** line you ship; archived copies below illustrate the topics—open the matching release under [Jetson documentation](https://docs.nvidia.com/jetson/) if your version differs.

| Topic | Why it matters for production |
|-------|------------------------------|
| [**Jetson Module Adaptation and Bring-Up**](https://docs.nvidia.com/jetson/archives/r35.3.1/DeveloperGuide/text/HR/JetsonModuleAdaptationAndBringUp.html) | Moving from a **developer kit** to a **custom carrier**: board naming, rootfs configuration, MB1/MB2 (pinmux, EEPROM), **device tree** porting, PCIe/USB, **flashing** the build image, and hardware/software bring-up **checklists**. |
| [**Kernel customization**](https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/kernel_custom.html) | Syncing kernel sources with **Git**, building and installing the kernel, **DTB** and signing/encryption where required, external modules, and optional **real-time** kernel package workflow. |

---

## BCT reference (in this module)

- [Deployment.md](Deployment.md) — reformatted **T23x BCT** (DU-10990-001): MB1/MB2 boot configuration tables, DTS vs legacy CFG, pinmux/prod/PMIC/storage/UPHY/security. Use the preamble and table of contents to navigate; the body is still dense NVIDIA reference material.

---

## EchoPilot AI (worked example / vendor reference)

**EchoMAV** publishes a concrete **L4T bring-up** path for **EchoPilot AI** (Orin NX / Orin Nano on their carrier): Jetson Linux BSP, sample rootfs, `apply_binaries.sh`, default user creation, **device-tree overlays**, and **initrd flash** to NVMe. Treat it as a **reference implementation** for “custom carrier + headless Orin” alongside NVIDIA’s module adaptation guide.

| Resource | What to use it for |
|----------|-------------------|
| [**echopilot_ai_bsp** (GitHub)](https://github.com/EchoMAV/echopilot_ai_bsp) | Upstream BSP scripts and branches (e.g. `board_revision_1b` for Rev1B). Clone this on your build host; run `install_l4t_orin.sh` against your `Linux_for_Tegra` tree as in their docs. |
| **EchoPilot AI documentation** (EchoMAV MkDocs, e.g. *Building L4T (Orin NX and Orin Nano)*) | Step-by-step host setup (they target **Ubuntu 22.04**), download names for a given **Jetson Linux** drop (example in their guide: **36.4.3**), `l4t_initrd_flash.sh` invocation for **external NVMe**, and operational notes (USB autosuspend, `--flash-only` after first image build). |
| **Local snapshot** — [echopilot_ai_bsp-board_revision_1b](echopilot_ai_bsp-board_revision_1b/README.md) | Copy of the **board_revision_1b** tree pinned in this repo: overlays under `Linux_for_Tegra/kernel/dtb/` (e.g. disable display, enable serial), BCT-related files, patches, and `install_l4t_orin.sh`. Diff this against your own `Linux_for_Tegra` when debugging DT or flash layout. |

**Hardware / software deltas** called out in EchoPilot’s Orin guide (useful pattern for any third-party carrier): carrier may **not** use the same **board ID EEPROM** scheme as the NVIDIA dev kit; **headless** products often need **display paths disabled** in DT so boot completes; product-specific **UART** routing is enabled via **overlays**. Always match **branch / revision** of the BSP to the silkscreen **board revision** on the unit.

---

## Deep dives (Platform module 1)

Use these Platform guides as **implementation** detail for L4T-facing production work:

- [Rootfs and A/B redundancy](../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Rootfs-and-AB-Redundancy/Guide.md)
- [OTA deep dive](../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-OTA-Deep-Dive/Guide.md)
- [Kernel internals](../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Kernel-Internals/Guide.md)
- [RT Linux](../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-RT-Linux-Deep-Dive/Guide.md) (when latency matters)
- [Security hardening](../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Security/Guide.md)

---

## Relationship to Edge AI (module 3)

**L4T customization** defines **what runs on the OS** (kernel, drivers, rootfs, CUDA install path, services). **Edge AI Optimization** assumes that foundation and focuses on **models** (TensorRT, quantization, DeepStream). Finish or parallel the platform deep dives above before treating inference as the main bottleneck.
