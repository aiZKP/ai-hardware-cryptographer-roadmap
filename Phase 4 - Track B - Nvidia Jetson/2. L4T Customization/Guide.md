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
