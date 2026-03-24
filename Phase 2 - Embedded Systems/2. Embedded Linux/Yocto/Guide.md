# Yocto Project — Embedded Linux Distribution Engineering

A structured course for hardware and software engineers who need to **build, own, and ship** a custom embedded Linux—not just install one. The goal is clarity first: you should finish with a **mental model** that survives release renames and new boards.

**Time investment (typical):** 4–10 weeks part-time for Modules 0–6; full course including production topics 3–6 months alongside real hardware.

**What you will be able to do**

- Explain how BitBake, recipes, layers, and images relate end-to-end.
- Reproduce a build from metadata alone (team-ready, not “works on my laptop”).
- Add a package, patch a component, and ship a smaller, testable image.
- Debug failures using logs and task graphs instead of guessing.
- Know when Yocto is the right tool—and when Buildroot or a vendor BSP is faster.

---

## Table of contents

1. [How to use this course](#1-how-to-use-this-course)
2. [Module 0 — Prerequisites and host setup](#2-module-0--prerequisites-and-host-setup)
3. [Module 1 — What Yocto is (and is not)](#3-module-1--what-yocto-is-and-is-not)
4. [Module 2 — Architecture in one picture](#4-module-2--architecture-in-one-picture)
5. [Module 3 — First successful build](#5-module-3--first-successful-build)
6. [Module 4 — Recipes: the atoms of the system](#6-module-4--recipes-the-atoms-of-the-system)
7. [Module 5 — Layers: how metadata stays maintainable](#7-module-5--layers-how-metadata-stays-maintainable)
8. [Module 6 — Images, packages, and features](#8-module-6--images-packages-and-features)
9. [Module 7 — Kernel, bootloader, device tree (integrator level)](#9-module-7--kernel-bootloader-device-tree-integrator-level)
10. [Module 8 — SDK and application workflow](#10-module-8--sdk-and-application-workflow)
11. [Module 9 — Debugging builds like an engineer](#11-module-9--debugging-builds-like-an-engineer)
12. [Module 10 — Licenses, compliance, and supply-chain hygiene](#12-module-10--licenses-compliance-and-supply-chain-hygiene)
13. [Module 11 — Performance, caching, and CI](#13-module-11--performance-caching-and-ci)
14. [Capstone projects](#14-capstone-projects)
15. [Glossary and quick reference](#15-glossary-and-quick-reference)
16. [Further reading](#16-further-reading)

---

## 1. How to use this course

**Study pattern that works**

- Read one module, then **do** the lab at the end before moving on.
- Keep a single “course notes” repo: your `meta-mylayer`, `local.conf` snippets, and commands that worked.
- When something fails, capture **the exact command**, **the first ERROR line**, and **the task log path**—that habit will save weeks over a career.

**Plain-English rule used throughout**

- **Formal term** → what it does in the build.
- **Why it exists** → maintainability, reproducibility, or scale.

---

## 2. Module 0 — Prerequisites and host setup

### 2.1 Skills you should already have

- Comfortable on the Linux command line (paths, environment variables, `grep`, basic scripting).
- Basic embedded concepts: bootloader → kernel → root filesystem; what a **device tree** is at a high level.
- Git workflows (clone, branch, commit). Yocto *is* a Git-heavy ecosystem.

### 2.2 Host machine expectations

Yocto builds are **disk- and I/O-heavy** more than they are “mystical.”

| Resource | Practical minimum | Comfortable |
|----------|-------------------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB (small images only) | 16–32 GB |
| Disk (SSD strongly preferred) | 80 GB free | 200+ GB |
| OS | Linux (supported distro for your Yocto release) | Same, on ext4 or similar |

Use a **case-sensitive** filesystem for the build tree. On some platforms, case-insensitivity causes bizarre, hard-to-debug failures.

### 2.3 Lab 0 — Verify your host

- Install your distro’s build dependencies using the **official Yocto guide for your release** (package names drift by distro).
- Confirm `git`, `python3`, and a working compiler toolchain for the *host* are present.
- Create a dedicated user or directory convention so paths stay stable (teams will thank you).

**Done when:** you can clone a large Git repository and compile a small native C “hello world” on the host without errors.

---

## 3. Module 1 — What Yocto is (and is not)

### 3.1 The problem

Embedded products need an OS that is:

- **Repeatable** (same inputs → same enough outputs for your release process).
- **Auditable** (licenses, versions, patches known and recorded).
- **Minimal** where possible (fewer packages → fewer CVEs and smaller update payloads).
- **Aligned to hardware** (kernel, firmware, boot chain, drivers).

### 3.2 What the Yocto Project provides

**Yocto Project** is an umbrella and governance model. Practically, you work with:

- **OpenEmbedded-Core (OE-Core)** — base recipes, classes, and policies.
- **BitBake** — the task scheduler / build engine.
- **Poky** — a reference integration (OE-Core + BitBake + tooling) you can build and learn from.

**Plain English:** Yocto is not “a Linux distro.” It is a **factory** that can produce distro-like artifacts (images, packages, SDKs) from metadata.

### 3.3 Yocto vs Buildroot (decision guidance)

| Dimension | Yocto / OpenEmbedded | Buildroot |
|-----------|----------------------|-----------|
| Philosophy | Layers, sharing metadata across products | Single-tree config, very direct |
| Package ecosystem | Large, community + vendor layers | Smaller default set; extensions possible |
| Learning curve | Steeper | Gentler |
| Multi-product reuse | Strong (layers, distros, configs) | Possible; often more manual |
| When vendors ship layers | Often the “expected” integration path | Less common (but not never) |

Neither is always “better.” **If your silicon vendor maintains a mature Yocto layer**, that often decides the default.

### 3.4 Lab 1 — Write your product requirements in Yocto terms

Answer in one page:

- **Target hardware** (SoC, machine name if known).
- **Connectivity** (Ethernet, Wi-Fi, USB gadgets, CAN, etc.).
- **Storage layout** (eMMC, NAND, NVMe, SD-only demo).
- **Update strategy** (A/B, single partition, package manager, image-based).
- **Must-have userspace** (containers or not, graphics or headless, real-time needs).

**Done when:** you can name what *must* be in the image vs what is “nice to have.”

---

## 4. Module 2 — Architecture in one picture

### 4.1 Data flow (mental model)

```
Metadata (layers: conf/, recipes-*)
        ↓
BitBake parses recipes + config (MACHINE, DISTRO, images)
        ↓
Task graph (fetch → unpack → patch → configure → compile → install → package → image)
        ↓
Artifacts (RPM/IPK/DEB packages, rootfs, kernel, bootloader, SDK)
```

### 4.2 Key objects

- **Recipe (`.bb`)** — how to build *one piece of software* (or fetch a binary).
- **Append file (`.bbappend`)** — your overlay changes without editing upstream recipes.
- **Layer** — a folder of metadata + configuration, versioned as a unit.
- **Image recipe** — defines what packages land on the root filesystem *for this product image*.
- **`MACHINE`** — selects board-specific kernel/bootloader/firmware assumptions.
- **`DISTRO`** — policy knobs: libc choices, init system preferences, security posture defaults (varies by distro layer).

### 4.3 Two configs you touch constantly

- **`conf/bblayers.conf`** — which layers BitBake is allowed to read.
- **`conf/local.conf`** — local machine settings: parallelism, download directory, extra image features, temporary tweaks.

**Plain English:** `bblayers.conf` is “which rulebooks exist.” `local.conf` is “how *my laptop/build server* applies them.”

### 4.4 Lab 2 — Draw your layer stack

On paper or a whiteboard, draw:

- Poky / OE-Core at the bottom.
- BSP layer for your hardware (when you have one).
- Your product layer on top.

Mark which layer should own: kernel tweaks, rootfs packages, systemd units, version pins.

**Done when:** you can explain where a change should live *before* you open an editor.

---

## 5. Module 3 — First successful build

### 5.1 Get Poky (reference flow)

Exact branch names change; always prefer the **documentation for the release you chose**.

Typical sequence (conceptual):

1. Clone Poky at a named release branch.
2. `source oe-init-build-env build` — sets environment variables for this shell.
3. Set `MACHINE` to a reference machine supported out of the box (often QEMU targets for learning).
4. Build a small image such as `core-image-minimal` first.

### 5.2 What “success” produces

You should be able to point to:

- The **image artifacts** under the build directory’s deploy area (layout varies slightly by version).
- A **kernel** and **rootfs** suitable for the selected `MACHINE`.

### 5.3 Lab 3 — Minimal image, boot under QEMU (or hardware)

- Build `core-image-minimal` for a QEMU-capable machine if available in your checkout.
- Boot it the way the docs for that release describe.
- Log in and run `uname -a` and `cat /etc/os-release`.

**Done when:** you have repeated the build from a clean shell using only your notes (no copy-paste from random blogs).

---

## 6. Module 4 — Recipes: the atoms of the system

### 6.1 Anatomy of a recipe

A recipe answers predictable questions:

- Where do sources come from (Git, tarball, license file)?
- How is it configured and compiled (autotools, CMake, Meson, Makefile, prebuilt)?
- What files are installed into staged roots?
- How is it split into runtime packages (`-dev`, `-dbg`, etc.)?

### 6.2 Inheritance and classes

Instead of copying boilerplate, recipes **inherit** behavior:

- **`autotools`**, **`cmake`**, **`meson`**, **`kernel`**, **`image`**, etc.

**Plain English:** classes encode “how this kind of software is usually built.”

### 6.3 Variables you will see constantly (conceptual)

Names evolve slightly, but the ideas do not:

- **`SRC_URI`** — what to download; patches are often listed here.
- **`S`** — where unpacked sources live.
- **`B`** — separate build dir when using out-of-tree builds.
- **`WORKDIR`** — per-recipe scratch space.
- **`PN`, `PV`, `PR`** — name/version/revision identifiers for packages.

### 6.4 Lab 4 — Inspect a real recipe

Pick a small recipe in OE-Core (a tiny utility). Trace:

- Which **class** it inherits.
- Which tasks exist (`do_compile`, `do_install`, etc.).
- What package names it produces.

Commands (conceptual; exact flags depend on version):

- `bitbake -e <recipe>` — effective environment (large output: learn to search it).
- `bitbake -c menuconfig linux-yocto` — only when you are ready for kernel exploration (optional this early).

**Done when:** you can name the tasks between source and package for that recipe.

---

## 7. Module 5 — Layers: how metadata stays maintainable

### 7.1 Why layers exist

Layers let teams **separate concerns**:

- Board support (BSP) stays with hardware knowledge.
- Corporate policy (distro) stays consistent across products.
- Product-specific tweaks stay isolated and reviewable.

### 7.2 Creating a custom layer (pattern)

Use the official `bitbake-layers create-layer` workflow for your release. You should end with:

- `conf/layer.conf` — compatibility and priority.
- `recipes-*` trees for your metadata.
- Optional `README` for layer intent and maintainers.

### 7.3 `bbappend` discipline

**Rule of thumb:** if you did not write the upstream recipe, prefer **`.bbappend`** in your layer.

Good reasons to append:

- Add a patch, tweak `PACKAGECONFIG`, add a runtime dependency.
- Install a config file with `FILESPATH` / `FILESEXTRAPATHS` patterns appropriate to your release.

Bad reasons:

- Forking a recipe to change one variable you could append (creates merge pain).

### 7.4 Lab 5 — Your own layer, minimal change

- Create `meta-mycourse`.
- Add `bblayers.conf` entry.
- Append a trivial change to a recipe (for example, a harmless `PACKAGECONFIG` toggle on a small library you actually use) **or** add a `FILES` tweak if appropriate.

**Done when:** `bitbake-layers show-layers` lists your layer in the expected priority order and the build still succeeds.

---

## 8. Module 6 — Images, packages, and features

### 8.1 Image recipes vs packagegroups

- **Image recipe** — names the packages that become the rootfs for a flashable artifact.
- **`packagegroup-*`** — bundles related packages for reuse across images.

### 8.2 `IMAGE_INSTALL` and friends

Teams usually extend images with:

- **`IMAGE_INSTALL:append`** — additive, review-friendly in many setups.

Avoid editing Poky images directly; extend via `bbappend` or a custom image recipe in your layer.

### 8.3 Image features (conceptual)

Features like debugging helpers, tools, or init tweaks are often gated as **image features** (exact names vary by distro/release). Treat them as **policy switches**, not random variables.

### 8.4 Lab 6 — Custom image

Create `recipes-core/images/mycourse-image.bb` that:

- `require`s or `inherit`s appropriately from a minimal base image for your release.
- Adds two packages you need (for example, `strace` and your favorite tiny shell utility).

**Done when:** `bitbake mycourse-image` produces a new deployable artifact and you can identify its size difference vs minimal.

---

## 9. Module 7 — Kernel, bootloader, device tree (integrator level)

### 9.1 What Yocto expects you to know

At this stage you are not required to be a kernel maintainer. You *are* responsible for:

- Selecting the right **kernel provider** / recipe for your BSP.
- Carrying **board device trees** and any **kernel config fragments** your hardware needs.
- Understanding **where** boot artifacts come from in the deploy directory.

### 9.2 Typical customization paths

- **Config fragments** — preferred for maintainable kernel option changes when supported.
- **Patches** — for drivers, DTS fixes, or backports (with review and upstreaming plan).
- **Out-of-tree modules** — sometimes the right boundary for proprietary or fast-moving drivers.

### 9.3 Lab 7 — Trace boot artifacts

For your `MACHINE`, list which tasks produce:

- Kernel image / device tree blobs (if applicable).
- Bootloader binary(ies).

**Done when:** you can open the deploy directory and point to each file the flashing script would use.

---

## 10. Module 8 — SDK and application workflow

### 10.1 Why the SDK exists

The image build proves the system fits together. The **SDK** gives application developers a cross-toolchain and sysroot-matched headers/libs **without** building the whole world in one tree.

### 10.2 Standard pattern

- Build `meta-toolchain` or the image-specific SDK target your docs recommend.
- Install the SDK shell script output into a predictable path.
- `source` the environment script and build your app with the provided `CC`, `PKG_CONFIG_PATH`, etc.

### 10.3 Lab 8 — Cross-compile “hello”

Cross-compile a trivial C program against the SDK sysroot and run it on the target image.

**Done when:** you trust the SDK version string matches the image you flashed (no silent ABI skew).

---

## 11. Module 9 — Debugging builds like an engineer

### 11.1 Read failures from the top of the log

When BitBake reports failure:

1. Identify the **recipe** and **task** (`do_compile`, `do_configure`, …).
2. Open the **task log** file path printed in the error.
3. Scroll to the **first** compiler or configuration error, not the last cascading line.

### 11.2 High-signal commands (conceptual)

- `bitbake -g <target>` — dependency graph artifacts (learn where dependency surprises hide).
- `devtool` (when available in your workflow) — modify sources in a controlled workspace.

### 11.3 Common failure classes

| Symptom | Often means |
|---------|-------------|
| Fetch failures | Network, proxy, missing `SRC_URI` checksum, retired URLs |
| Patch rejects | Version drift; branch pin wrong |
| CMake/autotools missing deps | `DEPENDS` incomplete; wrong `PACKAGECONFIG` |
| Rootfs size explosions | accidental `-dev` packages in image |

### 11.4 Lab 9 — Break it on purpose

Remove a dependency in a *throwaway* layer copy, rebuild, and practice tracing the log back to root cause. Revert the change.

**Done when:** you can articulate *why* the failure belongs to fetch vs compile vs packaging.

---

## 12. Module 10 — Licenses, compliance, and supply-chain hygiene

### 12.1 Why this matters in hardware products

Your firmware image is a **bundle of third-party copyrights**. Regulated customers and acquirers will ask for:

- **Bill of materials** for software.
- **License texts** and offer source where required.

### 12.2 Practical habits

- Enable license manifest generation appropriate to your release (names and variables evolve—follow current docs).
- Treat `LICENSE` and `LIC_FILES_CHKSUM` in recipes as **first-class review items**.
- Pin branches or use mirrors for anything that must be rebuildable in five years.

### 12.3 Lab 10 — Inspect the manifests

Generate your image and locate the license manifest artifacts. Pick three packages and verify:

- SPDX or license file recorded.
- Version corresponds to what you thought you shipped.

---

## 13. Module 11 — Performance, caching, and CI

### 13.1 Shared state (`sstate`) in one sentence

`sstate` caches **task outputs** so clean builds skip work that has not changed—when configured correctly.

### 13.2 CI design goals

- **Deterministic branches** (release streams), not “moving main” surprises.
- **Separate download and sstate** directories on shared storage with locking discipline.
- **Pin** layers with `SRCREV` / manifest tools appropriate to your process.

### 13.3 Lab 11 — Measure a rebuild

Time a no-op rebuild with warm cache vs after wiping tmp—but **keep downloads**. Record the delta.

**Done when:** you can explain what you would never delete in CI vs what is safe to scrub.

---

## 14. Capstone projects

Pick one as a “certificate you can demo.”

**A. Productized minimal image**

- Custom layer, custom image, serial console login, SSH optional, documented flash steps.

**B. Carry a hardware fix**

- Device tree tweak for a peripheral **you** can test (LED, I2C sensor, UART), with before/after photos or logs.

**C. Reproducible release bundle**

- `repo` manifest or pinned submodule set, README for exact checkout, CI job that emits image + SDK + license manifest.

---

## 15. Glossary and quick reference

| Term | Plain English |
|------|----------------|
| BitBake | Task executor: reads metadata, runs compile/package/image steps with parallelism |
| Recipe | Build instructions for one piece of software |
| Layer | Versioned bundle of metadata + configuration |
| Image | Root filesystem + kernel/boot artifacts for a product configuration |
| MACHINE | Hardware/BSP selection |
| DISTRO | Policy selection across images (where configured) |
| Sysroot | Headers/libs representing “what exists on target” for linking |
| BSP | Board Support Package metadata: machine configs, kernels, boot firmware glue |

**Commands to memorize slowly (not all on day one)**

- `source oe-init-build-env`
- `bitbake <target>`
- `bitbake-layers show-layers`

---

## 16. Further reading

Official sources (always prefer these over random tutorials):

- [Yocto Project Documentation](https://docs.yoctoproject.org/) — release-specific procedures and variables.
- [Yocto Project Quick Build](https://docs.yoctoproject.org/brief-yoctoprojectqs/index.html) — shortest supported path to a first image.
- [BitBake User Manual](https://docs.yoctoproject.org/bitbake.html) — when you need precise semantics.

Book-length depth (optional):

- *Embedded Linux Systems with the Yocto Project* — practical, widely used as a second pass after your first build.

Related roadmap note: for **Jetson-specific** production BSP practice, see the Jetson platform module *Orin-Nano-Yocto-BSP-Production* in **Phase 4 Track B**; this course is intentionally **vendor-neutral**.

---

*Course maintainer tip:* when documentation and the internet disagree, trust the **docs matching your exact Yocto release branch**—metadata syntax and variable names do evolve.
