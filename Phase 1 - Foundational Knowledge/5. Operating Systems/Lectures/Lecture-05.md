# Lecture 5: Bootstrap Process, Bootloaders, Firmware

**Source:** [CS124 Lec05](https://users.cms.caltech.edu/~donnie/cs124/lectures/CS124Lec05.pdf)

---

## Bootstrap Process

**Problem:** CPU starts executing from a fixed address (e.g., 0xFFFFFFF0 on x86). How does the OS get loaded?

**Answer:** Chain of increasingly sophisticated software:

1. **Firmware** (BIOS/UEFI) — initializes hardware, finds boot device
2. **Bootloader** — loads OS kernel from disk
3. **Kernel** — initializes system, starts init process

---

## PC BIOS (Legacy)

- **Stored in ROM/Flash** — runs on power-on
- **POST** (Power-On Self-Test) — check hardware
- **Loads first sector** of boot device (512 bytes) into memory at 0x7C00
- **Jumps** to that address — **boot sector** code runs

### Master Boot Record (MBR)

- **First 512 bytes** of disk
- **446 bytes:** Boot code
- **64 bytes:** Partition table (4 entries)
- **2 bytes:** Magic 0x55 0xAA

### Chain Loading

- **MBR boot code** too small for full bootloader
- **Chain load:** MBR loads **Volume Boot Record (VBR)** of active partition
- **VBR** loads second-stage bootloader (e.g., GRUB)

---

## IA32 Bootstrap Process

1. **CPU starts** in real mode (16-bit), at 0xFFFFFFF0
2. **BIOS** runs, loads MBR to 0x7C00, jumps
3. **Bootloader** (e.g., GRUB) loads kernel, switches to protected mode
4. **Kernel** takes over — sets up paging, IDT, etc.

---

## Plug-and-Play

- **PnP:** BIOS/OS discovers devices dynamically
- **Resource allocation** — IRQs, I/O ports, DMA channels
- **Avoids** manual jumpers and conflicts

---

## Intel MultiProcessor Specification

- **MP tables:** Describe multiple CPUs, buses, interrupts
- **Used by OS** to find and start APs (Application Processors)
- **BSP** (Bootstrap Processor) starts first; starts APs

---

## ACPI (Advanced Configuration and Power Interface)

- **Power management** — sleep states, CPU frequency
- **Device configuration** — replaces PnP BIOS
- **Tables** in memory — ACPI tables describe hardware
- **OS** reads tables, manages power

---

## UEFI (Unified Extensible Firmware Interface)

### Replaces BIOS

- **32/64-bit** from the start (no real mode)
- **Modular** — drivers loaded from EFI System Partition (ESP)
- **GUID Partition Table (GPT)** — replaces MBR; supports >2TB disks, >4 partitions

### Preboot Environment

- **UEFI** can run applications (e.g., boot manager, shell)
- **EFI System Partition:** FAT32 partition with bootloader (e.g., `\EFI\BOOT\bootx64.efi`)
- **Secure Boot** — verify signatures of boot components

### GPT (GUID Partition Table)

- **LBA 0:** Protective MBR (for compatibility)
- **LBA 1:** GPT header
- **LBA 2–33:** Partition entries (128 bytes each)
- **Supports** many partitions, large disks

---

## Summary

| Component | Role |
|-----------|------|
| BIOS | Legacy firmware, loads MBR |
| MBR | First sector, partition table, chain load |
| UEFI | Modern firmware, GPT, Secure Boot |
| Bootloader | Load kernel from disk into memory |
| ACPI | Power management, device config |
