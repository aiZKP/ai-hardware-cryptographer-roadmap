# Peripheral Access

**Phase 4 — Track B — Module 5.1** · Application Development

> **Focus:** Access and control hardware peripherals on the **Jetson Orin Nano 8GB** from Linux userspace — GPIO, PWM, UART, SPI, I2C, CAN, USB, storage, and backlight. Every interface is shown with both sysfs/chardev and library approaches.

**Hub:** [5. Application Development](../Guide.md)

---

## Table of Contents

1. [GPIO (Linux)](#1-gpio-linux)
2. [GPIO Naming — Alphanumeric to Numeric Assignment](#2-gpio-naming--alphanumeric-to-numeric-assignment)
3. [PWM (Linux)](#3-pwm-linux)
4. [ADC (Linux)](#4-adc-linux)
5. [UART (Linux)](#5-uart-linux)
6. [SPI (Linux)](#6-spi-linux)
7. [I2C (Linux)](#7-i2c-linux)
8. [CAN (Linux)](#8-can-linux)
9. [USB Host Mode (Linux)](#9-usb-host-mode-linux)
10. [USB Device Mode (Linux)](#10-usb-device-mode-linux)
11. [NVMe / PCIe Storage (Linux)](#11-nvme--pcie-storage-linux)
12. [SD/MMC Card (Linux)](#12-sdmmc-card-linux)
13. [Backlight (Linux)](#13-backlight-linux)
14. [Projects](#14-projects)
15. [Resources](#15-resources)

---

## 1. GPIO (Linux)

Jetson Orin Nano exposes GPIOs through the **gpiochip** character device interface (`/dev/gpiochipN`). The legacy sysfs interface (`/sys/class/gpio/`) is deprecated — use `libgpiod` instead.

### libgpiod tools

```bash
# List all GPIO chips
gpiodetect

# Show all lines on a chip
gpioinfo gpiochip0

# Read a GPIO input from a real line offset
gpioget gpiochip0 43

# Set a GPIO output high on a real line offset
gpioset gpiochip0 43=1

# Monitor GPIO events (rising/falling edge) on a real line offset
gpiomon gpiochip0 43
```

### What `gpiodetect` means on Jetson

On a Jetson Orin Nano dev kit you will typically see something like:

```text
gpiochip0 [tegra234-gpio] (164 lines)
gpiochip1 [tegra234-gpio-aon] (32 lines)
```

- `gpiochip0` is the main Tegra234 GPIO controller
- `gpiochip1` is the **always-on** GPIO controller used for low-power and wake-related signals
- the number in parentheses is how many line offsets exist on that controller

For most 40-pin header work, you will usually spend your time on `gpiochip0`.

### How to read `gpioinfo`

`gpioinfo gpiochip0` prints one row per line offset. Example:

```text
line  35:      "PG.00" "Force Recovery" input active-low [used]
line  43:      "PH.00"       unused   input  active-high
line 131:      "PZ.01"  "interrupt"   input  active-high [used]
```

Read each field like this:

| Field | Meaning |
|-------|---------|
| `line 43` | The **chip-relative line offset**. This is the number you pass to `gpioget`, `gpioset`, and `gpiomon`. |
| `"PH.00"` | The SoC pad or line name exposed by the driver/device tree. |
| `unused` / `"Force Recovery"` / `"interrupt"` | The current consumer. `unused` means no driver or userspace process owns it right now. |
| `input` / `output` | Current direction. |
| `active-high` / `active-low` | Logical polarity. |
| `[used]` | The line is already requested by the kernel or another process. Do not reuse it casually. |

### What names like `PH.00` actually mean

Names like `PH.00`, `PG.00`, or `PZ.01` are **Tegra GPIO port/bit names**.

Read `PH.00` like this:

- `P` = GPIO pad group prefix used in NVIDIA's naming
- `H` = GPIO bank or port
- `00` = bit number inside that bank

So:

- `PH.00` means port `H`, bit `0`
- `PZ.01` means port `Z`, bit `1`
- `PQ.05` means port `Q`, bit `5`

This naming style is useful because it stays close to NVIDIA's low-level documentation, pinmux data, and board-support files.

The important point is that `PH.00` is **not**:

- a 40-pin header number
- a Linux `gpiochip` line number
- a user-friendly board label like `GPIO09`

It is the **SoC-side identity** of that signal.

### Why `gpioget gpiochip0 <line>` failed in your shell

`<line>` in the docs is a **placeholder**, not something you type literally.

In `bash`, angle brackets mean shell redirection. So:

```bash
gpioget gpiochip0 <line>
```

is interpreted as "read stdin from a file named `line`", which is why it fails.

Replace the placeholder with a real line offset from `gpioinfo`, for example:

```bash
# Read line 43 on gpiochip0
gpioget gpiochip0 43

# Drive line 43 high
gpioset gpiochip0 43=1

# Monitor line 43 for edges
gpiomon gpiochip0 43
```

Also note:

- `gpioget` is for **reading**
- `gpioset` is for **driving outputs**
- `gpioget gpiochip0 <line>=1` is invalid because `gpioget` does not set values

### Practical workflow on Jetson

Use this sequence when you are trying to identify a GPIO safely:

```bash
# 1. See which controllers exist
gpiodetect

# 2. Inspect the main controller
gpioinfo gpiochip0

# 3. Pick an UNUSED line only after checking pinmux and board function
gpioget gpiochip0 43
```

Before driving a line with `gpioset`, verify all of these:

- the pin is really muxed as GPIO
- it is not marked `[used]`
- it is not a boot, recovery, regulator, wake, or other board-critical signal
- you know what external hardware is attached to it

### Finding a line by name

If the line names are populated, `gpiofind` is often easier than scrolling through the full list:

```bash
gpiofind "PH.00"
```

That returns the owning chip and line offset for that named line.

### 40-pin header vs GPIO line: the easy mental model

This is the part that confuses most people at first:

- the **40-pin header** is the physical connector on the board
- a **header pin number** is just the metal pin position, such as **pin 7** or **pin 22**
- a **GPIO line** is the Linux-facing handle you use through `gpiochipN`
- the **pinmux** decides whether that physical pin is acting as GPIO, SPI, I2C, UART, PWM, or something else

Think of it as a translation chain:

```text
Physical header pin
  -> board signal label
  -> SoC pad / pinmux entry
  -> Linux-visible function
  -> gpiochip line offset (only if configured as GPIO)
```

On the dev kit, the 40-pin header gives you a convenient physical starting point. On a custom board, the same logic still applies even if the connector is completely different.

### Start from the header pin, not from the Linux line

When wiring hardware, always begin with the physical header.

Example:

- `Pin 7` means the 7th pin on the 40-pin header
- on the Orin Nano dev kit, that header position is labeled `GPIO09`
- that board signal maps to a Tegra pad defined in NVIDIA's pinmux data
- only after Linux boots and the pad is configured as GPIO do you get a usable `gpiochip` line for `gpioget` or `gpioset`

That is why this is wrong:

- "I found line 43 in `gpioinfo`, so I know what board pin it is"

You only know the board pin after you trace it back through the pinmux and header mapping.

### Three practical examples

| What you touch on the board | What it means |
|-----------------------------|---------------|
| `Pin 7` | Physical header position |
| `GPIO09` | Friendly board-level signal name for that header pin |
| `Pin 22` | Physical header position |
| `GPIO22` | Friendly board-level signal name for that header pin |
| `Pin 19` | Physical header position |
| `SPI1_MOSI` | That pin is normally part of the SPI bus, not a generic GPIO while SPI is enabled |

The important distinction:

- `GPIO09`, `GPIO15`, `GPIO22` are usually pins you may choose to use as GPIO in a project
- `SPI1_MOSI`, `SPI1_MISO`, `I2C1_SDA`, `UART1_TXD` are often already assigned to a peripheral role

If a pin is muxed to SPI, I2C, or UART, do not expect `gpioset` on that pin to behave like a normal GPIO.

### What pinmux actually changes

Pinmux does **not** move the physical pin.

Pinmux changes the function of the SoC pad behind that pin.

For one header position, the pad may be able to act as:

- GPIO
- SPI
- I2C
- UART
- PWM
- audio or other SoC-specific functions

So if you change pinmux, you are answering this question:

> "What job should this physical pin do after boot?"

Examples:

- make pin 19 act as `SPI1_MOSI`
- make pin 15 act as plain `GPIO`
- make pin 32 act as `PWM`

### The easiest way to think about pinmux spreadsheet work

When you later customize pinmux, use this order:

1. Pick the **physical header pin** you want to use
2. Confirm the **board signal name** on the Jetson header table
3. Confirm whether that pin must be **GPIO** or a dedicated peripheral like SPI/I2C/UART
4. Change the **pinmux** so the pad matches that intended role
5. Boot Linux and then discover the runtime `gpiochip` line with `gpioinfo` or `gpiofind`

Do not do it in reverse. Do not start from an arbitrary `gpiochip0` line number and try to design hardware from that.

### A useful rule for Jetson bring-up

Use each naming scheme for the job it is good at:

- **Header pin number** for wiring
- **Board signal name** for schematics and connector tables
- **SoC pad / pinmux name** for spreadsheet and device-tree work
- **`gpiochip` line offset** for Linux userspace tools

If you keep those four layers separate, pinmux customization becomes much easier to reason about.

### What changes on a fully custom carrier board

On a fully custom Jetson carrier, you may not have the 40-pin header at all.

That means one layer disappears:

- there may be no familiar `Pin 7`, `Pin 22`, `Pin 19` style reference

But the lower layers are still the same:

- the Jetson module still exposes the same SoC pads
- those pads still need pinmux configuration
- Linux still exposes GPIOs through `gpiochip`
- `gpioinfo`, `gpiofind`, `gpioget`, and `gpioset` still work the same way

So the translation chain becomes:

```text
Custom connector pin or schematic net
  -> board net name
  -> SoC pad / pinmux entry
  -> Linux-visible function
  -> gpiochip line offset (only if configured as GPIO)
```

### Dev kit vs custom carrier

| Dev kit world | Custom carrier world |
|---------------|----------------------|
| `Pin 7 on 40-pin header` | `J5 pin 12` or `camera trigger net` or `radio reset net` |
| `GPIO09` board label | your own schematic net name |
| Jetson-IO may help for simple cases | you usually own the full pinmux and device-tree configuration |
| easy visual pin lookup | you must trace the signal in schematics and layout |

### What stays the same on a custom board

These concepts do not change:

- the SoC pad name such as `PH.00`
- the pinmux decision: GPIO vs SPI vs I2C vs UART vs PWM
- the runtime `gpiochip` model in Linux
- the need to avoid reusing pads already assigned to critical functions

### What changes on a custom board

These are now your responsibility:

- connector naming
- schematic net naming
- which pads are actually routed out
- pull-ups, level shifting, buffers, and signal integrity
- whether a pad is safe for boot, wake, reset, interrupts, or power sequencing
- the final pinmux and device-tree configuration shipped in the BSP

### How to think about pinmux for a custom carrier

For a custom carrier, the correct workflow is:

1. choose the **SoC pad** based on product needs and NVIDIA design constraints
2. route that pad to the connector, device, or subsystem you want on your carrier
3. assign the pad's role in the **pinmux spreadsheet**
4. generate and integrate the pinmux/device-tree data into the BSP
5. boot Linux and confirm the runtime behavior with `gpioinfo` and the real peripheral tools

In other words:

- on the dev kit, you often start from the **header pin**
- on a custom carrier, you usually start from the **SoC pad and schematic net**

That is why learning both naming styles matters. The 40-pin header is convenient for bring-up, but the **SoC pad name** is the more durable reference once you move into real carrier-board engineering.

### How to read NVIDIA's pinmux spreadsheet

The NVIDIA pinmux spreadsheet is easier to understand if you split each row into three parts:

1. **What pin is this on the module?**
   Examples:
   - `SODIMM Pin #`
   - `Jetson SODIMM Signal Name`

2. **What is NVIDIA's default or reset-time view of this pin?**
   Examples:
   - `Jetson Orin NX and Nano Function`
   - `POR Pull Strength`
   - `POR Pin State`

3. **What do I want this pin to do on my board?**
   Examples:
   - `Customer Usage`
   - `Pin Direction`
   - `Req. Initial State`
   - `3.3V Tolerance Enable`
   - internal and external pull-up / pull-down values
   - `IO Block Voltage`

Short version:

- the **left side** tells you which module pin you are talking about
- the **middle** tells you the default/reset behavior
- the **right side** is your actual board configuration

### One row, explained simply

Take a row like this:

```text
106  SPI1_SCK  SPI1_SCK  50k  pd  GPIO3_PY.00  Input  Int PD  ...  1.8V
```

Read it like this:

- `106` = the physical **SODIMM pin number** on the Jetson module
- `SPI1_SCK` = the connector signal name on the module
- `50k pd` = NVIDIA's default pull / reset-state view for that pin
- `GPIO3_PY.00` = the function you selected in the spreadsheet for your board
- `Input` = direction you want after boot
- `Int PD` = requested initial bias or state
- `1.8V` = this pin belongs to a 1.8 V I/O voltage domain

So that row is really saying:

> "Module pin 106 exists as the signal `SPI1_SCK`, but on my board I want the pad behind it to behave as `GPIO3_PY.00`, as an input, with a pull-down, in a 1.8 V domain."

### What names like `GPIO3_PY.00` mean

When the selected function is a GPIO, NVIDIA often uses names like:

- `GPIO3_PY.00`
- `GPIO3_PEE.02`

Read `GPIO3_PY.00` as:

- `GPIO3` = GPIO controller instance
- `PY` = SoC GPIO port or bank name
- `00` = bit index inside that port

That is a **pinmux / SoC function name**, not a 40-pin header label and not a Linux `gpiochip` line number.

### Why this matters for custom carriers

On the developer kit, you can often think:

- "I want to use header pin 22"

On a custom carrier, you usually need to think:

- "I want to route module pin 106 to my connector"
- "I want that pad to behave as GPIO, not SPI"
- "I need the right pull, voltage domain, and initial state for my hardware"

That is exactly why the spreadsheet exists. It is not just a pin list. It is the **boot-time electrical and functional definition** of your carrier board.

### The safest way to use the spreadsheet

When editing NVIDIA's pinmux sheet, ask these questions in order:

1. Which **module pin / SODIMM signal** am I routing?
2. What **SoC function** do I actually want there: GPIO, SPI, I2C, UART, PWM, CAN, etc.?
3. Should it be **input**, **output**, or **bidirectional**?
4. What should it do **immediately at boot**?
5. Is this pin in a **1.8 V** or **3.3 V** domain?
6. Do I need **internal** or **external** pulls for my real board?

If you answer those six clearly, the spreadsheet becomes much easier to use.

### libgpiod in C

```c
#include <gpiod.h>

struct gpiod_chip *chip = gpiod_chip_open("/dev/gpiochip0");
struct gpiod_line *line = gpiod_chip_get_line(chip, 42);

/* Output */
gpiod_line_request_output(line, "my-app", 0);
gpiod_line_set_value(line, 1);

/* Input */
gpiod_line_request_input(line, "my-app");
int val = gpiod_line_get_value(line);

gpiod_chip_close(chip);
```

### libgpiod in Python

```python
import gpiod

chip = gpiod.Chip('gpiochip0')
line = chip.get_line(42)

# Output
line.request(consumer="my-app", type=gpiod.LINE_REQ_DIR_OUT)
line.set_value(1)

# Input
line.request(consumer="my-app", type=gpiod.LINE_REQ_DIR_IN)
val = line.get_value()
```

### Device tree GPIO configuration

GPIOs are assigned functions in the **pinmux device tree**. To use a pin as GPIO, it must be configured as GPIO (not an alternate function) in the pinmux spreadsheet. See [Module 2 — Custom Carrier Board](../../2.%20Custom%20Carrier%20Board%20Design%20and%20Bring-Up/Guide.md) for pinmux configuration.

---

## 2. GPIO naming — alphanumeric to numeric assignment

Jetson uses several GPIO naming schemes at once. They are easy to confuse, and they are **not interchangeable**.

### Finding the mapping

```bash
# Show all lines on the main chip
gpioinfo gpiochip0

# Show all lines on the always-on chip
gpioinfo gpiochip1

# Find a line by SoC pad name, if the line has a name
gpiofind "PH.00"

# Kernel debug view
sudo cat /sys/kernel/debug/gpio
```

`gpioinfo | grep -i "GPIO"` is often not very useful on Jetson, because many lines are named like `PA.00`, `PH.00`, `PQ.05`, or have board-specific labels such as `Force Recovery`, not literal strings like `GPIO09`.

### The four names you will see

| Name style | Example | Where you see it | What it means |
|------------|---------|------------------|---------------|
| Board/header label | `GPIO09`, `GPIO15`, `GPIO22` | Carrier schematics, 40-pin header tables, Jetson pinout diagrams | A board-facing label for a header pin or connector signal |
| SoC pad / port.pin | `PH.00`, `PQ.05`, `PZ.01` | `gpioinfo`, pinmux docs, low-level bring-up notes | The Tegra GPIO bank and bit name |
| `gpiochip` line offset | `43`, `105`, `131` | `gpioinfo`, `gpioget`, `gpioset`, `gpiomon` | The number libgpiod uses inside one chip |
| Legacy sysfs/global GPIO number | `348`, `391` | Older Jetson docs, `/sys/class/gpio` workflows | Deprecated numbering model; avoid for new code |

### Mental model

When using `libgpiod`, the address format is:

```text
gpiochipN + line offset
```

Example:

```bash
gpioget gpiochip0 43
```

That means:

- controller: `gpiochip0`
- line offset on that controller: `43`

It does **not** mean:

- 40-pin header pin 43
- sysfs GPIO 43
- board label `GPIO43`

### How this connects to the 40-pin header

The 40-pin header gives you a **board view** of the connector. `libgpiod` gives you a **Linux view** of whatever pads are currently configured as GPIO.

Those are related, but they are not the same table.

Use this quick reference:

| Question you are asking | Use this source first |
|-------------------------|-----------------------|
| "Which hole on the header do I wire to?" | 40-pin header pinout |
| "Should this pin be SPI, I2C, UART, PWM, or GPIO?" | pinmux spreadsheet / Jetson-IO |
| "What line number do I pass to `gpioget`?" | `gpioinfo` / `gpiofind` |
| "Why is this pin unavailable?" | `gpioinfo`, device tree, and current peripheral assignment |

### One worked example

Suppose you want to use header **pin 22** in a project.

You would think about it like this:

1. **Board view:** pin 22 on the 40-pin header is labeled `GPIO22`
2. **Design intent:** you want it to be a plain input for an interrupt or handshake signal
3. **Pinmux step:** configure the underlying pad as GPIO, not as some alternate peripheral
4. **Linux step:** after boot, use `gpioinfo` or `gpiofind` to locate the runtime line offset
5. **Userspace step:** use that discovered line offset with `gpioget`, `gpioset`, or `gpiomon`

That same flow is what you will use later for custom carrier work and pinmux-table customization.

### What to trust when numbers disagree

For modern userspace work on Jetson, prefer this order:

1. `gpioinfo` and `gpiofind` for the current Linux-visible line offsets
2. NVIDIA pinmux spreadsheet and board schematic for signal identity
3. 40-pin header tables for physical pin location
4. old sysfs numbers only when reading legacy documentation

Always verify on the actual device and JetPack release you are using. The visible line offsets and consumers can change across BSP versions.

---

## 3. PWM (Linux)

Orin Nano exposes PWM channels via the Linux PWM subsystem.

### sysfs interface

```bash
# Export PWM channel (chip 0, channel 0)
echo 0 > /sys/class/pwm/pwmchip0/export

# Configure: 1 kHz, 50% duty cycle
echo 1000000 > /sys/class/pwm/pwmchip0/pwm0/period      # ns
echo 500000  > /sys/class/pwm/pwmchip0/pwm0/duty_cycle   # ns
echo 1       > /sys/class/pwm/pwmchip0/pwm0/enable
```

### Common uses on Jetson

| Use case | PWM channel | Notes |
|----------|------------|-------|
| **Fan control** | Dedicated fan PWM | Controlled by `jetson_clocks` or device tree `pwm-fan` node |
| **LED brightness** | GPIO-capable PWM pin | Requires pinmux set to PWM function |
| **Servo motor** | GPIO-capable PWM pin | 50 Hz, 1–2 ms pulse width |
| **Backlight** | Display backlight PWM | See [Backlight](#13-backlight-linux) |

---

## 4. ADC (Linux)

The Jetson Orin Nano SoM has **limited ADC** capability — no general-purpose ADC pins are exposed on the carrier connector. For analog sensing:

### External ADC options

| ADC | Interface | Resolution | Channels | Common use |
|-----|-----------|-----------|----------|-----------|
| **ADS1115** | I2C | 16-bit | 4 | Voltage, current, temperature sensors |
| **MCP3008** | SPI | 10-bit | 8 | General-purpose analog input |
| **INA226** | I2C | 16-bit | 2 (V + I) | Power monitoring |

### Reading external ADC via IIO subsystem

If your external ADC has a Linux IIO driver:

```bash
# List IIO devices
ls /sys/bus/iio/devices/

# Read raw value
cat /sys/bus/iio/devices/iio:device0/in_voltage0_raw

# Read scale
cat /sys/bus/iio/devices/iio:device0/in_voltage0_scale
```

Voltage = raw * scale (in millivolts).

---

## 5. UART (Linux)

Orin Nano exposes multiple UART ports. The debug console uses `ttyTCU0` (Tegra Combined UART). Application UARTs appear as `/dev/ttyTHS*`.

### Available UARTs

| Device | Use | Baud rate |
|--------|-----|-----------|
| `/dev/ttyTCU0` | Debug console (bootloader + kernel) | 115200 |
| `/dev/ttyTHS0` | Application UART 0 | Configurable |
| `/dev/ttyTHS1` | Application UART 1 | Configurable |

### Using UART from userspace

```bash
# Quick test with minicom
minicom -D /dev/ttyTHS0 -b 115200

# Or with stty + cat
stty -F /dev/ttyTHS0 115200 cs8 -cstopb -parenb
echo "hello" > /dev/ttyTHS0
cat /dev/ttyTHS0
```

### Python (pyserial)

```python
import serial

ser = serial.Serial('/dev/ttyTHS0', 115200, timeout=1)
ser.write(b'hello\n')
data = ser.readline()
ser.close()
```

---

## 6. SPI (Linux)

SPI buses appear as `/dev/spidevX.Y` (bus X, chip-select Y).

### Enable SPI in device tree

SPI buses must be enabled in the pinmux and device tree. Use the pinmux spreadsheet to assign SPI function to the correct pins.

### spidev test

```bash
# Loopback test (connect MOSI to MISO)
spidev_test -D /dev/spidev0.0 -s 1000000 -v
```

### Python (spidev)

```python
import spidev

spi = spidev.SpiDev()
spi.open(0, 0)           # bus 0, CS 0
spi.max_speed_hz = 1000000
response = spi.xfer2([0x01, 0x02, 0x03])
spi.close()
```

---

## 7. I2C (Linux)

I2C buses appear as `/dev/i2c-N`.

### Scanning for devices

```bash
# Scan all addresses on bus 1
sudo i2cdetect -y -r 1
```

### Reading/writing registers

```bash
# Read register 0x00 from device 0x50 on bus 1
sudo i2cget -y 1 0x50 0x00

# Write 0xFF to register 0x01
sudo i2cset -y 1 0x50 0x01 0xFF
```

### Python (smbus2)

```python
from smbus2 import SMBus

with SMBus(1) as bus:
    data = bus.read_byte_data(0x50, 0x00)
    bus.write_byte_data(0x50, 0x01, 0xFF)
```

### Common I2C devices on Jetson carriers

| Device | Address | Purpose |
|--------|---------|---------|
| **EEPROM** (carrier ID) | 0x50–0x57 | Board identification |
| **INA3221** | 0x40–0x43 | Power monitoring (VIN, 3.3V, 5V) |
| **IMU** (BMI088, ICM-42688) | 0x68–0x69 | Inertial measurement (robotics) |
| **RTC** (DS3231) | 0x68 | Real-time clock (battery-backed) |

---

## 8. CAN (Linux)

CAN bus on Jetson requires an external CAN transceiver on the carrier board (the SoC has CAN controllers but no integrated PHY).

### Setup

```bash
# Configure CAN interface (500 kbps)
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up

# For CAN-FD
sudo ip link set can0 type can bitrate 500000 dbitrate 2000000 fd on
sudo ip link set can0 up
```

### Send and receive

```bash
# Send a frame
cansend can0 123#DEADBEEF

# Receive (dump all frames)
candump can0

# Generate test traffic
cangen can0
```

### Python (python-can)

```python
import can

bus = can.interface.Bus(channel='can0', interface='socketcan')

# Send
msg = can.Message(arbitration_id=0x123, data=[0xDE, 0xAD, 0xBE, 0xEF])
bus.send(msg)

# Receive
msg = bus.recv(timeout=1.0)
print(f"ID: {msg.arbitration_id:#x}, Data: {msg.data.hex()}")
```

---

## 9. USB host mode (Linux)

Jetson Orin Nano supports USB 3.2 Gen 2 (10 Gbps) and USB 2.0 host ports.

### Enumeration

```bash
# List connected USB devices
lsusb

# Show USB tree with speeds
lsusb -t

# Detailed device info
lsusb -v -d <vendor>:<product>
```

### Common USB peripherals

| Device type | Driver | Notes |
|------------|--------|-------|
| **USB camera** | UVC (`uvcvideo`) | Works out of the box, see [Multimedia](../4.%20Multimedia/Guide.md) |
| **USB serial** (FTDI, CP210x) | `ftdi_sio`, `cp210x` | Appears as `/dev/ttyUSB*` |
| **USB Ethernet** | `cdc_ether`, `r8152` | Appears as `eth*` or `usb*` |
| **USB storage** | `usb-storage` | Appears as `/dev/sd*` |
| **USB audio** | `snd-usb-audio` | ALSA device |

---

## 10. USB device mode (Linux)

The Orin Nano dev kit's USB-C port supports device (gadget) mode for flashing and development.

### USB gadget framework

```bash
# Load the USB gadget configfs
modprobe libcomposite

# Common gadgets:
# - g_ether: USB Ethernet gadget (device appears as network adapter on host)
# - g_serial: USB serial gadget (device appears as ttyACM on host)
# - g_mass_storage: USB mass storage gadget
```

### Typical use cases

| Gadget | Use case |
|--------|----------|
| **Ethernet (RNDIS/ECM)** | SSH into Jetson over USB without network cable |
| **Serial (ACM)** | Debug console over USB |
| **Mass storage** | Expose a partition or file as USB drive |

---

## 11. NVMe / PCIe storage (Linux)

Most Jetson Orin Nano deployments use NVMe SSD for the root filesystem (faster and more reliable than SD card).

### NVMe operations

```bash
# List NVMe devices
nvme list

# Check health
sudo nvme smart-log /dev/nvme0

# Benchmark
sudo fio --name=test --rw=randread --bs=4k --numjobs=4 \
    --size=1G --runtime=30 --time_based --filename=/dev/nvme0n1

# Check PCIe link speed
sudo lspci -vv | grep -A 20 "Non-Volatile"
```

### PCIe link verification

```bash
# Should show Gen3 x4 or Gen4 x4 depending on module
sudo lspci -vv | grep "LnkSta:"
```

If the link trains at a lower speed than expected, check trace routing (see [Module 2 — Carrier Board](../../2.%20Custom%20Carrier%20Board%20Design%20and%20Bring-Up/Guide.md)).

---

## 12. SD/MMC card (Linux)

SD card slot (if present on carrier) appears as `/dev/mmcblk*`.

```bash
# Check SD card info
sudo fdisk -l /dev/mmcblk1

# Mount
sudo mount /dev/mmcblk1p1 /mnt

# Check speed class
cat /sys/block/mmcblk1/device/speed_class
```

**Production note:** SD cards have limited write endurance. Use NVMe for root filesystem; reserve SD for optional data logging or field updates only.

---

## 13. Backlight (Linux)

If your carrier has an LCD with PWM-controlled backlight:

```bash
# List backlight devices
ls /sys/class/backlight/

# Get current brightness
cat /sys/class/backlight/<device>/brightness

# Get max brightness
cat /sys/class/backlight/<device>/max_brightness

# Set brightness (0 = off, max = full)
echo 128 > /sys/class/backlight/<device>/brightness
```

Configure the backlight PWM channel in the device tree for your display panel.

---

## 14. Projects

- **Sensor dashboard:** Read an I2C temperature sensor (e.g., TMP102) and a SPI ADC (e.g., MCP3008 for analog input), display values on a UART terminal at 1 Hz.
- **CAN bus monitor:** Build a CAN bus sniffer that logs all frames to a file with timestamps. Add filtering by arbitration ID.
- **GPIO interrupt counter:** Use `gpiomon` or libgpiod event monitoring to count rising edges on an input pin. Measure maximum event rate.
- **USB gadget network:** Configure the Jetson as a USB Ethernet gadget so a host laptop can SSH into it over a single USB-C cable.

---

## 15. Resources

| Resource | Description |
|----------|-------------|
| **Jetson Orin Nano Developer Kit User Guide** | Pin header mapping, UART/SPI/I2C bus assignments |
| **libgpiod documentation** | Character device GPIO API (replaces sysfs) |
| **Linux kernel SPI/I2C docs** | `Documentation/spi/` and `Documentation/i2c/` in kernel source |
| **SocketCAN documentation** | Linux CAN subsystem, `ip link`, `candump`, `cansend` |
| **NVMe CLI** | `nvme-cli` package for NVMe management and diagnostics |
| [2. Custom Carrier Board](../../2.%20Custom%20Carrier%20Board%20Design%20and%20Bring-Up/Guide.md) | Pinmux, peripheral validation checklist |
