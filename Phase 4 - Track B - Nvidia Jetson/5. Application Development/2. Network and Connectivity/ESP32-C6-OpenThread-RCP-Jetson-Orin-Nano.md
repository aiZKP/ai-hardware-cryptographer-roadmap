# ESP32-C6 OpenThread RCP on Jetson Orin Nano - Project Guide

> **Goal:** Bring up a **second ESP32-C6** as an **OpenThread Radio Co-Processor (RCP)** for the **Jetson Orin Nano 8GB Developer Kit**, so the Jetson can run **OpenThread Border Router (OTBR)** or **OpenThread Daemon** while keeping your first ESP32-C6 dedicated to **ESP-Hosted Wi-Fi/BLE**.

**Hub:** [Network and Connectivity](Guide.md)  
**Related local guides:** [ESP32-C6 ESP-Hosted over SPI on Jetson Orin Nano](ESP32-C6-ESP-Hosted-SPI-Jetson-Orin-Nano.md) · [Peripheral Access](../1.%20Peripheral%20Access/Guide.md)

---

## 1. Why this project matters

Once your Jetson already has:

- `wlan0` from **ESP-Hosted** over SPI
- `hci0` from the same ESP32-C6 for BLE

the clean next step for **Thread** is not to overload that same chip. The cleaner architecture is:

- keep the first ESP32-C6 as the Jetson's Wi-Fi/BLE coprocessor
- add a **second ESP32-C6** dedicated to **802.15.4 / Thread**
- let the Jetson run the host-side OpenThread stack

This follows the standard **RCP design** from OpenThread:

- the **host processor** runs the OpenThread stack
- the **RCP** only handles the Thread radio / MAC layer
- host and RCP talk over **Spinel**

That model is a good fit for Jetson because the Linux host is always on and already powerful enough to run OTBR and other edge services.

---

## 2. Target architecture

```text
Jetson Orin Nano
  |
  |-- SPI -> ESP32-C6 #1 -> ESP-Hosted -> wlan0 + hci0
  |
  |-- UART (recommended first) -> ESP32-C6 #2 -> OpenThread RCP
  |                                 ^
  |                                 |
  |                             Spinel protocol
  |
  +--> Linux host side
        |-- otbr-agent   (for Thread Border Router)
        |-- or ot-daemon (for lighter RCP host use)
        |
        +--> wpan0
```

**Target outcome**

- Jetson keeps using `wlan0` from the first ESP32-C6 as its normal Wi-Fi interface
- Jetson gains a Thread radio path through `wpan0`
- `otbr-agent` or `ot-daemon` can talk to the second ESP32-C6 over Spinel
- the second ESP32-C6 is dedicated to Thread RCP use and is not shared with ESP-Hosted

---

## 3. Why use a second ESP32-C6

This guide intentionally avoids the single-chip “do everything on one ESP32-C6” path.

Why:

- **ESP-Hosted** is a Linux Wi-Fi/BLE coprocessor solution
- **OpenThread RCP** is a different host/co-processor model
- both want control of the same radio and host transport path
- the integration effort to merge them into one stable firmware is much higher than using two chips

Espressif officially documents:

- **ESP32-C6** supports **RCP mode**
- the RCP transport can be **SPI or UART**
- for Wi-Fi-based Thread Border Router products, a **dual-SoC** architecture is recommended for better coexistence behavior

So this guide uses the safer engineering split:

- **ESP32-C6 #1** for Jetson Wi-Fi/BLE
- **ESP32-C6 #2** for Thread RCP

---

## 4. Hardware and software prerequisites

### Hardware

- Jetson Orin Nano 8GB Developer Kit
- your existing **ESP32-C6 #1** already working with ESP-Hosted over SPI
- a **second ESP32-C6 dev board**
- jumper wires for a direct UART link between Jetson and the second ESP32-C6
- USB cable for flashing and monitoring the second ESP32-C6
- optionally, one more Thread-capable device for validation:
  - another ESP32-C6
  - ESP32-H2
  - a Matter-over-Thread end device

### Software

- JetPack 6.x / L4T 36.x on Jetson
- ESP-IDF installed on a Linux build/flash machine
- OpenThread Border Router host software on Jetson:
  - `ot-br-posix` if you want a real Thread Border Router
  - or OpenThread POSIX `ot-daemon` if you want a lighter RCP host setup

### Recommended first transport

Although ESP-IDF says RCP can use **SPI or UART**, this guide uses the Jetson's **40-pin header UART1** as the first real host transport.

Why:

- less risk of colliding with the SPI bus already used by ESP-Hosted
- it gives you a stable Jetson device path on this validated setup: **`/dev/ttyTHS1`**
- it reflects the real host/RCP wiring model instead of hiding it behind a USB-UART bridge

You can still use the ESP board's USB connection for:

- flashing
- serial monitor logs
- power during bring-up

You can optimize to SPI later if you need lower latency.

---

## 5. The official model you are implementing

OpenThread's **RCP design** means:

- the OpenThread core lives on the **host processor**
- the radio chip runs a minimal controller firmware
- host and controller talk through **Spinel**

On Jetson, that means:

- **Jetson** runs `otbr-agent` or `ot-daemon`
- **ESP32-C6 #2** runs the **`ot_rcp`** firmware from ESP-IDF

OpenThread's own coprocessor docs describe this as an **RCP design**, and OT Daemon is explicitly documented as the POSIX-side component for RCP setups.

For the backbone / infrastructure side, you have multiple valid choices on Jetson:

- `l4tbr0` if you want to use the Jetson USB device networking path to a host PC
- `wlan0` if the Jetson reaches the network through your first ESP32-C6 and ESP-Hosted
- `enP8p1s0` if you later use Ethernet as the OTBR backbone

On your current setup, **`l4tbr0` is the right choice for USB networking**, not `usb0` or `usb1`, because those interfaces are members of the `l4tbr0` bridge.

---

## 6. Build and flash the ESP32-C6 RCP firmware

Use the **ESP-IDF `ot_rcp` example** for the second ESP32-C6.

On your Linux build host:

```bash
cd $IDF_PATH/examples/openthread/ot_rcp

# Select the chip once
idf.py set-target esp32c6

# Optional: inspect settings
idf.py menuconfig

# Build
idf.py build
```

Flash it to the second ESP32-C6:

```bash
# Use the actual serial port for your board
# On many ESP32-C6 dev boards with CP210x this is /dev/ttyUSB0
idf.py -p /dev/ttyUSB0 flash monitor
```

### Important `menuconfig` choices for this guide

If the `ot_rcp` menu shows:

```text
[*] Configure RCP UART pin manually
(4)     The number of RX pin
(5)     The number of TX pin
```

that matches the direct Jetson UART wiring documented below.

For this guide:

- keep **`Configure RCP UART pin manually`** enabled
- set **RCP RX pin = `4`**
- set **RCP TX pin = `5`**
- leave **external coexist wire** disabled for first bring-up

That means:

- Jetson **TX** must go to ESP **GPIO4** (`RCP RX`)
- Jetson **RX** must come from ESP **GPIO5** (`RCP TX`)

### Port naming note

Do not hardcode `/dev/ttyACM0` just because some upstream OTBR examples show it.

On Espressif dev boards:

- one board may show up as `/dev/ttyUSB0`
- another may show up as `/dev/ttyACM0`

Use the actual port that belongs to **ESP32-C6 #2** on your machine.

If you are using an Espressif board with a CP210x bridge, `/dev/ttyUSB0` is common.

This USB serial port is for:

- flashing the `ot_rcp` image
- reading ESP boot and log output

It is **not** the same thing as the Jetson's real UART transport used later by `otbr-agent`.

---

## 7. Enable UART1 on the Jetson and wire it to the RCP

The Jetson Orin Nano 40-pin header exposes **UART1** as:

| Function | Jetson pin | Linux device | Direction |
|---|---:|---|---|
| UART1_TXD | `8` | `/dev/ttyTHS1` on your current image | Jetson -> ESP |
| UART1_RXD | `10` | `/dev/ttyTHS1` on your current image | ESP -> Jetson |
| Ground | `6` or `9` or `14` | -- | common reference |

For the ESP32-C6 RCP side in this guide:

| Function | ESP32-C6 pin | Direction |
|---|---:|---|
| RCP RX | `GPIO4` | Jetson TX -> ESP RX |
| RCP TX | `GPIO5` | ESP TX -> Jetson RX |
| Ground | `GND` | common reference |

So the exact wiring is:

```text
Jetson pin 8  (UART1_TXD) -> ESP32-C6 GPIO4
Jetson pin 10 (UART1_RXD) <- ESP32-C6 GPIO5
Jetson GND                -> ESP32-C6 GND
```

### Jetson UART prerequisites

On Jetson, the main thing that usually blocks use of the header UART is `nvgetty`, which can claim the user UART device.

Check the device first:

```bash
ls -l /dev/ttyTHS1
```

Then disable the serial getty if it is active:

```bash
sudo systemctl stop nvgetty
sudo systemctl disable nvgetty
```

Verify it is no longer active:

```bash
systemctl status nvgetty
```

### Basic Jetson UART configuration

Configure the port to the baud rate you intend to use with the RCP host link:

```bash
sudo stty -F /dev/ttyTHS1 460800 cs8 -cstopb -parenb raw -echo
```

This guide uses `460800` because that is a common `ot_rcp` / OTBR pairing on Espressif examples. If you later choose a different baud in your OpenThread configuration, keep the Jetson and ESP sides aligned.

### Electrical warning

The Jetson 40-pin UART is **3.3 V logic only**.

Do not connect it to:

- RS-232 voltage levels
- 5 V UART
- any external adapter that drives outside 3.3 V logic levels

For an ESP32-C6 dev board, direct 3.3 V UART wiring is fine.

### Optional sanity test

Before involving the ESP board, you can do a Jetson loopback test:

1. temporarily short **pin 8** to **pin 10**
2. run:

```bash
sudo cat /dev/ttyTHS1 &
echo "JETSON_UART_OK" | sudo tee /dev/ttyTHS1
```

If the text echoes back, the Jetson side UART path is alive.

---

## 8. Install OTBR on the Jetson

If you want a real Thread Border Router on the Jetson, use **`ot-br-posix`**.

```bash
sudo apt update
sudo apt install -y git

git clone --depth=1 https://github.com/openthread/ot-br-posix
cd ot-br-posix

./script/bootstrap
INFRA_IF_NAME=l4tbr0 ./script/setup
```

Why `INFRA_IF_NAME=l4tbr0` on your current Jetson:

- OTBR needs a **backbone / infrastructure interface**
- your Jetson currently exposes the USB networking path as the bridge **`l4tbr0`**
- `usb0` and `usb1` are members of that bridge, so `l4tbr0` is the real backbone interface

If you later want the backbone to be:

- ESP-Hosted Wi-Fi, use `INFRA_IF_NAME=wlan0`
- Ethernet, use `INFRA_IF_NAME=enP8p1s0` once that link is active

After installation:

```bash
sudo service otbr-agent status
```

The official OTBR native install guide shows the agent running like this at a high level:

```text
/usr/sbin/otbr-agent -I wpan0 -B wlan0 spinel+hdlc+uart:///dev/ttyACM0
```

For your setup, the important part is the **`-B <backbone>`** choice. On your current Jetson USB networking path, that backbone should be **`l4tbr0`**.

---

## 9. Point OTBR at the real ESP32-C6 RCP port

OTBR uses `/etc/default/otbr-agent` to define the Radio URL.

Edit it:

```bash
sudoedit /etc/default/otbr-agent
```

Set the RCP path and baud rate. For the **direct Jetson header UART** path in this guide:

```bash
OTBR_AGENT_OPTS="-I wpan0 -B l4tbr0 spinel+hdlc+uart:///dev/ttyTHS1?uart-baudrate=460800"
```

If you intentionally choose a USB-serial path instead, replace `/dev/ttyTHS1` with the real USB serial device such as `/dev/ttyUSB0` or `/dev/ttyACM0`.

If you later move the OTBR backbone to Wi-Fi, change `-B l4tbr0` to `-B wlan0`.

Then restart the service:

```bash
sudo systemctl restart otbr-agent
sudo systemctl status otbr-agent
sudo journalctl -u otbr-agent -n 100 --no-pager
```

Why `460800`:

- Espressif's Thread BR FAQ notes that OTBR often defaults to `115200`
- but the Espressif `ot_rcp` example commonly uses **`460800`**
- if the baud rate is wrong, host/RCP communication will fail even though the serial device exists

---

## 10. Validate the host/RCP link first

Before forming a Thread network, prove that the Jetson can talk to the RCP reliably.

Check that the expected Jetson UART device exists:

```bash
ls -l /dev/ttyTHS1
```

Check OTBR service health:

```bash
sudo service otbr-agent status
sudo journalctl -u otbr-agent -n 100 --no-pager
ip link show wpan0
```

What success looks like:

- `otbr-agent` is `active (running)`
- `wpan0` exists
- logs do not show repeated Spinel timeouts

If `wpan0` does not appear, stop there and fix the RCP link before trying to form a network.

---

## 11. Form a Thread network on the Jetson

Once OTBR is healthy, use `ot-ctl` on the Jetson.

```bash
sudo ot-ctl state
sudo ot-ctl dataset init new
sudo ot-ctl dataset commit active
sudo ot-ctl ifconfig up
sudo ot-ctl thread start
sudo ot-ctl state
```

Typical progression:

- first `disabled`
- then `detached`
- finally `leader` if this is the first Thread node in the new network

Useful follow-up commands:

```bash
sudo ot-ctl dataset active
sudo ot-ctl ipaddr
sudo ot-ctl netdata show
```

At this point, the Jetson is no longer just “connected to an RCP.” It is actively running the OpenThread host stack.

### How to read the real OTBR and `ot-ctl` output

The most useful habit in this project is to read the host-side state in layers instead of looking for one magic success line.

For this validated Jetson setup, the first important line is:

```text
Radio Co-processor version: openthread-esp32/... esp32c6
```

That line means the Linux host successfully talked to the ESP32-C6 over:

* `spinel+hdlc+uart`
* `/dev/ttyTHS1`
* `460800` baud

If that line is missing and `wpan0` never appears, the problem is still in the transport layer: wrong UART device, wrong baud, wrong wiring, bad RCP image, or an RCP reset.

Once the link is healthy, the next useful snapshot is usually:

```bash
sudo ot-ctl extaddr
sudo ot-ctl rloc16
sudo ot-ctl ipaddr
sudo ot-ctl state
```

For example, a real session may show:

```text
extaddr: fac5eb4acbada19e
rloc16: a400
ipaddr:
  fd3f:d825:5faf:9782:0:ff:fe00:a400
  fd3f:d825:5faf:9782:8b89:772a:39cd:737a
  fe80::f8c5:eb4a:cbad:a19e
state: detached
```

Read that output like this:

* `extaddr` is the 64-bit IEEE 802.15.4 radio identity.
* `rloc16` is the Thread mesh locator assigned inside the partition logic.
* `fd...ff:fe00:a400` is the RLOC IPv6 address derived from the mesh-local prefix and `rloc16`.
* `fd...8b89:...` is the Mesh-Local EID, a more stable mesh-local identity.
* `fe80::...` is the normal IPv6 link-local address.

This is the key nuance: **you can have valid Thread addresses and still be `detached`**. That means the dataset is present and the host/RCP stack is alive, but the node has not yet completed attachment to a partition.

### What the common OTBR log lines mean

These log lines are the most important ones to recognize:

```text
Mle-----------: Send Link Request (ff02::2)
MeshForwarder-: Sent IPv6 UDP msg ... dst:[ff02::2]:19788
Settings------: Read NetworkInfo {rloc:0xa400, extaddr:..., role:leader, ...}
BorderAgent---: Registering service OpenThread BorderRouter #A19E _meshcop._udp
```

How to interpret them:

* `Mle-----------` means the Thread control plane is actively trying to discover or attach.
* `MeshForwarder-` proves the radio path is transmitting real Thread traffic.
* `Settings------` shows persisted OpenThread state being read or written.
* `BorderAgent--- ... _meshcop._udp` shows the commissioning-facing Border Agent service being registered by OTBR.

One subtle line confuses many people:

```text
Settings------: Read NetworkInfo { ... role:leader, ... }
```

This does **not** prove the node is currently leader. It only means the stack read previously saved network state that remembered a leader role. The live role still comes from `sudo ot-ctl state`.

### Why `detached` can still be a healthy intermediate state

In a fresh lab setup, the expected state progression is:

* `disabled`
* `detached`
* `leader` for a one-node partition, or `router` / `child` when joining an existing mesh

So `detached` is not the same as "the UART is broken." In your actual logs, it appeared together with:

* a real `Radio Co-processor version` line
* `wpan0` creation
* valid `extaddr`, `rloc16`, and `ipaddr` output
* MLE attach attempts in the logs

That combination means the host/RCP link is already working. The remaining work is in **Thread attachment and partition formation**, not low-level serial bring-up.

### How to read the attach-attempt logs

Lines like these:

```text
Attach attempt 8, AnyPartition
Send Parent Request to routers
Send Parent Request to routers and REEDs
Attach attempt 8 unsuccessful, will try again in 32.128 seconds
```

mean the node is alive and behaving like a Thread node, but it has not yet completed the attach process. This is a protocol-state problem, not automatically a transport problem.

If you instead see:

```text
Wait for response timeout
no response from RCP during initialization
```

that points back to the host/RCP path and you should debug the UART, baud, RCP image, or board stability first.

### Session-socket and restart messages

Two more lines are easy to over-interpret:

```text
P-Daemon------: Session socket is ready
P-Daemon------: Daemon read: Connection reset by peer
```

In normal use, those often just mean that `ot-ctl` connected to OTBR's control socket and then exited. They are not automatically a radio failure.

Similarly, after restarting `otbr-agent`, it is normal to see:

* `wpan0` exist but still be `state DOWN`
* `sudo ot-ctl state` return `disabled`

until you run:

```bash
sudo ot-ctl dataset init new
sudo ot-ctl dataset commit active
sudo ot-ctl ifconfig up
sudo ot-ctl thread start
```

For this Jetson project, the known-good host-side baseline is:

* RCP transport: `spinel+hdlc+uart`
* serial device: `/dev/ttyTHS1`
* baud: `460800`
* OTBR backbone: `l4tbr0`

---

## 12. Optional lighter path: OT Daemon instead of OTBR

If you want an RCP host path without the full border-router stack, use **OpenThread Daemon**.

According to OpenThread's official coprocessor docs:

- `ot-daemon` is the POSIX service for RCP designs
- it uses a Spinel Radio URL just like OTBR

Build it:

```bash
git clone --depth=1 https://github.com/openthread/openthread
cd openthread

./script/bootstrap
./script/cmake-build posix -DOT_DAEMON=ON
```

Run it against the real RCP:

```bash
./build/posix/src/posix/ot-daemon \
  'spinel+hdlc+uart:///dev/ttyTHS1?uart-baudrate=460800'
```

In another terminal:

```bash
./build/posix/src/posix/ot-ctl state
```

Use this path if:

- you want to validate host/RCP behavior first
- you do not yet need a full Thread Border Router
- you want a smaller debugging surface than OTBR

If you choose the USB-serial path instead of Jetson header UART, substitute the real USB device path.

---

## 13. Common failure modes

### `otbr-agent` is running, but `wpan0` is missing

Usually means:

- wrong serial device
- wrong baud rate
- RCP firmware is not actually `ot_rcp`
- `nvgetty` is still owning the user UART device

Check:

```bash
sudo journalctl -u otbr-agent -n 100 --no-pager
```

### Spinel timeout warnings

Typical causes:

- wrong UART baud rate
- unstable USB serial path
- wrong serial port after replug or reflashing
- Jetson UART1 is not actually free for application use

Espressif's FAQ shows this family of symptom as:

```text
Wait for response timeout
```

### `/dev/ttyTHS1` exists, but OTBR still cannot talk to the RCP

Usually means one of:

- Jetson pin `8/10` are not actually cross-wired to ESP `GPIO4/5`
- TX/RX are swapped incorrectly
- `nvgetty` still owns the port
- the ESP side UART pins in `menuconfig` do not match the actual wiring

For this guide, the correct direct wiring is:

```text
Jetson pin 8  -> ESP GPIO4
Jetson pin 10 <- ESP GPIO5
```

### Confusing the two ESP32-C6 boards

Keep them clearly separated:

- **ESP32-C6 #1** = ESP-Hosted Wi-Fi/BLE over SPI
- **ESP32-C6 #2** = Thread RCP over UART

Do not flash `ot_rcp` onto the board currently providing `wlan0`.

### OTBR installs correctly but routing does not work

Make sure the backbone interface is right:

- `wlan0` if Jetson reaches the network through ESP-Hosted
- `eth0` only if you intentionally use Ethernet as the Thread BR backbone

---

## 14. Good next steps after first bring-up

- add a second Thread device and have it join the network
- validate `ipaddr`, `ping`, and service discovery over Thread
- keep the RCP on UART first, then evaluate SPI later only if needed
- if you want Matter later, keep this Jetson + RCP split and build on top of it

---

## 15. References

### Official upstream references

- [OpenThread co-processor designs](https://openthread.io/platforms/co-processor)
- [OpenThread Daemon (RCP host mode)](https://openthread.io/platforms/co-processor/ot-daemon)
- [OpenThread Border Router native install](https://openthread.io/guides/border-router/build-native)
- [ESP-IDF OpenThread on ESP32-C6](https://docs.espressif.com/projects/esp-idf/en/latest/esp32c6/api-guides/openthread.html)
- [ESP-IDF Thread / esp_openthread API reference](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/network/esp_openthread.html)
- [ESP Thread Border Router SDK](https://docs.espressif.com/projects/esp-thread-br/en/latest/)
- [ESP Thread BR FAQ: OTBR with `ot_rcp`](https://docs.espressif.com/projects/esp-thread-br/en/latest/qa.html#using-ot-rcp-with-otbr-ot-br-posix)
- [ESP32-C6 RF coexistence guidance](https://docs.espressif.com/projects/esp-idf/en/stable/esp32c6/api-guides/coexist.html)

### Local roadmap references

- [ESP32-C6 ESP-Hosted over SPI on Jetson Orin Nano](ESP32-C6-ESP-Hosted-SPI-Jetson-Orin-Nano.md)
- [Network and Connectivity hub](Guide.md)
- [Orin Nano GPIO/SPI/I2C/CAN deep-dive](../../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-GPIO-SPI-I2C-CAN/Guide.md)
