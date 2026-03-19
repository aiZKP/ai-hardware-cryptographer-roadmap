# OrinCraw — getting started on Jetson

Checklist for bringing up **OrinCraw** on **Jetson Orin Nano 8GB** (dev kit first, custom PCB later). Full spec: [../Guide.md](../Guide.md).

## 1. Base system

- [ ] Flash **JetPack 5.1.2** (L4T) to **512GB NVMe** (default OrinCraw storage; boot + rootfs + models + logs/OTA) for **initial bring-up**
- [ ] Plan upgrade to **JetPack 6.2.1** for the production-aligned stack; rebuild inference engines, ESP-Hosted host driver, and containers for the new kernel/L4T (see Guide §3 *JetPack / L4T version strategy*)
- [ ] Track **JetPack 7.x** for when NVIDIA **officially supports** your target module/carrier; treat as a major migration with full regression
- [ ] First boot: create user, set hostname to **`orincraw`** (or `orincraw-<room>`)
- [ ] Enable **SSH** (keys only for production), apply updates
- [ ] Install **NVIDIA Container Toolkit** if using Docker for inference
- [ ] Reserve **`/data`** layout:
  ```bash
  sudo mkdir -p /data/{models,skills,logs,config}
  sudo chown -R "$USER:$USER" /data
  chmod 700 /data/config
  ```

## 2. ESP32-C6 + ESP-Hosted (WiFi/BT to Linux)

- [ ] Wire **SPI** (and power/ground) per Espressif **ESP-Hosted** docs for your carrier
- [ ] Flash **ESP-Hosted** slave firmware matching host driver version on **ESP32-C6**
- [ ] Build or install **ESP-Hosted-NG** host driver for your **L4T kernel**
- [ ] Confirm **`wlan0`** (or equivalent) appears; configure with **NetworkManager** or `wpa_supplicant`
- [ ] Optional: bring up **second UART** for LED/button **DeviceService** (not for IP)

## 3. Audio

- [ ] Verify capture (mic array / I2S) and playback (speaker) at ALSA/PipeWire level
- [ ] Document device names for Wake/STT/TTS containers or services

## 4. OpenClaw Gateway

- [ ] Install **Node.js ≥ 22** (arm64)
- [ ] Install OpenClaw: `npm install -g openclaw@latest` (or build from source)
- [ ] Run gateway: `openclaw gateway --port 18789` (test)
- [ ] Install daemon: `openclaw onboard --install-daemon` (persistent)
- [ ] Store config under **`/data/config`** or symlink from `~/.openclaw` as desired

## 5. Local inference services

- [ ] Deploy STT / LLM / TTS (containers or systemd) with **`runtime: nvidia`** where needed
- [ ] Register tools/skills in OpenClaw pointing at local service URLs
- [ ] Load **one** resident LLM; measure RAM with **`tegrastats`** under full load

## 6. Web UI & mDNS

- [ ] Confirm **`http://orincraw.local`** (or your hostname) resolves on LAN
- [ ] Complete setup wizard: WiFi (if used), privacy, optional **BYOK** keys in `/data/config`

## 7. OTA & security (before field use)

- [ ] Signed artifacts; staged update + rollback (see Guide §8)
- [ ] Firewall: only required ports open; no password SSH

## 8. Next steps

- [ ] Run **benchmark checklist** in Guide §10
- [ ] Track **milestone** list in Guide §11 through **custom PCB** bring-up
