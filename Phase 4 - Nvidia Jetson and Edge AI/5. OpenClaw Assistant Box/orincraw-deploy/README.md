# OrinCraw deployment templates

- **[docker-compose.yml](docker-compose.yml)** — skeleton Compose `name: orincraw`; uncomment services when your L4T images are ready (pin tags per JetPack: **5.1.2** → **6.2.1** → **7** when supported—see Guide §3). Validate syntax:  
  `docker compose --profile template-only config`
- **[Getting-started-Jetson.md](Getting-started-Jetson.md)** — bring-up checklist.

Full product spec: [../Guide.md](../Guide.md).
