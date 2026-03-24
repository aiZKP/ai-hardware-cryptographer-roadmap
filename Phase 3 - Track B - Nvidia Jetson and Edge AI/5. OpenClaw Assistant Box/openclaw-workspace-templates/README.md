# OpenClaw workspace templates (reference copies)

These files mirror the **OpenClaw “workspace root”** pattern: the agent reads **markdown at the repo/workspace root** for identity, memory, and periodic checks. Copy what you need into **your actual OpenClaw workspace** (often `~/.openclaw` or a dedicated project folder—not necessarily this git repo).

| File | Role |
|------|------|
| [**BOOTSTRAP.md**](BOOTSTRAP.md) | First-run script: conversation to fill identity, then **delete** this file. |
| [**IDENTITY.md**](IDENTITY.md) | Agent persona metadata (name, vibe, emoji, avatar path). |
| [**USER.md**](USER.md) | Who the human is—update over time; **not a dossier**. |
| [**HEARTBEAT.md**](HEARTBEAT.md) | Periodic checks when heartbeat polling is enabled; **empty** = skip / minimal API use. |
| [**TOOLS.md**](TOOLS.md) | **Local** notes (SSH aliases, room names, TTS prefs)—skills stay generic. |

**Elsewhere in the same ecosystem (not all duplicated here):**

- **AGENTS.md** — procedures: session startup (`SOUL.md`, `USER.md`, `memory/…`), memory rules, red lines, group-chat behavior.
- **SOUL.md** — values and tone (helpful without filler, boundaries, continuity via files).
- **memory/YYYY-MM-DD.md** — daily raw log; **MEMORY.md** — curated long-term (main session only per typical rules).

**OrinClaw:** product requirements stay in [../Guide.md](../Guide.md). These templates are for **OpenClaw agent workspace** hygiene on the box or your dev machine.

**Privacy:** Do not commit filled **USER.md** / **IDENTITY.md** / **memory/** to a **public** repo if they contain real names or infra details.
