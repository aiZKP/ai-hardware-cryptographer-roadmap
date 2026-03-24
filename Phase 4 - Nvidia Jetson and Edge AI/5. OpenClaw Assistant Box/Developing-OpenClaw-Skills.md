# Developing skills for OpenClaw

This note ties together **theory**, **authoring steps**, and **where it lives in code** (from the [openclaw/openclaw](https://github.com/openclaw/openclaw) tree). Official product docs: [docs.openclaw.ai](https://docs.openclaw.ai) (navigation often mirrors `docs/` in the repo).

## Theory: what a “skill” is

- A **skill** is a **folder** whose main contract is **`SKILL.md`**: YAML frontmatter (metadata) + Markdown instructions for the model.
- OpenClaw aligns with the **[Agent Skills](https://agentskills.io/specification)** idea: portable, file-based capability packs—not separate binaries unless you document calling them via `bash` or another tool.
- At runtime, eligible skills are turned into a **compact XML-ish list** in the **system prompt** (names, descriptions, paths). The model decides when to follow a skill’s instructions; there is no separate “skill executor” unless you wire **slash commands** to tools (see frontmatter options in the upstream skills doc).
- **Security**: treat skills as **prompt + optional scripts**. Third-party skills are **untrusted code**; use sandboxing for risky sessions and read skills before enabling. Secrets belong in `openclaw.json` / env, not in `SKILL.md`.

## Do you need to clone the OpenClaw repo?

| Goal | Clone repo? |
|------|-------------|
| Add a **personal / project** skill | **No.** Put files under `<workspace>/skills/<name>/`. |
| Install from **ClawHub** | **No.** Use `clawhub` CLI (see below). |
| **Publish** to ClawHub | **No** for the OpenClaw app; you publish the skill folder. |
| Change **bundled** skills or **core** loading | **Yes**, fork/clone [openclaw/openclaw](https://github.com/openclaw/openclaw). |

Upstream CONTRIBUTING points skill contributions to **[ClawHub](https://clawhub.com)** rather than the main repo.

## Authoring workflow (minimal)

1. **Directory**: `mkdir -p ~/.openclaw/workspace/skills/<skill-id>` (or your configured workspace; see `agents.defaults.workspace`).
2. **`SKILL.md`**: at minimum:

   ```markdown
   ---
   name: my_skill
   description: One line: what this skill does for the assistant.
   ---

   # My skill

   When the user …, do … (clear triggers and steps).
   ```

   Use **`{baseDir}`** in prose to mean “this skill’s folder” (OpenClaw documents this substitution).

3. **Optional**: scripts, reference files, or instructions to use existing tools (`bash`, `browser`, etc.).
4. **Reload**: restart the Gateway or rely on the skills **watcher** (`skills.load.watch`, default on) so changes apply on the next agent turn / session (see snapshot behavior below).

## Loading order and config (technical)

From upstream docs (`docs/zh-CN/tools/skills.md` mirrors English `tools/skills.md`):

1. **Bundled** skills (npm / app)
2. **`~/.openclaw/skills`** (managed / shared on machine)
3. **`<workspace>/skills`** (highest priority for name conflicts)

Also: **`skills.load.extraDirs`** — lowest priority. **`skills.entries.<key>`** can disable skills or inject `env` / `apiKey`. **`skills.allowBundled`** can whitelist bundled-only skills.

**Gating** uses `metadata` (single-line JSON in YAML) under `metadata.openclaw`: `requires.bins`, `requires.env`, `requires.config`, `os`, `always`, install hints, etc. If there is no `metadata.openclaw`, the skill is generally eligible (unless disabled in config).

**Snapshots**: eligible skills are snapshotted per session; heavy changes may need a **new session** or watcher refresh—see doc section “会话快照”.

**Sandbox**: host `env` from skills does not apply inside Docker sandboxes; use sandbox docker env or images (see `skills-config` doc).

## ClawHub CLI (install / publish)

- Install CLI: `npm i -g clawhub` (or pnpm).
- Typical: `clawhub search "…"`, `clawhub install <slug>`, `clawhub update --all`, `clawhub publish <path> …`, `clawhub sync`.

Install target defaults to `./skills` or falls back to the OpenClaw workspace; override with `--workdir` / `CLAWHUB_WORKDIR`.

## Where to read code in a clone

- **Merge load + prompt**: `src/agents/skills/workspace.ts` — uses `@mariozechner/pi-coding-agent` (`loadSkillsFromDir`, `formatSkillsForPrompt`), OpenClaw filtering, path compaction.
- **Frontmatter / OpenClaw metadata**: `src/agents/skills/frontmatter.ts`, `src/shared/frontmatter.js` (related).
- **Eligibility / config**: `src/agents/skills/config.ts`, `filter.ts`.
- **Bundled skills directory**: `src/agents/skills/bundled-dir.ts`; repo root **`skills/`** contains many examples (e.g. `skills/summarize/SKILL.md`).
- **Runtime hook-in**: `src/agents/pi-embedded-runner/skills-runtime.ts`, `src/agents/skills.ts` (re-exports).

## Hardware-attached skills (Jetson / OrinClaw-class carriers)

Skills can drive **real hardware** when Linux exposes devices and your skill’s instructions (or scripts) call **`bash`**, **`i2cget`/`i2cset`**, **GPIO sysfs** or **libgpiod**, **USB** nodes under `/dev`, or small **HTTP/REST** daemons you run on the host.

**For product owners and open-source developers**

1. **Hardware assumptions** — If your skill depends on **I2C, SPI, GPIO, or USB gadgets**, document the **exact Linux paths**, **kernel modules**, and **carrier pinout** your stack expects. Skills should use **`requires.bins`** / **`requires.env`** so they **fail clearly** when hardware is absent.
2. **Skill metadata** — Use OpenClaw **`metadata.openclaw`** / **`requires.bins`** (and similar) so skills **degrade gracefully** when `i2c-tools`, a kernel module, or a device node is missing—see upstream `tools/skills.md`.
3. **Security** — Hardware skills often run **privileged** helpers; keep **scopes narrow** (specific bus + address), avoid blanket `sudo` in instructions, and document **trust model** for third-party modules.
4. **Distribution** — Publish hardware-specific skills on **ClawHub** (or git) with a **Bill of Materials** for the add-on module and a **tested JetPack / image** revision.

**Relationship to OpenClaw upstream:** Nothing in the Agent Skills format is Jetson-specific; **portability** comes from documenting **OS-level assumptions** in `SKILL.md` and `requires.*`.

### Optional USB FT232H expansion kit (OrinClaw — **no custom carrier IC**)

OrinClaw can support **hardware lab skills** by documenting an **optional** commodity **FT232H** (or similar) **USB module** plugged into a **USB host** port — see **§2.9**; the carrier exposes **2× USB-A + 1× USB-C host** (**§2.10**); use **BT mouse** or a **powered hub** if you need more than three downstream devices. The product **does not** need an FT232H chip on the main PCB.

- **Host:** Linux **`ftdi_sio`** + stable **`/dev/ttyUSB*`** or **`pyftdi`** / `libftdi` URLs like `ftdi://ftdi:232h/…`.
- **Permissions:** `udev` rules for **VID/PID** and **`dialout`** (or a dedicated group); document in the skill or factory README.
- **Containers:** If the Gateway runs in Docker, pass **`--device /dev/ttyUSB0`** (or use a udev symlink) for skills that shell out to Python.
- **Skill design:** Put helper scripts under `{baseDir}`; use **`requires.bins`** for `python3` and declare **`pip`** deps in the skill README or install script; **never** assume a fixed `/dev/ttyUSB0` without udev `SYMLINK+=` for `FT232H`.

## Testing

- CLI: `openclaw agent --message "…"` with a prompt that should trigger the skill.
- Iterate on description + “when to use” sections so the model picks it up reliably.

## Further reading

- [Agent Skills specification](https://agentskills.io/specification)
- [OpenClaw README](https://github.com/openclaw/openclaw) (workspace paths, onboarding)
- ClawHub: [clawhub.com](https://clawhub.com)
