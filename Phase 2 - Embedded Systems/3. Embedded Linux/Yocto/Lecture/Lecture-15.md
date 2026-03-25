# Lecture 15 — Glossary and quick reference

**Course:** [Yocto guide](../Guide.md) | **Phase 2 — Embedded Linux, Yocto**

**Previous:** [Lecture 14](Lecture-14.md) | **Next:** [Lecture 16 — Further reading](Lecture-16.md)

---

## Glossary

| Term | Plain English |
|------|----------------|
| BitBake | Task executor: reads metadata, runs compile/package/image steps with parallelism |
| Recipe | Build instructions for one piece of software |
| Layer | Versioned bundle of metadata + configuration |
| Image | Root filesystem plus kernel/boot artifacts for a product configuration |
| MACHINE | Hardware/BSP selection |
| DISTRO | Policy selection across images (where configured) |
| Sysroot | Headers/libs representing what exists on target for linking |
| BSP | Board Support Package metadata: machine configs, kernels, boot firmware glue |

## Commands to memorize slowly (not all on day one)

- source oe-init-build-env
- bitbake with your image or recipe target
- bitbake-layers show-layers

---

**Previous:** [Lecture 14](Lecture-14.md) | **Next:** [Lecture 16 — Further reading](Lecture-16.md)
