# C++ and SIMD

Part of [Phase 1 section 4 hub](../Guide.md).

Goal: Modern C++ for numerics and systems, plus how scalar loops map to vector instructions on CPUs.

## 1. Modern C++ for systems and numerics

- Core: value vs reference, const/constexpr, enums, namespaces.
- Memory: RAII, rule of five/zero, smart pointers; avoid heap in hot paths.
- Move semantics and large buffers.
- Templates: function templates, std::vector, std::array.
- STL algorithms: copy, transform, reduce patterns.
- Errors: exceptions vs codes; std::optional / expected where available.

Resources: A Tour of C++ (Stroustrup); cppreference.com; compile with -Wall -Wextra.

## 2. SIMD on CPUs

- Why SIMD: many lanes per instruction; bridges to GPU SIMT thinking.
- Autovectorization: simple loops; compiler reports (-fopt-info-vec, MSVC /Qvec-report).
- Intrinsics: x86 AVX2/AVX-512, ARM NEON when you need explicit vectors.
- std::simd where your toolchain supports it.

Profiling: perf, VTune, Instruments to see memory vs compute bound.

## Next

[OpenMP and OneTBB](../OpenMP%20and%20OneTBB/Guide.md)
