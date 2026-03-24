# C++ and Parallel Computing (Phase 1 section 4)

Goal: SIMD on CPU, multi-thread frameworks, CUDA SIMT, and portable OpenCL before Phase 3 neural networks.

Placement: After Operating Systems and before Phase 3 Neural Networks.

## Sub-tracks

| Order | Track | Guide |
|------|--------|--------|
| 1 | C++ and SIMD | [C++ and SIMD/Guide.md](C++%20and%20SIMD/Guide.md) |
| 2 | OpenMP and OneTBB | [OpenMP and OneTBB/Guide.md](OpenMP%20and%20OneTBB/Guide.md) |
| 3 | CUDA and SIMT | [CUDA and SIMT/Guide.md](CUDA%20and%20SIMT/Guide.md) |
| 4 | OpenCL | [OpenCL/Guide.md](OpenCL/Guide.md) |

CUDA needs NVIDIA GPU + toolkit for hands-on; OpenCL can use CPU or many GPUs; SIMD and OpenMP run on a normal PC or WSL.

Prerequisites: C from OS work; Phase 1 section 3 helps for threads and VM.

## Roadmap

- Phase 1 section 2: memory hierarchy and SIMD width.
- Phase 3: tensors map to these execution models.
- Phase 4 Track B Jetson: CUDA with power limits.
- Phase 4 Track A Xilinx: HLS/RTL; OpenCL here is the portable API layer.

## Next phase

[Phase 3 — Neural Networks](../../Phase 3 - Artificial Intelligence/Neural Networks/Guide.md)
