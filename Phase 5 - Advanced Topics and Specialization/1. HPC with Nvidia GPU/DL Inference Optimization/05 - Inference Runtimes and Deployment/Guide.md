# 05 — Inference Runtimes & Deployment Targets

**Order:** Fifth. After graph, kernels, compiler, and quantization (01–04), you deploy and measure in production-like settings.

**Role target:** [DL Inference Optimization Engineer](../../../../../README.md#the-four-career-steps) · **MTS Kernels** (deployment, production reliability, measurable outcomes).

---

## Why this comes fifth

Your kernels and optimizations only matter if they run correctly in a runtime and meet latency/throughput goals. This unit covers the main inference runtimes and how to measure and compare them.

---

## 1. Runtimes

* **TensorRT** — Engine build, plugins, dynamic shapes, DLA. How your kernels and graph optimizations show up in the engine.
* **ONNX Runtime** — Execution providers (CUDA, TensorRT, OpenVINO). Graph optimizations and provider selection.
* **Triton Inference Server** — Batching, model concurrency, metrics. Serving multiple models and dynamic batching.

---

## 2. Measurable outcomes

* **Latency** — p50, p99; what to measure (single request, batch).
* **Throughput** — QPS, tokens/s; how batch size and concurrency affect it.
* **Memory footprint** — Peak GPU/system memory; impact of batching and precision.
* **Methodology** — Reproducible benchmarks; A/B comparison (e.g. before/after kernel change, or TensorRT vs ONNX Runtime).

---

## Resources

* [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/best-practices/)
* [Triton Inference Server](https://github.com/triton-inference-server/server)
* [MLPerf Inference](https://mlcommons.org/benchmarks/inference/) — Reference benchmarks and methodology.

---

## Projects

1. **Runtime comparison** — Deploy the same model with ONNX Runtime and TensorRT (same hardware). Compare latency and throughput; document configuration and measurement method.
2. **Triton server** — Set up a minimal Triton server with dynamic batching. Measure QPS vs batch size and document how batching affects latency and throughput.
3. **Benchmark report** — For one model and one runtime, produce a one-page benchmark report: latency (p50/p99), throughput, memory, and exact environment (GPU, driver, runtime version).

---

## Next

→ **[06 — tinygrad Deep Dive](../06%20-%20tinygrad%20Deep%20Dive/Guide.md)** (optional) — Hands-on compiler/kernel interface: IR, scheduler, backends, and adding a simple optimization.
