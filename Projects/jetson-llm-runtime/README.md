# jetson-llm

Memory-first LLM inference runtime for NVIDIA Jetson Orin.

**Target hardware:** Orin Nano Super 8 GB (SM 8.7, 102 GB/s, 77 TOPS)
**Not supported:** x86, discrete GPUs, Windows, macOS — Jetson only.

## Why

Existing runtimes are not designed for 8 GB unified memory:

- **llama.cpp** — portable but generic CUDA kernels, no Jetson memory awareness
- **TensorRT-LLM** — fast but heavy, designed for datacenter A100/H100
- **jetson-llm** — memory-first, power-aware, Orin-tuned kernels, zero-allocation inference

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   jetson-llm                          │
│                                                       │
│  Serving Layer (OpenAI-compatible REST API)           │
│    POST /v1/chat/completions | GET /health            │
│                                                       │
│  Engine (GGUF load → prefill → decode → sample)      │
│    transformer_layer() × N_layers per token           │
│    Memory guard + thermal check per token             │
│                                                       │
│  CUDA Kernels (SM 8.7 tuned)                          │
│    gemv_q4 | fused_rmsnorm | flash_attn | rope        │
│    swiglu | softmax | fp16↔int8                       │
│                                                       │
│  Memory Manager                                       │
│    MemoryBudget | OOMGuard | KVCachePool | ScratchPool│
│                                                       │
│  Jetson HAL                                           │
│    PowerState | ThermalState | LiveStats | JetsonInfo  │
└──────────────────────────────────────────────────────┘
```

## Project Status

### Completed (✅)

| Component | Files | Lines | Status |
|-----------|-------|-------|--------|
| **Memory Manager** | budget.cpp, kv_cache.cpp, pool.cpp | ~300 | ✅ Implemented, tested |
| **Jetson HAL** | power.cpp, thermal.cpp, sysinfo.cpp | ~250 | ✅ Reads sysfs, tested |
| **CUDA Kernels** | 6 .cu files | ~500 | ✅ Implemented, 5 correctness tests pass |
| **GGUF Config Parser** | model.cpp (load_gguf_config) | ~80 | ✅ Reads model architecture |
| **GGUF Tensor Parser** | model.cpp (parse_tensor_infos) | ~150 | ✅ Parses tensor name/shape/offset |
| **Weight Mapping** | model.cpp (load_and_map_weights) | ~80 | ✅ Maps tensor names → struct pointers |
| **Tokenizer** | tokenizer.cpp | ~160 | ✅ Reads GGUF vocab, encode/decode |
| **Sampling** | sample.cpp | ~120 | ✅ Top-k, top-p, temperature, repeat penalty |
| **Transformer Forward** | decode.cpp (transformer_layer) | ~100 | ✅ Wires all 12 ops per layer |
| **Decode Loop** | decode.cpp (generate) | ~80 | ✅ Prefill + decode + streaming |
| **CLI** | main.cpp | ~120 | ✅ Interactive + single prompt + OOM pre-check |
| **HTTP Server** | http_server.cpp, main_server.cpp | ~250 | ✅ /health, /v1/chat/completions, /v1/models |
| **Scripts** | setup, bench, profile | ~180 | ✅ First-time setup, benchmark, nsys profiling |
| **Tests** | test_memory, test_kernels, test_model_load | ~280 | ✅ Memory, 5 kernel tests, 8 model load tests |
| **Total** | **30 files** | **~4,200** | |

### Known Bugs (🐛 — fix on Jetson hardware)

| # | Bug | Location | Impact | Fix |
|---|-----|----------|--------|-----|
| 1 | **GGUF KV skip may miscalculate offsets** | `model.cpp` parse_tensor_infos() | Tensor data pointers may be wrong → garbage output | Test with real GGUF, verify first tensor offset matches file layout |
| 2 | **Residual connection not chained** | `decode.cpp` transformer_layer() | Attention output and FFN output residuals not accumulated correctly: should be `x = residual + attn_out` then `x2 = x + ffn_out` | Add explicit residual add kernels between attention and FFN |
| 3 | **Embedding memcpy direction** | `decode.cpp` decode_step() | `tok_embd` is in mmap'd host memory, `cudaMemcpyAsync` uses DeviceToDevice | Change to `cudaMemcpyHostToDevice` or let unified memory handle it |
| 4 | **Missing include** | `decode.cpp` | Calls `munmap()` without `#include <sys/mman.h>` | Add include |
| 5 | **CUDA graph skeleton** | `decode.cpp` build_cuda_graph() | Graph capture body is empty — no actual kernels captured | Implement after verifying decode loop works without graph |
| 6 | **Attention output accumulator simplified** | `attention.cu` | `acc[d % 4]` loses data when head_dim > 4 × blockDim | Use proper register-tiled accumulator per output dimension |
| 7 | **No FP32 logit GEMV** | `decode.cpp` decode_step() | Uses FP16 GEMV for logits then converts — should use FP32 output for numerical stability | Add gemv_q4_fp32out kernel variant |
| 8 | **Tokenizer greedy encode is O(V×L)** | `tokenizer.cpp` encode() | Slow for large vocab (128K) — linear scan per character | Build hash map of token→id at load time |

### Not Yet Implemented (📋)

| Feature | Priority | Effort | Notes |
|---------|----------|--------|-------|
| Fix bugs #1–4 | **Critical** | 1 day | Required for first tokens to flow |
| Fix bug #6 (attention accumulator) | **Critical** | 1 day | Wrong output without this |
| Fix bugs #5, #7, #8 | Medium | 1 day | Performance + correctness |
| Streaming SSE in server | Medium | 0.5 day | Server currently returns full response |
| CUDA graph for decode | Low | 1 day | ~1ms saving per forward pass |
| Chat template formatting | Low | 0.5 day | `<|user|>...<|assistant|>` wrapping |
| Multi-turn conversation KV reuse | Low | 1 day | Don't re-prefill system prompt |
| INT8 KV scale storage in cache | Low | 0.5 day | Per-head scales for dequant |
| Tensor Core WMMA for prefill GEMM | Low | 2 days | Batch matmul for prompt processing |
| systemd service file | Low | 0.5 day | Auto-start on boot |

### Milestone Roadmap

```
v0.1 — First Tokens (on real Jetson)
  □ Fix bugs #1–4 (build + run without crash)
  □ Fix bug #6 (correct attention output)
  □ test_model_load passes with real GGUF
  □ Generate coherent text from Llama 3.2 3B Q4_K_M

v0.2 — Benchmark Baseline
  □ bench.sh produces tok/s numbers
  □ profile.sh identifies top 3 kernel bottlenecks
  □ Compare against stock llama.cpp (same model, same hardware)
  □ Fix bug #7 (FP32 logits)
  □ Fix bug #8 (tokenizer hash map)

v0.3 — Performance Target
  □ >20% faster than stock llama.cpp on decode
  □ CUDA graph for decode loop
  □ Streaming SSE in server
  □ Memory-stable over 1000+ tokens (no growth)

v0.4 — Production Ready
  □ Chat template support
  □ Multi-turn conversation
  □ systemd service for auto-start
  □ 24-hour stability test (no OOM, no crash, no thermal shutdown)
  □ Document tested models + performance table
```

## Build

```bash
# First-time setup on Jetson
./scripts/setup_jetson.sh

# Or manual build
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="87" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Run

```bash
# Validate model loading (run this first!)
./build/test_model_load models/Llama-3.2-3B-Q4_K_M.gguf

# Interactive chat
./build/jetson-llm -m models/Llama-3.2-3B-Q4_K_M.gguf -i

# Single prompt
./build/jetson-llm -m models/Llama-3.2-3B-Q4_K_M.gguf -p "Hello, Jetson"

# API server
./build/jetson-llm-server -m models/Llama-3.2-3B-Q4_K_M.gguf -p 8080

# Benchmark
./scripts/bench.sh models/Llama-3.2-3B-Q4_K_M.gguf

# Profile
./scripts/profile.sh models/Llama-3.2-3B-Q4_K_M.gguf
```

## Test

```bash
# Memory subsystem (no GPU needed)
./build/test_memory

# CUDA kernel correctness (needs Jetson GPU)
./build/test_kernels

# Full model load pipeline (needs GGUF file)
./build/test_model_load model.gguf
```

## File Map

```
jetson-llm-runtime/                      ~4,200 lines
├── CMakeLists.txt                        Build (aarch64 guard, SM 87)
├── README.md                             This file
├── include/
│   ├── jllm.h                            Master header + Orin constants
│   ├── jllm_memory.h                     MemoryBudget, OOMGuard, KVCachePool, ScratchPool
│   ├── jllm_jetson.h                     PowerState, ThermalState, LiveStats
│   ├── jllm_kernels.h                    Kernel API (tile sizes, block sizes)
│   └── jllm_engine.h                     Engine, ModelWeights, LayerWeights, Tokenizer
├── src/
│   ├── main.cpp                          CLI entry (probe → check → load → generate)
│   ├── main_server.cpp                   Server entry
│   ├── memory/
│   │   ├── budget.cpp                    /proc/meminfo → MemoryBudget
│   │   ├── kv_cache.cpp                  Tiered GPU/CPU pool with eviction
│   │   └── pool.cpp                      Bump allocator (zero malloc in inference)
│   ├── jetson/
│   │   ├── power.cpp                     sysfs GPU/EMC/CPU freq, nvpmodel
│   │   ├── thermal.cpp                   Thermal zones, adaptive backoff
│   │   └── sysinfo.cpp                   System probe + tegrastats-style live
│   ├── kernels/
│   │   ├── gemv_q4.cu                    INT4 dequant-fused GEMV (38% of time)
│   │   ├── fused_norm.cu                 RMSNorm + residual (2× less traffic)
│   │   ├── attention.cu                  Flash attention decode (online softmax)
│   │   ├── rope.cu                       Rotary embedding (fused Q+K)
│   │   ├── softmax.cu                    Numerically stable softmax
│   │   └── convert.cu                    FP16↔INT8 + SwiGLU activation
│   ├── engine/
│   │   ├── decode.cpp                    Transformer forward pass + gen loop
│   │   ├── model.cpp                     GGUF parser + mmap + tensor mapping
│   │   ├── sample.cpp                    Top-k, top-p, temp, repeat penalty
│   │   └── tokenizer.cpp                GGUF vocab + greedy encode/decode
│   └── server/
│       └── http_server.cpp               OpenAI-compatible REST (raw sockets)
├── scripts/
│   ├── setup_jetson.sh                   First-time Jetson config
│   ├── bench.sh                          Benchmark with system state
│   └── profile.sh                        Nsight Systems kernel profiling
└── tests/
    ├── test_memory.cpp                   Memory subsystem (3 tests)
    ├── test_kernels.cu                   Kernel correctness (5 tests)
    └── test_model_load.cpp              Full loading pipeline (8 tests)
```

## License

MIT
