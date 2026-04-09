# LLM Optimization on Jetson — From Cloud Techniques to Edge Reality

**Parent:** [ML and AI](../Guide.md)

> **Goal:** Take the optimization techniques used by cloud LLM platforms (vLLM, TensorRT-LLM, RunInfra, etc.) and adapt them to the extreme constraints of Jetson Orin Nano 8 GB — 51 GB/s bandwidth, shared memory, 15–40W power.

---

## 0. The Optimization Stack — Cloud vs Jetson

Cloud platforms like RunInfra, Together AI, and Fireworks AI deploy LLMs using a layered optimization stack. Every technique has a Jetson equivalent — but the priorities are reversed because Jetson is severely memory-bandwidth-bound rather than compute-bound.

```
Cloud GPU (H100 80GB, 3,350 GB/s, 989 TFLOPS):
  ┌─────────────────────────────────────────────┐
  │ 1. Quantization (FP8, AWQ 4-bit)           │ ← saves VRAM, improves throughput
  │ 2. FlashAttention-2                         │ ← saves SRAM, fuses memory ops
  │ 3. Fused Kernels (RMSNorm, rotary, SwiGLU) │ ← fewer kernel launches
  │ 4. PagedAttention (vLLM)                    │ ← KV cache memory efficiency
  │ 5. Speculative Decoding                     │ ← higher tokens/sec
  │ 6. Batching (continuous batching)           │ ← amortize compute over requests
  │ 7. Tensor Parallelism (multi-GPU)          │ ← scale beyond 1 GPU
  └─────────────────────────────────────────────┘

Jetson Orin Nano 8GB (51 GB/s, 40 TOPS):
  ┌─────────────────────────────────────────────┐
  │ 1. Quantization (INT4/INT8) ★★★★★          │ ← MANDATORY: model must fit in 5 GB
  │ 2. Model Selection ★★★★★                    │ ← choose models that fit (≤3B params)
  │ 3. KV Cache Management ★★★★                │ ← memory is the #1 constraint
  │ 4. FlashAttention / Fused Ops ★★★★         │ ← reduce bandwidth pressure
  │ 5. TensorRT-LLM Engine ★★★                 │ ← compiled, optimized execution
  │ 6. Speculative Decoding ★★★                │ ← higher tokens/sec within power budget
  │ 7. Batching ★★                              │ ← limited by memory, not compute
  └─────────────────────────────────────────────┘
  ★ = importance on Jetson (more ★ = more critical)
```

---

## 1. Quantization — The Most Important Optimization

On cloud GPUs, quantization is optional (saves cost). On Jetson, **quantization is mandatory** — without it, nothing fits.

### 1.1 Why Quantization Matters More on Jetson

```
Model: Llama 3.2 3B parameters

FP16:  3B × 2 bytes = 6.0 GB   ← won't fit (only ~5 GB free after OS/CMA)
INT8:  3B × 1 byte  = 3.0 GB   ← fits, but tight
INT4:  3B × 0.5 byte = 1.5 GB  ← fits comfortably, room for KV cache

Model: Phi-3 Mini 3.8B parameters

FP16:  3.8B × 2 bytes = 7.6 GB  ← impossible on 8 GB
INT4:  3.8B × 0.5 byte = 1.9 GB ← fits with room for context
```

### 1.2 Quantization Methods Ranked for Jetson

| Method | Bits | Quality loss | Speed on Jetson | When to use |
|--------|------|-------------|-----------------|-------------|
| **AWQ (Activation-Aware Weight)** | 4-bit | Very low | Fast (INT4 GEMM) | Best quality/size for Jetson |
| **GPTQ** | 4-bit | Low | Fast | Alternative to AWQ, well-supported |
| **INT8 PTQ (TensorRT)** | 8-bit | Minimal | Fastest | If model fits at INT8 |
| **FP8 (E4M3)** | 8-bit | Minimal | Fast (Ampere+) | When you need FP-like quality |
| **GGUF (llama.cpp)** | 2–8 bit | Mixed-precision | Good (CPU+GPU) | Easy deployment, any model |
| **SqueezeLLM** | 3-4 bit | Low | Moderate | Extreme compression |

### 1.3 AWQ Quantization (Recommended for Jetson)

AWQ preserves quality by protecting salient weight channels — the 1% of weights that matter most for output quality.

```python
# Quantize on your workstation (not on Jetson — too slow)
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_name = "microsoft/Phi-3-mini-4k-instruct"
quant_path = "phi3-mini-awq-int4"

model = AutoAWQForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# AWQ calibration (needs ~128 samples)
model.quantize(
    tokenizer,
    quant_config={
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,           # 4-bit weights
        "version": "GEMM"     # optimized GEMM kernel
    }
)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

### 1.4 GGUF with llama.cpp (Easiest Path)

llama.cpp runs on Jetson with CUDA support and handles mixed-precision quantization:

```bash
# On Jetson: install llama.cpp with CUDA
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)

# Download a pre-quantized GGUF model
# (Llama 3.2 3B in Q4_K_M = ~2 GB, good quality/size balance)
wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/\
Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Run inference
./build/bin/llama-cli \
    -m Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    -ngl 99 \           # offload all layers to GPU
    -c 2048 \           # context length
    -p "Explain how Jetson unified memory works:"
```

**GGUF quantization levels for Jetson:**

| Quantization | Bits | Size (3B model) | Quality | Recommended? |
|-------------|------|-----------------|---------|-------------|
| Q2_K | 2.6 | ~1.0 GB | Poor | Only if nothing else fits |
| Q3_K_M | 3.4 | ~1.3 GB | Acceptable | Memory-critical deployments |
| **Q4_K_M** | **4.5** | **~1.8 GB** | **Good** | **Best balance for Jetson** |
| Q5_K_M | 5.5 | ~2.2 GB | Very good | If memory allows |
| Q6_K | 6.6 | ~2.5 GB | Excellent | Best quality that fits |
| Q8_0 | 8.0 | ~3.0 GB | Near-FP16 | Only for small models |

---

## 2. Model Selection — Choose What Fits

The most important "optimization" is choosing the right model. A well-quantized small model beats a poorly-fitting large model every time.

### 2.1 Models That Fit on Orin Nano 8 GB

| Model | Params | INT4 size | Context | Tokens/sec (est.) | Use case |
|-------|--------|-----------|---------|-------------------|----------|
| **Llama 3.2 1B** | 1B | ~0.6 GB | 128K | ~30–40 | Lightweight chat, classification |
| **Llama 3.2 3B** | 3B | ~1.5 GB | 128K | ~15–25 | General chat, summarization |
| **Phi-3 Mini** | 3.8B | ~1.9 GB | 4K/128K | ~12–20 | Reasoning, code |
| **Gemma 2 2B** | 2B | ~1.0 GB | 8K | ~20–30 | Multilingual, general |
| **Qwen 2.5 3B** | 3B | ~1.5 GB | 32K | ~15–25 | Chinese + English |
| **StableLM 2 1.6B** | 1.6B | ~0.8 GB | 4K | ~25–35 | Compact, fast |
| **TinyLlama 1.1B** | 1.1B | ~0.6 GB | 2K | ~35–45 | Ultra-lightweight |

### 2.2 Models That DON'T Fit (Without Extreme Tricks)

| Model | Params | INT4 size | Why it doesn't fit |
|-------|--------|-----------|-------------------|
| Llama 3.1 8B | 8B | ~4.0 GB | Leaves <1 GB for KV cache + OS |
| Mistral 7B | 7B | ~3.5 GB | Tight, possible with short context |
| Qwen 2.5 7B | 7B | ~3.5 GB | Same — barely possible |
| Llama 70B | 70B | ~35 GB | Impossible on any Jetson |

> **7B models on 8 GB Jetson:** Technically possible with INT4 + 512-token context + minimal OS footprint, but impractical for production. Use Orin NX 16 GB or AGX Orin 64 GB for 7B+ models.

---

## 3. KV Cache Management — The Hidden Memory Consumer

During autoregressive generation, the KV cache stores key/value tensors for every token in the context. This can consume more memory than the model itself.

### 3.1 KV Cache Size Calculation

```
KV cache size = 2 × num_layers × num_kv_heads × head_dim × context_length × bytes_per_element

Llama 3.2 3B (INT8 KV cache, 2048 context):
  = 2 × 26 layers × 8 kv_heads × 128 head_dim × 2048 tokens × 1 byte
  = 2 × 26 × 8 × 128 × 2048
  = ~109 MB

Same model at 8192 context:
  = ~435 MB  ← significant on 8 GB!

Same model at 128K context:
  = ~6.8 GB  ← impossible, exceeds total free memory
```

### 3.2 KV Cache Optimization Techniques

| Technique | Memory saving | Quality impact | Jetson support |
|-----------|-------------|----------------|----------------|
| **INT8 KV cache** | 2× vs FP16 | Minimal | llama.cpp, TensorRT-LLM |
| **INT4 KV cache** | 4× vs FP16 | Small | llama.cpp (experimental) |
| **GQA (Grouped Query Attention)** | 4–8× vs MHA | None (model-level) | Built into modern models |
| **Sliding window attention** | Bounded | Loses long context | Mistral, some models |
| **KV cache eviction** | Bounded | Loses old context | Custom implementation |
| **PagedAttention (vLLM)** | No fragmentation waste | None | Not on Jetson (vLLM = server) |

**GQA is the most important:** Llama 3.2 uses GQA with 8 KV heads (vs 32 query heads). This means the KV cache is 4× smaller than traditional MHA. Always prefer GQA models on Jetson.

### 3.3 Context Length Budget

```
Orin Nano 8GB Memory Budget for LLM:

Total DRAM:                    8.0 GB
  - Firmware carveouts:       -0.4 GB
  - OS + kernel:              -0.5 GB
  - CMA:                      -0.5 GB (reduced for LLM workload)
  - CUDA runtime:             -0.3 GB
  ────────────────────────────────────
  Available for LLM:           6.3 GB

Model (Llama 3.2 3B INT4):   -1.5 GB
KV cache (INT8, ctx=2048):   -0.1 GB
Activation memory:            -0.2 GB
  ────────────────────────────────────
  Remaining:                   4.5 GB  ← room for longer context or larger model

Model (Phi-3 Mini INT4):     -1.9 GB
KV cache (INT8, ctx=4096):   -0.3 GB
Activation memory:            -0.3 GB
  ────────────────────────────────────
  Remaining:                   3.8 GB  ← still comfortable
```

---

## 4. FlashAttention and Fused Kernels

### 4.1 Why Fused Kernels Matter on Jetson

On H100, fused kernels save SRAM bandwidth and improve Tensor Core utilization. On Jetson, they save **DRAM bandwidth** — the #1 bottleneck.

```
Unfused attention (naive):
  Q × K^T → write attention scores to DRAM → read back → softmax → write →
  read back → multiply by V → write output

  Total DRAM traffic: ~4× the minimum

FlashAttention-2 (fused):
  Q × K^T → softmax → × V  (all in SRAM/registers, ONE read + ONE write)

  Total DRAM traffic: ~1× the minimum → 4× less bandwidth used
```

On Jetson's 51 GB/s bandwidth, this is the difference between 10 tokens/sec and 25 tokens/sec.

### 4.2 FlashAttention on Jetson

```bash
# llama.cpp automatically uses FlashAttention when available
./llama-cli -m model.gguf -ngl 99 -fa  # -fa enables FlashAttention

# TensorRT-LLM compiles FlashAttention into the engine
trtllm-build --model_dir ./model --output_dir ./engine \
    --use_fused_mlp \
    --use_flash_attn \
    --max_batch_size 1 \
    --max_input_len 2048 \
    --max_seq_len 2560
```

### 4.3 Other Fused Operations

Each fused operation eliminates a DRAM round-trip:

| Fused operation | What it combines | Bandwidth saved |
|----------------|-----------------|-----------------|
| **Fused RMSNorm** | Norm + scale in one kernel | ~2× less traffic |
| **Fused SwiGLU** | Gate + activation + multiply | ~3× less traffic |
| **Fused Rotary Embedding** | Position encoding + Q/K projection | ~2× less traffic |
| **Fused Add + Norm** | Residual connection + layer norm | ~2× less traffic |

TensorRT-LLM enables these automatically. llama.cpp has many fused CUDA kernels built in.

---

## 5. TensorRT-LLM on Jetson

TensorRT-LLM is NVIDIA's optimized inference engine for LLMs. It compiles the model into a TensorRT engine with fused kernels, quantization, and Tensor Core usage.

### 5.1 Build a TensorRT-LLM Engine for Jetson

```bash
# Install TensorRT-LLM (JetPack 6.x includes TensorRT, add LLM extension)
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com

# Convert Hugging Face model to TensorRT-LLM checkpoint
python convert_checkpoint.py \
    --model_dir ./Llama-3.2-3B-Instruct \
    --output_dir ./checkpoint \
    --dtype float16 \
    --tp_size 1          # single GPU on Jetson

# Build optimized engine
trtllm-build \
    --checkpoint_dir ./checkpoint \
    --output_dir ./engine \
    --gemm_plugin float16 \
    --max_batch_size 1 \
    --max_input_len 1024 \
    --max_seq_len 2048 \
    --max_num_tokens 2048 \
    --use_fused_mlp enable \
    --use_flash_attn enable \
    --strongly_typed

# Run inference
python run.py \
    --engine_dir ./engine \
    --tokenizer_dir ./Llama-3.2-3B-Instruct \
    --max_output_len 256 \
    --input_text "How does Jetson unified memory work?"
```

### 5.2 INT4 AWQ Engine for Minimum Memory

```bash
# Build with INT4 AWQ quantization
trtllm-build \
    --checkpoint_dir ./checkpoint-awq \
    --output_dir ./engine-int4 \
    --gemm_plugin auto \
    --max_batch_size 1 \
    --max_input_len 1024 \
    --max_seq_len 2048 \
    --use_fused_mlp enable \
    --weight_only_precision int4_awq
```

### 5.3 TensorRT-LLM vs llama.cpp on Jetson

| | TensorRT-LLM | llama.cpp |
|---|-------------|-----------|
| **Setup complexity** | High (build engine) | Low (download GGUF, run) |
| **Performance** | Best (compiled kernels) | Very good (hand-tuned CUDA) |
| **Quantization** | FP16, INT8, INT4 AWQ/GPTQ | Q2–Q8 mixed precision (GGUF) |
| **Flexibility** | Fixed engine (rebuild for changes) | Dynamic (change at runtime) |
| **Memory efficiency** | Excellent (preallocated) | Good (dynamic allocation) |
| **Model support** | Major models (Llama, Phi, Mistral, etc.) | Almost everything on HF |
| **Best for** | Production deployment | Prototyping + production |

**Recommendation:** Start with **llama.cpp** for prototyping (5-minute setup). Switch to **TensorRT-LLM** for production when you need maximum tokens/sec.

---

## 6. Speculative Decoding — Free Speed

Speculative decoding uses a small **draft model** to guess N tokens, then the large **target model** verifies all N in one forward pass. If the guess is correct, you get N tokens for the price of 1.

```
Without speculative decoding:
  Target model: generate token 1 → token 2 → token 3 → token 4
  Time: 4 forward passes × 50ms = 200ms

With speculative decoding (draft model guesses 4 tokens):
  Draft model:  generate 4 candidate tokens (fast, ~5ms total)
  Target model: verify all 4 in ONE forward pass (~55ms)
  If 3/4 accepted: 3 tokens in 60ms instead of 150ms → 2.5× faster

Speedup: 1.5–3× depending on acceptance rate
```

### 6.1 On Jetson

```bash
# llama.cpp supports speculative decoding
./llama-speculative \
    -m Llama-3.2-3B-Q4_K_M.gguf \       # target model (3B)
    -md TinyLlama-1.1B-Q4_K_M.gguf \    # draft model (1.1B)
    -ngl 99 \
    --draft 8 \                           # speculate 8 tokens ahead
    -p "Write a comprehensive guide to..."
```

**Memory budget for speculative decoding:**
- Target: Llama 3.2 3B INT4 = 1.5 GB
- Draft: TinyLlama 1.1B INT4 = 0.6 GB
- Total: 2.1 GB — fits easily on 8 GB

**When NOT to use on Jetson:** if the draft model pushes you over the memory budget, speculative decoding hurts more than it helps.

---

## 7. Runtime Optimizations

### 7.1 Power Mode Selection

```bash
# Check available power modes
sudo nvpmodel -q --verbose

# Set to maximum performance (15W on Orin Nano, 25W on Orin NX)
sudo nvpmodel -m 0
sudo jetson_clocks    # lock GPU/CPU at max frequency

# Or set power-efficient mode (7W)
sudo nvpmodel -m 1   # fewer CPU cores, lower GPU clock
```

Higher power mode = higher clock = more tokens/sec. But thermal design must support it.

### 7.2 GPU Frequency and Memory Clock

```bash
# Check current clocks
tegrastats --interval 500

# Lock GPU to max clock (prevents dynamic frequency scaling during inference)
sudo jetson_clocks

# Check GPU clock
cat /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/cur_freq
```

Dynamic frequency scaling adds latency jitter. For consistent inference latency, lock clocks.

### 7.3 NUMA-Aware Allocation (AGX Orin)

On AGX Orin with larger memory, ensure CUDA allocations use the optimal memory controller:

```bash
# Pin process to specific CPU cores close to memory controller
taskset -c 0-5 ./llama-cli -m model.gguf -ngl 99
```

### 7.4 Swap / zram for Emergency Overflow

If a model barely doesn't fit, zram (compressed RAM swap) can help:

```bash
# Enable 4 GB zram (compressed in-memory swap)
sudo zramctl --find --size 4G --algorithm zstd
sudo mkswap /dev/zram0
sudo swapon /dev/zram0 -p 5

# Now models slightly over RAM can run (with performance penalty)
```

zram compresses pages in memory — ~2:1 ratio for model weights. A 6 GB model on 5.5 GB available might work via zram, but with 30–50% speed penalty due to compression/decompression overhead.

---

## 8. Complete Optimization Checklist

```
Before deployment — run through this checklist:

□ Model Selection
  □ Model fits in INT4 with room for KV cache
  □ GQA-based model preferred (smaller KV cache)
  □ Context length budgeted against available memory

□ Quantization
  □ AWQ or GPTQ 4-bit for best quality/size
  □ GGUF Q4_K_M for llama.cpp deployment
  □ Calibration data representative of production input

□ KV Cache
  □ INT8 KV cache enabled
  □ Maximum context length capped to fit memory
  □ GQA model chosen to minimize KV memory

□ Inference Engine
  □ llama.cpp with -ngl 99 (full GPU offload)
  □ Or TensorRT-LLM engine compiled for target batch/context
  □ FlashAttention enabled

□ System Configuration
  □ nvpmodel set to appropriate power mode
  □ jetson_clocks to lock frequencies
  □ CMA reduced (LLM doesn't need large CMA)
  □ Unnecessary services disabled (GUI, bluetooth)

□ Profiling
  □ tegrastats monitored during inference
  □ Tokens/sec measured at steady state
  □ Memory usage verified (no slow growth / leak)
  □ Thermal verified (no throttling under sustained load)
```

---

## 9. Benchmark Reference

Expected performance on Orin Nano 8 GB (15W mode, Q4_K_M, llama.cpp):

| Model | Params | GGUF size | Prompt eval | Generation | Context |
|-------|--------|-----------|-------------|-----------|---------|
| TinyLlama 1.1B | 1.1B | 0.6 GB | ~120 tok/s | ~40 tok/s | 2048 |
| Llama 3.2 1B | 1.3B | 0.7 GB | ~100 tok/s | ~35 tok/s | 2048 |
| Gemma 2 2B | 2.6B | 1.5 GB | ~50 tok/s | ~20 tok/s | 2048 |
| Llama 3.2 3B | 3.2B | 1.8 GB | ~40 tok/s | ~15 tok/s | 2048 |
| Phi-3 Mini 3.8B | 3.8B | 2.2 GB | ~30 tok/s | ~12 tok/s | 2048 |

> These are estimates. Actual performance depends on power mode, thermal design, context length, and prompt content. Always benchmark your specific configuration.

---

## 10. Projects

| # | Project | What you learn |
|---|---------|---------------|
| 1 | **llama.cpp on Jetson** | Download Llama 3.2 3B Q4_K_M, build llama.cpp with CUDA, measure tokens/sec at different context lengths |
| 2 | **Quantization comparison** | Run same model at Q2_K, Q4_K_M, Q6_K, Q8_0. Measure tokens/sec, memory, and output quality (perplexity) |
| 3 | **TensorRT-LLM engine** | Build a TRT-LLM engine for Phi-3 Mini INT4. Compare latency with llama.cpp on same prompts |
| 4 | **Speculative decoding** | Set up TinyLlama as draft for Llama 3.2 3B. Measure acceptance rate and speedup |
| 5 | **Memory budget audit** | Run tegrastats during inference. Map every MB: OS, CMA, model, KV cache, activations. Verify against Section 3.3 |
| 6 | **Power vs performance** | Benchmark same model at 7W, 15W, 25W (if Orin NX). Plot tokens/sec vs power. Calculate tokens/joule |
| 7 | **Context length scaling** | Measure tokens/sec at context 512, 1024, 2048, 4096. Plot. Identify where KV cache pressure causes degradation |
| 8 | **Production chatbot** | Build a REST API serving Llama 3.2 3B on Jetson using llama.cpp server mode. Measure P50/P95 latency under concurrent requests |

---

## 11. Resources

| Resource | What it covers |
|----------|---------------|
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | Best open-source LLM inference engine for edge |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA's optimized LLM engine |
| [NVIDIA Jetson AI Lab](https://www.jetson-ai-lab.com/) | Pre-built containers and tutorials for LLMs on Jetson |
| [Jetson Generative AI Playground](https://developer.nvidia.com/embedded/generative-ai) | NVIDIA's LLM deployment guides for Jetson |
| [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) | AWQ quantization library |
| [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691) | Algorithm behind fused attention |
| [Speculative Decoding paper](https://arxiv.org/abs/2302.01318) | Original speculative decoding paper |
| [vLLM paper (PagedAttention)](https://arxiv.org/abs/2309.06180) | KV cache memory management (server reference) |
| [Orin Nano Memory Architecture](../../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Memory-Architecture/Guide.md) | Unified memory deep dive (this roadmap) |

---

## Next

→ Back to [ML and AI hub](../Guide.md)
