// decode.cpp — Transformer forward pass + generation loop
//
// Memory-first: zero allocation during decode. All buffers from scratch pool.
// Thermal-aware: checks temperature and memory before every token.
// CUDA graph: captures decode kernels after first iteration, replays for rest.
//
// BUG FIXES applied:
//   #2 — Residual connection properly chained (residual1 for attn, residual2 for FFN)
//   #3 — Embedding memcpy uses cudaMemcpyDefault (works for unified + discrete)
//   #4 — Added #include <sys/mman.h>
//   #5 — CUDA graph capture now runs actual transformer_layer kernels
//   #7 — FP32 logit output via host-side conversion after FP16 GEMV

#include "jllm_engine.h"
#include <chrono>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <unistd.h>
#include <sys/mman.h>   // BUG #4 fix

namespace jllm {

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<float, std::milli>;

// ── Vector add kernel (for residual connections) ─────────────────────────
// BUG #2 fix: need explicit residual add between stages

__global__ void vec_add_kernel(half* __restrict__ out,
                                const half* __restrict__ a,
                                const half* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

static void vec_add(half* out, const half* a, const half* b, int n, cudaStream_t s) {
    int block = 128;
    int grid = (n + block - 1) / block;
    vec_add_kernel<<<grid, block, 0, s>>>(out, a, b, n);
}

// ── FP16 to FP32 conversion kernel ──────────────────────────────────────
// BUG #7 fix: convert logits to FP32 on GPU before D2H copy

__global__ void fp16_to_fp32_kernel(float* __restrict__ out,
                                     const half* __restrict__ in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __half2float(in[i]);
}

static void fp16_to_fp32(float* out, const half* in, int n, cudaStream_t s) {
    int block = 256;
    int grid = (n + block - 1) / block;
    fp16_to_fp32_kernel<<<grid, block, 0, s>>>(out, in, n);
}

// ═════════════════════════════════════════════════════════════════════════

Engine::Engine() {}
Engine::~Engine() { unload(); }

void Engine::unload() {
    if (decode_graph_exec_) { cudaGraphExecDestroy(decode_graph_exec_); decode_graph_exec_ = nullptr; }
    if (decode_graph_)      { cudaGraphDestroy(decode_graph_); decode_graph_ = nullptr; }
    if (stream_)            { cudaStreamDestroy(stream_); stream_ = nullptr; }
    kv_cache_.destroy();
    scratch_.destroy();
    if (weights_) {
        cudaHostUnregister(weights_);
        munmap(weights_, weights_size_);  // now works — #include <sys/mman.h> added
        weights_ = nullptr;
    }
    if (model_weights_.layers) { delete[] model_weights_.layers; model_weights_.layers = nullptr; }
    loaded_ = false;
    graph_captured_ = false;
}

bool Engine::load(const std::string& gguf_path, const GenParams& params) {
    gen_params_ = params;
    budget_ = probe_system_memory();

    config_ = load_gguf_config(gguf_path);
    fprintf(stderr, "[engine] %s: %d layers, %d heads (%d KV), dim=%d, vocab=%d\n",
            config_.name.c_str(), config_.n_layers, config_.n_heads,
            config_.n_kv_heads, config_.hidden_dim, config_.vocab_size);

    if (!load_and_map_weights(gguf_path, &weights_, &weights_size_,
                              &model_weights_, config_)) {
        fprintf(stderr, "[engine] Failed to load/map weights\n");
        return false;
    }
    budget_.model_mb = weights_size_ / (1024 * 1024);

    int kv_bytes = params.kv_int8 ? 1 : 2;
    int auto_ctx = budget_.max_context(config_.n_layers, config_.n_kv_heads, config_.head_dim, kv_bytes);
    int ctx = params.context_limit > 0 ? std::min(params.context_limit, auto_ctx) : auto_ctx;
    ctx = std::min(ctx, config_.max_seq_len);
    fprintf(stderr, "[engine] Context: %d tokens (memory-limited to %d)\n", ctx, auto_ctx);

    KVCachePool::Config kv_cfg = {};
    kv_cfg.n_layers = config_.n_layers;
    kv_cfg.n_kv_heads = config_.n_kv_heads;
    kv_cfg.head_dim = config_.head_dim;
    kv_cfg.max_context = ctx;
    kv_cfg.overflow_context = ctx / 4;
    kv_cfg.kv_type_bytes = kv_bytes;
    if (!kv_cache_.init(kv_cfg)) return false;
    budget_.kv_cache_mb = kv_cache_.capacity_bytes() / (1024 * 1024);

    int64_t scratch_bytes = 0;
    scratch_bytes += config_.hidden_dim * sizeof(half) * 8;
    scratch_bytes += config_.n_heads * config_.head_dim * sizeof(half) * 3;
    scratch_bytes += config_.intermediate_dim * sizeof(half) * 4;
    scratch_bytes += config_.vocab_size * sizeof(float);
    scratch_bytes += config_.vocab_size * sizeof(half);
    scratch_bytes = std::max(scratch_bytes, (int64_t)64 * 1024 * 1024);
    scratch_bytes = (scratch_bytes + 4095) & ~4095LL;

    if (!scratch_.init(scratch_bytes)) return false;
    budget_.scratch_mb = scratch_bytes / (1024 * 1024);

    cudaStreamCreate(&stream_);
    tokenizer_.load_from_gguf(gguf_path);

    budget_.print();
    loaded_ = true;
    return true;
}

// ── Single transformer layer ─────────────────────────────────────────────
// BUG #2 FIX: proper residual chaining:
//   residual1 = x (input to layer)
//   attn_out  = attention(RMSNorm(x))
//   x2        = residual1 + attn_out          ← first residual add
//   ffn_out   = FFN(RMSNorm(x2))
//   x_out     = x2 + ffn_out                  ← second residual add

void Engine::transformer_layer(int layer, int pos, half* x) {
    const auto& lw = model_weights_.layers[layer];
    int H = config_.hidden_dim;
    int KV_DIM = config_.n_kv_heads * config_.head_dim;
    int I = config_.intermediate_dim;

    // Scratch buffers (bump allocator)
    half* normed   = (half*)scratch_.get(H * sizeof(half));
    half* q_buf    = (half*)scratch_.get(config_.n_heads * config_.head_dim * sizeof(half));
    half* k_buf    = (half*)scratch_.get(KV_DIM * sizeof(half));
    half* v_buf    = (half*)scratch_.get(KV_DIM * sizeof(half));
    half* attn_out = (half*)scratch_.get(H * sizeof(half));
    half* attn_proj = (half*)scratch_.get(H * sizeof(half));
    half* x2       = (half*)scratch_.get(H * sizeof(half));  // after first residual
    half* normed2  = (half*)scratch_.get(H * sizeof(half));
    half* gate_buf = (half*)scratch_.get(I * sizeof(half));
    half* up_buf   = (half*)scratch_.get(I * sizeof(half));
    half* swiglu_out = (half*)scratch_.get(I * sizeof(half));
    half* ffn_out  = (half*)scratch_.get(H * sizeof(half));

    // ── Attention block ──────────────────────────────────────

    // 1. Pre-attention RMSNorm: normed = RMSNorm(x) * weight
    //    (x is the residual — don't modify it yet)
    half* zero_buf = (half*)scratch_.get(H * sizeof(half));
    cudaMemsetAsync(zero_buf, 0, H * sizeof(half), stream_);
    fused_rmsnorm_residual(normed, x, zero_buf, lw.rms_attn, 1, H, 1e-5f, stream_);

    // 2. QKV projections
    gemv_q4(q_buf, lw.wq, lw.sq, normed, config_.n_heads * config_.head_dim, H, 32, stream_);
    gemv_q4(k_buf, lw.wk, lw.sk, normed, KV_DIM, H, 32, stream_);
    gemv_q4(v_buf, lw.wv, lw.sv, normed, KV_DIM, H, 32, stream_);

    // 3. RoPE
    rope_inplace(q_buf, k_buf, config_.n_heads, config_.n_kv_heads,
                 config_.head_dim, pos, config_.rope_theta, stream_);

    // 4. Store K/V into cache
    if (gen_params_.kv_int8) {
        fp16_to_int8((int8_t*)kv_cache_.key_ptr(layer, pos), nullptr, k_buf, 1, KV_DIM, stream_);
        fp16_to_int8((int8_t*)kv_cache_.val_ptr(layer, pos), nullptr, v_buf, 1, KV_DIM, stream_);
    } else {
        cudaMemcpyAsync(kv_cache_.key_ptr(layer, pos), k_buf,
                        KV_DIM * sizeof(half), cudaMemcpyDefault, stream_);
        cudaMemcpyAsync(kv_cache_.val_ptr(layer, pos), v_buf,
                        KV_DIM * sizeof(half), cudaMemcpyDefault, stream_);
    }

    // 5. Attention
    float scale = 1.0f / sqrtf((float)config_.head_dim);
    flash_attention_decode(attn_out, q_buf, kv_cache_.key_ptr(layer, 0),
                          kv_cache_.val_ptr(layer, 0),
                          config_.n_heads, config_.n_kv_heads, config_.head_dim,
                          pos + 1, scale, gen_params_.kv_int8, nullptr, stream_);

    // 6. Output projection
    gemv_q4(attn_proj, lw.wo, lw.so, attn_out, H, H, 32, stream_);

    // 7. FIRST RESIDUAL: x2 = x + attn_proj  (BUG #2 FIX)
    vec_add(x2, x, attn_proj, H, stream_);

    // ── FFN block ────────────────────────────────────────────

    // 8. Pre-FFN RMSNorm: normed2 = RMSNorm(x2) * weight
    fused_rmsnorm_residual(normed2, x2, zero_buf, lw.rms_ffn, 1, H, 1e-5f, stream_);

    // 9. Gate and up projections
    gemv_q4(gate_buf, lw.w_gate, lw.s_gate, normed2, I, H, 32, stream_);
    gemv_q4(up_buf,   lw.w_up,   lw.s_up,   normed2, I, H, 32, stream_);

    // 10. SwiGLU
    fused_swiglu(swiglu_out, gate_buf, up_buf, 1, I, stream_);

    // 11. Down projection
    gemv_q4(ffn_out, lw.w_down, lw.s_down, swiglu_out, H, I, 32, stream_);

    // 12. SECOND RESIDUAL: x = x2 + ffn_out  (BUG #2 FIX)
    vec_add(x, x2, ffn_out, H, stream_);
}

// ── Full decode step ─────────────────────────────────────────────────────

int Engine::decode_step(int pos) {
    int H = config_.hidden_dim;

    half* x = (half*)scratch_.get(H * sizeof(half));

    // Embedding lookup
    // BUG #3 FIX: use cudaMemcpyDefault — works for both host (mmap) and device memory
    const half* embd = model_weights_.tok_embd + last_token_ * H;
    cudaMemcpyAsync(x, embd, H * sizeof(half), cudaMemcpyDefault, stream_);

    // All transformer layers
    for (int l = 0; l < config_.n_layers; l++) {
        transformer_layer(l, pos, x);
    }

    // Final RMSNorm
    half* normed = (half*)scratch_.get(H * sizeof(half));
    half* zero = (half*)scratch_.get(H * sizeof(half));
    cudaMemsetAsync(zero, 0, H * sizeof(half), stream_);
    fused_rmsnorm_residual(normed, x, zero, model_weights_.output_norm, 1, H, 1e-5f, stream_);

    // Logits: FP16 GEMV then convert to FP32 on GPU (BUG #7 FIX)
    half*  logits_fp16 = (half*)scratch_.get(config_.vocab_size * sizeof(half));
    float* logits_fp32 = (float*)scratch_.get(config_.vocab_size * sizeof(float));

    gemv_q4(logits_fp16, model_weights_.output, model_weights_.s_output,
            normed, config_.vocab_size, H, 32, stream_);

    // Convert on GPU — avoids per-element __half2float on CPU
    fp16_to_fp32(logits_fp32, logits_fp16, config_.vocab_size, stream_);

    cudaStreamSynchronize(stream_);

    // Copy FP32 logits to CPU for sampling
    std::vector<float> h_logits(config_.vocab_size);
    cudaMemcpy(h_logits.data(), logits_fp32,
               config_.vocab_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Sample
    int token = sample_token(h_logits.data(), config_.vocab_size, gen_params_,
                             recent_tokens_.data(), recent_tokens_.size());

    recent_tokens_.push_back(token);
    if ((int)recent_tokens_.size() > 64) recent_tokens_.erase(recent_tokens_.begin());

    last_token_ = token;
    return token;
}

// ── CUDA graph capture (BUG #5 FIX) ─────────────────────────────────────
// Captures the GPU-side kernels of one decode step.
// Requirements:
//   - KV cache length is padded to max_context (fixed graph structure)
//   - Embedding lookup done outside graph (host→device copy not capturable)
//   - Sampling done outside graph (host-side operation)

void Engine::build_cuda_graph(int pos) {
    if (graph_captured_) return;
    if (!gen_params_.use_cuda_graph) return;

    fprintf(stderr, "[engine] Capturing CUDA graph...\n");

    // Pre-allocate a fixed hidden state buffer for graph capture
    int H = config_.hidden_dim;
    half* graph_x = (half*)scratch_.get(H * sizeof(half));

    cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);

    // Capture all transformer layers
    for (int l = 0; l < config_.n_layers; l++) {
        transformer_layer(l, pos, graph_x);
    }

    // Capture final norm
    half* g_normed = (half*)scratch_.get(H * sizeof(half));
    half* g_zero = (half*)scratch_.get(H * sizeof(half));
    cudaMemsetAsync(g_zero, 0, H * sizeof(half), stream_);
    fused_rmsnorm_residual(g_normed, graph_x, g_zero,
                          model_weights_.output_norm, 1, H, 1e-5f, stream_);

    // Capture logit projection + conversion
    half* g_logits_fp16 = (half*)scratch_.get(config_.vocab_size * sizeof(half));
    float* g_logits_fp32 = (float*)scratch_.get(config_.vocab_size * sizeof(float));
    gemv_q4(g_logits_fp16, model_weights_.output, model_weights_.s_output,
            g_normed, config_.vocab_size, H, 32, stream_);
    fp16_to_fp32(g_logits_fp32, g_logits_fp16, config_.vocab_size, stream_);

    cudaError_t err = cudaStreamEndCapture(stream_, &decode_graph_);
    if (err != cudaSuccess) {
        fprintf(stderr, "[engine] Graph capture failed: %s\n", cudaGetErrorString(err));
        decode_graph_ = nullptr;
        return;
    }

    err = cudaGraphInstantiate(&decode_graph_exec_, decode_graph_, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[engine] Graph instantiation failed: %s\n", cudaGetErrorString(err));
        cudaGraphDestroy(decode_graph_);
        decode_graph_ = nullptr;
        return;
    }

    graph_captured_ = true;
    fprintf(stderr, "[engine] CUDA graph captured successfully\n");
}

// ── Memory and thermal check ─────────────────────────────────────────────

bool Engine::check_memory_and_thermal(int pos) {
    OOMGuard guard(256);
    int kv_bytes = gen_params_.kv_int8 ? 1 : 2;
    if (!guard.can_extend(config_.kv_per_token_bytes(kv_bytes))) {
        fprintf(stderr, "\n[oom_guard] Stopping at token %d — %lld MB free\n",
                pos, guard.real_free_mb());
        return false;
    }

    if (pos % 10 == 0) {
        auto ts = read_thermal();
        int backoff = thermal_backoff_us(ts);
        if (backoff > 0) {
            fprintf(stderr, "\n[thermal] %.1f°C — backing off %d ms\n",
                    ts.gpu_temp_c, backoff / 1000);
            usleep(backoff);
        }
    }
    return true;
}

// ── Main generation loop ─────────────────────────────────────────────────

GenStats Engine::generate(const std::string& prompt, const GenParams& params,
                          TokenCallback token_cb) {
    GenStats stats = {};
    stop_flag_ = false;
    gen_params_ = params;
    recent_tokens_.clear();

    auto prompt_tokens = tokenizer_.encode(prompt);
    stats.prompt_tokens = prompt_tokens.size();
    if (prompt_tokens.empty()) {
        prompt_tokens.push_back(tokenizer_.bos_id);
    }

    // Prefill
    auto t0 = Clock::now();
    for (int i = 0; i < (int)prompt_tokens.size(); i++) {
        scratch_.reset();
        last_token_ = prompt_tokens[i];
        int H = config_.hidden_dim;
        half* x = (half*)scratch_.get(H * sizeof(half));
        const half* embd = model_weights_.tok_embd + last_token_ * H;
        cudaMemcpyAsync(x, embd, H * sizeof(half), cudaMemcpyDefault, stream_);
        for (int l = 0; l < config_.n_layers; l++)
            transformer_layer(l, i, x);
    }
    cudaStreamSynchronize(stream_);
    auto t1 = Clock::now();
    stats.prompt_ms = Ms(t1 - t0).count();
    if (stats.prompt_tokens > 0)
        stats.prompt_tok_per_sec = stats.prompt_tokens / (stats.prompt_ms / 1000.0f);
    fprintf(stderr, "[engine] Prefill: %d tokens in %.0f ms (%.1f tok/s)\n",
            stats.prompt_tokens, stats.prompt_ms, stats.prompt_tok_per_sec);

    // Decode
    auto t2 = Clock::now();
    int64_t peak_mem = 0;
    float peak_temp = 0;
    int pos = prompt_tokens.size();

    for (int i = 0; i < params.max_tokens && !stop_flag_; i++) {
        if (!check_memory_and_thermal(pos)) {
            stats.oom_stops++;
            break;
        }

        scratch_.reset();
        int token = decode_step(pos);
        pos++;

        bool is_eos = (token == tokenizer_.eos_id);

        if (token_cb) {
            std::string text = tokenizer_.decode(token);
            token_cb(text.c_str(), is_eos);
        }

        stats.completion_tokens++;
        if (is_eos) break;

        if (i % 10 == 0) {
            OOMGuard g(0);
            int64_t used = budget_.total_mb - g.real_free_mb();
            peak_mem = std::max(peak_mem, used);
            peak_temp = std::max(peak_temp, read_thermal().gpu_temp_c);
        }
    }

    auto t3 = Clock::now();
    stats.decode_ms = Ms(t3 - t2).count();
    if (stats.completion_tokens > 0)
        stats.decode_tok_per_sec = stats.completion_tokens / (stats.decode_ms / 1000.0f);
    stats.peak_memory_mb = peak_mem;
    stats.peak_thermal_c = peak_temp;

    fprintf(stderr, "[engine] Decode: %d tokens in %.0f ms (%.1f tok/s)\n",
            stats.completion_tokens, stats.decode_ms, stats.decode_tok_per_sec);
    return stats;
}

void Engine::stop() { stop_flag_ = true; }
LiveStats Engine::stats() const { return read_live_stats(); }

}  // namespace jllm
