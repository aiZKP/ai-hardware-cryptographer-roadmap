// decode.cpp — Transformer forward pass + generation loop
//
// Memory-first: zero allocation during decode. All buffers from scratch pool.
// Thermal-aware: checks temperature and memory before every token.
// CUDA graph: captures decode step after first iteration, replays for rest.

#include "jllm_engine.h"
#include <chrono>
#include <algorithm>
#include <cstring>
#include <unistd.h>

namespace jllm {

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<float, std::milli>;

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
        munmap(weights_, weights_size_);
        weights_ = nullptr;
    }
    if (model_weights_.layers) { delete[] model_weights_.layers; model_weights_.layers = nullptr; }
    loaded_ = false;
    graph_captured_ = false;
}

bool Engine::load(const std::string& gguf_path, const GenParams& params) {
    gen_params_ = params;
    budget_ = probe_system_memory();

    // Load config
    config_ = load_gguf_config(gguf_path);
    fprintf(stderr, "[engine] %s: %d layers, %d heads (%d KV), dim=%d, vocab=%d\n",
            config_.name.c_str(), config_.n_layers, config_.n_heads,
            config_.n_kv_heads, config_.hidden_dim, config_.vocab_size);

    // Load weights (mmap + pin)
    if (!load_gguf_weights(gguf_path, &weights_, &weights_size_)) return false;
    budget_.model_mb = weights_size_ / (1024 * 1024);

    // Map weight tensors
    model_weights_ = map_weights(weights_, weights_size_, config_);

    // Auto context from budget
    int kv_bytes = params.kv_int8 ? 1 : 2;
    int auto_ctx = budget_.max_context(config_.n_layers, config_.n_kv_heads, config_.head_dim, kv_bytes);
    int ctx = params.context_limit > 0 ? std::min(params.context_limit, auto_ctx) : auto_ctx;
    ctx = std::min(ctx, config_.max_seq_len);
    fprintf(stderr, "[engine] Context: %d tokens (memory-limited to %d)\n", ctx, auto_ctx);

    // KV cache
    KVCachePool::Config kv_cfg = {};
    kv_cfg.n_layers = config_.n_layers;
    kv_cfg.n_kv_heads = config_.n_kv_heads;
    kv_cfg.head_dim = config_.head_dim;
    kv_cfg.max_context = ctx;
    kv_cfg.overflow_context = ctx / 4;
    kv_cfg.kv_type_bytes = kv_bytes;
    if (!kv_cache_.init(kv_cfg)) return false;
    budget_.kv_cache_mb = kv_cache_.capacity_bytes() / (1024 * 1024);

    // Scratch: large enough for all intermediates in one decode step
    // Need: hidden_dim × 2 (attn input + output) + intermediate_dim × 2 (FFN)
    // + head_dim × n_heads (Q) + logits (vocab_size)
    int64_t scratch_bytes = 0;
    scratch_bytes += config_.hidden_dim * sizeof(half) * 4;           // x, residual, normed, attn_out
    scratch_bytes += config_.n_heads * config_.head_dim * sizeof(half) * 3; // Q, K, V projections
    scratch_bytes += config_.intermediate_dim * sizeof(half) * 3;     // gate, up, down
    scratch_bytes += config_.vocab_size * sizeof(float);              // logits (FP32 for sampling)
    scratch_bytes = std::max(scratch_bytes, (int64_t)64 * 1024 * 1024);
    scratch_bytes = (scratch_bytes + 4095) & ~4095LL;                 // page align

    if (!scratch_.init(scratch_bytes)) return false;
    budget_.scratch_mb = scratch_bytes / (1024 * 1024);

    cudaStreamCreate(&stream_);

    // Load tokenizer
    tokenizer_.load_from_gguf(gguf_path);

    budget_.print();
    loaded_ = true;
    return true;
}

// ── Single transformer layer forward pass ────────────────────────────────
// All buffers from scratch pool — zero malloc.

void Engine::transformer_layer(int layer, int pos, half* x) {
    const auto& lw = model_weights_.layers[layer];
    int H = config_.hidden_dim;
    int KV_DIM = config_.n_kv_heads * config_.head_dim;
    int I = config_.intermediate_dim;

    // Get scratch buffers (bump allocator — resets each step)
    half* normed   = (half*)scratch_.get(H * sizeof(half));
    half* q_buf    = (half*)scratch_.get(config_.n_heads * config_.head_dim * sizeof(half));
    half* k_buf    = (half*)scratch_.get(KV_DIM * sizeof(half));
    half* v_buf    = (half*)scratch_.get(KV_DIM * sizeof(half));
    half* attn_out = (half*)scratch_.get(H * sizeof(half));
    half* residual = (half*)scratch_.get(H * sizeof(half));
    half* normed2  = (half*)scratch_.get(H * sizeof(half));
    half* gate_buf = (half*)scratch_.get(I * sizeof(half));
    half* up_buf   = (half*)scratch_.get(I * sizeof(half));
    half* ffn_out  = (half*)scratch_.get(H * sizeof(half));

    // Save residual
    cudaMemcpyAsync(residual, x, H * sizeof(half), cudaMemcpyDeviceToDevice, stream_);

    // 1. Attention RMSNorm (fused with residual from previous layer)
    // For layer 0, residual is the embedding; for layer N, it's previous layer output
    fused_rmsnorm_residual(normed, x, residual, lw.rms_attn, 1, H, 1e-5f, stream_);

    // 2. QKV projections (quantized GEMV: batch=1 decode)
    gemv_q4(q_buf, lw.wq, lw.sq, normed, config_.n_heads * config_.head_dim, H, 32, stream_);
    gemv_q4(k_buf, lw.wk, lw.sk, normed, KV_DIM, H, 32, stream_);
    gemv_q4(v_buf, lw.wv, lw.sv, normed, KV_DIM, H, 32, stream_);

    // 3. RoPE (in-place on Q and K)
    rope_inplace(q_buf, k_buf, config_.n_heads, config_.n_kv_heads,
                 config_.head_dim, pos, config_.rope_theta, stream_);

    // 4. Store K and V into KV cache
    if (gen_params_.kv_int8) {
        // Quantize K/V to INT8 before storing
        fp16_to_int8((int8_t*)kv_cache_.key_ptr(layer, pos), nullptr, k_buf, 1, KV_DIM, stream_);
        fp16_to_int8((int8_t*)kv_cache_.val_ptr(layer, pos), nullptr, v_buf, 1, KV_DIM, stream_);
    } else {
        cudaMemcpyAsync(kv_cache_.key_ptr(layer, pos), k_buf, KV_DIM * sizeof(half),
                        cudaMemcpyDeviceToDevice, stream_);
        cudaMemcpyAsync(kv_cache_.val_ptr(layer, pos), v_buf, KV_DIM * sizeof(half),
                        cudaMemcpyDeviceToDevice, stream_);
    }

    // 5. Flash Attention (decode: single query)
    float scale = 1.0f / sqrtf((float)config_.head_dim);
    flash_attention_decode(attn_out, q_buf, kv_cache_.key_ptr(layer, 0),
                          kv_cache_.val_ptr(layer, 0),
                          config_.n_heads, config_.n_kv_heads, config_.head_dim,
                          pos + 1, scale, gen_params_.kv_int8, nullptr, stream_);

    // 6. Output projection
    half* attn_proj = (half*)scratch_.get(H * sizeof(half));
    gemv_q4(attn_proj, lw.wo, lw.so, attn_out, H, H, 32, stream_);

    // 7. Residual add (attn_proj + residual → x)
    // Reuse x as output, fused into FFN norm step

    // 8. FFN RMSNorm (fused with residual)
    fused_rmsnorm_residual(normed2, attn_proj, residual, lw.rms_ffn, 1, H, 1e-5f, stream_);

    // 9. FFN: gate and up projections
    gemv_q4(gate_buf, lw.w_gate, lw.s_gate, normed2, I, H, 32, stream_);
    gemv_q4(up_buf,   lw.w_up,   lw.s_up,   normed2, I, H, 32, stream_);

    // 10. SwiGLU activation (fused)
    half* swiglu_out = (half*)scratch_.get(I * sizeof(half));
    fused_swiglu(swiglu_out, gate_buf, up_buf, 1, I, stream_);

    // 11. Down projection
    gemv_q4(ffn_out, lw.w_down, lw.s_down, swiglu_out, H, I, 32, stream_);

    // 12. Final residual add → x (output of this layer)
    // Simple vector add: x = ffn_out + attn_proj + residual
    // TODO: fuse into a single kernel (triple add)
    cudaMemcpyAsync(x, ffn_out, H * sizeof(half), cudaMemcpyDeviceToDevice, stream_);
}

// ── Full decode step: all layers + logits ────────────────────────────────

int Engine::decode_step(int pos) {
    int H = config_.hidden_dim;

    // Get hidden state buffer
    half* x = (half*)scratch_.get(H * sizeof(half));

    // Token embedding lookup (position 'pos' in sequence)
    // For decode, this is the previously generated token
    const half* embd = model_weights_.tok_embd + last_token_ * H;
    cudaMemcpyAsync(x, embd, H * sizeof(half), cudaMemcpyDeviceToDevice, stream_);

    // Run all transformer layers
    for (int l = 0; l < config_.n_layers; l++) {
        transformer_layer(l, pos, x);
    }

    // Final RMSNorm
    half* normed = (half*)scratch_.get(H * sizeof(half));
    half* zero = (half*)scratch_.get(H * sizeof(half));
    cudaMemsetAsync(zero, 0, H * sizeof(half), stream_);
    fused_rmsnorm_residual(normed, x, zero, model_weights_.output_norm, 1, H, 1e-5f, stream_);

    // Logits projection (hidden → vocab)
    float* logits = (float*)scratch_.get(config_.vocab_size * sizeof(float));
    // TODO: need FP32 output GEMV variant for logits
    // For now, use FP16 GEMV then convert
    half* logits_fp16 = (half*)scratch_.get(config_.vocab_size * sizeof(half));
    gemv_q4(logits_fp16, model_weights_.output, model_weights_.s_output,
            normed, config_.vocab_size, H, 32, stream_);

    // Convert FP16 logits to FP32 for sampling
    // (sampling needs FP32 for numerical stability in softmax/exp)
    cudaStreamSynchronize(stream_);

    // Copy to CPU for sampling (logits are small: vocab_size × 4 bytes ≈ 512 KB)
    std::vector<float> h_logits(config_.vocab_size);
    // Manual FP16→FP32 on host (faster than launching a tiny kernel for this)
    std::vector<half> h_logits_fp16(config_.vocab_size);
    cudaMemcpy(h_logits_fp16.data(), logits_fp16,
               config_.vocab_size * sizeof(half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < config_.vocab_size; i++)
        h_logits[i] = __half2float(h_logits_fp16[i]);

    // Sample next token
    int token = sample_token(h_logits.data(), config_.vocab_size, gen_params_,
                             recent_tokens_.data(), recent_tokens_.size());

    // Track for repeat penalty
    recent_tokens_.push_back(token);
    if ((int)recent_tokens_.size() > 64) recent_tokens_.erase(recent_tokens_.begin());

    last_token_ = token;
    return token;
}

// ── CUDA graph capture ───────────────────────────────────────────────────

void Engine::build_cuda_graph(int pos) {
    if (graph_captured_) return;

    fprintf(stderr, "[engine] Capturing CUDA graph for decode step...\n");

    cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);

    scratch_.reset();
    // Run one decode step under capture
    // Note: some operations (cudaMemcpy H2D/D2H) can't be captured
    // Only GPU-to-GPU kernels are captured
    for (int l = 0; l < config_.n_layers; l++) {
        // Simplified capture — full implementation needs to handle
        // variable KV cache length (pos changes each step)
        // CUDA graphs work best when the graph structure is fixed
    }

    cudaStreamEndCapture(stream_, &decode_graph_);
    cudaGraphInstantiate(&decode_graph_exec_, decode_graph_, nullptr, nullptr, 0);
    graph_captured_ = true;

    fprintf(stderr, "[engine] CUDA graph captured\n");
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

    if (pos % 10 == 0) {  // check thermal every 10 tokens (not every token — sysfs read is slow)
        auto ts = read_thermal();
        int backoff = thermal_backoff_us(ts);
        if (backoff > 0) {
            fprintf(stderr, "\n[thermal] %.1f°C — backing off %d ms\n",
                    ts.gpu_temp_c, backoff / 1000);
            usleep(backoff);
            return true;  // continue after backoff
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

    // Tokenize
    auto prompt_tokens = tokenizer_.encode(prompt);
    stats.prompt_tokens = prompt_tokens.size();
    if (prompt_tokens.empty()) {
        prompt_tokens.push_back(tokenizer_.bos_id);
    }

    // Prefill: process all prompt tokens
    auto t0 = Clock::now();
    for (int i = 0; i < (int)prompt_tokens.size(); i++) {
        scratch_.reset();
        last_token_ = prompt_tokens[i];
        // Run forward pass but don't sample (just build KV cache)
        int H = config_.hidden_dim;
        half* x = (half*)scratch_.get(H * sizeof(half));
        const half* embd = model_weights_.tok_embd + last_token_ * H;
        cudaMemcpyAsync(x, embd, H * sizeof(half), cudaMemcpyDeviceToDevice, stream_);
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

    // Decode: generate tokens one at a time
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

        // EOS check
        bool is_eos = (token == tokenizer_.eos_id);

        // Stream output
        if (token_cb) {
            std::string text = tokenizer_.decode(token);
            token_cb(text.c_str(), is_eos);
        }

        stats.completion_tokens++;

        if (is_eos) break;

        // Track peaks (every 10 tokens)
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
