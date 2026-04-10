// model.cpp — GGUF model loader (minimal, Jetson-optimized)
//
// We only read what we need from GGUF:
//   - Model config (n_layers, n_heads, etc.)
//   - Weight tensors (mmap for zero-copy, then pin for GPU access)
//   - Tokenizer vocabulary
//
// No support for FP32 models (waste of memory on 8 GB).
// Only Q4_K_M, Q5_K_M, Q8_0, FP16 quantizations.

#include "jllm_engine.h"
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace jllm {

// ── GGUF magic and header ────────────────────────────────────────────────

static constexpr uint32_t GGUF_MAGIC = 0x46475547;  // "GGUF" in little-endian

struct GGUFHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
};

// Minimal KV metadata reader
struct GGUFMetaKV {
    char     key[256];
    uint32_t value_type;
    union {
        uint32_t u32;
        int32_t  i32;
        float    f32;
        uint64_t u64;
    };
};

// ── Read model config from GGUF ──────────────────────────────────────────

ModelConfig load_gguf_config(const std::string& path) {
    ModelConfig cfg = {};
    cfg.name = path;

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "[gguf] Cannot open %s\n", path.c_str());
        return cfg;
    }

    GGUFHeader hdr;
    fread(&hdr, sizeof(hdr), 1, f);

    if (hdr.magic != GGUF_MAGIC) {
        fprintf(stderr, "[gguf] Invalid magic: 0x%08X (expected 0x%08X)\n",
                hdr.magic, GGUF_MAGIC);
        fclose(f);
        return cfg;
    }

    fprintf(stderr, "[gguf] Version %u, %lu tensors, %lu metadata entries\n",
            hdr.version, hdr.n_tensors, hdr.n_kv);

    // Read metadata key-value pairs
    // This is a simplified reader — production code should handle all GGUF types
    for (uint64_t i = 0; i < hdr.n_kv && i < 256; i++) {
        // Read key length + key string
        uint64_t key_len;
        fread(&key_len, sizeof(key_len), 1, f);
        if (key_len > 255) { fseek(f, key_len, SEEK_CUR); continue; }

        char key[256] = {};
        fread(key, 1, key_len, f);

        // Read value type
        uint32_t vtype;
        fread(&vtype, sizeof(vtype), 1, f);

        // Parse based on type (simplified — only handles uint32 and float)
        if (vtype == 4) {  // GGUF_TYPE_UINT32
            uint32_t val;
            fread(&val, sizeof(val), 1, f);

            if (strstr(key, "block_count"))        cfg.n_layers = val;
            else if (strstr(key, "head_count_kv")) cfg.n_kv_heads = val;
            else if (strstr(key, "head_count"))    cfg.n_heads = val;
            else if (strstr(key, "embedding_length")) cfg.hidden_dim = val;
            else if (strstr(key, "feed_forward_length")) cfg.intermediate_dim = val;
            else if (strstr(key, "vocab_size"))    cfg.vocab_size = val;
            else if (strstr(key, "context_length")) cfg.max_seq_len = val;
        } else if (vtype == 6) {  // GGUF_TYPE_FLOAT32
            float val;
            fread(&val, sizeof(val), 1, f);
            if (strstr(key, "rope.freq_base")) cfg.rope_theta = val;
        } else if (vtype == 8) {  // GGUF_TYPE_STRING
            uint64_t str_len;
            fread(&str_len, sizeof(str_len), 1, f);
            if (strstr(key, "general.name") && str_len < 256) {
                char name[256] = {};
                fread(name, 1, str_len, f);
                cfg.name = name;
            } else {
                fseek(f, str_len, SEEK_CUR);
            }
        } else {
            // Skip unknown types (need proper GGUF parser for production)
            // For now, try to skip common sizes
            fseek(f, 8, SEEK_CUR);  // rough skip
        }
    }

    // Derive head_dim
    if (cfg.n_heads > 0 && cfg.hidden_dim > 0)
        cfg.head_dim = cfg.hidden_dim / cfg.n_heads;

    // Defaults for missing fields
    if (cfg.rope_theta == 0.0f) cfg.rope_theta = 10000.0f;
    if (cfg.max_seq_len == 0) cfg.max_seq_len = 2048;
    if (cfg.n_kv_heads == 0) cfg.n_kv_heads = cfg.n_heads;  // MHA fallback

    fclose(f);
    return cfg;
}

// ── Load weights via mmap + pin ──────────────────────────────────────────

bool load_gguf_weights(const std::string& path, void** weights, int64_t* size) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[gguf] Cannot open %s\n", path.c_str());
        return false;
    }

    struct stat st;
    fstat(fd, &st);
    *size = st.st_size;

    fprintf(stderr, "[gguf] Mapping %lld MB...\n", *size / (1024 * 1024));

    // mmap the entire file (read-only)
    void* mapped = mmap(nullptr, *size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (mapped == MAP_FAILED) {
        fprintf(stderr, "[gguf] mmap failed\n");
        return false;
    }

    // Advise kernel: we'll read sequentially during load, then random during inference
    madvise(mapped, *size, MADV_SEQUENTIAL);

    // On Jetson unified memory, mmap'd file is already in DRAM.
    // For GPU access, we need it pinned or use cudaHostRegister.
    // cudaHostRegister pins existing pages without copying.
    cudaError_t err = cudaHostRegister(mapped, *size, cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        fprintf(stderr, "[gguf] cudaHostRegister failed: %s (continuing without pin)\n",
                cudaGetErrorString(err));
        // Not fatal — GPU can still read via page faults (slower but works)
    }

    madvise(mapped, *size, MADV_RANDOM);  // switch to random access for inference

    *weights = mapped;
    fprintf(stderr, "[gguf] Loaded and pinned %lld MB\n", *size / (1024 * 1024));
    return true;
}

int64_t ModelConfig::weight_bytes() const {
    // Rough estimate based on param count and quant type
    int64_t params = (int64_t)n_layers *
        (4LL * hidden_dim * hidden_dim +       // Q, K, V, O projections
         2LL * hidden_dim * intermediate_dim);  // gate + up projections
    params += (int64_t)vocab_size * hidden_dim; // embedding + output

    // Q4_K_M: ~0.5 bytes per param + scales
    return params / 2 + params / 32 * 2;  // weights + scales
}

// ── Weight tensor mapping ────────────────────────────────────────────────
// GGUF stores tensor info (name, shape, offset) in the header.
// After loading the blob via mmap, we compute pointers into it.
//
// This is a simplified mapper — assumes standard Llama-style tensor names.
// A full implementation would parse the GGUF tensor info section.

ModelWeights map_weights(const void* blob, int64_t blob_size, const ModelConfig& cfg) {
    ModelWeights mw = {};
    mw.n_layers = cfg.n_layers;
    mw.layers = new LayerWeights[cfg.n_layers];
    memset(mw.layers, 0, sizeof(LayerWeights) * cfg.n_layers);

    // TODO: Parse GGUF tensor info to get actual offsets.
    // For now, this is a placeholder structure.
    // In production, iterate the tensor metadata section of GGUF,
    // match tensor names to our LayerWeights fields, and compute:
    //   ptr = (char*)blob + data_offset + tensor_offset
    //
    // Tensor name patterns (Llama-style):
    //   "token_embd.weight"           → tok_embd
    //   "blk.{i}.attn_q.weight"       → layers[i].wq
    //   "blk.{i}.attn_k.weight"       → layers[i].wk
    //   "blk.{i}.attn_v.weight"       → layers[i].wv
    //   "blk.{i}.attn_output.weight"  → layers[i].wo
    //   "blk.{i}.ffn_gate.weight"     → layers[i].w_gate
    //   "blk.{i}.ffn_up.weight"       → layers[i].w_up
    //   "blk.{i}.ffn_down.weight"     → layers[i].w_down
    //   "blk.{i}.attn_norm.weight"    → layers[i].rms_attn
    //   "blk.{i}.ffn_norm.weight"     → layers[i].rms_ffn
    //   "output_norm.weight"           → output_norm
    //   "output.weight"                → output

    fprintf(stderr, "[model] Weight tensor mapping: needs GGUF tensor info parser\n");
    fprintf(stderr, "[model] Allocated %d layer weight structs\n", cfg.n_layers);

    return mw;
}

}  // namespace jllm
