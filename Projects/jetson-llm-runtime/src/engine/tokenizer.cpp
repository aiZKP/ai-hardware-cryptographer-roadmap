// tokenizer.cpp — BPE tokenizer with GGUF vocabulary loading
//
// Reads vocabulary directly from GGUF metadata.
// Simple greedy BPE encode (not optimal but functional).
// For production: link against sentencepiece or use HF tokenizers C API.

#include "jllm_engine.h"
#include <cstdio>
#include <cstring>
#include <algorithm>

namespace jllm {

bool Tokenizer::load_from_gguf(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    // Read GGUF header
    uint32_t magic, version;
    uint64_t n_tensors, n_kv;
    fread(&magic, 4, 1, f);
    fread(&version, 4, 1, f);
    fread(&n_tensors, 8, 1, f);
    fread(&n_kv, 8, 1, f);

    if (magic != 0x46475547) { fclose(f); return false; }

    // Scan metadata for tokenizer entries
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t key_len;
        fread(&key_len, 8, 1, f);
        if (key_len > 1024) { fseek(f, key_len, SEEK_CUR); fseek(f, 4, SEEK_CUR); continue; }

        char key[1025] = {};
        fread(key, 1, key_len, f);

        uint32_t vtype;
        fread(&vtype, 4, 1, f);

        // tokenizer.ggml.tokens = array of strings
        if (strcmp(key, "tokenizer.ggml.tokens") == 0 && vtype == 9) { // GGUF_TYPE_ARRAY
            uint32_t arr_type;
            uint64_t arr_len;
            fread(&arr_type, 4, 1, f);
            fread(&arr_len, 8, 1, f);

            vocab.resize(arr_len);
            for (uint64_t t = 0; t < arr_len; t++) {
                uint64_t str_len;
                fread(&str_len, 8, 1, f);
                vocab[t].resize(str_len);
                fread(&vocab[t][0], 1, str_len, f);
            }
            fprintf(stderr, "[tokenizer] Loaded %zu tokens from GGUF\n", vocab.size());
        }
        // tokenizer.ggml.bos_token_id
        else if (strcmp(key, "tokenizer.ggml.bos_token_id") == 0 && vtype == 4) {
            fread(&bos_id, 4, 1, f);
        }
        // tokenizer.ggml.eos_token_id
        else if (strcmp(key, "tokenizer.ggml.eos_token_id") == 0 && vtype == 4) {
            fread(&eos_id, 4, 1, f);
        }
        else {
            // Skip value based on type
            switch (vtype) {
                case 0: fseek(f, 1, SEEK_CUR); break;   // uint8
                case 1: fseek(f, 1, SEEK_CUR); break;   // int8
                case 2: fseek(f, 2, SEEK_CUR); break;   // uint16
                case 3: fseek(f, 2, SEEK_CUR); break;   // int16
                case 4: fseek(f, 4, SEEK_CUR); break;   // uint32
                case 5: fseek(f, 4, SEEK_CUR); break;   // int32
                case 6: fseek(f, 4, SEEK_CUR); break;   // float32
                case 7: fseek(f, 1, SEEK_CUR); break;   // bool
                case 8: {                                 // string
                    uint64_t slen;
                    fread(&slen, 8, 1, f);
                    fseek(f, slen, SEEK_CUR);
                    break;
                }
                case 9: {                                 // array
                    uint32_t atype;
                    uint64_t alen;
                    fread(&atype, 4, 1, f);
                    fread(&alen, 8, 1, f);
                    // Skip array elements (rough)
                    for (uint64_t a = 0; a < alen; a++) {
                        if (atype == 8) {  // string array
                            uint64_t slen;
                            fread(&slen, 8, 1, f);
                            fseek(f, slen, SEEK_CUR);
                        } else if (atype == 6) fseek(f, 4, SEEK_CUR);
                        else if (atype == 4) fseek(f, 4, SEEK_CUR);
                        else fseek(f, 4, SEEK_CUR);  // guess
                    }
                    break;
                }
                case 10: fseek(f, 8, SEEK_CUR); break;  // uint64
                case 11: fseek(f, 8, SEEK_CUR); break;  // int64
                case 12: fseek(f, 8, SEEK_CUR); break;  // float64
                default: fseek(f, 8, SEEK_CUR); break;   // unknown, skip 8
            }
        }
    }

    fclose(f);
    fprintf(stderr, "[tokenizer] vocab=%zu, bos=%d, eos=%d\n", vocab.size(), bos_id, eos_id);
    return !vocab.empty();
}

// Simple greedy longest-match encode (not full BPE, but functional)
std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    tokens.push_back(bos_id);

    size_t pos = 0;
    while (pos < text.size()) {
        int best_id = -1;
        size_t best_len = 0;

        // Greedy: find longest matching token starting at pos
        for (int id = 0; id < (int)vocab.size(); id++) {
            const auto& tok = vocab[id];
            if (tok.size() > best_len && tok.size() <= text.size() - pos) {
                if (text.compare(pos, tok.size(), tok) == 0) {
                    best_id = id;
                    best_len = tok.size();
                }
            }
        }

        if (best_id >= 0) {
            tokens.push_back(best_id);
            pos += best_len;
        } else {
            // Unknown byte — encode as byte fallback
            // Most GGUF tokenizers have byte tokens like <0x41>
            char byte_tok[8];
            snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>", (unsigned char)text[pos]);
            for (int id = 0; id < (int)vocab.size(); id++) {
                if (vocab[id] == byte_tok) { tokens.push_back(id); break; }
            }
            pos++;
        }
    }

    return tokens;
}

std::string Tokenizer::decode(int token_id) const {
    if (token_id >= 0 && token_id < (int)vocab.size()) {
        const auto& tok = vocab[token_id];
        // Handle byte tokens like <0x0A> → actual byte
        if (tok.size() == 6 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x') {
            unsigned int byte_val;
            if (sscanf(tok.c_str(), "<0x%02X>", &byte_val) == 1)
                return std::string(1, (char)byte_val);
        }
        return tok;
    }
    return "";
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string result;
    for (int id : ids) {
        if (id != bos_id && id != eos_id)
            result += decode(id);
    }
    return result;
}

}  // namespace jllm
